"""Inference scheduler for single-GPU continuous batching with paged KV cache."""

import logging
import threading
import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from astrai.inference.cache import STOP, PagedCache
from astrai.inference.sampling import sample
from astrai.model.automodel import AutoModel
from astrai.tokenize.tokenizer import AutoTokenizer

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task states in the continuous batching lifecycle."""

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


class Task:
    """Represents a single generation request with paged KV cache tracking."""

    def __init__(
        self,
        task_id: str,
        prompt_ids: List[int],
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        stream_callback: Optional[Callable[[str], None]] = None,
    ):
        self.task_id = task_id
        self.prompt_ids = prompt_ids
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        self.status = TaskStatus.PENDING
        self.output_ids: List[int] = []
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.page_table: List[int] = []
        self.n_pages: int = 0
        self._prefix_cached_tokens: int = 0
        self.arrival_time = time.time()
        self.finish_time: Optional[float] = None
        self.stream_callback = stream_callback
        self._pages_freed: bool = False

    @property
    def next_pos(self) -> int:
        return self.input_tokens + len(self.output_ids)

    def is_finished(self, stop_ids: List[int]) -> bool:
        if self.output_tokens >= self.max_tokens:
            return True
        if self.output_ids and self.output_ids[-1] in stop_ids:
            return True
        return False


class InferenceScheduler:
    """Continuous batching scheduler with paged KV cache.

    Runs a background generation loop with four phases per iteration:
      1. Cleanup finished tasks and release resources.
      2. Refill active batch from the waiting queue.
      3. Prefill newly activated tasks.
      4. Decode the largest same-position group of active tasks.
    """

    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        max_batch_size: int = 16,
        max_seq_len: Optional[int] = None,
        max_prompt_len: int = 512,
        page_size: int = 64,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        config = model.config

        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len or config.max_len
        self.max_prompt_len = max_prompt_len
        self.page_size = page_size
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

        n_kv_heads = config.n_kv_heads
        head_dim = config.dim // config.n_heads
        n_layers = config.n_layers
        n_pages = (
            max_batch_size * (self.max_seq_len + page_size) + page_size - 1
        ) // page_size

        self.page_cache = PagedCache(
            n_layers,
            n_pages,
            page_size,
            n_kv_heads,
            head_dim,
            self.device,
            self.dtype,
        )

        self.waiting_queue: List[Task] = []
        self.active_tasks: List[Task] = []

        self._running = False
        self._task_event = threading.Event()
        self._lock = threading.Lock()

        self._total_tasks = 0
        self._total_tokens = 0

    def _n_pages_for(self, n_tokens: int) -> int:
        return (n_tokens + self.page_size - 1) // self.page_size

    def add_task(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        task_id = f"task_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        prompt_ids = self.tokenizer.encode(prompt)
        if len(prompt_ids) > self.max_prompt_len:
            prompt_ids = prompt_ids[-self.max_prompt_len :]

        if len(prompt_ids) >= self.max_seq_len:
            if stream_callback:
                stream_callback(STOP)
            return task_id

        max_tokens = min(max_tokens, self.max_seq_len - len(prompt_ids))

        task = Task(
            task_id=task_id,
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream_callback=stream_callback,
        )

        with self._lock:
            self.waiting_queue.append(task)
            self._total_tasks += 1

        self._task_event.set()
        return task_id

    def remove_task(self, task_id: str) -> None:
        with self._lock:
            removed_active = [t for t in self.active_tasks if t.task_id == task_id]
            self.waiting_queue = [t for t in self.waiting_queue if t.task_id != task_id]
            self.active_tasks = [t for t in self.active_tasks if t.task_id != task_id]

        for task in removed_active:
            if not task._pages_freed:
                self._free_pages(task.page_table)
                task.page_table.clear()
                task.n_pages = 0
                task._pages_freed = True

    def _free_pages(self, indices: List[int]) -> None:
        for idx in indices:
            self.page_cache.free(idx)

    def _record_page_hashes(self, task: Task, start_logical_page: int = 0) -> None:
        full_pages = len(task.prompt_ids) // self.page_size
        for i in range(start_logical_page, full_pages):
            self.page_cache.record_page(task.page_table[i], task.prompt_ids, i)

    def _remove_finished_tasks(self) -> None:
        finished = []
        for task in self.active_tasks:
            if task.status == TaskStatus.ABORTED:
                task.finish_time = time.time()
                finished.append(task)
            elif task.is_finished(self.tokenizer.stop_ids):
                task.status = TaskStatus.FINISHED
                task.finish_time = time.time()
                finished.append(task)
                self._total_tokens += task.output_tokens

        for task in finished:
            if not task._pages_freed:
                self._free_pages(task.page_table)
                task.page_table.clear()
                task.n_pages = 0
                task._pages_freed = True

        self.active_tasks = [
            t
            for t in self.active_tasks
            if t.status not in (TaskStatus.FINISHED, TaskStatus.ABORTED)
        ]

    def _refill_active_batch(self) -> None:
        available = self.max_batch_size - len(self.active_tasks)
        if available <= 0:
            return

        to_add: List[Task] = []
        with self._lock:
            n = min(available, len(self.waiting_queue))
            for _ in range(n):
                to_add.append(self.waiting_queue.pop(0))

        failed: List[Task] = []
        for task in to_add:
            prompt_len = len(task.prompt_ids)

            hit_pages = self.page_cache.lookup_prefix(task.prompt_ids)
            cached_tokens = len(hit_pages) * self.page_size
            for p in hit_pages:
                self.page_cache.inc_ref(p)

            remaining = prompt_len - cached_tokens
            n_new = self._n_pages_for(remaining) if remaining > 0 else 0
            new_pages = self.page_cache.alloc_n(n_new) if n_new > 0 else []

            if remaining > 0 and not new_pages:
                for p in hit_pages:
                    self.page_cache.free(p)
                failed.append(task)
                continue

            task.page_table = hit_pages + new_pages
            task.n_pages = len(task.page_table)
            task._prefix_cached_tokens = cached_tokens
            task.status = TaskStatus.RUNNING
            self.active_tasks.append(task)

        if failed:
            with self._lock:
                self.waiting_queue[:0] = failed

    def _execute_prefill(
        self, tasks: List[Task], prompt_len: int, start_pos: int = 0
    ) -> None:
        tasks = sorted(tasks, key=lambda t: t.task_id)
        batch_sz = len(tasks)

        seq_len = prompt_len - start_pos
        input_ids = torch.empty(batch_sz, seq_len, dtype=torch.long, device=self.device)
        input_mask = torch.ones(
            batch_sz, prompt_len, dtype=torch.bool, device=self.device
        )

        for i, t in enumerate(tasks):
            input_ids[i] = torch.tensor(
                t.prompt_ids[start_pos:prompt_len], device=self.device
            )

        page_tables = self._make_page_table_tensor(tasks)

        with torch.inference_mode():
            self.model(
                input_ids,
                input_mask=input_mask,
                start_pos=start_pos,
                paged_cache=self.page_cache.bind(page_tables, total_len=prompt_len),
            )

        start_logical_page = start_pos // self.page_size
        for t in tasks:
            self._record_page_hashes(t, start_logical_page=start_logical_page)

    def _execute_decode(self, tasks: List[Task], start_pos: int) -> None:
        if not tasks:
            return

        tasks = sorted(tasks, key=lambda t: t.task_id)

        valid: List[Task] = []
        for t in tasks:
            if self._maybe_alloc_page(t, start_pos):
                valid.append(t)
            else:
                t.status = TaskStatus.ABORTED
                if t.stream_callback:
                    t.stream_callback(STOP)

        if not valid:
            return

        tasks = valid
        batch_sz = len(tasks)

        input_ids = torch.tensor(
            [t.output_ids[-1] if t.output_ids else t.prompt_ids[-1] for t in tasks],
            dtype=torch.long,
            device=self.device,
        )

        active_mask = torch.ones((batch_sz, 1), dtype=torch.bool, device=self.device)

        page_tables = self._make_page_table_tensor(tasks)
        total_len = start_pos + 1

        temperatures = torch.tensor([t.temperature for t in tasks], device=self.device)
        top_ks = torch.tensor([t.top_k for t in tasks], device=self.device)
        top_ps = torch.tensor([t.top_p for t in tasks], device=self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids.unsqueeze(1),
                input_mask=active_mask,
                paged_cache=self.page_cache.bind(page_tables, total_len=total_len),
                start_pos=start_pos,
            )
            logits = outputs["logits"][:, -1, :]

        next_tokens = sample(
            logits,
            temperature=temperatures,
            top_k=top_ks,
            top_p=top_ps,
        ).tolist()

        for t, ntok in zip(tasks, next_tokens):
            t.output_ids.append(ntok)
            t.output_tokens += 1
            pos = t.input_tokens + t.output_tokens
            self._maybe_alloc_page(t, pos)
            if t.stream_callback:
                t.stream_callback(self.tokenizer.decode([ntok]))

        for t in tasks:
            if t.is_finished(self.tokenizer.stop_ids):
                if t.stream_callback:
                    t.stream_callback(STOP)

    def _make_page_table_tensor(self, tasks: List[Task]) -> Tensor:
        max_pages = max(t.n_pages for t in tasks)
        rows = [t.page_table + [-1] * (max_pages - t.n_pages) for t in tasks]
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def _maybe_alloc_page(self, task: Task, pos: int) -> bool:
        needed = self._n_pages_for(pos + 1)
        while task.n_pages < needed:
            p = self.page_cache.alloc()
            if p < 0:
                return False
            task.page_table.append(p)
            task.n_pages += 1
        return True

    def _run_generation_loop(self) -> None:
        try:
            while self._running:
                self._remove_finished_tasks()
                self._refill_active_batch()

                if not self.active_tasks and not self.waiting_queue:
                    self._task_event.clear()
                    self._task_event.wait(timeout=1.0)
                    continue

                to_prefill = [t for t in self.active_tasks if t.output_tokens == 0]
                if to_prefill:
                    for t in to_prefill:
                        t.input_tokens = len(t.prompt_ids)

                    groups: Dict[Tuple[int, int], List[Task]] = {}
                    for t in to_prefill:
                        key = (len(t.prompt_ids), t._prefix_cached_tokens)
                        groups.setdefault(key, []).append(t)

                    for (prompt_len, start_pos), group in groups.items():
                        if start_pos < prompt_len:
                            self._execute_prefill(group, prompt_len, start_pos)

                pos_groups: Dict[int, List[Task]] = {}
                for t in self.active_tasks:
                    pos_groups.setdefault(t.next_pos, []).append(t)

                if pos_groups:
                    best_pos = max(pos_groups, key=lambda p: len(pos_groups[p]))
                    self._execute_decode(pos_groups[best_pos], best_pos)
        except Exception as e:
            logger.error(f"Scheduler loop crashed: {e}", exc_info=True)
            for task in self.active_tasks:
                if task.stream_callback:
                    task.stream_callback(STOP)
            for task in self.waiting_queue:
                if task.stream_callback:
                    task.stream_callback(STOP)
            raise

    def start(self) -> None:
        if not self._running:
            self._running = True
            t = threading.Thread(target=self._run_generation_loop, daemon=True)
            t.start()
            self._loop_thread = t

    def stop(self) -> None:
        self._running = False
        self._task_event.set()
        if hasattr(self, "_loop_thread"):
            self._loop_thread.join(timeout=2.0)
        self.waiting_queue.clear()
        self.active_tasks.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_tasks": self._total_tasks,
            "total_tokens": self._total_tokens,
            "active_tasks": len(self.active_tasks),
            "waiting_queue": len(self.waiting_queue),
        }
