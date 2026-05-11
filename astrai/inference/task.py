import logging
import threading
import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from astrai.inference.cache import STOP, PagedCache
from astrai.tokenize.tokenizer import AutoTokenizer

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


class Task:
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


class TaskManager:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        page_cache: PagedCache,
        max_batch_size: int = 16,
        max_seq_len: int = 8192,
        max_prompt_len: int = 512,
        page_size: int = 64,
    ):
        self.tokenizer = tokenizer
        self.page_cache = page_cache
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.max_prompt_len = max_prompt_len
        self.page_size = page_size

        self.waiting_queue: List[Task] = []
        self.active_tasks: List[Task] = []

        self._task_event = threading.Event()
        self._lock = threading.Lock()

        self._total_tasks = 0
        self._total_tokens = 0

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

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_tasks": self._total_tasks,
            "total_tokens": self._total_tokens,
            "active_tasks": len(self.active_tasks),
            "waiting_queue": len(self.waiting_queue),
        }

    def remove_finished_tasks(self, stop_ids: List[int]) -> None:
        finished = []
        for task in self.active_tasks:
            if task.status == TaskStatus.ABORTED:
                task.finish_time = time.time()
                finished.append(task)
            elif task.is_finished(stop_ids):
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

    def refill_active_batch(self) -> None:
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

    def has_work(self) -> bool:
        return bool(self.active_tasks or self.waiting_queue)

    def wait_for_tasks(self, timeout: float = 1.0) -> None:
        self._task_event.clear()
        self._task_event.wait(timeout=timeout)

    def wake(self) -> None:
        self._task_event.set()

    def _n_pages_for(self, n_tokens: int) -> int:
        return (n_tokens + self.page_size - 1) // self.page_size

    def _free_pages(self, indices: List[int]) -> None:
        for idx in indices:
            self.page_cache.free(idx)
