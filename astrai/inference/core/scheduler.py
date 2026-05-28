import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch

from astrai.inference.core.cache import KVCache
from astrai.inference.core.executor import Executor
from astrai.inference.core.task import STOP, Task, TaskManager, TaskStatus
from astrai.model.automodel import AutoModel
from astrai.tokenize.tokenizer import AutoTokenizer

logger = logging.getLogger(__name__)


class InferenceScheduler:
    """Four-phase continuous batching loop: cleanup -> refill -> prefill -> decode."""

    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        max_batch_size: int = 16,
        max_seq_len: Optional[int] = None,
        max_prompt_len: int = 2048,
        page_size: int = 64,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        config = model.config

        if max_seq_len is not None:
            self.max_seq_len = max_seq_len
        elif config.max_len is not None:
            self.max_seq_len = config.max_len
        else:
            raise ValueError(
                "max_seq_len must be provided either as argument "
                "or in model config (config.max_len)"
            )
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

        n_pages = (
            max_batch_size * (self.max_seq_len + page_size) + page_size - 1
        ) // page_size

        self._page_cache = KVCache(
            config.n_layers,
            n_pages,
            page_size,
            config.n_kv_heads,
            config.dim // config.n_heads,
            self.device,
            self.dtype,
        )

        self._task_mgr = TaskManager(
            tokenizer=tokenizer,
            max_batch_size=max_batch_size,
            max_seq_len=self.max_seq_len,
            max_prompt_len=max_prompt_len,
        )

        self._executor = Executor(
            model=model,
            tokenizer=tokenizer,
            page_cache=self._page_cache,
            device=self.device,
            dtype=self.dtype,
        )

        self._running = False

    def add_task(self, prompt: str, **kwargs) -> str:
        return self._task_mgr.add_task(prompt, **kwargs)

    def remove_task(self, task_id: str):
        for task in self._task_mgr.remove_task(task_id):
            self._page_cache.task_free(task.task_id)

    def get_stats(self) -> Dict[str, Any]:
        return self._task_mgr.get_stats()

    def _run_generation_loop(self):
        stop_ids = self._task_mgr.tokenizer.stop_ids
        try:
            while self._running:
                finished = self._task_mgr.remove_finished_tasks(stop_ids)
                for task in finished:
                    self._page_cache.task_free(task.task_id)

                active = self._task_mgr.get_active_tasks()
                available = self._task_mgr.max_batch_size - len(active)
                if available > 0:
                    candidates = self._task_mgr.pull_candidates(available)
                    failed = []
                    for task in candidates:
                        if self._page_cache.task_alloc(task.task_id, task.prompt_ids):
                            self._task_mgr.activate(task)
                        else:
                            failed.append(task)
                    if failed:
                        self._task_mgr.return_to_waiting(failed)

                if not self._task_mgr.has_work():
                    self._task_mgr.wait_for_tasks(timeout=1.0)
                    continue

                to_prefill = [
                    t
                    for t in self._task_mgr.get_active_tasks()
                    if t.output_tokens == 0
                    and self._page_cache.task_cached(t.task_id) < len(t.prompt_ids)
                ]
                if to_prefill:
                    for t in to_prefill:
                        t.input_tokens = len(t.prompt_ids)

                    groups: Dict[Tuple[int, int], List[Task]] = {}
                    for t in to_prefill:
                        key = (
                            len(t.prompt_ids),
                            self._page_cache.task_cached(t.task_id),
                        )
                        groups.setdefault(key, []).append(t)

                    for (prompt_len, start_pos), group in groups.items():
                        self._executor.execute_prefill(group, prompt_len, start_pos)
                        start_logical_page = start_pos // self._page_cache.page_size
                        for t in group:
                            self._page_cache.task_record_hashes(
                                t.task_id,
                                t.prompt_ids,
                                start_logical_page=start_logical_page,
                            )

                pos_groups: Dict[int, List[Task]] = {}
                for t in self._task_mgr.get_active_tasks():
                    pos_groups.setdefault(t.next_pos, []).append(t)

                if pos_groups:
                    best_key = max(pos_groups, key=lambda k: len(pos_groups[k]))
                    group = sorted(pos_groups[best_key], key=lambda t: t.task_id)

                    valid: List[Task] = []
                    for t in group:
                        if self._page_cache.task_extend(t.task_id, t.next_pos):
                            valid.append(t)
                        else:
                            t.status = TaskStatus.ABORTED
                            if t.stream_callback:
                                t.stream_callback(STOP)

                    if valid:
                        next_tokens = self._executor.execute_decode(valid)

                        for t, ntok in zip(valid, next_tokens):
                            t.output_ids.append(ntok)
                            t.output_tokens += 1
                            pos = t.input_tokens + t.output_tokens
                            extend_ok = self._page_cache.task_extend(t.task_id, pos)
                            if t.stream_callback:
                                t.stream_callback(
                                    self._task_mgr.tokenizer.decode([ntok])
                                )
                            if not extend_ok:
                                t.status = TaskStatus.ABORTED
                                if t.stream_callback:
                                    t.stream_callback(STOP)

                        for t in valid:
                            if t.is_finished(stop_ids):
                                if t.stream_callback:
                                    t.stream_callback(STOP)

        except Exception as e:
            logger.error(f"Scheduler loop crashed: {e}", exc_info=True)
            for task in self._task_mgr.get_active_tasks():
                if task.stream_callback:
                    task.stream_callback(STOP)
                self._page_cache.task_free(task.task_id)
            for task in self._task_mgr.get_waiting_tasks():
                if task.stream_callback:
                    task.stream_callback(STOP)
            self._task_mgr.clear_queues()
            raise

    def start(self):
        if not self._running:
            self._running = True
            t = threading.Thread(target=self._run_generation_loop, daemon=True)
            t.start()
            self._loop_thread = t

    def stop(self):
        self._running = False
        self._task_mgr.wake()
        if hasattr(self, "_loop_thread"):
            self._loop_thread.join(timeout=2.0)
        for task in self._task_mgr.get_active_tasks():
            self._page_cache.task_free(task.task_id)
        self._task_mgr.clear_queues()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
