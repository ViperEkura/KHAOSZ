import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch

from astrai.inference.cache import PagedCache
from astrai.inference.executor import Executor
from astrai.inference.task import STOP, Task, TaskManager
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
        max_prompt_len: int = 512,
        page_size: int = 64,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        config = model.config

        self.max_seq_len = max_seq_len or config.max_len
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

        n_kv_heads = config.n_kv_heads
        head_dim = config.dim // config.n_heads
        n_layers = config.n_layers
        n_pages = (
            max_batch_size * (self.max_seq_len + page_size) + page_size - 1
        ) // page_size

        page_cache = PagedCache(
            n_layers,
            n_pages,
            page_size,
            n_kv_heads,
            head_dim,
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
            page_cache=page_cache,
            page_size=page_size,
            device=self.device,
            dtype=self.dtype,
        )

        self._running = False

    def add_task(self, prompt: str, **kwargs) -> str:
        return self._task_mgr.add_task(prompt, **kwargs)

    def remove_task(self, task_id: str) -> None:
        for task in self._task_mgr.remove_task(task_id):
            self._executor.free_task_pages(task)

    def get_stats(self) -> Dict[str, Any]:
        return self._task_mgr.get_stats()

    def _run_generation_loop(self) -> None:
        try:
            while self._running:
                finished = self._task_mgr.remove_finished_tasks(
                    self._task_mgr.tokenizer.stop_ids
                )
                for task in finished:
                    self._executor.free_task_pages(task)

                available = self._task_mgr.max_batch_size - len(
                    self._task_mgr.active_tasks
                )
                if available > 0:
                    candidates = self._task_mgr.pull_candidates(available)
                    failed = []
                    for task in candidates:
                        if self._executor.allocate_pages_for_activation(task):
                            self._task_mgr.activate(task)
                        else:
                            failed.append(task)
                    if failed:
                        self._task_mgr.return_to_waiting(failed)

                if not self._task_mgr.has_work():
                    self._task_mgr.wait_for_tasks(timeout=1.0)
                    continue

                to_prefill = [
                    t for t in self._task_mgr.active_tasks if t.output_tokens == 0
                ]
                if to_prefill:
                    for t in to_prefill:
                        t.input_tokens = len(t.prompt_ids)

                    groups: Dict[Tuple[int, int], List[Task]] = {}
                    for t in to_prefill:
                        key = (len(t.prompt_ids), self._executor.get_cached_tokens(t))
                        groups.setdefault(key, []).append(t)

                    for (prompt_len, start_pos), group in groups.items():
                        self._executor.execute_prefill(group, prompt_len, start_pos)

                pos_groups: Dict[int, List[Task]] = {}
                for t in self._task_mgr.active_tasks:
                    pos_groups.setdefault(t.next_pos, []).append(t)

                if pos_groups:
                    best_pos = max(pos_groups, key=lambda p: len(pos_groups[p]))
                    self._executor.execute_decode(pos_groups[best_pos], best_pos)

        except Exception as e:
            logger.error(f"Scheduler loop crashed: {e}", exc_info=True)
            for task in self._task_mgr.active_tasks:
                if task.stream_callback:
                    task.stream_callback(STOP)
            for task in self._task_mgr.waiting_queue:
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
        self._task_mgr.wake()
        if hasattr(self, "_loop_thread"):
            self._loop_thread.join(timeout=2.0)
        self._task_mgr.waiting_queue.clear()
        self._task_mgr.active_tasks.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
