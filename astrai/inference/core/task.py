import logging
import threading
import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from astrai.tokenize.tokenizer import AutoTokenizer

logger = logging.getLogger(__name__)

STOP = object()


class TaskStatus(Enum):
    """Task lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


class Task:
    """Single generation request: prompt, sampling params, output state."""

    def __init__(
        self,
        task_id: str,
        prompt_ids: List[int],
        max_tokens: Optional[int] = None,
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
        self.arrival_time = time.time()
        self.finish_time: Optional[float] = None
        self.stream_callback = stream_callback

    @property
    def next_pos(self) -> int:
        return self.input_tokens + len(self.output_ids)

    def is_finished(self, stop_ids: List[int]) -> bool:
        if self.max_tokens is not None and self.output_tokens >= self.max_tokens:
            return True
        if self.output_ids and self.output_ids[-1] in stop_ids:
            return True
        return False


class TaskManager:
    """Thread-safe task queues and lifecycle transitions (no page ops)."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_batch_size: int = 16,
        max_seq_len: int = 8192,
        max_prompt_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.max_prompt_len = max_prompt_len

        self.waiting_queue: List[Task] = []
        self.active_tasks: List[Task] = []

        self._task_event = threading.Event()
        self._lock = threading.Lock()

        self._total_tasks = 0
        self._total_tokens = 0

    def add_task(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
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

        if max_tokens is None:
            max_tokens = self.max_seq_len - len(prompt_ids)
        else:
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

    def remove_task(self, task_id: str) -> List[Task]:
        with self._lock:
            removed_active = [t for t in self.active_tasks if t.task_id == task_id]
            self.waiting_queue = [t for t in self.waiting_queue if t.task_id != task_id]
            self.active_tasks = [t for t in self.active_tasks if t.task_id != task_id]
        return removed_active

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_tasks": self._total_tasks,
            "total_tokens": self._total_tokens,
            "active_tasks": len(self.active_tasks),
            "waiting_queue": len(self.waiting_queue),
        }

    def remove_finished_tasks(self, stop_ids: List[int]) -> List[Task]:
        with self._lock:
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

            self.active_tasks = [
                t
                for t in self.active_tasks
                if t.status not in (TaskStatus.FINISHED, TaskStatus.ABORTED)
            ]
            return finished

    def pull_candidates(self, n: int) -> List[Task]:
        to_add: List[Task] = []
        with self._lock:
            take = min(n, len(self.waiting_queue))
            for _ in range(take):
                to_add.append(self.waiting_queue.pop(0))
        return to_add

    def activate(self, task: Task) -> None:
        task.status = TaskStatus.RUNNING
        with self._lock:
            self.active_tasks.append(task)

    def return_to_waiting(self, tasks: List[Task]) -> None:
        with self._lock:
            self.waiting_queue[:0] = tasks

    def has_work(self) -> bool:
        return bool(self.active_tasks or self.waiting_queue)

    def wait_for_tasks(self, timeout: float = 1.0) -> None:
        self._task_event.clear()
        self._task_event.wait(timeout=timeout)

    def get_active_tasks(self) -> List[Task]:
        with self._lock:
            return list(self.active_tasks)

    def clear_queues(self) -> None:
        with self._lock:
            self.waiting_queue.clear()
            self.active_tasks.clear()

    def wake(self) -> None:
        self._task_event.set()
