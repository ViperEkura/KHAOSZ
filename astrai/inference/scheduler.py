"""Inference scheduler for continuous batching."""

import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor

from astrai.config import ModelConfig


class TaskStatus:
    """Task state for continuous batching."""

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


class Task:
    """Individual task for continuous batching."""

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
        self.slot: int = -1
        self.arrival_time = time.time()
        self.finish_time: Optional[float] = None

        self.stream_callback = stream_callback

    def is_finished(self, stop_ids: List[int]) -> bool:
        """Check if task is finished."""
        if self.output_ids and self.output_ids[-1] in stop_ids:
            return True
        if self.output_tokens >= self.max_tokens:
            return True
        return False


def apply_sampling_strategies(
    logits: Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    filter_value: float = -float("inf"),
) -> Tensor:
    """Apply sampling strategies to the logits tensor."""
    if temperature != 1.0:
        logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )

        logits[indices_to_remove] = filter_value

    return logits


class InferenceScheduler:
    """Inference scheduler with continuous batching support."""

    def __init__(
        self,
        model,
        tokenizer,
        config: ModelConfig,
        max_batch_size: int = 16,
        max_seq_len: Optional[int] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len or config.max_len
        self.device = device
        self.dtype = dtype

        num_heads = config.n_kv_heads
        head_dim = config.dim // config.n_heads
        n_layers = config.n_layers

        k_cache = torch.empty(
            (
                max_batch_size,
                self.max_seq_len,
                n_layers,
                num_heads,
                head_dim,
            ),
            device=device,
            dtype=dtype,
        )
        v_cache = torch.empty(
            (
                max_batch_size,
                self.max_seq_len,
                n_layers,
                num_heads,
                head_dim,
            ),
            device=device,
            dtype=dtype,
        )
        self.kv_cache = (k_cache, v_cache)
        self.seq_mask = torch.ones(
            (max_batch_size, self.max_seq_len), device=device, dtype=torch.bool
        )

        self.waiting_queue: List[Task] = []
        self.active_tasks: List[Task] = []

        self._running = False
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
        """Add a new task to the waiting queue."""
        task_id = f"task_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        prompt_ids = self.tokenizer.encode(prompt)

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
        """Remove a task from the scheduler."""
        with self._lock:
            self.waiting_queue = [t for t in self.waiting_queue if t.task_id != task_id]
            self.active_tasks = [t for t in self.active_tasks if t.task_id != task_id]

    def _remove_finished_tasks(self) -> None:
        """Remove finished tasks from active batch."""
        finished = []
        for task in self.active_tasks:
            if task.is_finished(self.tokenizer.stop_ids):
                task.status = TaskStatus.FINISHED
                task.finish_time = time.time()
                finished.append(task)
                self._total_tokens += task.output_tokens

        for task in finished:
            slot = task.slot
            if slot >= 0 and slot < len(self.active_tasks):
                self.seq_mask[slot, :] = False
            task.slot = -1

        self.active_tasks = [
            t for t in self.active_tasks if t.status != TaskStatus.FINISHED
        ]

    def _refill_active_batch(self) -> None:
        """Refill active batch with waiting tasks."""
        available_slots = self.max_batch_size - len(self.active_tasks)
        if available_slots <= 0:
            return

        with self._lock:
            to_add = []
            for _ in range(min(available_slots, len(self.waiting_queue))):
                if self.waiting_queue:
                    task = self.waiting_queue.pop(0)
                    task.status = TaskStatus.RUNNING
                    to_add.append(task)

            for task in to_add:
                for i in range(self.max_batch_size):
                    if all(t.slot != i for t in self.active_tasks):
                        task.slot = i
                        break
                self.active_tasks.append(task)

    def _execute_prefill(self, tasks: List[Task]) -> None:
        """Execute Prefill phase."""
        if not tasks:
            return

        tasks = sorted(tasks, key=lambda t: t.slot)

        prompt_lens = [len(task.prompt_ids) for task in tasks]
        max_len = max(prompt_lens)

        input_ids = torch.zeros(
            len(tasks), max_len, dtype=torch.long, device=self.device
        )
        for i, task in enumerate(tasks):
            if len(task.prompt_ids) > 0:
                input_ids[i, : len(task.prompt_ids)] = torch.tensor(
                    task.prompt_ids, device=self.device
                )

        if self.tokenizer.pad_id is not None:
            input_mask = torch.ne(input_ids, self.tokenizer.pad_id)
        else:
            input_mask = torch.ones(
                input_ids.shape, dtype=torch.bool, device=self.device
            )

        with torch.inference_mode():
            outputs = self.model(
                input_ids,
                input_mask=input_mask,
                start_pos=0,
                persistent_key_values=self.kv_cache,
            )

        for i, task in enumerate(tasks):
            task.input_tokens = prompt_lens[i]
            task.output_tokens = 0

        for task in tasks:
            if task.slot >= 0:
                self.seq_mask[task.slot, : task.input_tokens] = True

    def _execute_decode(self, tasks: List[Task], start_pos: int) -> None:
        """Execute Decode phase."""
        if not tasks:
            return

        tasks = sorted(tasks, key=lambda t: t.slot)

        input_ids = torch.zeros(len(tasks), dtype=torch.long, device=self.device)
        for i, task in enumerate(tasks):
            if task.output_ids:
                input_ids[i] = task.output_ids[-1]
            else:
                input_ids[i] = task.prompt_ids[-1]

        input_tensor = input_ids.unsqueeze(1)
        active_mask = torch.ones((len(tasks), 1), dtype=torch.bool, device=self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_tensor,
                input_mask=active_mask,
                persistent_key_values=self.kv_cache,
                start_pos=start_pos,
            )
            logits = outputs["logits"][:, -1, :]

        next_token_ids = []
        for i, task in enumerate(tasks):
            logit = logits[i : i + 1]
            logit = apply_sampling_strategies(
                logit,
                task.temperature,
                task.top_k,
                task.top_p,
            )
            probs = torch.softmax(logit, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_ids.append(next_token.item())

        for task, next_token in zip(tasks, next_token_ids):
            task.output_ids.append(next_token)
            task.output_tokens += 1

            pos = task.input_tokens + task.output_tokens
            if task.slot >= 0 and pos < self.max_seq_len:
                self.seq_mask[task.slot, pos] = True

            if task.stream_callback:
                token_str = self.tokenizer.decode([next_token])
                task.stream_callback(token_str)

        for task in tasks:
            if task.output_tokens >= task.max_tokens or (
                task.output_ids and task.output_ids[-1] in self.tokenizer.stop_ids
            ):
                if task.stream_callback:
                    task.stream_callback("[DONE]")

    def _run_generation_loop(self) -> None:
        """Main generation loop."""
        while self._running:
            self._remove_finished_tasks()
            self._refill_active_batch()

            if not self.active_tasks:
                self._task_event.wait(timeout=0.01)
                self._task_event.clear()
                continue

            new_tasks = [t for t in self.active_tasks if t.output_tokens == 0]
            decode_tasks = [t for t in self.active_tasks if t.output_tokens > 0]

            if decode_tasks:
                start_pos = max(t.input_tokens + t.output_tokens for t in decode_tasks)
            else:
                start_pos = 0

            if new_tasks:
                self._execute_prefill(new_tasks)
                decode_tasks = new_tasks
                start_pos = max(t.input_tokens for t in decode_tasks)

            if decode_tasks:
                self._execute_decode(decode_tasks, start_pos)

            if not self.active_tasks and not self.waiting_queue:
                self._task_event.wait(timeout=0.05)
                self._task_event.clear()

    def start(self) -> None:
        """Start the generation loop."""
        if not self._running:
            self._running = True
            self._loop_thread = threading.Thread(target=self._run_generation_loop)
            self._loop_thread.daemon = True
            self._loop_thread.start()

    def stop(self) -> None:
        """Stop the generation loop."""
        self._running = False
        if hasattr(self, "_loop_thread"):
            self._loop_thread.join(timeout=1.0)

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "total_tasks": self._total_tasks,
            "total_tokens": self._total_tokens,
            "active_tasks": len(self.active_tasks),
            "waiting_queue": len(self.waiting_queue),
        }
