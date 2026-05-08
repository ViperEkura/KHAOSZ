"""Inference scheduler for single-GPU continuous batching.

Splits scheduling concerns across modules:
  - cache.py:   SlotAllocator (Object Pool), PrefixCacheManager
  - sampling.py: Strategy-pattern logit transformations
"""

import logging
import threading
import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from astrai.inference.cache import _STOP, PrefixCacheManager, SlotAllocator
from astrai.inference.sampling import sample
from astrai.model.automodel import AutoModel
from astrai.tokenize import AutoTokenizer

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task states in the continuous batching lifecycle."""

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


class Task:
    """Represents a single generation request within the scheduler.

    Tracks prompt tokens, generated output, sampling parameters,
    KV cache slot assignment, and prefix cache matching state.
    """

    __slots__ = (
        "task_id",
        "prompt_ids",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "status",
        "output_ids",
        "input_tokens",
        "output_tokens",
        "slot",
        "prefix_len",
        "arrival_time",
        "finish_time",
        "stream_callback",
    )

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
        """Initializes a new task.

        Args:
            task_id: Unique identifier for this task.
            prompt_ids: Tokenized prompt sequence.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability threshold.
            top_k: Top-k sampling count (0 disables).
            stream_callback: Optional callback invoked per decoded token.
        """
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
        self.prefix_len: int = 0
        self.arrival_time = time.time()
        self.finish_time: Optional[float] = None
        self.stream_callback = stream_callback

    @property
    def next_pos(self) -> int:
        """Returns the next KV cache position to write during decode."""
        return self.input_tokens + len(self.output_ids)

    def is_finished(self, stop_ids: List[int]) -> bool:
        """Checks whether the task has reached a stopping condition.

        Args:
            stop_ids: List of stop token IDs (e.g., EOS).

        Returns:
            True if max_tokens reached or the last output token is a stop ID.
        """
        if self.output_tokens >= self.max_tokens:
            return True
        if self.output_ids and self.output_ids[-1] in stop_ids:
            return True
        return False


class InferenceScheduler:
    """Continuous batching scheduler for single-GPU inference.

    Runs a background generation loop with four phases per iteration:
      1. Cleanup finished tasks and release resources.
      2. Refill active batch from the waiting queue.
      3. Prefill newly activated tasks (full, partial, or fully cached).
      4. Decode the largest same-position group of active tasks.

    Tasks at different positions are never batched together in decode,
    avoiding RoPE corruption from misaligned KV cache writes.
    """

    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        max_batch_size: int = 16,
        max_seq_len: Optional[int] = None,
        max_prompt_len: int = 512,
        cache_capacity: int = 1000,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initializes the scheduler and pre-allocates the KV cache.

        Args:
            model: The language model (must have config with n_layers, n_kv_heads, etc.).
            tokenizer: Tokenizer for encoding prompts and decoding outputs.
            max_batch_size: Maximum number of concurrent tasks.
            max_seq_len: Maximum sequence length (defaults to config.max_len).
            max_prompt_len: Maximum prompt tokens (longer prompts are truncated).
            cache_capacity: Maximum prefix cache node count.
            device: Target device for tensors.
            dtype: Data type for KV cache tensors.
        """
        config = model.config

        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len or config.max_len
        self.max_prompt_len = max_prompt_len
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

        self.prefix_cache = PrefixCacheManager(max_capacity=cache_capacity)

        n_kv_heads = config.n_kv_heads
        head_dim = config.dim // config.n_heads
        n_layers = config.n_layers
        self._n_layers = n_layers

        k_cache = torch.empty(
            (max_batch_size, self.max_seq_len, n_layers, n_kv_heads, head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        v_cache = torch.empty(
            (max_batch_size, self.max_seq_len, n_layers, n_kv_heads, head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.kv_cache = (k_cache, v_cache)

        self.seq_mask = torch.zeros(
            (max_batch_size, self.max_seq_len),
            device=self.device,
            dtype=torch.bool,
        )

        self.slot_allocator = SlotAllocator(max_batch_size)
        self.waiting_queue: List[Task] = []
        self.active_tasks: List[Task] = []

        self._running = False
        self._task_event = threading.Event()
        self._lock = threading.Lock()

        self._total_tasks = 0
        self._total_tokens = 0

    def _alloc_slot(self) -> int:
        """Allocates a free KV cache slot using the Object Pool.

        Returns:
            Slot index on success, -1 if all slots are occupied.
        """
        return self.slot_allocator.alloc()

    def _free_slot(self, idx: int) -> None:
        """Releases a KV cache slot back to the free pool.

        Args:
            idx: Slot index to free.
        """
        self.slot_allocator.free(idx)
        self.seq_mask[idx, :] = False

    def _try_reuse_slot(self, prefix: Tuple[int, ...]) -> Tuple[int, bool]:
        """Attempts to reuse a prefix-cached slot directly without KV copy.

        The slot is reusable only if it is free and its version matches
        the current slot version (no intervening allocation overwrote it).

        Args:
            prefix: The matched prefix token sequence.

        Returns:
            Tuple of (slot, True) on success, or (-1, False) if reuse is not possible.
        """
        _plen, cached_slot, cached_ver = self.prefix_cache.find(list(prefix))
        if cached_slot >= 0 and self.slot_allocator.is_free(cached_slot):
            if cached_ver == self.slot_allocator.version(cached_slot):
                self.slot_allocator.occupy(cached_slot)
                return cached_slot, True
        return -1, False

    def add_task(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Adds a generation task to the waiting queue.

        Encodes the prompt, queries the prefix cache for a match,
        and enqueues the task for the background generation loop.

        Args:
            prompt: Input text to generate from.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling count.
            stream_callback: Called per decoded token with the string representation.

        Returns:
            Unique task ID string.
        """
        task_id = f"task_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        prompt_ids = self.tokenizer.encode(prompt)

        if len(prompt_ids) > self.max_prompt_len:
            prompt_ids = prompt_ids[-self.max_prompt_len :]

        task = Task(
            task_id=task_id,
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream_callback=stream_callback,
        )

        prefix_len, _cached_slot, _cached_ver = self.prefix_cache.find(prompt_ids)
        task.prefix_len = prefix_len

        with self._lock:
            self.waiting_queue.append(task)
            self._total_tasks += 1

        self._task_event.set()
        return task_id

    def remove_task(self, task_id: str) -> None:
        """Removes a task from both the waiting queue and active tasks.

        Args:
            task_id: The task to remove.
        """
        with self._lock:
            removed_active = [t for t in self.active_tasks if t.task_id == task_id]
            self.waiting_queue = [t for t in self.waiting_queue if t.task_id != task_id]
            self.active_tasks = [t for t in self.active_tasks if t.task_id != task_id]

        for task in removed_active:
            if task.prefix_len > 0:
                prefix = tuple(task.prompt_ids[: task.prefix_len])
                self.prefix_cache.release(prefix)
            if task.prefix_len < len(task.prompt_ids):
                self.prefix_cache.release(tuple(task.prompt_ids))
            if task.slot >= 0:
                self._free_slot(task.slot)
            task.slot = -1

    def _remove_finished_tasks(self) -> None:
        """Removes all finished tasks from the active batch.

        Releases prefix cache references and frees the KV cache slot
        for each completed task.
        """
        finished = []
        for task in self.active_tasks:
            if task.is_finished(self.tokenizer.stop_ids):
                task.status = TaskStatus.FINISHED
                task.finish_time = time.time()
                finished.append(task)
                self._total_tokens += task.output_tokens

        for task in finished:
            if task.prefix_len > 0:
                prefix = tuple(task.prompt_ids[: task.prefix_len])
                self.prefix_cache.release(prefix)
            if task.prefix_len < len(task.prompt_ids):
                self.prefix_cache.release(tuple(task.prompt_ids))
            if task.slot >= 0:
                self._free_slot(task.slot)
            task.slot = -1

        self.active_tasks = [
            t for t in self.active_tasks if t.status != TaskStatus.FINISHED
        ]

    def _refill_active_batch(self) -> None:
        """Moves waiting tasks into the active batch, up to max_batch_size.

        Attempts direct slot reuse for prefix-matched tasks; falls back
        to allocating a fresh slot with KV cache copy when reuse is not possible.
        """
        available = self.max_batch_size - len(self.active_tasks)
        if available <= 0:
            return

        to_add: List[Task] = []
        with self._lock:
            n = min(available, len(self.waiting_queue))
            for _ in range(n):
                to_add.append(self.waiting_queue.pop(0))

        for i, task in enumerate(to_add):
            slot = -1
            reused = False
            if task.prefix_len > 0:
                prefix = tuple(task.prompt_ids[: task.prefix_len])
                cached_slot, reused = self._try_reuse_slot(prefix)
                if reused:
                    slot = cached_slot
            if slot < 0:
                slot = self._alloc_slot()
                if slot < 0:
                    with self._lock:
                        self.waiting_queue[:0] = to_add[i:]
                    break
            task.slot = slot
            task.status = TaskStatus.RUNNING
            self.active_tasks.append(task)

            if task.prefix_len > 0 and not reused:
                prefix = tuple(task.prompt_ids[: task.prefix_len])
                _plen, cached_slot, cached_ver = self.prefix_cache.find(list(prefix))
                if cached_slot >= 0 and cached_ver == self.slot_allocator.version(
                    cached_slot
                ):
                    self.prefix_cache.pin(prefix)
                    self.prefix_cache.copy_kv(
                        prefix, slot, self.kv_cache, self._n_layers
                    )
                else:
                    task.prefix_len = 0

    def _execute_prefill(self, tasks: List[Task]) -> None:
        """Runs batched prefill for newly activated tasks.

        Fully-cached tasks skip the model. Others are grouped by prefix_len
        so tasks sharing the same start_pos are batched together.
        """
        if not tasks:
            return

        groups: Dict[int, List[Task]] = {}
        for t in tasks:
            plen = len(t.prompt_ids)
            if t.prefix_len == plen:
                t.input_tokens = plen
                t.output_tokens = 0
                if t.slot >= 0:
                    self.seq_mask[t.slot, : t.input_tokens] = True
            else:
                groups.setdefault(t.prefix_len, []).append(t)

        for prefix_len, group in groups.items():
            slot_indices = torch.tensor([t.slot for t in group], device=self.device)
            self._execute_prefill_batch(group, prefix_len, slot_indices)

    def _execute_prefill_batch(
        self, tasks: List[Task], prefix_len: int, slot_indices: Tensor
    ) -> None:
        """Unified prefill for tasks sharing a common prefix_len.

        Args:
            tasks: Tasks with the same prefix_len < len(prompt_ids).
            prefix_len: Number of cached prefix tokens (0 for full prefill).
            slot_indices: Tensor of slot indices for KV cache mapping.
        """
        tasks = sorted(tasks, key=lambda t: t.slot)
        batch_sz = len(tasks)

        new_lens = [len(t.prompt_ids) - prefix_len for t in tasks]
        max_new_len = max(new_lens)

        input_ids = torch.zeros(
            batch_sz, max_new_len, dtype=torch.long, device=self.device
        )
        input_mask = torch.zeros(
            batch_sz, prefix_len + max_new_len, dtype=torch.bool, device=self.device
        )

        for i, t in enumerate(tasks):
            new_ids = t.prompt_ids[prefix_len:]
            nl = len(new_ids)
            if nl > 0:
                input_ids[i, :nl] = torch.tensor(new_ids, device=self.device)
            input_mask[i, : prefix_len + nl] = True

        with torch.inference_mode():
            self.model(
                input_ids,
                input_mask=input_mask,
                start_pos=prefix_len,
                persistent_key_values=self.kv_cache,
                slot_indices=slot_indices,
            )

        for i, t in enumerate(tasks):
            t.input_tokens = len(t.prompt_ids)
            t.output_tokens = 0
            self.prefix_cache.insert(
                tuple(t.prompt_ids), t.slot, self.slot_allocator.version(t.slot)
            )
            if t.slot >= 0:
                self.seq_mask[t.slot, : t.input_tokens] = True

    def _execute_decode(self, tasks: List[Task], start_pos: int) -> None:
        """Executes the decode phase for a group of tasks at the same position.

        Args:
            tasks: Tasks sharing the same next_pos value.
            start_pos: Common KV cache write position for the batch.
        """
        if not tasks:
            return

        tasks = sorted(tasks, key=lambda t: t.slot)
        batch_sz = len(tasks)
        slot_indices = torch.tensor([t.slot for t in tasks], device=self.device)

        input_ids = torch.zeros(batch_sz, dtype=torch.long, device=self.device)
        for i, t in enumerate(tasks):
            input_ids[i] = t.output_ids[-1] if t.output_ids else t.prompt_ids[-1]

        active_mask = torch.ones((batch_sz, 1), dtype=torch.bool, device=self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids.unsqueeze(1),
                input_mask=active_mask,
                persistent_key_values=self.kv_cache,
                start_pos=start_pos,
                slot_indices=slot_indices,
            )
            logits = outputs["logits"][:, -1, :]

        next_tokens = sample(
            logits,
            temperature=torch.tensor(
                [t.temperature for t in tasks], device=logits.device
            ),
            top_k=torch.tensor([t.top_k for t in tasks], device=logits.device),
            top_p=torch.tensor([t.top_p for t in tasks], device=logits.device),
        ).tolist()

        for t, ntok in zip(tasks, next_tokens):
            t.output_ids.append(ntok)
            t.output_tokens += 1
            pos = t.input_tokens + t.output_tokens
            if t.slot >= 0 and pos < self.max_seq_len:
                self.seq_mask[t.slot, pos] = True
            if t.stream_callback:
                t.stream_callback(self.tokenizer.decode([ntok]))

        for t in tasks:
            if t.is_finished(self.tokenizer.stop_ids):
                if t.stream_callback:
                    t.stream_callback(_STOP)

    def _run_generation_loop(self) -> None:
        """Main generation loop run in a daemon thread.

        Continuously cycles through cleanup, refill, prefill, and decode.
        Decode processes only the largest position group to ensure all
        batched tasks share the same KV cache write position.
        """
        try:
            while self._running:
                self._remove_finished_tasks()
                self._refill_active_batch()

                with self._lock:
                    if not self.active_tasks and not self.waiting_queue:
                        self._task_event.clear()
                        self._task_event.wait(timeout=0.01)
                        continue
                    tasks = self.active_tasks[:]

                to_prefill = [t for t in tasks if t.output_tokens == 0]
                if to_prefill:
                    self._execute_prefill(to_prefill)

                pos_groups: Dict[int, List[Task]] = {}
                for t in self.active_tasks:
                    pos_groups.setdefault(t.next_pos, []).append(t)

                if pos_groups:
                    best_pos = max(pos_groups, key=lambda p: len(pos_groups[p]))
                    self._execute_decode(pos_groups[best_pos], best_pos)

                if not self.waiting_queue and len(self.active_tasks) <= 1:
                    self._task_event.wait(timeout=0.005)
                    self._task_event.clear()
        except Exception as e:
            logger.error(f"Scheduler loop crashed: {e}", exc_info=True)
            for task in self.active_tasks:
                if task.stream_callback:
                    task.stream_callback(_STOP)
            for task in self.waiting_queue:
                if task.stream_callback:
                    task.stream_callback(_STOP)
            raise

    def start(self) -> None:
        """Starts the background generation loop thread."""
        if not self._running:
            self._running = True
            t = threading.Thread(target=self._run_generation_loop, daemon=True)
            t.start()

    def stop(self) -> None:
        """Stops the generation loop and releases all resources."""
        self._running = False
        self._task_event.set()
        if hasattr(self, "_loop_thread"):
            self._loop_thread.join(timeout=2.0)
        self.waiting_queue.clear()
        self.active_tasks.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_stats(self) -> Dict[str, Any]:
        """Returns current scheduler statistics.

        Returns:
            Dict with total_tasks, total_tokens, active_tasks, waiting_queue.
        """
        return {
            "total_tasks": self._total_tasks,
            "total_tokens": self._total_tokens,
            "active_tasks": len(self.active_tasks),
            "waiting_queue": len(self.waiting_queue),
        }
