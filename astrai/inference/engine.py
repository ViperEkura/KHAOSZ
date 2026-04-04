"""
Continuous Batching Inference Engine

This module provides the main continuous batching components:
- Task: Individual generation task with state management
- TaskStatus: Task state enumeration
- InferenceScheduler: Handles request scheduling and KV cache management
- InferenceEngine: Unified inference engine

Author: AstrAI Team
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from astrai.config import ModelConfig, ModelParameter
from astrai.tokenize.chat_template import HistoryType, build_prompt


# Use print for debugging instead of logging
def _debug(*args, **kwargs):
    pass


@dataclass
class GenerationRequest:
    """Request parameters for text generation."""

    top_k: int
    top_p: float
    temperature: float
    max_len: int

    query: Union[str, List[str]]
    history: Optional[Union[HistoryType, List[HistoryType]]] = None
    system_prompt: Optional[str] = None
    stream: bool = False

    def __post_init__(self):
        if not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if not isinstance(self.top_p, float) or self.top_p < 0.0 or self.top_p > 1.0:
            raise ValueError("top_p must be a float between 0.0 and 1.0")
        if not isinstance(self.temperature, float) or self.temperature < 0.0:
            raise ValueError("temperature must be a non-negative float")


class TaskStatus(Enum):
    """Task state enumeration for continuous batching.

    States:
        PENDING: Task is waiting to be scheduled
        RUNNING: Task is currently being processed
        FINISHED: Task completed successfully
        ABORTED: Task was cancelled or failed
    """

    PENDING = auto()
    RUNNING = auto()
    FINISHED = auto()
    ABORTED = auto()


@dataclass
class Task:
    """Individual task for continuous batching.

    Attributes:
        task_id: Unique task identifier
        prompt_ids: Input token IDs
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        status: Current task status
        output_ids: Generated token IDs
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens generated
        slot: Batch slot position (-1 if not assigned)
        arrival_time: Task arrival timestamp
        finish_time: Task completion timestamp
        stream_callback: Callback for streaming output
    """

    task_id: str
    prompt_ids: List[int]
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50

    status: TaskStatus = TaskStatus.PENDING
    output_ids: List[int] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    slot: int = -1
    arrival_time: float = field(default_factory=time.time)
    finish_time: Optional[float] = None

    stream_callback: Optional[Callable[[str], None]] = None

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
    """Inference scheduler with continuous batching support.

    Manages request scheduling, KV cache allocation, and generation loop.
    Supports dynamic batch composition where new requests can join at any time
    and completed requests are immediately released.
    """

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

        _debug(
            f"add_task: task_id={task_id}, prompt_len={len(prompt_ids)}, has_callback={stream_callback is not None}"
        )

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
        """Remove finished tasks from active batch and update caches."""
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
        """Execute Prefill phase: process entire prompt at once."""
        if not tasks:
            return

        _debug(f"_execute_prefill: processing {len(tasks)} tasks")

        # Sort tasks by slot to ensure correct batch indexing with KV cache
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

        # Create boolean mask for attention
        if self.tokenizer.pad_id is not None:
            input_mask = torch.ne(input_ids, self.tokenizer.pad_id)
        else:
            input_mask = torch.ones(
                input_ids.shape, dtype=torch.bool, device=self.device
            )

        _debug(
            f"_execute_prefill: input_ids shape={input_ids.shape}, max_len={max_len}"
        )

        try:
            with torch.inference_mode():
                outputs = self.model(
                    input_ids,
                    input_mask=input_mask,
                    start_pos=0,
                    persistent_key_values=self.kv_cache,
                )
            _debug(
                f"_execute_prefill: model forward done, output keys={outputs.keys() if hasattr(outputs, 'keys') else 'no keys'}"
            )
        except Exception as e:
            _debug(f"_execute_prefill: ERROR: {e}")
            raise

        for i, task in enumerate(tasks):
            task.input_tokens = prompt_lens[i]
            task.output_tokens = 0
            _debug(
                f"  task {task.task_id}: input_tokens={task.input_tokens}, output_tokens={task.output_tokens}"
            )

        for task in tasks:
            if task.slot >= 0:
                self.seq_mask[task.slot, : task.input_tokens] = True

        _debug(f"_execute_prefill: done, {len(tasks)} tasks marked as prefill complete")

    def _execute_decode(self, tasks: List[Task], start_pos: int) -> None:
        """Execute Decode phase: generate one token at a time."""
        if not tasks:
            return

        _debug(f"_execute_decode: processing {len(tasks)} tasks, start_pos={start_pos}")

        # Sort tasks by slot to ensure batch index aligns with slot (KV cache position)
        # Task at slot 0 → batch index 0 → KV stored at cache[0]
        # Task at slot 1 → batch index 1 → KV stored at cache[1]
        tasks = sorted(tasks, key=lambda t: t.slot)

        input_ids = torch.zeros(len(tasks), dtype=torch.long, device=self.device)
        for i, task in enumerate(tasks):
            if task.output_ids:
                input_ids[i] = task.output_ids[-1]
            else:
                input_ids[i] = task.prompt_ids[-1]

        input_tensor = input_ids.unsqueeze(1)  # shape: (batch, 1)

        # Create 2D attention mask: (batch, seq_len)
        active_mask = torch.ones((len(tasks), 1), dtype=torch.bool, device=self.device)
        _debug(
            f"_execute_decode: input_tensor shape={input_tensor.shape}, active_mask shape={active_mask.shape}"
        )

        try:
            with torch.inference_mode():
                outputs = self.model(
                    input_tensor,
                    input_mask=active_mask,
                    persistent_key_values=self.kv_cache,
                    start_pos=start_pos,
                )
            _debug(
                f"_execute_decode: model forward done, logits shape={outputs['logits'].shape}"
            )
            logits = outputs["logits"][:, -1, :]
        except Exception as e:
            _debug(f"_execute_decode: ERROR: {e}")
            raise

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

        _debug(f"_execute_decode: next_tokens={next_token_ids}")

        for task, next_token in zip(tasks, next_token_ids):
            task.output_ids.append(next_token)
            task.output_tokens += 1

            pos = task.input_tokens + task.output_tokens
            if task.slot >= 0 and pos < self.max_seq_len:
                self.seq_mask[task.slot, pos] = True

            if task.stream_callback:
                token_str = self.tokenizer.decode([next_token])
                task.stream_callback(token_str)

        # Check if any task reached max_tokens or stop token
        for task in tasks:
            if task.output_tokens >= task.max_tokens or (
                task.output_ids and task.output_ids[-1] in self.tokenizer.stop_ids
            ):
                _debug(
                    f"decode: task {task.task_id} finished, output_tokens={task.output_tokens}, max_tokens={task.max_tokens}"
                )
                if task.stream_callback:
                    task.stream_callback("[DONE]")

    def _run_generation_loop(self) -> None:
        """Main generation loop with continuous batching."""
        _debug("generation_loop: started")
        while self._running:
            self._remove_finished_tasks()
            self._refill_active_batch()

            if not self.active_tasks:
                self._task_event.wait(timeout=0.01)
                self._task_event.clear()
                continue

            _debug(
                f"generation_loop: active={len(self.active_tasks)}, waiting={len(self.waiting_queue)}"
            )

            new_tasks = [t for t in self.active_tasks if t.output_tokens == 0]
            decode_tasks = [t for t in self.active_tasks if t.output_tokens > 0]

            _debug(
                f"generation_loop: new_tasks={len(new_tasks)}, decode_tasks={len(decode_tasks)}"
            )
            for t in self.active_tasks:
                _debug(
                    f"  active task {t.task_id}: output_tokens={t.output_tokens}, input_tokens={t.input_tokens}"
                )

            if decode_tasks:
                start_pos = max(t.input_tokens + t.output_tokens for t in decode_tasks)
            else:
                start_pos = 0

            # First run prefill for new tasks
            if new_tasks:
                _debug(f"generation_loop: running prefill for {len(new_tasks)} tasks")
                self._execute_prefill(new_tasks)
                _debug(f"generation_loop: prefill done")

                # After prefill, convert these tasks to decode tasks in the same iteration
                decode_tasks = new_tasks
                start_pos = max(t.input_tokens for t in decode_tasks)
                _debug(
                    f"generation_loop: after prefill, decode_tasks={len(decode_tasks)}, start_pos={start_pos}"
                )

            if decode_tasks:
                _debug(
                    f"generation_loop: running decode for {len(decode_tasks)} tasks, start_pos={start_pos}"
                )
                self._execute_decode(decode_tasks, start_pos)
                _debug(f"generation_loop: decode done")

            if not self.active_tasks and not self.waiting_queue:
                time.sleep(0.001)

    def start(self) -> None:
        """Start the generation loop in a background thread."""
        if not self._running:
            _debug("InferenceScheduler.start: starting loop thread")
            self._running = True
            self._loop_thread = threading.Thread(target=self._run_generation_loop)
            self._loop_thread.daemon = True
            self._loop_thread.start()
            _debug("InferenceScheduler.start: loop thread started")

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


class InferenceEngine:
    """Unified inference engine for continuous batching.

    Provides a single interface for:
    - Single request generation (streaming or non-streaming)
    - Batch request generation (streaming or non-streaming)
    """

    def __init__(
        self,
        parameter: ModelParameter,
        max_batch_size: int = 16,
        max_seq_len: Optional[int] = None,
    ):
        self.model = parameter.model
        self.tokenizer = parameter.tokenizer
        self.config = parameter.config

        model_params = next(self.model.parameters())
        self.device = model_params.device
        self.dtype = model_params.dtype

        self.scheduler = InferenceScheduler(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            device=self.device,
            dtype=self.dtype,
        )

        self.kv_cache = self.scheduler.kv_cache
        self.seq_mask = self.scheduler.seq_mask

        _debug("InferenceEngine: starting scheduler")
        self.scheduler.start()
        _debug("InferenceEngine: scheduler started")

    def generate(
        self,
        prompt: Union[str, List[str]],
        stream: bool = False,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
    ) -> Union[Generator[str, None, None], str, List[str]]:
        """Unified generation interface."""
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        if stream:
            return self._generate_streaming(
                prompts, is_batch, max_tokens, temperature, top_p, top_k
            )
        else:
            return self._generate_non_streaming(
                prompts, is_batch, max_tokens, temperature, top_p, top_k
            )

    def generate_with_request(
        self, request: GenerationRequest
    ) -> Union[Generator[str, None, None], str, List[str]]:
        """Generate with GenerationRequest object."""
        prompt = build_prompt(request.query, request.history)

        return self.generate(
            prompt=prompt,
            stream=request.stream,
            max_tokens=request.max_len,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )

    def _generate_streaming(
        self,
        prompts: List[str],
        is_batch: bool,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Union[Generator[str, None, None], List[Generator[str, None, None]]]:
        """Generate with streaming output (synchronous)."""
        results = []
        _debug(f"_generate_streaming: prompts={len(prompts)}")

        if is_batch:
            raise NotImplementedError("Batch streaming is not implemented yet")

        def make_callback(idx: int):
            def cb(token: str):
                _debug(f"callback[{idx}]: token={token!r}")
                results.append(token)

            return cb

        for i, p in enumerate(prompts):
            _debug(f"_generate_streaming: adding task {i}: {p[:30]}...")
            self.scheduler.add_task(
                prompt=p,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stream_callback=make_callback(i),
            )

        def gen():
            _debug("generator: start yielding")
            while True:
                # Yield accumulated tokens
                while results:
                    token = results.pop(0)
                    if token == "[DONE]":
                        _debug("generator: got [DONE]")
                        return
                    _debug(f"generator: yielding {token!r}")
                    yield token
                time.sleep(0.01)

        return gen()

    def _generate_non_streaming(
        self,
        prompts: List[str],
        is_batch: bool,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Union[str, List[str]]:
        """Generate without streaming."""
        results = ["" for _ in range(len(prompts))]
        done_flags = [False] * len(prompts)
        lock = threading.Lock()

        def make_callback(idx: int):
            def cb(token: str):
                if token == "[DONE]":
                    done_flags[idx] = True
                else:
                    with lock:
                        results[idx] += token

            return cb

        for i, p in enumerate(prompts):
            self.scheduler.add_task(
                prompt=p,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stream_callback=make_callback(i),
            )

        while not all(done_flags):
            time.sleep(0.001)

        return results if is_batch else results[0]

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return self.scheduler.get_stats()

    def shutdown(self) -> None:
        """Shutdown the engine."""
        self.scheduler.stop()
