"""Unified inference engine."""

import gc
import logging
import threading
from typing import Any, Dict, Generator, List, Optional, Union

import torch
import torch.nn as nn

from astrai.tokenize.tokenizer import TextTokenizer
from astrai.inference.scheduler import InferenceScheduler

logger = logging.getLogger(__name__)


class GenerationRequest:
    """Request parameters for text generation."""

    def __init__(
        self,
        messages: List[Dict[str, str]],
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_len: int = 1024,
        stream: bool = False,
    ):
        self.messages = messages
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_len = max_len
        self.stream = stream

        self._validate()

    def _validate(self):
        """Validate request parameters."""
        if not (isinstance(self.top_k, int) and self.top_k >= 0):
            raise ValueError("top_k must be a non-negative integer")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be a float between 0.0 and 1.0")
        if not (isinstance(self.temperature, (int, float)) and self.temperature >= 0):
            raise ValueError("temperature must be a non-negative number")


class _StreamingResult:
    """Streaming result holder with event-based notification."""

    def __init__(self):
        self.tokens: List[str] = []
        self._event = threading.Event()
        self._lock = threading.Lock()

    def append(self, token: str):
        with self._lock:
            self.tokens.append(token)
        self._event.set()

    def pop_all(self) -> List[str]:
        with self._lock:
            tokens = self.tokens.copy()
            self.tokens.clear()
            if not tokens:
                self._event.clear()
            return tokens

    def wait(self, timeout: float = None) -> bool:
        return self._event.wait(timeout=timeout)


class _NonStreamingResult:
    """Non-streaming result holder with event-based completion notification."""

    def __init__(self, count: int):
        self.results: List[str] = [""] * count
        self.done_flags: List[bool] = [False] * count
        self._completed_count = 0
        self._event = threading.Event()
        self._lock = threading.Lock()

    def append(self, idx: int, token: str):
        with self._lock:
            if token == "[DONE]":
                if not self.done_flags[idx]:
                    self.done_flags[idx] = True
                    self._completed_count += 1
                    if self._completed_count == len(self.results):
                        self._event.set()
            else:
                self.results[idx] += token

    def is_all_done(self) -> bool:
        with self._lock:
            return all(self.done_flags)

    def wait(self, timeout: float = None) -> bool:
        return self._event.wait(timeout=timeout)

    def get_results(self) -> List[str]:
        with self._lock:
            return self.results.copy()


class InferenceEngine:
    """Unified inference engine for continuous batching."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: TextTokenizer,
        max_batch_size: int = 1,
        max_seq_len: Optional[int] = None,
    ):
        """
        Initialize inference engine with separate model and tokenizer.

        Args:
            model: The language model for inference (nn.Module, e.g., Transformer)
            tokenizer: The tokenizer for encoding/decoding text
            config: Model configuration
            max_batch_size: Maximum batch size for continuous batching
            max_seq_len: Maximum sequence length (defaults to config.max_len)
        """
        self.model = model
        self.tokenizer = tokenizer

        # Get device and dtype from model parameters
        try:
            first_param = next(model.parameters())
            device = first_param.device
            dtype = first_param.dtype
        except StopIteration:
            # Model has no parameters, use default device/dtype
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float32

        self.scheduler = InferenceScheduler(
            model=self.model,
            tokenizer=self.tokenizer,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

        self.kv_cache = self.scheduler.kv_cache
        self.seq_mask = self.scheduler.seq_mask

        self.scheduler.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Handle exceptions on exit."""
        self.shutdown()
        return False

    def generate(
        self,
        prompt: Union[str, List[str]],
        stream: bool = False,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        abort_on_exception: bool = True,
    ) -> Union[Generator[str, None, None], str, List[str]]:
        """Unified generation interface.

        Args:
            abort_on_exception: If True, abort the generation when consumer
                stops iterating (GeneratorExit/StopIteration). Default: True.
        """
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        if stream:
            return self._generate_streaming(
                prompts,
                is_batch,
                max_tokens,
                temperature,
                top_p,
                top_k,
                abort_on_exception,
            )
        else:
            return self._generate_non_streaming(
                prompts, is_batch, max_tokens, temperature, top_p, top_k
            )

    def generate_with_request(
        self, request: GenerationRequest
    ) -> Union[Generator[str, None, None], str, List[str]]:
        """Generate with GenerationRequest object."""
        # Use tokenizer's chat template with messages
        prompt = self.tokenizer.apply_chat_template(request.messages, tokenize=False)

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
        abort_on_exception: bool = True,
    ) -> Union[Generator[str, None, None], List[Generator[str, None, None]]]:
        """Generate with streaming output.

        Args:
            abort_on_exception: If True, abort the task when generator is
                stopped early by consumer (GeneratorExit/StopIteration).
        """
        if is_batch:
            raise NotImplementedError("Batch streaming is not implemented yet")

        result = _StreamingResult()

        task_id = self.scheduler.add_task(
            prompt=prompts[0],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream_callback=result.append,
        )

        def gen():
            try:
                while True:
                    tokens = result.pop_all()
                    for token in tokens:
                        if token == "[DONE]":
                            return
                        yield token
                    result.wait(timeout=0.05)
            except Exception:
                # Consumer stopped iterating - abort the task
                if abort_on_exception:
                    self.scheduler.remove_task(task_id)
                raise

        gen.task_id = task_id
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
        result = _NonStreamingResult(len(prompts))

        for i, p in enumerate(prompts):
            # Create closure to capture current index value using factory function
            def make_callback(idx):
                def callback(token):
                    result.append(idx, token)

                return callback

            self.scheduler.add_task(
                prompt=p,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stream_callback=make_callback(i),
            )

        result.wait()
        results = result.get_results()
        return results if is_batch else results[0]

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return self.scheduler.get_stats()

    def shutdown(self) -> None:
        """Shutdown the engine and release all resources."""
        self.scheduler.stop()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
