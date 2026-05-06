"""Unified inference engine for continuous batching."""

import asyncio
import gc
import logging
import threading
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

import torch
import torch.nn as nn

from astrai.inference.scheduler import _STOP, InferenceScheduler
from astrai.tokenize import AutoTokenizer

logger = logging.getLogger(__name__)


class GenerationRequest:
    """Request parameters for text generation.

    Encapsulates messages, sampling parameters, and streaming preference
    for a single generation request.
    """

    def __init__(
        self,
        messages: List[Dict[str, str]],
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_len: int = 1024,
        stream: bool = False,
    ):
        """Initializes a generation request.

        Args:
            messages: Conversation history as list of {"role": ..., "content": ...}.
            top_k: Top-k sampling count (0 disables).
            top_p: Nucleus sampling probability threshold.
            temperature: Sampling temperature.
            max_len: Maximum tokens to generate.
            stream: Whether to return output as a token stream.
        """
        self.messages = messages
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_len = max_len
        self.stream = stream
        self._validate()

    def _validate(self):
        """Validates sampling parameter ranges."""
        if not (isinstance(self.top_k, int) and self.top_k >= 0):
            raise ValueError("top_k must be a non-negative integer")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be a float between 0.0 and 1.0")
        if not (isinstance(self.temperature, (int, float)) and self.temperature >= 0):
            raise ValueError("temperature must be a non-negative number")


class _Result:
    """Thread-safe token accumulator for streaming and non-streaming modes.

    Supports multiple concurrent generation tasks with per-index result tracking.
    Uses a threading.Event for efficient waiting on completion.
    """

    def __init__(self, count: int = 1):
        """Initializes the accumulator.

        Args:
            count: Number of concurrent generation tasks to track.
        """
        self._lock = threading.Lock()
        self._event = threading.Event()
        self.tokens: List[str] = []
        self.results: List[str] = [""] * count
        self._done: List[bool] = [False] * count
        self._completed = 0
        self._total = count

    def append(self, token: str, idx: int = 0):
        """Appends a token to the result buffer.

        In non-streaming mode, tokens are concatenated into results[idx].
        The sentinel _STOP marks a task as complete.

        Args:
            token: The decoded token string, or _STOP sentinel.
            idx: Index of the generation task this token belongs to.
        """
        with self._lock:
            self.tokens.append(token)
            if token is not _STOP:
                self.results[idx] += token
            else:
                if not self._done[idx]:
                    self._done[idx] = True
                    self._completed += 1
        self._event.set()

    def pop_all(self) -> List[str]:
        """Returns and clears all accumulated tokens.

        Returns:
            List of token strings since the last call.
        """
        with self._lock:
            out = self.tokens.copy()
            self.tokens.clear()
            if not out:
                self._event.clear()
            return out

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Blocks until new tokens arrive or the timeout expires.

        Args:
            timeout: Maximum wait time in seconds (None = infinite).

        Returns:
            True if the event was set (new data available), False on timeout.
        """
        return self._event.wait(timeout=timeout)

    def get_results(self) -> List[str]:
        """Returns all accumulated results for non-streaming mode.

        Returns:
            List of complete generated strings, one per task index.
        """
        with self._lock:
            return self.results.copy()


class InferenceEngine:
    """Unified inference engine backed by continuous-batching scheduler.

    Usage:
        with InferenceEngine(model, tokenizer) as engine:
            for token in engine.generate("hello", stream=True):
                print(token, end="")

            text = engine.generate("hello")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        max_batch_size: int = 1,
        max_seq_len: Optional[int] = None,
        max_prompt_len: int = 512,
        cache_capacity: int = 1000,
    ):
        """Initializes the engine and starts the scheduler background thread.

        Args:
            model: The language model (nn.Module, e.g. Transformer).
            tokenizer: Tokenizer for encoding/decoding.
            max_batch_size: Maximum concurrent tasks in the scheduler.
            max_seq_len: Maximum sequence length (defaults to model config).
            max_prompt_len: Maximum prompt tokens (longer prompts truncated).
            cache_capacity: Maximum prefix cache nodes.
        """
        try:
            first_param = next(model.parameters())
            device = first_param.device
            dtype = first_param.dtype
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float32

        self.model = model
        self.tokenizer = tokenizer

        self.scheduler = InferenceScheduler(
            model=self.model,
            tokenizer=self.tokenizer,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            max_prompt_len=max_prompt_len,
            cache_capacity=cache_capacity,
            device=device,
            dtype=dtype,
        )

        self.scheduler.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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
    ) -> Union[Generator[str, None, None], str, List[str]]:
        """Generates text from a prompt.

        Args:
            prompt: Single string or list of strings for batch generation.
            stream: If True, returns a generator yielding tokens one by one.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability threshold.
            top_k: Top-k sampling count (0 disables).

        Returns:
            Generator (stream=True), single string (non-stream, single prompt),
            or list of strings (non-stream, batch prompts).
        """
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

    def generate_async(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
    ) -> AsyncGenerator[str, None]:
        """Async streaming generator that does not block the event loop.

        Runs the synchronous generator in a background thread pool executor,
        yielding tokens to the async consumer as they arrive.

        Args:
            prompt: Input text to generate from.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling count.

        Yields:
            Decoded token strings as they are generated.
        """
        sync_gen = self._generate_streaming(
            [prompt], False, max_tokens, temperature, top_p, top_k
        )

        async def _agen():
            loop = asyncio.get_event_loop()
            while True:
                token = await loop.run_in_executor(None, self._next_token, sync_gen)
                if token is None:
                    break
                yield token

        return _agen()

    @staticmethod
    def _next_token(gen: Generator) -> Optional[str]:
        """Retrieves the next token from a synchronous generator.

        Args:
            gen: A synchronous generator yielding token strings.

        Returns:
            The next token, or None if the generator is exhausted.
        """
        try:
            return next(gen)
        except StopIteration:
            return None

    def generate_with_request(
        self, request: GenerationRequest
    ) -> Union[Generator[str, None, None], str, List[str]]:
        """Generates text from a structured GenerationRequest.

        Applies the chat template to the request's messages before generation.

        Args:
            request: A GenerationRequest with messages and parameters.

        Returns:
            Generator, string, or list of strings (see generate()).
        """
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
    ) -> Generator[str, None, None]:
        """Internal streaming generator.

        Polls the _Result accumulator in a loop, yielding tokens as they arrive.
        Cleans up the scheduler task on GeneratorExit.

        Args:
            prompts: List of prompts (only first is used; batch not yet supported).
            is_batch: If True, raises NotImplementedError.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling count.

        Yields:
            Decoded token strings.
        """
        if is_batch:
            raise NotImplementedError("Batch streaming not yet supported")

        result = _Result()

        task_id = self.scheduler.add_task(
            prompt=prompts[0],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream_callback=lambda tok: result.append(tok, 0),
        )

        def gen():
            try:
                while True:
                    tokens = result.pop_all()
                    for token in tokens:
                        if token is _STOP:
                            return
                        yield token
                    if not result.wait(timeout=0.05):
                        pass
            finally:
                self.scheduler.remove_task(task_id)

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
        """Internal non-streaming generator.

        Submits all prompts to the scheduler and waits for all to complete.

        Args:
            prompts: List of prompt strings.
            is_batch: Whether multiple prompts were provided.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling count.

        Returns:
            Single string for one prompt, list of strings for batch.
        """
        result = _Result(count=len(prompts))

        for i, p in enumerate(prompts):

            def make_cb(idx):
                return lambda tok: result.append(tok, idx)

            self.scheduler.add_task(
                prompt=p,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stream_callback=make_cb(i),
            )

        result.wait()
        res = result.get_results()
        return res if is_batch else res[0]

    def get_stats(self) -> Dict[str, Any]:
        """Returns current engine statistics.

        Returns:
            Dict with total_tasks, total_tokens, active_tasks, waiting_queue.
        """
        return self.scheduler.get_stats()

    def shutdown(self) -> None:
        """Shuts down the engine, stops the scheduler, and frees GPU memory."""
        self.scheduler.stop()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
