"""Unified inference engine."""

import threading
from typing import Any, Dict, Generator, List, Optional, Union

from astrai.config import ModelParameter
from astrai.tokenize.chat_template import build_prompt

from astrai.inference.scheduler import InferenceScheduler


class GenerationRequest:
    """Request parameters for text generation."""

    def __init__(
        self,
        query: Union[str, List[str]],
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_len: int = 1024,
        history: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ):
        self.query = query
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_len = max_len
        self.history = history
        self.system_prompt = system_prompt
        self.stream = stream

        self._validate()

    def _validate(self):
        """Validate request parameters."""
        if not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if not isinstance(self.top_p, float) or self.top_p < 0.0 or self.top_p > 1.0:
            raise ValueError("top_p must be a float between 0.0 and 1.0")
        if not isinstance(self.temperature, float) or self.temperature < 0.0:
            raise ValueError("temperature must be a non-negative float")


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
        self.results: List[str] = ["" for _ in range(count)]
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

        self.scheduler.start()

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
        """Generate with streaming output."""
        if is_batch:
            raise NotImplementedError("Batch streaming is not implemented yet")

        result = _StreamingResult()

        self.scheduler.add_task(
            prompt=prompts[0],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream_callback=result.append,
        )

        def gen():
            while True:
                tokens = result.pop_all()
                for token in tokens:
                    if token == "[DONE]":
                        return
                    yield token
                result.wait(timeout=0.05)

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
            self.scheduler.add_task(
                prompt=p,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stream_callback=result.append,
            )

        result.wait()
        results = result.get_results()
        return results if is_batch else results[0]

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return self.scheduler.get_stats()

    def shutdown(self) -> None:
        """Shutdown the engine."""
        self.scheduler.stop()
