"""Protocol handlers for OpenAI and Anthropic chat completion APIs.

Template Method + Builder patterns eliminate the 45% code duplication between
stream/non-stream branches and across protocol adapters.
"""

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from astrai.inference.engine import InferenceEngine


class SSEBuilder:
    """Fluent builder for SSE (Server-Sent Events) formatted chunks."""

    @staticmethod
    def event(data: Dict[str, Any], event: Optional[str] = None) -> str:
        lines: List[str] = []
        if event:
            lines.append(f"event: {event}")
        lines.append(f"data: {json.dumps(data, ensure_ascii=False)}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def done() -> str:
        return "data: [DONE]\n\n"


@dataclass
class StreamContext:
    """Shared state across the streaming generation lifecycle."""

    resp_id: str
    created: int
    model: str
    prompt_tokens: int
    completion_tokens: int = 0
    accumulated: str = ""


class StopChecker:
    """Scans accumulated text for stop sequence matches."""

    def __init__(self, sequences: List[str]):
        self._sequences = [s for s in sequences if s]

    def check(self, text: str) -> Optional[str]:
        for seq in self._sequences:
            if seq in text:
                return seq
        return None

    def trim(self, text: str, matched: str) -> str:
        idx = text.rfind(matched)
        return text[:idx] if idx != -1 else text

    @property
    def has_sequences(self) -> bool:
        return len(self._sequences) > 0


class ProtocolHandler(ABC):
    """Template-method base for API protocol handlers.

    Subclasses implement format hooks; the base class orchestrates the
    generate-async loop and SSE/JSON response construction.

    Lifecycle::

        handle()
          ├─ build_prompt()          # protocol-specific prompt assembly
          ├─ create_response_id()    # unique response identifier
          ├─ [stream]
          │   ├─ format_stream_start()
          │   ├─ format_stream_token()  × N
          │   │   └─ on_token() hook for stop-sequence interception
          │   └─ format_stream_end()
          └─ [non-stream]
              ├─ (accumulate tokens)
              └─ format_non_stream_response()
    """

    request_model: type[BaseModel]

    def __init__(self, request: BaseModel, engine: InferenceEngine):
        self.request = request
        self.engine = engine

    @abstractmethod
    def build_prompt(self) -> str:
        """Build the full prompt string from the request messages."""

    @abstractmethod
    def create_response_id(self) -> str:
        """Generate a unique response ID following the protocol convention."""

    @abstractmethod
    def format_stream_start(self, ctx: StreamContext) -> List[str]:
        """Yield SSE events that open the stream (role marker, metadata)."""

    @abstractmethod
    def format_stream_token(self, ctx: StreamContext, token: str) -> str:
        """Yield an SSE event for a single generated token."""

    @abstractmethod
    def format_stream_end(self, ctx: StreamContext) -> List[str]:
        """Yield SSE events that close the stream (finish reason, usage stats)."""

    @abstractmethod
    def format_non_stream_response(
        self, ctx: StreamContext, content: str
    ) -> Dict[str, Any]:
        """Build the JSON response body for non-streaming mode."""

    def get_stop_sequences(self) -> List[str]:
        return []

    def create_stop_checker(self) -> StopChecker:
        return StopChecker(self.get_stop_sequences())

    def on_token(
        self, ctx: StreamContext, token: str, stop_checker: StopChecker
    ) -> Optional[str]:
        """Hook after each token is appended to accumulated.

        Return a matched stop-sequence string to break the loop,
        or None to continue.

        """
        return None

    async def handle(self) -> Union[StreamingResponse, Dict[str, Any]]:
        ctx = StreamContext(
            resp_id=self.create_response_id(),
            created=int(time.time()),
            model=self.request.model,
            prompt_tokens=self._count_prompt_tokens(),
        )

        agen = self.engine.generate_async(
            prompt=self.build_prompt(),
            max_tokens=self.request.max_tokens,
            temperature=self.request.temperature,
            top_p=self.request.top_p,
            top_k=self.request.top_k,
        )

        if self.request.stream:
            return self._handle_stream(agen, ctx)
        else:
            return await self._handle_non_stream(agen, ctx)

    def _count_prompt_tokens(self) -> int:
        return len(self.engine.tokenizer.encode(self.build_prompt()))

    def _handle_stream(self, agen, ctx: StreamContext) -> StreamingResponse:
        stop_checker = self.create_stop_checker()

        async def event_stream():
            for event in self.format_stream_start(ctx):
                yield event

            async for token in agen:
                ctx.completion_tokens += 1
                ctx.accumulated += token

                matched = self.on_token(ctx, token, stop_checker)
                if matched:
                    break

                yield self.format_stream_token(ctx, token)

            for event in self.format_stream_end(ctx):
                yield event
            yield SSEBuilder.done()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def _handle_non_stream(self, agen, ctx: StreamContext) -> Dict[str, Any]:
        stop_checker = self.create_stop_checker()
        chunks: List[str] = []

        async for token in agen:
            ctx.completion_tokens += 1
            ctx.accumulated += token
            chunks.append(token)

            matched = self.on_token(ctx, token, stop_checker)
            if matched:
                break

        content = "".join(chunks)
        return self.format_non_stream_response(ctx, content)


def _extract_text_content(content: Union[str, List[Dict[str, Any]]]) -> str:
    """Extract plain text from an Anthropic content block (string or list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
    return ""


class OpenAIHandler(ProtocolHandler):
    """OpenAI-compatible /v1/chat/completions handler."""

    def build_prompt(self) -> str:
        messages = [
            {"role": m.role, "content": m.content} for m in self.request.messages
        ]
        return self.engine.tokenizer.apply_chat_template(messages, tokenize=False)

    def create_response_id(self) -> str:
        return f"chatcmpl-{uuid.uuid4().hex[:12]}"

    def format_stream_start(self, ctx: StreamContext) -> List[str]:
        return [
            SSEBuilder.event(
                {
                    "id": ctx.resp_id,
                    "object": "chat.completion.chunk",
                    "created": ctx.created,
                    "model": ctx.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        ]

    def format_stream_token(self, ctx: StreamContext, token: str) -> str:
        return SSEBuilder.event(
            {
                "id": ctx.resp_id,
                "object": "chat.completion.chunk",
                "created": ctx.created,
                "model": ctx.model,
                "choices": [
                    {"index": 0, "delta": {"content": token}, "finish_reason": None}
                ],
            }
        )

    def format_stream_end(self, ctx: StreamContext) -> List[str]:
        return [
            SSEBuilder.event(
                {
                    "id": ctx.resp_id,
                    "object": "chat.completion.chunk",
                    "created": ctx.created,
                    "model": ctx.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            ),
            SSEBuilder.event(
                {
                    "prompt_tokens": ctx.prompt_tokens,
                    "completion_tokens": ctx.completion_tokens,
                    "total_tokens": ctx.prompt_tokens + ctx.completion_tokens,
                }
            ),
        ]

    def format_non_stream_response(
        self, ctx: StreamContext, content: str
    ) -> Dict[str, Any]:
        return {
            "id": ctx.resp_id,
            "object": "chat.completion",
            "created": ctx.created,
            "model": ctx.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": ctx.prompt_tokens,
                "completion_tokens": ctx.completion_tokens,
                "total_tokens": ctx.prompt_tokens + ctx.completion_tokens,
            },
        }


class AnthropicHandler(ProtocolHandler):
    """Anthropic-compatible /v1/messages handler."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._yielded = ""

    def build_prompt(self) -> str:
        messages: List[Dict[str, str]] = []
        system = getattr(self.request, "system", None)
        if system:
            messages.append({"role": "system", "content": system})
        for m in self.request.messages:
            content = _extract_text_content(m.content)
            if content:
                messages.append({"role": m.role, "content": content})
        return self.engine.tokenizer.apply_chat_template(messages, tokenize=False)

    def create_response_id(self) -> str:
        return f"msg_{uuid.uuid4().hex[:24]}"

    def get_stop_sequences(self) -> List[str]:
        return getattr(self.request, "stop_sequences", None) or []

    def on_token(
        self, ctx: StreamContext, token: str, stop_checker: StopChecker
    ) -> Optional[str]:
        matched = stop_checker.check(ctx.accumulated)
        if not matched:
            return None

        ctx._stop_matched = matched
        trimmed = ctx.accumulated[: ctx.accumulated.rfind(matched)]
        unyielded = trimmed[len(self._yielded) :]
        if unyielded:
            ctx._last_yield_trimmed = unyielded
        return matched

    def format_stream_start(self, ctx: StreamContext) -> List[str]:
        return [
            SSEBuilder.event(
                {
                    "type": "message_start",
                    "message": {
                        "id": ctx.resp_id,
                        "type": "message",
                        "role": "assistant",
                        "model": ctx.model,
                        "content": [],
                        "usage": {"input_tokens": ctx.prompt_tokens},
                    },
                },
                event="message_start",
            ),
            SSEBuilder.event(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                event="content_block_start",
            ),
        ]

    def format_stream_token(self, ctx: StreamContext, token: str) -> str:
        self._yielded += token
        return SSEBuilder.event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": token},
            },
            event="content_block_delta",
        )

    def format_stream_end(self, ctx: StreamContext) -> List[str]:
        matched = getattr(ctx, "_stop_matched", None)
        events: List[str] = []
        last_yielded = getattr(ctx, "_last_yield_trimmed", "")
        if last_yielded:
            events.append(
                SSEBuilder.event(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": last_yielded},
                    },
                    event="content_block_delta",
                )
            )
        events.append(
            SSEBuilder.event(
                {"type": "content_block_stop", "index": 0},
                event="content_block_stop",
            )
        )
        events.append(
            SSEBuilder.event(
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "stop_sequence" if matched else "end_turn",
                        "stop_sequence": matched,
                    },
                    "usage": {"output_tokens": ctx.completion_tokens},
                },
                event="message_delta",
            )
        )
        events.append(SSEBuilder.event({"type": "message_stop"}, event="message_stop"))
        return events

    def format_non_stream_response(
        self, ctx: StreamContext, content: str
    ) -> Dict[str, Any]:
        matched = getattr(ctx, "_stop_matched", None)
        if matched:
            content = content[: content.rfind(matched)]
        return {
            "id": ctx.resp_id,
            "type": "message",
            "role": "assistant",
            "model": ctx.model,
            "content": [{"type": "text", "text": content}],
            "stop_reason": "stop_sequence" if matched else "end_turn",
            "stop_sequence": matched,
            "usage": {
                "input_tokens": ctx.prompt_tokens,
                "output_tokens": ctx.completion_tokens,
            },
        }
