"""OpenAI chat completion response builder."""

import logging
import time
import uuid
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from astrai.inference.api.protocol import (
    GenContext,
    ResponseBuilder,
    StopInfo,
    sse_event,
)
from astrai.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)

_UNSUPPORTED_PARAMS = (
    "n",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
)


class OpenAIResponseBuilder(ResponseBuilder):
    def prepare(
        self, request: BaseModel, engine: InferenceEngine
    ) -> Tuple[str, GenContext, List[str]]:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = engine.tokenizer.apply_chat_template(messages, tokenize=False)

        self._resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self._model = request.model

        for param in _UNSUPPORTED_PARAMS:
            value = getattr(request, param, None)
            fields = getattr(type(request), "model_fields", {})
            default = fields[param].default if param in fields else None
            if value is not None and value != default:
                logger.warning(
                    "ChatCompletionRequest param '%s'=%r is not supported and will be ignored",
                    param,
                    value,
                )
            if value is not None and value != default:
                logger.warning(
                    "ChatCompletionRequest param '%s'=%r is not supported and will be ignored",
                    param,
                    value,
                )

        ctx = GenContext(
            resp_id=self._resp_id,
            created=int(time.time()),
            model=self._model,
            prompt_tokens=0,
        )
        stop = request.stop
        stop_sequences = (
            [] if stop is None else [stop] if isinstance(stop, str) else stop
        )
        return prompt, ctx, stop_sequences

    def format_stream_start(self, ctx: GenContext) -> List[str]:
        return [
            sse_event(
                {
                    "id": self._resp_id,
                    "object": "chat.completion.chunk",
                    "created": ctx.created,
                    "model": self._model,
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

    def format_chunk(self, token: str) -> str:
        return sse_event(
            {
                "id": self._resp_id,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": self._model,
                "choices": [
                    {"index": 0, "delta": {"content": token}, "finish_reason": None}
                ],
            }
        )

    def format_stream_end(self, ctx: GenContext, stop: StopInfo) -> List[str]:
        return [
            sse_event(
                {
                    "id": self._resp_id,
                    "object": "chat.completion.chunk",
                    "created": ctx.created,
                    "model": self._model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            ),
            sse_event(
                {
                    "prompt_tokens": ctx.prompt_tokens,
                    "completion_tokens": ctx.completion_tokens,
                    "total_tokens": ctx.prompt_tokens + ctx.completion_tokens,
                }
            ),
        ]

    def format_response(
        self, ctx: GenContext, content: str, stop: StopInfo
    ) -> Dict[str, Any]:
        return {
            "id": self._resp_id,
            "object": "chat.completion",
            "created": ctx.created,
            "model": self._model,
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
