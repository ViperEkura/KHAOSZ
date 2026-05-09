"""
OpenAI / Anthropic-compatible chat completion server backed by continuous-batching inference.
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from astrai.inference.engine import InferenceEngine
from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer

logger = logging.getLogger(__name__)

_project_root = Path(__file__).parent.parent.parent


class ServerState:
    def __init__(self):
        self.engine: Optional[InferenceEngine] = None
        self.config: Dict[str, Any] = {
            "device": "cuda",
            "dtype": torch.bfloat16,
            "param_path": None,
            "max_batch_size": 16,
        }


_state = ServerState()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completion API request body."""

    model: str = "astrai"
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=2048, ge=1)
    n: Optional[int] = Field(default=1, ge=1)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[int, float]] = None
    user: Optional[str] = None


class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class MessagesRequest(BaseModel):
    """Anthropic Messages API request body."""

    model: str = "astrai"
    max_tokens: int = Field(default=1024, ge=1)
    messages: List[AnthropicMessage]
    system: Optional[str] = None
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=1)
    stream: Optional[bool] = False
    stop_sequences: Optional[List[str]] = None


def configure_server(
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    param_path: Optional[Path] = None,
    max_batch_size: int = 16,
):
    _state.config.update(
        device=device,
        dtype=dtype,
        param_path=param_path,
        max_batch_size=max_batch_size,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model(
            param_path=_state.config["param_path"],
            device=_state.config["device"],
            dtype=_state.config["dtype"],
            max_batch_size=_state.config["max_batch_size"],
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    if _state.engine:
        _state.engine.shutdown()
        logger.info("Inference engine shutdown complete")


app = FastAPI(title="AstrAI Inference Server", version="0.2.0", lifespan=lifespan)


def load_model(
    param_path: Optional[Path] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_batch_size: int = 16,
):
    if param_path is None:
        param_path = _project_root / "params"
    if not param_path.exists():
        raise FileNotFoundError(f"Parameter directory not found: {param_path}")

    tokenizer = AutoTokenizer.from_pretrained(param_path)
    model = AutoModel.from_pretrained(param_path)
    model.to(device=device, dtype=dtype)
    logger.info(f"Model loaded on {device} with dtype {dtype}")

    _state.engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
    )
    logger.info(f"Inference engine initialized with max_batch_size={max_batch_size}")


def _get_engine() -> InferenceEngine:
    if _state.engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _state.engine


def _make_chunk(
    delta: Dict[str, str],
    finish_reason: Optional[str] = None,
    *,
    resp_id: str,
    created: int,
    model: str,
    index: int = 0,
) -> str:
    """Build a single SSE ``data:`` chunk matching OpenAI streaming format."""
    data = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _state.engine is not None,
    }


@app.get("/stats")
async def get_stats():
    return _get_engine().get_stats()


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint (streaming + non-streaming)."""
    engine = _get_engine()
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = request.model

    prompt = engine.tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in request.messages],
        tokenize=False,
    )
    prompt_tokens = len(engine.tokenizer.encode(prompt))

    if request.stream:
        agen = engine.generate_async(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )

        async def event_stream():
            yield _make_chunk(
                {"role": "assistant"},
                finish_reason=None,
                resp_id=resp_id,
                created=created,
                model=model,
            )

            completion_tokens = 0
            async for token in agen:
                yield _make_chunk(
                    {"content": token},
                    finish_reason=None,
                    resp_id=resp_id,
                    created=created,
                    model=model,
                )
                completion_tokens += 1

            yield _make_chunk(
                {},
                finish_reason="stop",
                resp_id=resp_id,
                created=created,
                model=model,
            )

            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            yield f"data: {json.dumps(usage, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    completion_tokens = 0
    chunks: List[str] = []
    agen = engine.generate_async(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
    )
    async for token in agen:
        chunks.append(token)
        completion_tokens += 1
    content = "".join(chunks)

    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_anthropic_sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _check_stop_sequence(text: str, stop_sequences: List[str]) -> Optional[str]:
    for seq in stop_sequences:
        if seq and seq in text:
            return seq
    return None


def _extract_text_content(content: Union[str, List[Dict[str, Any]]]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
    return ""


def _build_anthropic_messages(
    messages: List[AnthropicMessage], system: Optional[str]
) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    if system:
        result.append({"role": "system", "content": system})
    for m in messages:
        content = _extract_text_content(m.content)
        if content:
            result.append({"role": m.role, "content": content})
    return result


@app.post("/v1/messages")
async def create_message(request: MessagesRequest):
    """Anthropic-compatible Messages API endpoint (streaming + non-streaming)."""
    engine = _get_engine()
    resp_id = f"msg_{uuid.uuid4().hex[:24]}"
    model = request.model

    chat_messages = _build_anthropic_messages(request.messages, request.system)
    prompt = engine.tokenizer.apply_chat_template(chat_messages, tokenize=False)
    prompt_tokens = len(engine.tokenizer.encode(prompt))

    stop_sequences = request.stop_sequences or []

    if request.stream:
        agen = engine.generate_async(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )

        async def event_stream():
            yield _make_anthropic_sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": resp_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "usage": {"input_tokens": prompt_tokens},
                    },
                },
            )

            yield _make_anthropic_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )

            completion_tokens = 0
            accumulated = ""
            stopped_seq: Optional[str] = None
            async for token in agen:
                accumulated += token
                completion_tokens += 1

                matched = _check_stop_sequence(accumulated, stop_sequences)
                if matched:
                    text = accumulated[: accumulated.rfind(matched)]
                    stopped_seq = matched
                    if text:
                        yield _make_anthropic_sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": text},
                            },
                        )
                    break

                yield _make_anthropic_sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": token},
                    },
                )

            yield _make_anthropic_sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": 0},
            )

            stop_reason = "stop_sequence" if stopped_seq else "end_turn"
            yield _make_anthropic_sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": stopped_seq},
                    "usage": {"output_tokens": completion_tokens},
                },
            )

            yield _make_anthropic_sse(
                "message_stop",
                {"type": "message_stop"},
            )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    completion_tokens = 0
    chunks: List[str] = []
    agen = engine.generate_async(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
    )
    stopped_seq: Optional[str] = None
    accumulated = ""
    async for token in agen:
        chunks.append(token)
        completion_tokens += 1
        accumulated += token
        matched = _check_stop_sequence(accumulated, stop_sequences)
        if matched:
            stopped_seq = matched
            break

    content = "".join(chunks)
    if stopped_seq:
        idx = content.rfind(stopped_seq)
        if idx != -1:
            content = content[:idx]

    return {
        "id": resp_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": content}],
        "stop_reason": "stop_sequence" if stopped_seq else "end_turn",
        "stop_sequence": stopped_seq,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        },
    }


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    param_path: Optional[Path] = None,
    max_batch_size: int = 16,
):
    configure_server(
        device=device,
        dtype=dtype,
        param_path=param_path,
        max_batch_size=max_batch_size,
    )
    uvicorn.run(
        "astrai.inference.server:app",
        host=host,
        port=port,
        reload=reload,
    )
