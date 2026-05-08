"""
Inference Server with Continuous Batching Support

FastAPI server for inference with continuous batching.
Provides OpenAI-compatible chat completion endpoints.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """Encapsulates all server runtime state.

    Attributes:
        engine: The inference engine instance.
        model_param: The loaded model.
        config: Server configuration dict.
    """

    def __init__(self):
        self.engine: Optional[InferenceEngine] = None
        self.model_param: Optional[Any] = None
        self.config: Dict[str, Any] = {
            "device": "cuda",
            "dtype": torch.bfloat16,
            "param_path": None,
            "max_batch_size": 16,
        }


_state = ServerState()


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
    _state.model_param = AutoModel.from_pretrained(param_path)
    _state.model_param.to(device=device, dtype=dtype)
    logger.info(f"Model loaded on {device} with dtype {dtype}")

    _state.engine = InferenceEngine(
        model=_state.model_param,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
    )
    logger.info(f"Inference engine initialized with max_batch_size={max_batch_size}")


def _get_engine() -> InferenceEngine:
    if _state.engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _state.engine


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    max_tokens: int = Field(2048, ge=1)
    stream: bool = False
    system_prompt: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str = "chatcmpl-default"
    object: str = "chat.completion"
    created: int = 0
    model: str = "astrai"
    choices: List[Dict[str, Any]]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _state.model_param is not None,
        "engine_ready": _state.engine is not None,
    }


@app.get("/stats")
async def get_stats():
    return _get_engine().get_stats()


@app.post("/v1/chat/completions", response_model=CompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    engine = _get_engine()

    prompt = engine.tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in request.messages],
        tokenize=False,
    )

    if request.stream:
        agen = engine.generate_async(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )

        async def event_stream():
            async for token in agen:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': token}}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        result = engine.generate(
            prompt=prompt,
            stream=False,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )

        import time

        resp = CompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result},
                    "finish_reason": "stop",
                }
            ],
        )
        return resp


@app.post("/generate")
async def generate(
    query: str,
    history: Optional[List[List[str]]] = None,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    max_len: int = 2048,
    stream: bool = False,
):
    engine = _get_engine()

    messages = []
    if history:
        for h in history:
            if len(h) >= 2:
                messages.append({"role": "user", "content": h[0]})
                messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": query})

    prompt = engine.tokenizer.apply_chat_template(messages, tokenize=False)

    if stream:
        agen = engine.generate_async(
            prompt=prompt,
            max_tokens=max_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        async def text_stream():
            async for token in agen:
                yield token + "\n"

        return StreamingResponse(text_stream(), media_type="text/plain")
    else:
        result = engine.generate(
            prompt=prompt,
            stream=False,
            max_tokens=max_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return {"response": result}


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
