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

# Global model parameter and engine (loaded once)
_engine: Optional[InferenceEngine] = None
_model_param: Optional[Any] = None
_project_root = Path(__file__).parent.parent.parent

# Server configuration (set before running server)
_server_config: Dict[str, Any] = {
    "device": "cuda",
    "dtype": torch.bfloat16,
    "param_path": None,
    "max_batch_size": 16,
}


def configure_server(
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    param_path: Optional[Path] = None,
    max_batch_size: int = 16,
):
    """Configure server settings before starting.

    Args:
        device: Device to load model on (e.g., "cuda", "cpu", "cuda:0")
        dtype: Data type for model weights (e.g., torch.bfloat16, torch.float16)
        param_path: Path to model parameters directory
        max_batch_size: Maximum batch size for continuous batching
    """
    _server_config["device"] = device
    _server_config["dtype"] = dtype
    _server_config["param_path"] = param_path
    _server_config["max_batch_size"] = max_batch_size


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global _model_param, _engine
    # Startup: Load model with configured settings
    try:
        load_model(
            param_path=_server_config["param_path"],
            device=_server_config["device"],
            dtype=_server_config["dtype"],
            max_batch_size=_server_config["max_batch_size"],
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    # Shutdown: Cleanup engine
    if _engine:
        _engine.shutdown()
        logger.info("Inference engine shutdown complete")


app = FastAPI(title="AstrAI Inference Server", version="0.2.0", lifespan=lifespan)


def load_model(
    param_path: Optional[Path] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_batch_size: int = 16,
):
    """Load model parameters and initialize inference engine."""
    global _model_param, _engine
    if param_path is None:
        param_path = _project_root / "params"
    if not param_path.exists():
        raise FileNotFoundError(f"Parameter directory not found: {param_path}")

    # Load tokenizer separately
    tokenizer = AutoTokenizer.from_pretrained(param_path)
    _model_param = AutoModel.from_pretrained(param_path)
    _model_param.to(device=device, dtype=dtype)
    logger.info(f"Model loaded on {device} with dtype {dtype}")

    # Initialize inference engine with separate model and tokenizer
    _engine = InferenceEngine(
        model=_model_param,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
    )
    logger.info(f"Inference engine initialized with max_batch_size={max_batch_size}")


# Pydantic models for API request/response
class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
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
        "model_loaded": _model_param is not None,
        "engine_ready": _engine is not None,
    }


@app.get("/stats")
async def get_stats():
    """Get inference engine statistics."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine.get_stats()


@app.post("/v1/chat/completions", response_model=CompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint.

    Supports both streaming and non-streaming modes with continuous batching.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Convert messages to prompt using engine's tokenizer
    # Extract system prompt if present, then apply chat template
    # Apply chat template directly with messages
    prompt = _engine.tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in request.messages],
        tokenize=False,
    )

    if request.stream:
        # Streaming response (use synchronous generator)
        generator = _engine.generate(
            prompt=prompt,
            stream=True,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )

        def generate_stream():
            for token in generator:
                if token == "[DONE]":
                    break
                yield f"data: {json.dumps({'choices': [{'delta': {'content': token}}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        # Non-streaming response
        result = _engine.generate(
            prompt=prompt,
            stream=False,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )

        # Build OpenAI-style response
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
    """Simple generation endpoint.

    Args:
        query: Input query string
        history: Conversation history as list of [user, assistant] pairs
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        max_len: Maximum tokens to generate
        stream: Enable streaming output

    Returns:
        dict: Generation result with response field
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Build messages for chat template
    messages = []
    if history:
        # Convert history format: List[List[str]] -> List[Dict]
        for h in history:
            if len(h) >= 2:
                messages.append({"role": "user", "content": h[0]})
                messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": query})

    # Use tokenizer's chat template
    prompt = _engine.tokenizer.apply_chat_template(messages, tokenize=False)

    if stream:
        # Synchronous streaming
        result = _engine.generate(
            prompt=prompt,
            stream=True,
            max_tokens=max_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        def stream_generator():
            for token in result:
                yield token + "\n"

        return StreamingResponse(stream_generator(), media_type="text/plain")
    else:
        result = _engine.generate(
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
    """Run the FastAPI server with uvicorn.

    Args:
        host: Server host address
        port: Server port number
        reload: Enable auto-reload for development
        device: Device to load model on (e.g., "cuda", "cpu", "cuda:0")
        dtype: Data type for model weights (e.g., torch.bfloat16, torch.float16)
        param_path: Path to model parameters directory
        max_batch_size: Maximum batch size for continuous batching
    """
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
