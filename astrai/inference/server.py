import torch
import uvicorn
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from astrai.config.param_config import ModelParameter
from astrai.inference.generator import GeneratorFactory, GenerationRequest

logger = logging.getLogger(__name__)

# Global model parameter (loaded once)
_model_param: Optional[ModelParameter] = None
_project_root = Path(__file__).parent.parent.parent
app = FastAPI(title="AstrAI Inference Server", version="0.1.0")


def load_model(
    param_path: Optional[Path] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load model parameters into global variable."""
    global _model_param
    if param_path is None:
        param_path = _project_root / "params"
    if not param_path.exists():
        raise FileNotFoundError(f"Parameter directory not found: {param_path}")
    _model_param = ModelParameter.load(param_path, disable_init=True)
    _model_param.to(device=device, dtype=dtype)
    logger.info(f"Model loaded on {device} with dtype {dtype}")


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


class StreamCompletionResponse(BaseModel):
    id: str = "chatcmpl-default"
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = "astrai"
    choices: List[Dict[str, Any]]


def convert_messages_to_history(
    messages: List[ChatMessage],
) -> tuple[Optional[str], Optional[List[Tuple[str, str]]]]:
    """Convert OpenAI-style messages to system_prompt and history."""
    system_prompt = None
    history: List[Tuple[str, str]] = []
    user_buffer = []
    assistant_buffer = []
    for msg in messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            if assistant_buffer:
                # Flush previous pair
                history.append(("".join(user_buffer), "".join(assistant_buffer)))
                user_buffer = []
                assistant_buffer = []
            user_buffer.append(msg.content)
        elif msg.role == "assistant":
            assistant_buffer.append(msg.content)
        else:
            logger.warning(f"Unknown role {msg.role}")
    # If there is a pending user message without assistant, treat as current query
    # We'll handle this later
    return system_prompt, history if history else None


@app.on_event("startup")
async def startup_event():
    """Load model on server startup."""
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model_param is not None}


@app.post("/v1/chat/completions", response_model=CompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    """OpenAI‑compatible chat completion endpoint.

    Supports both streaming and non‑streaming modes.
    """
    if _model_param is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Convert messages to query/history
    # For simplicity, assume the last user message is the query, previous messages are history
    system_prompt, history = convert_messages_to_history(request.messages)
    # Extract last user message as query
    user_messages = [m.content for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    query = user_messages[-1]
    # If there are multiple user messages, we could merge them, but for demo we keep simple

    gen_request = GenerationRequest(
        query=query,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_len=request.max_tokens,
        history=history,
        system_prompt=system_prompt,
        stream=request.stream,
    )

    if request.stream:
        # Return streaming response
        def generate_stream():
            generator = GeneratorFactory.create(_model_param, gen_request)
            for chunk in generator.generate(gen_request):
                # chunk is the cumulative response string
                # For OpenAI compatibility, we send incremental delta
                # For simplicity, we send the whole chunk each time
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        # Non‑streaming
        generator = GeneratorFactory.create(_model_param, gen_request)
        if gen_request.stream:
            # Should not happen because we set stream=False
            pass
        response_text = generator.generate(gen_request)
        # Build OpenAI‑style response
        import time

        resp = CompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
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
    """Simple generation endpoint compatible with existing GenerationRequest."""
    if _model_param is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Convert history format
    hist: Optional[List[Tuple[str, str]]] = None
    if history:
        hist = [
            (h[0], h[1]) for h in history
        ]  # assuming each item is [user, assistant]
    gen_request = GenerationRequest(
        query=query,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_len=max_len,
        history=hist,
        stream=stream,
    )
    if stream:

        def stream_generator():
            generator = GeneratorFactory.create(_model_param, gen_request)
            for chunk in generator.generate(gen_request):
                yield chunk + "\n"

        return StreamingResponse(stream_generator(), media_type="text/plain")
    else:
        generator = GeneratorFactory.create(_model_param, gen_request)
        result = generator.generate(gen_request)
        return {"response": result}


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server with uvicorn."""
    uvicorn.run("astrai.inference.server:app", host=host, port=port, reload=reload)
