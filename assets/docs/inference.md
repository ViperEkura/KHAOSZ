# Inference

## KV Cache

At decode time, only the last query token matters. All previous K/V are cached to avoid recomputation:

$$
o_n = \sum_j \text{softmax}\left(\frac{q_n k_j}{\sqrt{d_k}}\right) v_j
$$

RoPE is applied **before** KV cache write, not after — otherwise position encoding drift occurs.

## KVCache System

Six classes (plus two helpers) working together:

```
KVCache (facade)
  ├── PagePool         orchestrates page allocation + prefix matching
  │     ├── Allocator   bitmask-based page allocator + ref-count + LRU eviction (inside PagePool)
  │     └── PrefixCache hash-based prefix matching (page_hash via polynomial hash) (inside PagePool)
  ├── TaskTable        maps task_id → page_table + cached token count
  ├── Storage          k_cache / v_cache tensors (n_layers × n_pages × page_size × n_kv_heads × head_dim)
  └── KvcacheView      bundles Storage + page_table + total_len for attention layers (returned by bind())
```

`KVCache.bind(page_table, total_len)` returns a `KvcacheView` used by attention layers via `write()` / `gather()`.

## Continuous Batching

`InferenceScheduler` runs a daemon thread with a 4-phase loop:

```
1. Cleanup → Remove finished tasks, free KV pages
2. Refill  → Pop from waiting_queue, task_alloc pages, activate
3. Prefill → Group by (prompt_len, start_pos), run full forward
4. Decode  → Pick largest same-position group, single-token forward
```

## Sampling (Strategy Pattern)

```
BaseSamplingStrategy (ABC)
  ├── TemperatureStrategy
  ├── TopKStrategy
  ├── TopPStrategy
  └── SamplingPipeline
```

`SamplingPipeline` composes them: Temperature → Top-K → Top-P → softmax → multinomial.  
`sample()` is a convenience shortcut for one-shot usage.

## Protocol Handlers (Strategy Pattern)

```python
class ProtocolHandler:  # concrete orchestrator
    def __init__(self, request, engine, builder): ...
    async def handle(self):
        prompt, ctx, stops = builder.prepare(request, engine)
        agen = engine.generate_async(prompt, ...)
        if stream: self._handle_stream(agen, ctx, stops)
        else:      return await self._handle_non_stream(agen, ctx, stops)
```

`ResponseBuilder` (ABC): `prepare()`, `format_stream_start()`, `format_chunk()`, `format_stream_end()`, `format_response()`.

`OpenAIResponseBuilder` → `/v1/chat/completions`, `AnthropicResponseBuilder` → `/v1/messages`.

Adding a protocol = one builder file, no handler subclassing needed.

## Engine & GenerateResult

```
InferenceEngine
  ├── generate(prompt, stream, ...) → str | List[str] | Generator
  ├── generate_with_request(req)    → same
  ├── generate_async(prompt, ...)   → AsyncGenerator
  ├── get_stats()                   → Dict
  └── shutdown()
```

`GenerateResult` uses `Condition` for non-streaming (`wait_completion()`) and `Event` for streaming (`wait()`). Stream callback is `cb(token)`.

## HTTP API

```
POST /v1/chat/completions   OpenAI
POST /v1/messages            Anthropic
GET  /health                 {"status":"ok","model_loaded":true}
GET  /stats                  scheduler statistics
```

### OpenAI

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'
```

Response:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1717000000,
  "model": "astrai",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
}
```

Streaming SSE: `object: "chat.completion.chunk"` — starts with role delta, then token chunks, ends with finish chunk + usage stats, then `data: [DONE]`.

### Anthropic

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"astrai","system":"You are helpful.","messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'
```

Supports `stop_sequences` and streaming via `event: content_block_delta`.

### GenerationRequest Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | List[dict] | required | Chat messages (role, content) |
| `top_k` | int | 50 | Top-k count |
| `top_p` | float | 1.0 | Nucleus threshold |
| `temperature` | float | 1.0 | Sampling temperature (> 0.0) |
| `max_tokens` | Optional[int] | None | Max generation length |
| `stream` | bool | False | Stream output |

## Engine API

```python
# Non-streaming
engine.generate("Hello", stream=False)          # -> str
engine.generate(["A", "B"], stream=False)       # -> List[str]

# Streaming
engine.generate("Hello", stream=True)           # -> Generator[str]
engine.generate(["A", "B"], stream=True)        # -> Generator[Tuple[int, str]]

# Async
async for token in engine.generate_async("Hello", ...):    # -> AsyncGenerator[str]
    print(token)
```

> Document Update Time: 2026-05-30
