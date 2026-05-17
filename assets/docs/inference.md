# Inference

## KV Cache

At decode time, only the last query token matters. All previous K/V are cached to avoid recomputation:

$$
o_n = \sum_j \text{softmax}\left(\frac{q_n k_j}{\sqrt{d_k}}\right) v_j
$$

RoPE is applied **before** KV cache write, not after — otherwise position encoding drift occurs.

## KVCache System

Six classes working together:

```
KVCache (facade)
  ├── Allocator        bitmask-based page allocator + ref-count + LRU eviction
  ├── PrefixCache      hash-based prefix matching (page_hash via rolling hash)
  ├── PagePool         orchestrates Allocator + PrefixCache
  ├── TaskTable        maps task_id → page_table + cached token count
  ├── Storage          k_cache / v_cache tensors (n_layers × n_pages × page_size × n_kv_heads × head_dim)
  └── KvcacheView      bundles Storage + page_table + total_len for attention layers
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
BaseSamplingStrategy → TemperatureStrategy → TopKStrategy → TopPStrategy
```

`SamplingPipeline` composes them: Temperature → Top-K → Top-P → softmax → multinomial.  
`sample()` is a convenience shortcut for one-shot usage.

## Protocol Handlers (Template Method)

```python
class ProtocolHandler(ABC):
    def handle(self):
        ctx = StreamContext(...)
        agen = engine.generate_async(prompt, ...)
        if stream: self._handle_stream(agen, ctx)
        else:      self._handle_non_stream(agen, ctx)
```

Subclass hooks: `build_prompt()`, `create_response_id()`, `format_stream_start/token/end()`, `format_non_stream_response()`.

`OpenAIHandler` → `/v1/chat/completions`, `AnthropicHandler` → `/v1/messages`.

## Engine & GenerateResult

```
InferenceEngine
  ├── generate(prompt, stream, ...) → str | List[str] | Generator
  ├── generate_with_request(req)    → same
  └── generate_async(prompt, ...)   → AsyncGenerator
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
  "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
}
```

Streaming SSE: `data: {"choices":[{"delta":{"role":"assistant"}}]}` → token chunks → `data: [DONE]`

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
| `temperature` | float | 1.0 | Sampling temperature (0.0–2.0) |
| `top_p` | float | 1.0 | Nucleus threshold |
| `top_k` | int | 50 | Top-k count |
| `max_tokens` | int | None | Max generation length |
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
await engine.generate_async("Hello", ...)       # -> AsyncGenerator[str]
```

> Document Update Time: 2026-05-17
