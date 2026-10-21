# Internals

Mathematical foundations and internal algorithms for AstrAI's training, inference, and preprocessing pipelines. For practical usage guides, see [Training](../guides/training.md), [Inference](../guides/inference.md), and [Preprocessing](../guides/preprocessing.md).

For training component ownership, startup and failure cleanup, see [Trainer internals](trainer/README.md).

## Contents

- [Autoregression & Causal Masking](#autoregression--causal-masking)
- [Rotary Position Embedding (RoPE)](#rotary-position-embedding-rope)
- [Training Loss Formulas](#training-loss-formulas)
- [Training Loop Internals](#training-loop-internals)
- [Callback Lifecycle](#callback-lifecycle)
- [KV Cache Mathematics](#kv-cache-mathematics)
- [Mask Algorithm Internals](#mask-algorithm-internals)
- [Gradient Accumulation Mechanics](#gradient-accumulation-mechanics)

## Autoregression & Causal Masking

Given a token sequence, the model predicts the probability of the next token. Each generated token is appended to the input and fed back, repeating until an end-of-sequence token or max length.

```
sequence : [[1, 2, 3, 4, 5, 6]]
input_ids: [[1, 2, 3, 4, 5]]
target_ids: [[2, 3, 4, 5, 6]]
```

A lower-triangular causal mask prevents attending to future positions:

```
[[0, -inf, -inf, -inf, -inf],
 [0,    0, -inf, -inf, -inf],
 [0,    0,    0, -inf, -inf],
 [0,    0,    0,    0, -inf],
 [0,    0,    0,    0,    0]]
```

This ensures position $i$ can only attend to positions $\leq i$, which is essential for autoregressive generation.

## Rotary Position Embedding (RoPE)

RoPE embeds position into Q/K vectors via complex rotation:

$$ q_i = R_i W_q x_i, \quad k_j = R_j W_k x_j, \quad q_i^T k_j = x_i^T W_q^T R_{i-j} W_k x_j $$

`RotaryEmbedding` pre-computes a cos/sin table `freqs_cis` of shape
`[max_len, dim/2, 2]` (f32 — `[cos, sin]` pairs). `forward()` returns
a `[batch, seq_len, dim/2, 2]` slice indexed by `position_ids`.
`apply_rotary_emb` applies the rotation: during training it uses torch
complex multiply (autograd-compatible); during inference it auto-dispatches
to a fused CUDA kernel when available. The key property is that the dot
product $q_i^T k_j$ depends only on the relative position $i - j$, not the
absolute positions.

**Critical for inference**: RoPE is applied **before** KV cache write, not after. If applied after caching, position encoding drift occurs because cached K/V would have stale rotation factors.

## Training Loss Formulas

### SEQ (Pre-training)

Next-token cross-entropy with optional label smoothing:

$$ L_{\text{PT}} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{\lt t}; \theta) $$

### SFT (Supervised Fine-Tuning)

Masked cross-entropy (`ignore_index=-100`) over response tokens only:

$$ L_{\text{SFT}} = -\frac{1}{L}\sum_{t=P+1}^{P+L} \log P(s_t \mid s_{\lt t}; \theta) $$

Prompt tokens are masked out via `loss_mask`; only response tokens contribute to the loss.

### DPO (Direct Preference Optimization)

Frozen reference model, preference margin via log-ratio:

$$ L_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right] $$

Parameters: `beta=0.1`, `reduction="sum"`.

### GRPO (Group Relative Policy Optimization)

Token-level PPO with group-normalized advantages:

$$ \text{Advantage}_i = \frac{r_i - \mu}{\sigma + \epsilon} $$

$$ L_{\text{GRPO}} = -\mathbb{E}_t\left[\min\left(\rho_t A,\; \text{clip}\left(\rho_t, 1-\epsilon, 1+\epsilon\right)A\right)\right] + \lambda \cdot \mathbb{E}_t\left[\frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1\right] $$

Where $\rho_t = \pi_\theta(a_t|s_t) / \pi_{\text{old}}(a_t|s_t)$ is the per-token importance sampling ratio. Online rollout records $\log \pi_{\text{old}}$ when each token is sampled and reuses those values directly during training; offline batches may fall back to a synchronized `old_model`. Advantages are derived from scalar per-response rewards, group-normalized, and broadcast across all response tokens. Only response tokens contribute to the loss.

Parameters: `group_size=4`, `clip_eps=0.2`, `kl_coef=0.01`. Optional
`clip_eps_low`/`clip_eps_high` values enable DAPO-style asymmetric clipping;
unset values inherit `clip_eps` for backward-compatible symmetric clipping.
The `loss_aggregation` switch selects token-level DAPO weighting or equal
sequence weighting. Optional `overlong_max_len`/`overlong_buffer_len` settings
add the DAPO linear soft-overlong penalty before group advantage normalization.

### MoE Load Balancing

MoE layers add a differentiable load-balancing term based on mean router probabilities and top-k expert assignment frequency. The training objective is:

$$ L = L_{\text{task}} + \lambda_{\text{MoE}} L_{\text{aux}} $$

`TrainConfig.moe_aux_loss_coef` controls $\lambda_{\text{MoE}}$ (default `0.01`). The unweighted and weighted auxiliary losses are logged separately.

## Training Loop Internals

Two-level loop: **epoch** → **batch**. Optimizer step fires every `grad_accum_steps` batches.

```
on_train_begin
  model.train()
  on_epoch_begin
    for batch in dataloader:
      with executor.accumulate(model):
        on_batch_begin
        loss_output = strategy(batch)
        context.loss = loss_output["loss"].item()
        context.metrics = loss_output["metrics"]
        stand_loss = loss_output["loss"] / executor.grad_accum_steps
        executor.backward(stand_loss)
        context.consumed_samples += (
            context.config.batch_per_device * context.dp_size
        )
        on_batch_end

        if executor.sync_gradients:
          before_optimizer_step
          strategy.optimizer_step(optimizer)
          optimizer.zero_grad()
          if scheduler:
            scheduler.step()
          after_optimizer_step
    on_epoch_end
on_train_end
```

The loss is divided by `grad_accum_steps` before `backward()`, so accumulated gradients sum to the correct mean.
Strategy metrics are detached and converted to Python `float` values before the
`LossOutput` is returned; only `LossOutput.loss` remains a differentiable tensor.

With `cp_size > 1`, `TrainContextBuilder` wraps the strategy in `CPStrategy`
(`astrai/parallel/cp.py`): the same `strategy(batch)` call shards the batch
across the cp group, runs the strategy's forward/reduction on the local slice,
and rescales the local token loss to the global mean. Metric and validation
reductions go through `ParallelTopology.reduce_mean`/`reduce_sum` over the dp
group — cp peers hold values derived from the same batch, so summing them
would double-count.

## Callback Lifecycle

| Hook | Fires | Default callback |
|------|-------|-----------------|
| `on_train_begin` | Before training starts | `GradientCheckpointingCallback`, `CheckpointCallback`, `MetricCallback` |
| `on_epoch_begin` | Start of each epoch | `ProgressBarCallback` |
| `on_batch_begin` | Every batch | — |
| `before_optimizer_step` | Every accumulation window, before `optimizer.step()` | `MetricCallback`, `ProgressBarCallback`, `GradientClippingCallback` |
| `on_batch_end` | Every batch | — |
| `after_optimizer_step` | Every accumulation window, after `optimizer.step()` and `scheduler.step()` | `CheckpointCallback` |
| `on_epoch_end` | End of each epoch | `MetricCallback`, `ProgressBarCallback` |
| `on_error` | On exception after context construction, or requested stop | `CheckpointCallback`, `MetricCallback` |
| `on_train_end` | Session teardown after context construction, including a failed start callback | `GradientCheckpointingCallback`, `CheckpointCallback`, `MetricCallback` |

Default callbacks (in order): `gradient_checkpointing` (activation checkpointing, optional), `checkpoint` (safetensors, rank-0), `metric` (JSONL + validation, rank-0), `progress_bar` (tqdm, rank-0), `gradient_clipping`. The gradient-clipping callback is always registered and always calls `executor.clip_grad_norm()` with the numeric `max_grad_norm` value.

## KV Cache Mathematics

At decode time, only the last query token matters. All previous K/V are cached to avoid recomputation:

$$ o_n = \sum_j \text{softmax}\left(\frac{q_n k_j}{\sqrt{d_k}}\right) v_j $$

The cache stores $k_j$ and $v_j$ for all previous positions. At each decode step, only $q_n$ (the current query) is computed fresh, and attention is computed against the cached K/V.

**RoPE ordering**: RoPE is applied to Q/K **before** writing to the KV cache. This is essential because:
1. The cached K values already contain the rotation for their original positions.
2. The new Q is rotated for its current position.
3. The dot product $q_n^T k_j$ then correctly depends on $n - j$ (relative position).

If RoPE were applied after caching, the rotation factors would be inconsistent between cached and new tokens.

### Cache Architecture

Three-layer separation (SGLang-inspired):

- **KVStorage**: Flat token-level buffers `[n_layers, size, n_kv_heads, head_dim]`.
- **ReqToTokenPool**: Index table `[req_idx, pos] → physical token slot`, shared across all layers.
- **Allocator + RadixCache**: Paged-mode allocation with ref-counting, LRU eviction, and exact page-aligned prefix sharing when `page_size > 1`.

`BlockPool` orchestrates all three (the per-request accounting facade is `KVCacheManager`); In contiguous mode (default), `req_to_token` is a trivial linear mapping. In paged mode, slots are allocated on demand. `RadixCache` walks exact token-page edges from the root, preserving parent-prefix context instead of treating a page hash as a globally unique key. Only complete pages whose KV entries have been materialized are shared; partial pages remain request-private and are released at completion. The final sampled token is excluded because it has not yet been decoded into KV.

`bind_tasks()` returns a `KVCache` dataclass with `kv_indptr`, a prefix-sum index over sequence lengths computed once per step and shared across layers. Attention layers access buffers directly — no methods, no abstraction.

### Attention dispatch

The extension separates runtime choice, per-operator policy, and exact kernels:

- `runtime/context.py` owns nested `op_backend(...)` scopes and snapshots.
- `runtime/operation.py` defines `OpSpec` and `OpDispatcher`.
- `backend/attention/dispatcher.py` selects CUDA, FlashAttention, or torch
  functions according to the call's `AttentionSpec`.
- `kernel/` adapters invoke one compiled kernel and raise if it is unavailable.

CUDA attention writes and reads the paged KV cache for supported inference
prefill and decode calls. FlashAttention supports compatible inference and
training calls; torch SDPA is the reference fallback. Automatic selection
tries CUDA, FlashAttention, then torch. Explicit arguments and contexts are
strict; `ASTR_OPS` and `set_op` are soft process defaults.

```python
from astrai.extension import attention, op_backend

with op_backend(attention="torch"):
    output = attention(q, k, v)
```

Create an inference scheduler inside the desired scope to capture that choice
for its worker. Rotary embedding dispatches independently and uses torch
autograd whenever backward is needed. CUDA Graph eligibility is checked in
the inference worker against its captured attention choice.

## Mask Algorithm Internals

### Template mode (`template: true`)

1. Prepend BOS token (masked)
2. For each message in the field's array:
   1. Render through `chat_template` for that single message
   2. Encode rendered text
   3. Apply mask rule for the message's role

### Non-template mode

Encode the field value as text. Mask value is 1 (train) or 0 (mask) per the section's `action`.

### Text config detection

When no section uses `template` and all sections have `action: "train"`, the builder omits `loss_mask` from the output — all tokens are trained.

### Position ID strategies

| Mode | Behavior |
|------|----------|
| `none` | No position IDs generated |
| `doc_reset` | Reset position to 0 at each document boundary in packed sequences |
| `continuous` | Continuous position IDs across packed documents |

Default is `doc_reset`, which ensures each document in a packed bin starts from position 0, preventing position encoding drift between unrelated documents.

## Gradient Accumulation Mechanics

Three cooperating layers enable gradient accumulation:

1. **`GradientState`** — tracks the micro-step counter. Fires `sync_gradients=True` every `grad_accum_steps` micro-batches. The counter is incremented at the **start** of `accumulate()`, before the forward pass.

2. **`executor._no_sync(model)`** — suppresses gradient synchronization on non-sync micro-steps:
   - `NoneExecutor`: `nullcontext` (nothing to skip)
   - `DDPExecutor`: `model.no_sync()` (PyTorch's built-in — skips all-reduce of gradient buckets)
   - `FSDPExecutor`: `set_requires_gradient_sync(False, recurse=True)` on each `FSDPModule` (FSDP2's native mechanism)

3. **`AccumOptimizer` / `AccumScheduler`** — wrap the real optimizer/scheduler. `step()` and `zero_grad()` are gated on `sync_gradients` — they only forward to the inner optimizer when the sync flag is True.

The loss is divided by `grad_accum_steps` before `backward()`, so gradients sum to the correct mean across micro-steps. `consumed_samples` increments by `batch_per_device * dp_size` every micro-batch (cp ranks share a batch, so they do not scale it).

### Effective batch size

```
effective_batch = dp_size × batch_per_device × grad_accum_steps
```

### Total optimizer steps

```
samples_per_replica = ceil(dataset_len / dp_size)
batches_per_replica  = ceil(samples_per_replica / batch_per_device)
total_steps          = (batches_per_replica // grad_accum_steps) * n_epoch
```

This accounts for data-parallel sharding — each dp replica processes `1/dp_size` of the dataset.

> Document Update Time: 2026-09-05
