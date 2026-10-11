# Quantize

FP8 quantization has two explicit entrypoints. `quantize` scales and casts
the requested output orientations. `quantize_scaled` also measures the raw
input amax, updates a history window and publishes the next scale in the
same kernel. Neither entry owns a cache, history cursor or training state.

## Interfaces and math

```python
quantize(x, inv_scale, *, dtype=None, tdtype=None)
quantize_scaled(x, inv_scale, ring_state, *, hist_len, hist_idx,
                dtype=None, tdtype=None, margin=0, fp8_max=None,
                next_scale=None, next_inv_scale=None)
```

Both return `(out, out_t, amax)`. `dtype` requests an output with x's shape;
`tdtype` requests the contiguous `[cols, rows]` transpose after folding the
leading axes into rows. At least one dtype is required. An absent dtype
produces `None`; it does not inherit the other dtype. Input types are BF16,
FP16 and FP32; output types are E4M3FN and E5M2, independently selected.
Transposed output requires a 2D+ input. Empty inputs preserve the original
output shape and use the existing `[1, 0]` empty transpose convention.

`inv_scale` is a float32 device scalar. Both operations compute
`cast(x * inv_scale)`; mixed output formats use the same multiplier.
Pure `quantize` returns `amax=None`.

```python
out, _, _ = quantize(x, inv_scale, dtype=torch.float8_e4m3fn)
_, out_t, _ = quantize(x, inv_scale, tdtype=torch.float8_e5m2)
out, out_t, _ = quantize(x, inv_scale,
                         dtype=torch.float8_e4m3fn, tdtype=torch.float8_e5m2)
```

`quantize_scaled` requires a contiguous float32 CUDA ring on x's device.
Its last finishing block performs the history update and publication:

```text
hist[hist_idx] = max(abs(x))                 # zero for an empty input
next_scale = max(max(hist) / fp8_max / 2**margin, 1e-12)
next_inv_scale = reciprocal(next_scale)     # correctly rounded __frcp_rn
```

The range defaults to 448 for E4M3FN or 57344 for E5M2, following `dtype`,
or `tdtype` for a transpose-only request. An explicit `fp8_max` overrides
that choice. Margin retains AstrAI's division by `2**margin` and the kernel
retains the existing calculation order and reciprocal rounding.

Provide `next_scale` and `next_inv_scale` together to publish into external
scalar tensors; otherwise publication uses ring pair zero. Scalar tensors
and publication slots must share the input device. The returned amax aliases
the ring slot and changes on subsequent updates. The entry does not advance
`hist_idx` or switch the current scale pair.

The ring layout remains:

```text
[hist n | scale0 | recip0 | amax | done | scratch 32 | scale1 | recip1]
```

A standalone ring may omit the final pair. `hist_len` is explicit because
the optional publication pair makes it ambiguous from numel. The Torch
boundary validates shape, dtype, device, capacity and integer range before
converting geometry and history indices to native `int`. Casting needs
sm_80+; FP8 tensor-core GEMM separately requires sm_89+.

## Parameters and dependencies

The two entries use three data structures:

| Structure | Responsibility |
|---|---|
| `QuantParams` | Data required by one CUDA execution |
| `QuantizeOutputs` | The `out`, `out_t` and `amax` result tensors |
| `RingLayout` | Ring extents and shared relative slot offsets |

`api/native/quantize.h` contains the native parameter contract and ring
layout, with no Torch, Python or CUDA dependency. `QuantParams` has fifteen
fields and no member defaults; the entry starts with `QuantParams p{}` and
fills the fields needed for the requested operation:

```cpp
struct QuantParams {
    const void* input_ptr;
    const float* inv_scale;
    void* output_ptr;
    void* output_transposed_ptr;

    int rows;
    int cols;
    int out_row_stride;
    int out_col_stride;

    float* ring_state;
    int hist_len;
    int hist_idx;
    float* next_scale;
    float* next_inv_scale;

    float fp8_max;
    float pow2_margin;
};
```

Output pointer presence selects the written orientations. Native original
output strides retain their element-based row/column semantics; transposed
placement is derived from rows. `ring_state == nullptr` selects pure
quantization. With a ring, the kernel derives history, amax, scratch and
completion-counter pointers from its base and `RingLayout` constants.
There is no separate fold flag or per-slot pointer set. Publication pointers
remain independent so a caller can publish outside the ring.

`api/torch/dtype.h` owns the CUDA-free `fp8_max_of` query for the two FP8
ranges. Bindings and `ScaleRing` use this metadata without including CUDA
dispatch. The entry uses shared Torch dtype visitors to select
`<InT, OutT, TransposedOutT = OutT>`; the input, original output and
transposed output types keep the same order through launch and kernel.

## Execution ownership

- `api/torch/quantize.h` declares the complete Tensor interfaces and result.
  `csrc/quantize/entry.cu` checks parameters, makes input storage contiguous,
  allocates requested outputs, binds the native parameters and dispatches
  dtypes. It keeps the input Tensor alive until launch submission.
- `csrc/quantize/bindings.cu` registers named functions directly. It resolves
  Python defaults and converts the result to a tuple; C++ interfaces have no
  default arguments or options containers.
- `launcher/quantize.cuh` configures the grid/block and checks the launch.
  `kernel/quantize/kernel.cuh` performs the cast, transpose and optional
  fused ring update. Empty inputs still launch one block for a ring update.

Both outputs and the optional history update share one input read and one
kernel launch. Execution uses the current stream; argument checking does
not read GPU values back to the CPU.

FP8 linear owns `ScaleRing`, caches, metadata registration and checkpoint
restoration. It calls `quantize` for dynamic scaling and `quantize_scaled`
for history scaling, passing the current reciprocal and next publication
pair. `version_of` belongs to cached-cast invalidation in that caller layer.
Cache hits, no-grad calls and forward/backward transitions retain their
existing ring behavior. See [gemm.md](gemm.md) for the training composition.

## Interface migration and int8

Old `quantize`/`transposed` and `quantize_dual` calls are replaced by explicit
`dtype`/`tdtype` requests and the fixed three-element result. A history update
uses `quantize_scaled`; the public margin is an integer exponent. There are
no compatibility wrappers. The Python adapter does not forward native
scratch constants through module `__getattr__`.

`quantize_int8(x)` is the shared Torch implementation for `[N, K]` weights
and `[..., K]` activations. It returns contiguous int8 data with x's shape
and flat float32 last-axis row scales, using the unchanged
`clamp_min(amax, 1e-12) / 127` and round/clamp arithmetic. It is detached
from autograd and does not require the compiled quantize module.
