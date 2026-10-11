# Quantize

Quantization has two explicit entrypoints shared by FP8 and int8 outputs.
`quantize` scales and casts the requested output orientations.
`quantize_scaled` also measures the raw input amax, updates a global history
window and publishes the next scale in the same kernel. Neither entry owns
a cache, history cursor or training state.

## Interfaces and math

```python
quantize(x, inv_scale, *, dtype=None, tdtype=None)
quantize_scaled(x, inv_scale, ring_state, *, hist_len, hist_idx,
                dtype=None, tdtype=None, margin=0, quant_max=None,
                next_scale=None, next_inv_scale=None)
```

Both return `(out, out_t, amax)`. `dtype` requests an output with x's shape;
`tdtype` requests the contiguous `[cols, rows]` transpose after folding the
leading axes into rows. At least one dtype is required. An absent dtype
produces `None`; it does not inherit the other dtype. Input types are BF16,
FP16 and FP32; output types are E4M3FN, E5M2 and int8, independently selected.
The two outputs may use any combination of these formats.

The logical matrix has `cols=x.shape[-1]` and `rows` equal to the product
of leading extents; a 1D input has one row. Transposed output requires a 2D+
input. Empty inputs preserve the original output shape and use the existing
`rows=0`, `cols=1`, `[1, 0]` execution/transpose convention. Multiplier
broadcast checks still use the original logical rows and columns; empty
inputs do not read multiplier storage.

`inv_scale` is the current float32 quantization multiplier on x's CUDA
device. Shape defines its broadcast axes:

| Multiplier shape | Meaning in original input coordinates |
|---|---|
| Any shape with one element | Scalar for every row and column |
| `[cols]` or `[1, cols]` | Per column, shared across rows |
| `[rows, 1]` | Per row, shared across columns |
| `[rows, cols]` | Per element |

A non-scalar multiplier is 1D or 2D. A 1D vector uses standard column
broadcasting; each axis of a 2D multiplier must be one or the corresponding
logical matrix extent. Shape determines the axis even for square inputs;
vector length is never used to guess a row axis.

Non-contiguous views and expanded tensors are accepted. Tensor strides
describe storage independently of the broadcast shape. Both output
orientations compute `cast(x[row, col] * multiplier[row, col])` in original
input coordinates, with the same multiplier for mixed output formats.
Transpose changes output placement and keeps the multiplier attached to
the same input element. Pure `quantize` returns `amax=None` and does not
infer a scale.

FP8 conversion keeps its existing round-to-nearest, finite-saturating
conversion. Int8 conversion rounds to nearest with ties to even and clamps
to `[-127, 127]`. A scaled NaN becomes zero; scaled positive/negative
infinity becomes positive/negative 127. No zero-point is used.

```python
out, _, _ = quantize(x, inv_scale, dtype=torch.float8_e4m3fn)
_, out_t, _ = quantize(x, inv_scale, tdtype=torch.int8)
out, out_t, _ = quantize(x, inv_scale,
                         dtype=torch.float8_e4m3fn, tdtype=torch.int8)
```

## History and next-scale publication

`quantize_scaled` accepts only a one-element `inv_scale`, regardless of
its shape. The history and published scale are global per tensor, for every
output format; row/column multiplier broadcasting belongs to `quantize`.
It requires a
contiguous float32 CUDA ring on x's device. Its last finishing block performs:

```text
hist[hist_idx] = max(abs(x))                  # zero for an empty input
next_scale = max(max(hist) / quant_max / 2**margin, 1e-12)
next_inv_scale = reciprocal(next_scale)      # correctly rounded __frcp_rn
```

The range defaults to 448 for E4M3FN, 57344 for E5M2 or 127 for int8,
following `dtype`, or `tdtype` for a transpose-only request. A mixed request
uses that same precedence; an explicit `quant_max` overrides the choice.
Margin retains AstrAI's division by `2**margin` and the kernel retains the
existing calculation order and reciprocal rounding.

The current `inv_scale` controls this call's casts. The returned amax and
newly published `next_scale`/`next_inv_scale` describe the history update for
a subsequent call; publication does not change the multiplier already used.
Provide both publication tensors together to write external scalar slots;
otherwise publication uses ring pair zero. All scalar tensors and publication
slots must share the input device. The returned amax aliases the ring slot
and changes on subsequent updates. The entry does not advance `hist_idx`
or switch the current scale pair.

The ring layout remains:

```text
[hist n | scale0 | recip0 | amax | done | scratch 32 | scale1 | recip1]
```

A standalone ring may omit the final pair. `hist_len` is explicit because
the optional publication pair makes it ambiguous from numel. The Torch
boundary validates shape, dtype, device, capacity and integer range before
converting geometry and history indices to native `int`. The compiled
quantization entry requires sm_80+; FP8 tensor-core GEMM separately requires
sm_89+.

## Parameters and dependencies

The two entries use three data structures:

| Structure | Responsibility |
|---|---|
| `QuantParams` | Data required by one CUDA execution |
| `QuantizeOutputs` | The `out`, `out_t` and `amax` result tensors |
| `RingLayout` | Ring extents and shared relative slot offsets |

`api/native/quantize.h` contains the native parameter contract and ring
layout, with no Torch, Python or CUDA dependency. `QuantParams` has seventeen
fields and no member defaults; the entry starts with `QuantParams p{}` and
fills the fields needed for the requested operation:

```cpp
struct QuantParams {
    /* Contiguous input matrix extent. */
    const void* input_ptr;
    int rows;
    int cols;

    /* Requested output orientations and primary element strides. */
    void* output_ptr;
    void* output_transposed_ptr;
    int out_row_stride;
    int out_col_stride;

    /* Multiplier element strides in input coordinates; zero broadcasts an axis. */
    const float* inv_scale;
    int inv_scale_row_stride;
    int inv_scale_col_stride;

    /* Null ring skips history publication; publication targets may be outside it. */
    float* ring_state;
    int hist_len;
    int hist_idx;
    float* next_scale;
    float* next_inv_scale;

    /* next_scale = max(max(history) / quant_max / pow2_margin, 1e-12). */
    float quant_max;
    float pow2_margin;
};
```

The kernel reads the original input coordinate (r, c) from
`inv_scale[r * inv_scale_row_stride + c * inv_scale_col_stride]`. Both
strides are in elements. A singleton multiplier axis uses zero stride;
other axes retain the Tensor's actual storage stride, including zero for
expanded storage. A scalar uses two zero strides. Native callers provide
these same addressing strides directly.

Output pointer presence selects the written orientations. Native primary
output strides count elements in row/column coordinates; transposed
placement is derived from rows. `ring_state == nullptr` selects pure quantization. With
a ring, the kernel derives history, amax, scratch and completion-counter
pointers from its base and `RingLayout` constants. There is no separate fold
flag or per-slot pointer set. Publication pointers remain independent so a
caller can publish outside the ring.

`api/torch/dtype.h` owns the CUDA-free `quant_max_of` query for FP8 and int8
ranges. Bindings and `ScaleRing` use this metadata without including CUDA
dispatch. The entry uses shared Torch dtype visitors to select
`<InT, OutT, TransposedOutT = OutT>`; input, original output and transposed
output types keep the same order through launch and kernel.

## Execution ownership

- `api/torch/quantize.h` declares the complete Tensor interfaces and result.
  `csrc/quantize/entry.cu` checks parameters, makes input storage contiguous,
  allocates requested outputs, binds native parameters and dispatches dtypes.
  It keeps the input Tensor alive until launch submission.
- `csrc/quantize/bindings.cu` registers named functions directly. It resolves
  Python defaults and converts the result to a tuple; C++ interfaces have no
  default arguments or options containers.
- `launcher/quantize.cuh` configures the grid/block in `launch_quantize` and
  checks the launch. `kernel/quantize/kernel.cuh` implements
  `quantize_strided_kernel` for the cast, transpose and optional fused ring
  update. Empty inputs still launch one block for a ring update.

Both outputs and the optional history update share one input read and one
kernel launch. Execution uses the current stream; argument checking does
not read GPU values back to the CPU.

FP8 linear owns `ScaleRing`, caches, metadata registration and checkpoint
restoration. It calls `quantize` for dynamic scaling and `quantize_scaled`
for history scaling, passing the current reciprocal and next publication
pair. `version_of` belongs to cached-cast invalidation in that caller layer.
Cache hits, no-grad calls and forward/backward transitions retain their
existing ring behavior. See [gemm.md](gemm.md) for the training composition.

## Migration and caller-selected row scales

The separate int8 helper is removed. Automatic row-scale selection belongs
to the caller or strategy; the execution entry receives the multiplier
explicitly. For ordinary nonempty weights `[N, K]` or activations `[..., K]`,
the symmetric row-scale recipe is expressed as:

```python
rows = x.detach().to(torch.float32).reshape(-1, x.shape[-1])
scale = rows.abs().amax(dim=-1).clamp_min(1e-12) / 127.0
inv_scale = scale.reciprocal().reshape(-1, 1)
out, _, _ = quantize(x, inv_scale, dtype=torch.int8)
```

`scale` stays flat for GEMM; quantization receives its reciprocal as
`[rows, 1]` to declare row broadcasting. A per-column multiplier instead
uses `[cols]` or `[1, cols]`. `scale` is the dequantization factor:
`x ~= out.float() * scale[:, None]` after flattening the input to rows.
The previous int8
helper evaluated `x / scale`; the shared entry evaluates
`x * reciprocal(scale)`, with the float32 reciprocal rounded separately.
Values near a rounding boundary may differ by one int8 code, so this
migration does not promise bitwise equality with the old division. The
existing FP8 scalar-multiplier calculation remains unchanged.

For W8A8 linear, pass
the input-row scale as GEMM `a_scale` and weight-row scale as `b_scale` with
the usual transposed-weight layout; see [GEMM scales](gemm.md#scales).

Older `transposed` and `quantize_dual` calls use explicit `dtype`/`tdtype`
requests and the fixed three-element result. History updates use
`quantize_scaled` with integer-exponent margin and `quant_max`. There are no
compatibility wrappers or separate int8 exports. The Python adapter does
not forward native scratch constants through module `__getattr__`.
