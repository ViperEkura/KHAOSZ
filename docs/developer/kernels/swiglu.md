# SwiGLU

Native kernel: `csrc/swiglu.cu`. Python operator: `astrai.extension.kernel.swiglu.swiglu`. Dispatch: `astrai.extension.backend.swiglu.swiglu`.

## Math contract

With `s = sigmoid(gate)`, forward computes `up * gate * s`. Given upstream gradient `dy`, backward returns `dy * gate * s` for `up` and `dy * up * s * (1 + gate * (1 - s))` for `gate`. Forward and backward each use one native CUDA launch; the registered first-order autograd function saves the two inputs and does not save the forward intermediate.

The fused path supports contiguous CUDA tensors of identical shape and dtype in FP16, BF16, or FP32. Aligned lengths divisible by four use packed loads and stores; other inputs use the scalar kernel. Auto dispatch uses Torch for unsupported calls. `with op_backend(swiglu="cuda")` requires the fused path and raises if it cannot handle the call.

## Performance measurement

`CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/benchmark/swiglu.py --shapes 2048x6912 8192x6912 --warmup 20 --iterations 100 --trials 5`

BF16, identical inputs and upstream gradients, interleaved Torch/CUDA trials, median CUDA event times. Times include the public per-operator dispatcher. The large shape had these results:

| Mode | Torch eager | Fused eager | Torch CUDA Graph | Fused CUDA Graph |
|---|---:|---:|---:|---:|
| Forward | 363.30 µs | 219.39 µs | 363.32 µs | 218.47 µs |
| Forward + backward | 1085.18 µs | 1352.55 µs | 1010.91 µs | 588.78 µs |

The fused path improved graph-captured training latency by 1.72× at this shape. Eager training remained slower, so this measurement does not claim a universal speedup. The benchmark script reports both shapes.
