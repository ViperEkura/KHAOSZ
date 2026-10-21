# Rotary Embedding

Native kernel: `csrc/rotary_emb.cu`. Python operator: `astrai.extension.kernel.rotary.rotary_emb`. Dispatch: `astrai.extension.backend.rotary.apply_rotary_emb`.

## Math contract

For each adjacent pair `(e, o)` and frequency pair `(c, s)`, forward returns `(ec - os, es + oc)`. Given upstream gradient `(de, do)`, backward returns `(de*c + do*s, do*c - de*s)`. The backward kernel needs the frequency tensor only; autograd does not save the input or forward output.

The CUDA path accepts BF16 packed `[tokens, heads, dim]` or dense `[batch, seq, heads, dim]` input, FP32 frequencies with matching token axes and `[dim/2, 2]` pair axes, and a positive head dimension divisible by four. The innermost input dimension must be contiguous; outer strides may differ, including partial rotary slices. The output and input gradient are contiguous. Forward and backward each use one CUDA launch.

The frequency tensor is treated as fixed. If it requires a gradient, dispatch uses the Torch implementation. Unsupported device, dtype, shape, or input layout also use Torch in auto mode; an explicit `rotary="cuda"` selection raises an error when unsupported. The fused operator provides a first-order gradient for the input.

## RTX 5090 measurement

`CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/benchmark/rotary.py --shape 4x2048x16x128 --warmup 10 --iterations 100 --trials 3`

BF16 training forward plus backward, identical input and upstream gradient, interleaved Torch/CUDA trials; medians of CUDA event times. Peak allocation is measured above the preallocated input, frequency, and gradient tensors.

| Path | Peak allocation | Eager | CUDA Graph |
|---|---:|---:|---:|
| Torch | 168 MiB | 2075.17 µs | 321.91 µs |
| Fused CUDA | 64 MiB | 1362.12 µs | 78.09 µs |

The fused path used 61.9% less peak allocated memory in this case. These are operator-level measurements on one RTX 5090, not model throughput.
