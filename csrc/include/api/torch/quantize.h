#pragma once
/*
 * Quantize's caller contract: the composed one-call pass shared by
 * quantize/bindings.cu and gemm/fp8/linear.cu; the implementation,
 * quantize/entry.cu, is listed in both modules' source lists. Torch-tensor
 * level, no pybind (the gemm.h header rules).
 */

#include <c10/util/Optional.h>
#include <cstdint>
#include <torch/types.h>

#include <api/native/quantize.h>

namespace astrai {
namespace quant {

struct QuantizeOutputs {
    torch::Tensor out;   // row-major orientation (undefined if not asked)
    torch::Tensor out_t; // [cols][rows] transpose (undefined if not asked)
    torch::Tensor amax;  // the round's raw-domain amax; undefined without a ring
};

/*
 * One quantize pass end to end: validate, allocate, launch. ``layout`` picks
 * the orientations produced; ``dtype_b`` (default: ``dtype_a``) recasts the
 * transposed side (hybrid fwd/bwd pair from one read). ``ring`` switches on
 * the delayed-scaling fold and requires ``hist_len`` (RingLayout,
 * api/native/quantize.h); ``pub_scale``/``pub_recip`` redirect where the
 * fold publishes (default: the ring's own slots). Semantics at the definition
 * (quantize/entry.cu).
 */
QuantizeOutputs run_quantize(torch::Tensor x,
                             torch::Tensor scale,
                             QuantLayout layout,
                             at::ScalarType dtype_a,
                             c10::optional<at::ScalarType> dtype_b,
                             c10::optional<torch::Tensor> ring,
                             int64_t hist_idx,
                             double fp8_max,
                             double pow2_margin,
                             c10::optional<torch::Tensor> pub_scale = c10::nullopt,
                             c10::optional<torch::Tensor> pub_recip = c10::nullopt,
                             c10::optional<int64_t> hist_len = c10::nullopt);

} // namespace quant
} // namespace astrai
