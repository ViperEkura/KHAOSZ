#pragma once
/* Torch contracts shared by quantize and GEMM; no Python binding types. */
#include <c10/util/Optional.h>
#include <cstdint>
#include <torch/types.h>

namespace astrai {
namespace quant {

/* Undefined tensors represent unrequested outputs; amax is a ring view. */
struct QuantizeOutputs {
    torch::Tensor out;
    torch::Tensor out_t;
    torch::Tensor amax;
};

/* Multiply by inv_scale and cast each explicitly requested orientation. */
QuantizeOutputs quantize(const torch::Tensor& x,
                         const torch::Tensor& inv_scale,
                         c10::optional<at::ScalarType> dtype,
                         c10::optional<at::ScalarType> tdtype);

/* Also fold raw amax and publish the next scale and reciprocal. No cursor
 * advance; undefined publication tensors select pair zero in ring_state.
 */
QuantizeOutputs quantize_scaled(const torch::Tensor& x,
                                const torch::Tensor& inv_scale,
                                const torch::Tensor& ring_state,
                                int64_t hist_len,
                                int64_t hist_idx,
                                c10::optional<at::ScalarType> dtype,
                                c10::optional<at::ScalarType> tdtype,
                                int64_t margin,
                                double fp8_max,
                                const torch::Tensor& next_scale,
                                const torch::Tensor& next_inv_scale);
} // namespace quant
} // namespace astrai
