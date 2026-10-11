#pragma once
/* Tile-grid configuration for the native fused FP8 quantize kernel. */
#include <algorithm>
#include <kernel/quantize/kernel.cuh>
#include <utils/launch.cuh>

namespace astrai {
namespace quant {
/* Empty inputs still launch one block to publish the history update. */
template <typename InT, typename OutT, typename TransposedOutT = OutT>
void launch_fp8_quantize(const QuantParams& p, cudaStream_t stream) {
    const dim3 grid(std::max(1, (p.cols + 63) / 64), std::max(1, (p.rows + 31) / 32));
    fp8_quantize_strided_kernel<InT, OutT, TransposedOutT>
        <<<grid, dim3(32, kQuantWarps), 0, stream>>>(p);
    ASTRAI_LAUNCH_CHECK();
}
} // namespace quant
} // namespace astrai
