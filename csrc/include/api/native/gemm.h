#pragma once

#include <cstdint>

#include <utils/shape.cuh>

/*
 * GEMM native parameter contract — dtype-neutral: layout tags
 * and the unified parameter POD shared by every element-type specialization.
 * Element storage and conversion live in datatype/element.cuh. This
 * header stays torch-free for both host and device callers.
 */

namespace astrai {
namespace gemm {

// Operand storage tags (CUTLASS-style) relative to the canonical matrices
struct RowMajor {};
struct ColMajor {};

/*
 * Tile-geometry Shape: the shared static-shape vocabulary imported so the
 * Shape<M, N, K> CTA recipes keep their spelling (policy/gemm/manifest.h).
 */
using astrai::Shape;

/*
 * Unified GEMM parameter POD: one struct flows through the kernels; each
 * kernel touches only the fields it needs.
 */
struct GemmParams {
    /*
     * Operands, optional fused bias, and output.
     * Bias is added in FP32 before the single output rounding; null disables it.
     */
    const void* __restrict__ a_ptr = nullptr;
    const void* __restrict__ b_ptr = nullptr;
    const void* __restrict__ bias_ptr = nullptr;
    void* __restrict__ out_ptr = nullptr;

    /*
     * Per-operand dequantization scales, multiplied into the epilogue.
     * a_scale_m: 0 = device scalar, m = per-row scale.
     * b_scale_n: 0 = device scalar, n = per-output-channel scale.
     * Null pointers disable scaling. Scales grouped along K cannot fold here.
     */
    const float* __restrict__ a_scale = nullptr;
    const float* __restrict__ b_scale = nullptr;
    int a_scale_m = 0;
    int b_scale_n = 0;

    /*
     * Matrix geometry and strides, measured in elements.
     * batch selects the BMM count; grid.z advances by the batch strides.
     * A zero input batch stride broadcasts that operand across the batch.
     * Batch strides may exceed INT32. out_ld permits strided outputs;
     * output orientation is supplied by the policy's LayoutOut tag.
     */
    int batch = 1;
    int m, n, k;
    int a_ld, b_ld, out_ld;
    int64_t a_batch_stride = 0;
    int64_t b_batch_stride = 0;
    int64_t out_batch_stride = 0;

    /*
     * Runtime raster order selected from the problem's aspect ratio.
     * Positive: group M rows and walk M fastest to reuse B columns.
     * Negative: group N columns and walk N fastest to reuse A rows.
     * Zero: plain N-fastest raster.
     */
    int raster = 8;
};

} // namespace gemm
} // namespace astrai
