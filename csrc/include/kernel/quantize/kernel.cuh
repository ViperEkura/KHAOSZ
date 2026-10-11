#pragma once
/* FP8 device kernels use QuantParams plus compile-time input/output types. */

#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>

#include <api/native/quantize.h>
#include <arith/reduce.cuh>
#include <datatype/element.cuh>
#include <utils/define.cuh>

namespace astrai {
namespace quant {
/* Must match the launcher's y-dimension and publish_amax's per-warp slots. */
inline constexpr int kQuantWarps = 8;

/* Last block folds amax into history, publishes scale, and resets scratch. */
template <int kWarps> DEVICE_FORCEINLINE void publish_amax(const QuantParams& p, float v) {
    float* const hist = p.ring_state;
    float* const aux = hist + p.hist_len;
    float* const scratch = aux + RingLayout::kScratch;
    auto* const done = reinterpret_cast<unsigned int*>(aux + RingLayout::kDone);
    v = warp_reduce<maximum<float>>(v);
    __shared__ float slots[kWarps];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if ((tid & 31) == 0)
        slots[tid >> 5] = v;
    __syncthreads();
    if (tid == 0) {
#pragma unroll
        for (int w = 1; w < kWarps; ++w)
            v = fmaxf(v, slots[w]);
        const int slot = (blockIdx.y * gridDim.x + blockIdx.x) & (kFoldSlots - 1);
        atomic_max_float(scratch + slot, v);
        __threadfence();
        const unsigned int ticket = atomicAdd(done, 1u);
        __threadfence();
        /* The kernel uses a 2D grid, so the completion count must include both axes. */
        const unsigned int total = gridDim.x * gridDim.y;
        if (ticket != total - 1u)
            return;
        float peak = scratch[0];
        for (int s = 1; s < kFoldSlots; ++s)
            peak = fmaxf(peak, scratch[s]);
        hist[p.hist_idx] = peak;
        float win = hist[0];
        for (int i = 1; i < p.hist_len; ++i)
            win = fmaxf(win, hist[i]);
        const float next = fmaxf(win / p.fp8_max / p.pow2_margin, 1e-12f);
        *p.next_scale = next;
        /* Match the host ATen reciprocal's rounding; do not use a fast reciprocal. */
        *p.next_inv_scale = __frcp_rn(next);
        aux[RingLayout::kAmax] = peak;
        for (int s = 0; s < kFoldSlots; ++s)
            scratch[s] = 0.0f;
        *done = 0u;
    }
}

/*
 * One tiled kernel handles RowMajor, Transposed, and Dual; the host flattens
 * leading dimensions into [rows, cols]. Strides select the primary output,
 * while transposed output uses c*rows+r. A 17-word shared-tile pitch
 * coalesces transposed writes and avoids bank conflicts; other strides remain
 * correct but may not coalesce. Same-format Dual shares conversions; mixed
 * formats convert per orientation.
 */
template <typename Fp8TA, typename Fp8TB> struct dual_cvt {
    static constexpr bool kMixed = !std::is_same_v<Fp8TA, Fp8TB>;

    static DEVICE_FORCEINLINE void store(uint8_t (*q)[2], uint8_t (*q2)[2], int j, int k, float v) {
        q[j][k] = ElemTrait<Fp8TA>::cvt_byte(v);
        if constexpr (kMixed)
            q2[j][k] = ElemTrait<Fp8TB>::cvt_byte(v);
    }

    static DEVICE_FORCEINLINE void zero(uint8_t (*q)[2], uint8_t (*q2)[2], int j) {
        q[j][0] = 0;
        q[j][1] = 0;
        if constexpr (kMixed) {
            q2[j][0] = 0;
            q2[j][1] = 0;
        }
    }

    static DEVICE_FORCEINLINE uint8_t pick(const uint8_t (*q)[2],
                                           const uint8_t (*q2)[2],
                                           int j,
                                           int k) {
        if constexpr (kMixed)
            return q2[j][k];
        return q[j][k];
    }
};

template <typename InT, typename OutT, typename TransposedOutT = OutT>
__global__ void fp8_quantize_strided_kernel(QuantParams p) {
    constexpr int kTileC = 64, kTileR = 32;
    __shared__ uint8_t tile[kTileC][kTileR + 2];
    const float mult = *p.inv_scale;
    const auto* x = static_cast<const InT*>(p.input_ptr);
    const int r0 = blockIdx.y * kTileR;
    const int c0 = blockIdx.x * kTileC;
    const int r = r0 + threadIdx.y * 4;
    const int c = c0 + threadIdx.x * 2;

    using Cvt = dual_cvt<OutT, TransposedOutT>;
    uint8_t q[4][2];
    uint8_t q2[4][2]; // live only when the two orientations differ in format
    float local_amax = 0.0f;
    constexpr int kPairAlign = 2 * (int)sizeof(InT);
    using PairT = typename ElemTrait<InT>::Pair;
    const bool full_tile = r0 + kTileR <= p.rows && c0 + kTileC <= p.cols &&
                           (reinterpret_cast<uintptr_t>(x) & (kPairAlign - 1)) == 0 &&
                           ((p.cols & 1) == 0);
    if (full_tile) {
        const InT* a = x + (int64_t)r * p.cols + c;
        const PairT raws[4] = {*reinterpret_cast<const PairT*>(a),
                               *reinterpret_cast<const PairT*>(a + (int64_t)p.cols),
                               *reinterpret_cast<const PairT*>(a + 2 * (int64_t)p.cols),
                               *reinterpret_cast<const PairT*>(a + 3 * (int64_t)p.cols)};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float f[2];
            astrai::load_pair<InT>(reinterpret_cast<const InT*>(&raws[j]), f);
            local_amax = fmaxf(local_amax, fmaxf(fabsf(f[0]), fabsf(f[1])));
            Cvt::store(q, q2, j, 0, f[0] * mult);
            Cvt::store(q, q2, j, 1, f[1] * mult);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            Cvt::zero(q, q2, j);
            if (r + j < p.rows && c < p.cols) {
                const InT* a = x + (int64_t)(r + j) * p.cols + c;
                if (c + 1 < p.cols && (reinterpret_cast<uintptr_t>(a) & (kPairAlign - 1)) == 0) {
                    float f[2];
                    astrai::load_pair<InT>(a, f);
#pragma unroll
                    for (int k = 0; k < 2; ++k) {
                        local_amax = fmaxf(local_amax, fabsf(f[k]));
                        Cvt::store(q, q2, j, k, f[k] * mult);
                    }
                } else {
                    const float v0 = ElemTrait<InT>::to_float(a[0]);
                    local_amax = fmaxf(local_amax, fabsf(v0));
                    Cvt::store(q, q2, j, 0, v0 * mult);
                    if (c + 1 < p.cols) {
                        const float v1 = ElemTrait<InT>::to_float(a[1]);
                        local_amax = fmaxf(local_amax, fabsf(v1));
                        Cvt::store(q, q2, j, 1, v1 * mult);
                    }
                }
            }
        }
    }
    /* Primary output placement is selected by the stride pair. */
    uint8_t* out = static_cast<uint8_t*>(p.output_ptr);
    if (out != nullptr) {
        const int64_t s_r = p.out_row_stride, s_c = p.out_col_stride;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (r + j < p.rows && c < p.cols) {
                const int64_t off = (int64_t)(r + j) * s_r + (int64_t)c * s_c;
                uint8_t* o = out + off;
                if (s_c == 1 && c + 1 < p.cols && (off & 1) == 0)
                    *reinterpret_cast<unsigned short*>(o) =
                        (unsigned short)(q[j][0] | (q[j][1] << 8));
                else {
                    o[0] = q[j][0];
                    if (c + 1 < p.cols)
                        o[s_c] = q[j][1];
                }
            }
    }
    /* Transposed output uses c*rows+r and a shared tile for coalesced stores. */
    uint8_t* out_t = static_cast<uint8_t*>(p.output_transposed_ptr);
    if (out_t != nullptr) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int k = 0; k < 2; ++k)
                tile[threadIdx.x * 2 + k][threadIdx.y * 4 + j] = Cvt::pick(q, q2, j, k);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int oc = c0 + threadIdx.y * 8 + i;
            if (oc < p.cols && r0 + threadIdx.x < p.rows)
                out_t[(int64_t)oc * p.rows + r0 + threadIdx.x] =
                    tile[threadIdx.y * 8 + i][threadIdx.x];
        }
    }
    if (p.ring_state != nullptr)
        publish_amax<kQuantWarps>(p, local_amax);
}

} // namespace quant
} // namespace astrai
