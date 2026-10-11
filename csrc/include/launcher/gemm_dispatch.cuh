#pragma once
/* Typed GEMM query, layout routing, dispatch, and probe. */
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>
#include <utility>

#include <api/native/gemm.h>
#include <api/native/gemm_plan.h>
#include <launcher/gemm_tiles.cuh>
#include <memory/tma.cuh>
#include <mma/mma.cuh>
#include <policy/gemm/traits.cuh>
#include <utils/device.cuh>

namespace astrai {
namespace gemm {

/*
 * Dtype-class derivation (returns api/native/gemm_plan.h's GemmPerfClass): the mma
 * promotion rule plus operand widths (mixed bf16xfp8 lands with W8A16 —
 * same bytes, same promoted bf16 k16 mma). int8 is tested FIRST: it
 * promotes to an int8 mma, so the "not bf16 -> fp8" arm would swallow it
 * and every W8A8 row would be unreachable.
 */
template <typename ElemA, typename ElemB> constexpr GemmPerfClass gemm_perf_class() {
    using MmaT = typename gemm_mma_traits<ElemA, ElemB>::MmaT;
    if constexpr (std::is_same_v<ElemA, int8_t> && std::is_same_v<ElemB, int8_t>) {
        return GemmPerfClass::kW8A8;
    } else if constexpr (!std::is_same_v<MmaT, __nv_bfloat16>) {
        return GemmPerfClass::kF8A8; // native fp8 symmetric pair
    } else if constexpr (std::is_same_v<ElemA, __nv_bfloat16> &&
                         std::is_same_v<ElemB, __nv_bfloat16>) {
        return GemmPerfClass::kW16A16;
    } else {
        return GemmPerfClass::kW8A16;
    }
}

static_assert(gemm_perf_class<int8_t, int8_t>() == GemmPerfClass::kW8A8,
              "int8 x int8 is its own class (see the ordering note above)");
static_assert(gemm_perf_class<__nv_fp8_e4m3, __nv_fp8_e4m3>() == GemmPerfClass::kF8A8,
              "fp8 x fp8 keys the F8A8 table");
static_assert(gemm_perf_class<__nv_bfloat16, __nv_bfloat16>() == GemmPerfClass::kW16A16,
              "bf16 x bf16 keys the W16A16 table");
static_assert(gemm_perf_class<__nv_bfloat16, int8_t>() == GemmPerfClass::kW8A16,
              "a quantized weight against bf16 activations keys W8A16");

// One spelling for the typed policy binding used by query, probe, and launch.
template <typename ElemA,
          typename ElemB,
          typename LayoutA,
          typename LayoutB,
          typename OutT,
          typename Schedule,
          typename LayoutOut>
using GemmDispatchFor = GemmTileDispatch<ElemA, ElemB, LayoutA, LayoutB, LayoutOut, OutT, Schedule>;

/* Derive the query from typed operands and the launch geometry. */
template <typename ElemA,
          typename ElemB,
          typename LayoutA,
          typename LayoutB,
          typename OutT = __nv_bfloat16,
          typename Schedule = MmaSync,
          typename LayoutOut = RowMajor>
PlanQuery plan_query(const GemmParams& p, const DeviceFacts* dev = nullptr) {
    PlanQuery q;
    q.dev = dev != nullptr ? *dev : device_facts();
    q.m = p.m;
    q.n = p.n;
    q.k = p.k;
    q.batch = p.batch;
    q.perf_class = (int)gemm_perf_class<ElemA, ElemB>();
    q.crosswise = crosswise_of<LayoutA, LayoutB>();
    q.ba = (int)sizeof(ElemA);
    q.bb = (int)sizeof(ElemB);
    q.out_elem_bytes = (int)sizeof(OutT);
    using MmaT = typename gemm_mma_traits<ElemA, ElemB>::MmaT;
    using MmaShape = typename astrai::MmaShapeFor<MmaT>::type;
    q.mma_k = MmaShape::kK;
    // Match the descriptor's alignment check before pricing TMA residency.
    // Driver encode can still reject a map; the launcher then falls back.
    q.tma = tma_eligible_v<Schedule, ElemA, ElemB, LayoutA, LayoutB> && q.dev.cc >= 90 &&
            !gemm_tma_staging_disabled() &&
            astrai::tma_aligned16(p.a_ptr, p.a_ld * sizeof(ElemA),
                                  (p.batch > 1 ? p.a_batch_stride : 0) * sizeof(ElemA)) &&
            astrai::tma_aligned16(p.b_ptr, p.b_ld * sizeof(ElemB),
                                  (p.batch > 1 ? p.b_batch_stride : 0) * sizeof(ElemB));
    q.contiguous = q.crosswise == 0 && p.a_ld == p.k && p.b_ld == p.k &&
                   (reinterpret_cast<uintptr_t>(p.a_ptr) % 32) == 0 &&
                   (reinterpret_cast<uintptr_t>(p.b_ptr) % 32) == 0 &&
                   (p.batch <= 1 || ((p.a_batch_stride * sizeof(ElemA)) % 32 == 0 &&
                                     (p.b_batch_stride * sizeof(ElemB)) % 32 == 0));
    q.rank3a = p.batch > 1 && p.a_batch_stride > 0;
    q.rank3b = p.batch > 1 && p.b_batch_stride > 0;
    q.resources =
        &GemmDispatchFor<ElemA, ElemB, LayoutA, LayoutB, OutT, Schedule, LayoutOut>::resources;
    return q;
}

/* Carry the effective staging decision into the typed launch. */
template <typename ElemA,
          typename ElemB,
          typename LayoutA,
          typename LayoutB,
          typename OutT = __nv_bfloat16,
          typename Schedule = MmaSync,
          typename LayoutOut = RowMajor>
LaunchPlan plan_dispatch_for(const GemmParams& p, const DeviceFacts* dev = nullptr) {
    const PlanQuery q =
        plan_query<ElemA, ElemB, LayoutA, LayoutB, OutT, Schedule, LayoutOut>(p, dev);
    return {plan_dispatch(q), q.tma};
}

/* Lower the problem and visit the layout tags shared by launch and probe. */
template <typename ElemA, typename ElemB, typename F>
auto with_gemm_layout(const GemmParams& p, bool trans_a, bool trans_b, F&& fn) {
    if constexpr (std::is_same_v<ElemA, ElemB>) {
        if (!trans_a && !trans_b) {
            // Symmetric NN runs as B^T A^T with a transposed output.
            GemmParams swapped = p;
            swapped.m = p.n;
            swapped.n = p.m;
            swapped.a_ptr = p.b_ptr;
            swapped.b_ptr = p.a_ptr;
            swapped.a_ld = p.b_ld;
            swapped.b_ld = p.a_ld;
            swapped.a_batch_stride = p.b_batch_stride;
            swapped.b_batch_stride = p.a_batch_stride;
            swapped.out_ld = p.n;
            return fn(swapped, ColMajor{}, ColMajor{}, ColMajor{});
        }
    }
    if (trans_a && trans_b)
        return fn(p, ColMajor{}, ColMajor{}, RowMajor{});
    if (trans_b)
        return fn(p, RowMajor{}, ColMajor{}, RowMajor{});
    if (trans_a)
        return fn(p, ColMajor{}, RowMajor{}, RowMajor{});
    if constexpr (std::is_same_v<ElemA, ElemB>) {
        // NN was lowered above; avoid instantiating the direct symmetric kernel.
        return fn(p, ColMajor{}, ColMajor{}, RowMajor{});
    } else {
        return fn(p, RowMajor{}, RowMajor{}, RowMajor{});
    }
}

/* Dispatch symmetric NN through the transpose; mixed NN stays direct. */
template <typename ElemA,
          typename ElemB = ElemA,
          typename OutT = __nv_bfloat16,
          typename Schedule = MmaSync>
void gemm_dispatch(const GemmParams& p,
                   cudaStream_t stream,
                   bool trans_a,
                   bool trans_b,
                   const DeviceFacts* dev = nullptr) {
    DeviceFacts local_dev;
    if (dev == nullptr) {
        local_dev = device_facts();
        dev = &local_dev;
    }
    const auto launch = [&](const GemmParams& lowered, auto la, auto lb, auto lout) {
        using Dispatch = GemmDispatchFor<ElemA, ElemB, decltype(la), decltype(lb), OutT, Schedule,
                                         decltype(lout)>;
        Dispatch::launch(lowered,
                         plan_dispatch_for<ElemA, ElemB, decltype(la), decltype(lb), OutT, Schedule,
                                           decltype(lout)>(lowered, dev),
                         stream, *dev);
    };
    with_gemm_layout<ElemA, ElemB>(p, trans_a, trans_b, launch);
}

/* Host-only probe: use the launch layout route without launching a kernel. */
template <typename ElemA, typename ElemB, typename Schedule = MmaSync>
std::pair<PlanDecision, PlanQuery>
plan_probe_for(const GemmParams& p, bool trans_a, bool trans_b, const DeviceFacts& dev) {
    return with_gemm_layout<ElemA, ElemB>(
        p, trans_a, trans_b, [&](const GemmParams& lowered, auto la, auto lb, auto lo) {
            PlanQuery q = plan_query<ElemA, ElemB, decltype(la), decltype(lb), __nv_bfloat16,
                                     Schedule, decltype(lo)>(lowered, &dev);
            return std::make_pair(plan_dispatch(q), std::move(q));
        });
}

/* One signature spelling for explicit definitions and external instantiations. */
#define ASTRAI_GEMM_SPECIALIZATIONS(PREFIX, TA, TB, SCHEDULE)                                      \
    PREFIX void gemm_dispatch<TA, TB, __nv_bfloat16, SCHEDULE>(const GemmParams&, cudaStream_t,    \
                                                               bool, bool, const DeviceFacts*);    \
    PREFIX std::pair<PlanDecision, PlanQuery> plan_probe_for<TA, TB, SCHEDULE>(                    \
        const GemmParams&, bool, bool, const DeviceFacts&)

#ifndef ASTRAI_GEMM_SCHEDULE
#define ASTRAI_GEMM_SCHEDULE MmaSync
#endif
#define ASTRAI_GEMM_INSTANTIATE(TA, TB)                                                            \
    ASTRAI_GEMM_SPECIALIZATIONS(template, TA, TB, ASTRAI_GEMM_SCHEDULE)

/*
 * The extension entry includes gemm_build.h before this header. Its manifest
 * selects externally compiled specializations, preserving per-schedule CUDA
 * images. Standalone native callers instantiate the definitions above.
 */
#ifdef ASTRAI_BUILD_FP8
#define ASTRAI_GEMM_BASE_EXTERN(SCHEDULE)                                                          \
    ASTRAI_GEMM_SPECIALIZATIONS(extern template, __nv_bfloat16, __nv_bfloat16, SCHEDULE);          \
    ASTRAI_GEMM_SPECIALIZATIONS(extern template, __nv_bfloat16, int8_t, SCHEDULE);                 \
    ASTRAI_GEMM_SPECIALIZATIONS(extern template, int8_t, int8_t, SCHEDULE)
#define ASTRAI_GEMM_FP8_EXTERN(SCHEDULE)                                                           \
    ASTRAI_GEMM_SPECIALIZATIONS(extern template, __nv_fp8_e4m3, __nv_fp8_e4m3, SCHEDULE);          \
    ASTRAI_GEMM_SPECIALIZATIONS(extern template, __nv_fp8_e5m2, __nv_fp8_e5m2, SCHEDULE)

ASTRAI_GEMM_BASE_EXTERN(MmaSync);
#if ASTRAI_BUILD_FP8
ASTRAI_GEMM_FP8_EXTERN(MmaSync);
#endif
#if ASTRAI_BUILD_TMA
ASTRAI_GEMM_BASE_EXTERN(TmaMma);
ASTRAI_GEMM_FP8_EXTERN(TmaMma);
#endif
#if ASTRAI_BUILD_MX
ASTRAI_GEMM_FP8_EXTERN(Sm120Mma);
#endif
#undef ASTRAI_GEMM_BASE_EXTERN
#undef ASTRAI_GEMM_FP8_EXTERN
#endif

} // namespace gemm
} // namespace astrai
