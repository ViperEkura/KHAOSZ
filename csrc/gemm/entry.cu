/* GEMM typed entry, dtype/schedule selection, and planner probe. */

#include <c10/core/ScalarType.h>
#include <cstdint>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <api/torch/dispatch.cuh>
#include <api/torch/gemm.h>
#include <gemm_build.h>
#include <launcher/gemm_dispatch.cuh>

#include "params.h"

using namespace astrai;
using namespace astrai::quant;

namespace astrai {
namespace gemm {

#define ASTRAI_GEMM_BASE_PAIRS(X)                                                                  \
    X(bf16, bf16)                                                                                  \
    X(bf16, int8_t)                                                                                \
    X(int8_t, int8_t)

#if ASTRAI_BUILD_FP8
#define ASTRAI_GEMM_FP8_PAIRS(X)                                                                   \
    X(fp8_e4m3, fp8_e4m3)                                                                          \
    X(fp8_e5m2, fp8_e5m2)
#else
#define ASTRAI_GEMM_FP8_PAIRS(X)
#endif
#define ASTRAI_GEMM_PAIRS(X) ASTRAI_GEMM_BASE_PAIRS(X) ASTRAI_GEMM_FP8_PAIRS(X)

#define GEMM_EXTERN_FOR(TA, TB, SCHEDULE) \
    ASTRAI_GEMM_SPECIALIZATIONS(extern template, TA, TB, SCHEDULE);

#define GEMM_EXTERN_BASE(TA, TB) GEMM_EXTERN_FOR(TA, TB, MmaSync)
ASTRAI_GEMM_PAIRS(GEMM_EXTERN_BASE)
#undef GEMM_EXTERN_BASE
#if ASTRAI_BUILD_TMA
#define GEMM_EXTERN_TMA(TA, TB) GEMM_EXTERN_FOR(TA, TB, TmaMma)
ASTRAI_GEMM_PAIRS(GEMM_EXTERN_TMA)
#undef GEMM_EXTERN_TMA
#endif
#if ASTRAI_BUILD_MX
#define GEMM_EXTERN_MX(TA, TB) GEMM_EXTERN_FOR(TA, TB, Sm120Mma)
ASTRAI_GEMM_FP8_PAIRS(GEMM_EXTERN_MX)
#undef GEMM_EXTERN_MX
#endif
#undef GEMM_EXTERN_FOR

static GemmCapabilities capabilities_for(const DeviceFacts& dev) {
    const int cc = dev.cc;
    return {cc,
            supports(build::kBase, cc),
            supports(build::kFp8, cc),
            supports(build::kTma, cc),
            supports(build::kMx, cc),
            build::kTargets};
}

GemmCapabilities capabilities() { return capabilities_for(device_facts()); }

namespace {

/*
 * The shared typed visitor packs case labels with:
 * one u16 per pair, so every label is a compile-time constant and the switch
 * lowers to one indexed branch with no runtime-initialized state. An
 * unsupported pair raises with the actual operand dtypes in the message
 * instead of a hardcoded list that can drift.
 */
using GemmDispatchFn = void (*)(const GemmParams&, cudaStream_t, bool, bool, const DeviceFacts&);

/* The same pair list drives typed dispatch and unsupported-pair diagnostics. */
std::string unsupported_pair_message(c10::ScalarType a, c10::ScalarType b) {
    return std::string("unsupported operand dtype pair ") + toString(a) + " x " + toString(b) +
           " (instantiated: " + dtype_pair_names({ASTRAI_GEMM_PAIRS(ASTRAI_SCALAR_TYPE_PAIR)}) +
           ")";
}

ASTRAI_DEFINE_DTYPE_PAIR_VISITOR(visit_gemm_pair, ASTRAI_GEMM_PAIRS, unsupported_pair_message)

/* All kernel variants have distinct schedule types and architecture images. */
template <typename A, typename B, typename F> auto with_schedule(const DeviceFacts& dev, F&& fn) {
    constexpr bool fp8 = sizeof(A) == 1 && !std::is_same_v<A, int8_t>;
    const GemmCapabilities caps = capabilities_for(dev);
    TORCH_CHECK(fp8 ? caps.fp8 : caps.mma, "GEMM kernels were not built for this device/dtype (SM",
                caps.cc, ", compiled targets: ", caps.targets, ")");
#if ASTRAI_BUILD_MX
    if constexpr (fp8) {
        if (caps.mx && !gemm_mx_cell_disabled())
            return fn(Sm120Mma{});
    }
#endif
#if ASTRAI_BUILD_TMA
    if (caps.tma && !gemm_tma_staging_disabled())
        return fn(TmaMma{});
#endif
    return fn(MmaSync{});
}

template <typename F>
auto with_gemm_variant(c10::ScalarType a, c10::ScalarType b, const DeviceFacts& dev, F&& fn) {
    return visit_gemm_pair(a, b, [&](auto pair) {
        using Pair = decltype(pair);
        return with_schedule<typename Pair::A, typename Pair::B>(
            dev, [&](auto schedule) { return fn(pair, schedule); });
    });
}

using GemmProbeFn = std::pair<PlanDecision, PlanQuery> (*)(
    const GemmParams&, bool, bool, const DeviceFacts&);

/* One selection binds both operations to the same dtype, schedule and device. */
struct GemmVariant {
    GemmDispatchFn dispatch;
    GemmProbeFn probe;
    DeviceFacts device;

    void launch(const PreparedGemm& prepared, cudaStream_t stream) const {
        dispatch(prepared.params, stream, prepared.a.transposed, prepared.b.transposed, device);
    }

    auto inspect(const GemmParams& params, bool trans_a, bool trans_b) const {
        return probe(params, trans_a, trans_b, device);
    }
};

GemmVariant select_gemm_variant(c10::ScalarType a, c10::ScalarType b) {
    const DeviceFacts dev = device_facts();
    return with_gemm_variant(a, b, dev, [&](auto pair, auto schedule) -> GemmVariant {
        using Pair = decltype(pair);
        using Schedule = decltype(schedule);
        return {&gemm_dispatch<typename Pair::A, typename Pair::B, __nv_bfloat16, Schedule>,
                &plan_probe_for<typename Pair::A, typename Pair::B, Schedule>, dev};
    });
}

} // namespace

/*
 * Planner introspection: the Python tooling's C++ face. The heuristic reads
 * CUDA function metadata, but the probe launches nothing. Rows reach the planner
 * through configure()'s rows channel; injected rows rank BELOW the override
 * rows (plan/table.cpp), keeping the override tier authoritative.
 */

PlanProbe plan_probe(int64_t m,
                     int64_t n,
                     int64_t k,
                     at::ScalarType dt_a,
                     at::ScalarType dt_b,
                     bool trans_a,
                     bool trans_b,
                     int64_t batch) {
    TORCH_CHECK(m > 0 && n > 0 && k > 0, "GEMM probe: M, N and K must be greater than zero (got ",
                m, ", ", n, ", ", k, ")");
    GemmParams p{}; // Contiguous operands with aligned base addresses for the probe.
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(n);
    p.k = static_cast<int>(k);
    p.batch = static_cast<int>(batch);
    p.a_ld = trans_a ? m : k;
    p.b_ld = trans_b ? k : n;
    p.a_batch_stride = batch > 1 ? m * k : 0;
    p.b_batch_stride = batch > 1 ? n * k : 0;
    const auto variant = select_gemm_variant(dt_a, dt_b);
    const auto [decision, query] = variant.inspect(p, trans_a, trans_b);
    PlanProbe r;
    r.source = decision.source;
    r.cta = decision.recipe.cta;
    r.k_stages = decision.recipe.k_stages;
    r.raster = decision.raster;
    r.k_tile = decision.recipe.k_tile;
    r.perf_class = query.perf_class;
    r.crosswise = query.crosswise;
    r.tma = query.tma;
    if (config_state().planner == "heuristic" && query.resources) {
        for (const auto& v : tile_vocabulary()) {
            if (v[0] != (query.crosswise > 0) || v[1] != query.ba || v[2] != query.bb)
                continue;
            GemmRecipe recipe{v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11]};
            const auto resource = query.resources(recipe, query);
            const auto& e = resource.effective;
            r.resources.push_back({recipe.cta, recipe.k_stages, recipe.k_tile, e.bm, e.bn, e.k_tile,
                                   e.wm, e.wn, e.threads, resource.resident, resource.registers,
                                   resource.local_bytes});
        }
    }
    return r;
}

/* Torch entry owns the device guard, prepared storage and selected variant. */
torch::Tensor quant_gemm_impl(torch::Tensor a,
                              torch::Tensor b,
                              c10::optional<torch::Tensor> a_scale,
                              c10::optional<torch::Tensor> b_scale,
                              bool trans_a,
                              bool trans_b,
                              c10::optional<torch::Tensor> bias) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors required");
    TORCH_CHECK((a.dim() == 2 || a.dim() == 3) && (b.dim() == 2 || b.dim() == 3),
                "a and b must be 2D or 3D (batched)");
    TORCH_CHECK(a.device() == b.device(), "a and b must share device");
    const at::cuda::OptionalCUDAGuard guard(a.device());
    const GemmInputs inputs{std::move(a), std::move(b), std::move(a_scale), std::move(b_scale),
                            trans_a, trans_b, std::move(bias)};
    const auto prepared = prepare_gemm(inputs);
    const auto variant = select_gemm_variant(inputs.a.scalar_type(), inputs.b.scalar_type());
    variant.launch(prepared, at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_CHECK(cudaGetLastError());
    return prepared.output;
}

} // namespace gemm
} // namespace astrai
