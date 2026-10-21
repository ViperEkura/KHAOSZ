/* GEMM typed entry, dtype/schedule selection, and planner probe. */

#include <algorithm>
#include <c10/core/ScalarType.h>
#include <c10/util/StringUtil.h>
#include <climits>
#include <cstdint>
#include <string>
#include <utility>

#include <ATen/cuda/CUDAContext.h>
#include <api/torch/dispatch.cuh>
#include <api/torch/gemm.h>
#include <c10/cuda/CUDAGuard.h>
#include <gemm_build.h>
#include <launcher/gemm_dispatch.cuh>
#include <policy/gemm/config.h>
#include <utils/device.cuh>

using namespace astrai;
using namespace astrai::quant;

namespace astrai {
namespace gemm {

using GemmBasePairs =
    TypeList<DTypePair<bf16, bf16>, DTypePair<bf16, int8_t>, DTypePair<int8_t, int8_t>>;
#if ASTRAI_BUILD_FP8
using GemmFp8Pairs = TypeList<DTypePair<fp8_e4m3, fp8_e4m3>, DTypePair<fp8_e5m2, fp8_e5m2>>;
#else
using GemmFp8Pairs = TypeList<>;
#endif
using GemmPairs = concat_type_lists_t<GemmBasePairs, GemmFp8Pairs>;

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
 * Inner-layout resolution for one GEMM operand. The user flag names the math
 * (0 = last two dims are [rows][contract], 1 = transposed); the storage may
 * independently be a col-major view (.t() of a contiguous buffer), which
 * folds into the returned dispatch flag at zero copy — the kernel's
 * LayoutA/LayoutB tags cover both storages. m/n/k derive from the user flag
 * only. Tensors whose inner dims are neither natural layout fall back to
 * .contiguous().
 */
struct GemmOperand {
    torch::Tensor storage;
    int ld;
    int64_t batch_stride;
    bool transposed;
};

GemmOperand resolve_operand(const torch::Tensor& input, bool transposed) {
    torch::Tensor storage = input;
    bool col_major = false;
    if (storage.stride(-1) != 1) {
        if (storage.stride(-2) == 1) {
            col_major = true;
        } else {
            storage = storage.contiguous();
        }
    }
    const int64_t ld = col_major ? storage.stride(-1) : storage.stride(-2);
    TORCH_CHECK(ld >= 0 && ld <= INT_MAX, "operand leading stride must fit native int");
    const int64_t batch_stride = storage.dim() == 3 ? storage.stride(0) : 0;
    return {std::move(storage), static_cast<int>(ld), batch_stride, transposed != col_major};
}

// A resolved dequant scale: n == 0 marks the per-tensor device scalar.
struct QuantScale {
    const float* ptr = nullptr;
    int n = 0;
};

QuantScale resolve_quant_scale(const torch::Tensor& s,
                               int64_t extent,
                               const char* name,
                               const c10::Device& device) {
    TORCH_CHECK(s.defined(), name, " is required");
    TORCH_CHECK(s.is_cuda() && s.scalar_type() == torch::kFloat32 && s.is_contiguous(), name,
                " must be a contiguous CUDA float32 tensor");
    TORCH_CHECK(s.device() == device, name, " must be on the same device as operands");
    TORCH_CHECK(s.numel() == 1 || s.numel() == extent, name,
                " must hold 1 element (per-tensor) or ", extent, " (per-row/per-channel)");
    return {s.data_ptr<float>(), s.numel() == 1 ? 0 : (int)extent};
}

/* Resolved storage must outlive the launch, including contiguous fallbacks. */
struct PreparedGemm {
    GemmOperand a;
    GemmOperand b;
    torch::Tensor output;
    GemmParams params;
};

/* Launch and probe share the same native integer contract. */
void check_gemm_geometry(int64_t m, int64_t n, int64_t k, int64_t batch) {
    TORCH_CHECK(m > 0 && n > 0 && k > 0, "GEMM: M, N and K must be greater than zero (got ", m,
                ", ", n, ", ", k, ")");
    TORCH_CHECK(m <= INT_MAX && n <= INT_MAX && k <= INT_MAX,
                "GEMM: M, N and K must fit native int");
    TORCH_CHECK(batch > 0 && batch <= 65535, "GEMM: batch must be in [1, 65535]");
}

/* Check problem, scales and bias before materializing operand storage. */
PreparedGemm prepare_gemm(const torch::Tensor& a,
                          const torch::Tensor& b,
                          const c10::optional<torch::Tensor>& a_scale,
                          const c10::optional<torch::Tensor>& b_scale,
                          bool trans_a,
                          bool trans_b,
                          const c10::optional<torch::Tensor>& bias) {
    const auto dt_a = a.scalar_type(), dt_b = b.scalar_type();
    const int64_t m = trans_a ? a.size(-1) : a.size(-2);
    const int64_t n = trans_b ? b.size(-2) : b.size(-1);
    const int64_t k = trans_a ? a.size(-2) : a.size(-1);
    const int64_t batch_a = a.dim() == 3 ? a.size(0) : 1;
    const int64_t batch_b = b.dim() == 3 ? b.size(0) : 1;
    TORCH_CHECK(batch_a > 0 && batch_b > 0, "GEMM: input batch dimensions must be positive");
    TORCH_CHECK(batch_a == batch_b || batch_a == 1 || batch_b == 1, "batch dim mismatch (got ",
                batch_a, " and ", batch_b, ")");
    const int64_t batch = std::max(batch_a, batch_b);
    check_gemm_geometry(m, n, k, batch);
    TORCH_CHECK(k == (trans_b ? b.size(-1) : b.size(-2)), "inner dim mismatch");
    const auto opt_scale = [&](const c10::optional<torch::Tensor>& s, int64_t extent,
                               const char* name, at::ScalarType dtype) -> QuantScale {
        if (!s.has_value()) {
            TORCH_CHECK(dtype != at::kChar, "quant_gemm: ", name,
                        " is required for an int8 operand");
            return {nullptr, 0};
        }
        TORCH_CHECK(dtype != at::kBFloat16, "quant_gemm: ", name,
                    " given for a bf16 operand (nothing to dequant)");
        return resolve_quant_scale(*s, extent, name, a.device());
    };
    const QuantScale sa = opt_scale(a_scale, m, "a_scale", dt_a);
    const QuantScale sb = opt_scale(b_scale, n, "b_scale", dt_b);
    const void* bias_ptr = nullptr;
    if (bias.has_value() && bias->defined() && bias->numel() > 0) {
        TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == at::kBFloat16,
                    "quantized gemm bias must be a CUDA bf16 tensor");
        TORCH_CHECK(bias->device() == a.device(),
                    "quantized gemm bias must be on the same device as operands");
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == n,
                    "quantized gemm bias must be 1D of length n=", n);
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        bias_ptr = bias->data_ptr();
    }

    auto resolved_a = resolve_operand(a, trans_a);
    auto resolved_b = resolve_operand(b, trans_b);
    const bool batched_out = a.dim() == 3 || b.dim() == 3;
    torch::Tensor output = batched_out
                               ? torch::empty({batch, m, n}, a.options().dtype(at::kBFloat16))
                               : torch::empty({m, n}, a.options().dtype(at::kBFloat16));
    GemmParams p{};
    p.a_ptr = resolved_a.storage.data_ptr();
    p.b_ptr = resolved_b.storage.data_ptr();
    p.out_ptr = output.data_ptr();
    p.bias_ptr = bias_ptr;
    p.a_scale = sa.ptr;
    p.a_scale_m = sa.n;
    p.b_scale = sb.ptr;
    p.b_scale_n = sb.n;
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(n);
    p.k = static_cast<int>(k);
    p.a_ld = resolved_a.ld;
    p.b_ld = resolved_b.ld;
    p.batch = static_cast<int>(batch);
    p.a_batch_stride = (batch_a == 1 && batch > 1) ? 0 : resolved_a.batch_stride;
    p.b_batch_stride = (batch_b == 1 && batch > 1) ? 0 : resolved_b.batch_stride;
    p.out_batch_stride = m * n;
    p.out_ld = static_cast<int>(n);
    return {std::move(resolved_a), std::move(resolved_b), std::move(output), p};
}

/* The launcher header binds these addresses to per-pair compiled specializations. */
using GemmDispatchFn = void (*)(const GemmParams&, cudaStream_t, bool, bool, const DeviceFacts*);

/* The same pair list drives typed dispatch and unsupported-pair diagnostics. */
std::string unsupported_pair_message(c10::ScalarType a, c10::ScalarType b) {
    return c10::str("unsupported operand dtype pair ", toString(a), " x ", toString(b),
                    " (instantiated: ", dtype_pair_names<GemmPairs>(), ")");
}

/* All kernel variants have distinct schedule types and architecture images. */
template <typename A, typename B, typename F> auto with_schedule(const DeviceFacts& dev, F&& fn) {
    constexpr bool fp8 = sizeof(A) == 1 && !std::is_same_v<A, int8_t>;
    if constexpr (fp8)
        TORCH_CHECK(dev.cc >= 89, "FP8 MMA requires compute capability 8.9+");
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
    return visit_dtype_pair<GemmPairs>(
        a, b,
        [&](auto pair) {
            using Pair = decltype(pair);
            return with_schedule<typename Pair::A, typename Pair::B>(
                dev, [&](auto schedule) { return fn(pair, schedule); });
        },
        unsupported_pair_message);
}

using GemmProbeFn = std::pair<PlanDecision, PlanQuery> (*)(const GemmParams&,
                                                           bool,
                                                           bool,
                                                           const DeviceFacts&);

/* One selection binds both operations to the same dtype, schedule and device. */
struct GemmVariant {
    GemmDispatchFn dispatch;
    GemmProbeFn probe;
    DeviceFacts device;

    void launch(const PreparedGemm& prepared, cudaStream_t stream) const {
        dispatch(prepared.params, stream, prepared.a.transposed, prepared.b.transposed, &device);
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
    check_gemm_geometry(m, n, k, batch);
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
    const auto variant = select_gemm_variant(a.scalar_type(), b.scalar_type());
    const auto prepared = prepare_gemm(a, b, a_scale, b_scale, trans_a, trans_b, bias);
    variant.launch(prepared, at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_CHECK(cudaGetLastError());
    return prepared.output;
}

} // namespace gemm
} // namespace astrai
