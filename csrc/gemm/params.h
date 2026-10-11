#pragma once
/* Torch operand validation and packing; no planner or kernel selection. */
#include <algorithm>
#include <utility>

#include <torch/extension.h>

#include <api/native/gemm.h>

#include <api/torch/dtype.h>
#include <api/torch/fp8_checks.h>

namespace astrai {
namespace gemm {

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
    int64_t ld;
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
    return {storage, col_major ? storage.stride(-1) : storage.stride(-2),
            storage.dim() == 3 ? storage.stride(0) : 0, transposed != col_major};
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

/* Inputs retain scales and bias until the asynchronous launch is submitted. */
struct GemmInputs {
    torch::Tensor a;
    torch::Tensor b;
    c10::optional<torch::Tensor> a_scale;
    c10::optional<torch::Tensor> b_scale;
    bool trans_a;
    bool trans_b;
    c10::optional<torch::Tensor> bias;
};

/* Resolved storage must outlive the launch, including contiguous fallbacks. */
struct PreparedGemm {
    GemmOperand a;
    GemmOperand b;
    torch::Tensor output;
    GemmParams params;
};

/* The caller validates devices/ranks and holds the CUDA device guard. */
PreparedGemm prepare_gemm(const GemmInputs& inputs) {
    const auto& a = inputs.a;
    const auto& b = inputs.b;
    const auto& a_scale = inputs.a_scale;
    const auto& b_scale = inputs.b_scale;
    const auto& bias = inputs.bias;
    const bool trans_a = inputs.trans_a, trans_b = inputs.trans_b;
    const auto dt_a = a.scalar_type(), dt_b = b.scalar_type();
    const bool i8a = dt_a == torch::kChar, i8b = dt_b == torch::kChar;
    const bool f8a = is_fp8(dt_a);
    const bool f8b = is_fp8(dt_b);
    const bool b16a = dt_a == torch::kBFloat16, b16b = dt_b == torch::kBFloat16;
    /*
     * The supported-pair set is validated once, by the dispatch switch's
     * default arm in entry.cu, with the operand dtypes in the message.
     */
    if (f8a || f8b) {
        astrai::quant::check_fp8_device(a.device().index());
    }
    const int64_t m = trans_a ? a.size(-1) : a.size(-2);
    const int64_t n = trans_b ? b.size(-2) : b.size(-1);
    const int64_t k = trans_a ? a.size(-2) : a.size(-1);
    TORCH_CHECK(m > 0 && n > 0 && k > 0, "quant_gemm: M, N and K must be greater than zero (got ",
                m, ", ", n, ", ", k, ")");
    auto opt_scale = [&](const c10::optional<torch::Tensor>& s, int64_t extent, const char* name,
                         bool i8_side, bool bf16_side) -> QuantScale {
        if (!s.has_value()) {
            TORCH_CHECK(!i8_side, "quant_gemm: ", name, " is required for an int8 operand");
            return {nullptr, 0};
        }
        TORCH_CHECK(!bf16_side, "quant_gemm: ", name,
                    " given for a bf16 operand (nothing to dequant)");
        return resolve_quant_scale(*s, extent, name, a.device());
    };
    const QuantScale sa = opt_scale(a_scale, m, "a_scale", i8a, b16a);
    const QuantScale sb = opt_scale(b_scale, n, "b_scale", i8b, b16b);

    torch::Tensor bias_t;
    if (bias.has_value())
        bias_t = *bias;

    const int64_t batch_a = a.dim() == 3 ? a.size(0) : 1;
    const int64_t batch_b = b.dim() == 3 ? b.size(0) : 1;
    TORCH_CHECK(batch_a == batch_b || batch_a == 1 || batch_b == 1, "batch dim mismatch (got ",
                batch_a, " and ", batch_b, ")");
    const int64_t batch = std::max(batch_a, batch_b);
    TORCH_CHECK(batch <= 65535, "batch dim exceeds the grid.z launch limit");

    auto resolved_a = resolve_operand(a, trans_a);
    auto resolved_b = resolve_operand(b, trans_b);
    TORCH_CHECK(k == (trans_b ? b.size(-1) : b.size(-2)), "inner dim mismatch");
    TORCH_CHECK(sa.ptr == nullptr || sa.n == 0 || sa.n == m, "a_scale extent must match m");
    TORCH_CHECK(sb.ptr == nullptr || sb.n == 0 || sb.n == n, "w_scale extent must match n");

    const bool batched_out = a.dim() == 3 || b.dim() == 3;
    torch::Tensor output = batched_out
                               ? torch::empty({batch, m, n}, a.options().dtype(torch::kBFloat16))
                               : torch::empty({m, n}, a.options().dtype(torch::kBFloat16));
    GemmParams p;
    p.a_ptr = resolved_a.storage.data_ptr();
    p.b_ptr = resolved_b.storage.data_ptr();
    p.out_ptr = output.data_ptr();
    p.a_scale = sa.ptr;
    p.a_scale_m = sa.n;
    p.b_scale = sb.ptr;
    p.b_scale_n = sb.n;
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(n);
    p.k = static_cast<int>(k);
    p.a_ld = static_cast<int>(resolved_a.ld);
    p.b_ld = static_cast<int>(resolved_b.ld);
    if (bias_t.defined() && bias_t.numel() > 0) {
        TORCH_CHECK(bias_t.is_cuda() && bias_t.scalar_type() == torch::kBFloat16,
                    "quantized gemm bias must be a CUDA bf16 tensor");
        TORCH_CHECK(bias_t.device() == a.device(),
                    "quantized gemm bias must be on the same device as operands");
        TORCH_CHECK(bias_t.dim() == 1 && bias_t.size(0) == n,
                    "quantized gemm bias must be 1D of length n=", n);
        TORCH_CHECK(bias_t.is_contiguous(), "bias must be contiguous");
        p.bias_ptr = bias_t.data_ptr();
    }
    p.batch = static_cast<int>(batch);
    p.a_batch_stride = (batch_a == 1 && batch > 1) ? 0 : resolved_a.batch_stride;
    p.b_batch_stride = (batch_b == 1 && batch > 1) ? 0 : resolved_b.batch_stride;
    p.out_batch_stride = m * n;
    p.out_ld = static_cast<int>(n);

    return {std::move(resolved_a), std::move(resolved_b), std::move(output), p};
}

} // namespace
} // namespace gemm
} // namespace astrai
