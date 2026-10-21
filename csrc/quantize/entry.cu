/* Torch validation, allocation and dtype dispatch, shared by both modules. */
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <cmath>

#include <api/native/quantize.h>
#include <api/torch/dispatch.cuh>
#include <api/torch/quantize.h>
#include <launcher/quantize.cuh>

namespace astrai {
namespace quant {
namespace {

using QuantInputTypes = TypeList<bf16, fp16, float>;
using QuantOutputTypes = TypeList<fp8_e4m3, fp8_e5m2, int8_t>;

void check_scalar(const torch::Tensor& t, c10::Device device, const char* name) {
    TORCH_CHECK(t.defined() && t.is_cuda() && t.device() == device &&
                    t.scalar_type() == at::kFloat && t.numel() == 1,
                name, " must be a CUDA float32 scalar on the input device");
}

/* Validate input metadata and bind geometry without allocating storage. */
void check_inputs(const torch::Tensor& x,
                  const torch::Tensor& inv_scale,
                  c10::optional<at::ScalarType> dtype,
                  c10::optional<at::ScalarType> tdtype,
                  QuantParams& p) {
    TORCH_CHECK(x.is_cuda(), "quantize requires a CUDA tensor");
    TORCH_CHECK(x.dim() >= 1, "quantize requires a 1D+ input");
    TORCH_CHECK(dtype_supported<QuantInputTypes>(x.scalar_type()),
                unsupported_dtype_message<QuantInputTypes>("quantize", x.scalar_type()));
    TORCH_CHECK(dtype.has_value() || tdtype.has_value(), "quantize requires dtype or tdtype");
    for (const auto dt : {dtype, tdtype})
        TORCH_CHECK(!dt.has_value() || dtype_supported<QuantOutputTypes>(*dt),
                    "quantize output dtype must be float8_e4m3fn, float8_e5m2 or int8");
    TORCH_CHECK(!tdtype.has_value() || x.dim() >= 2,
                "transposed quantize output needs a 2D+ tensor");
    const auto* device = at::cuda::getDeviceProperties(x.device().index());
    TORCH_CHECK(device->major >= 8, "quantize requires compute capability 8.0+");
    const int64_t input_cols = x.size(-1);
    int64_t input_rows = 1;
    for (int axis = 0; axis + 1 < x.dim(); ++axis) {
        const int64_t size = x.size(axis);
        TORCH_CHECK(size == 0 || input_rows <= INT64_MAX / size,
                    "quantize row extent overflows int64");
        input_rows *= size;
    }
    // Preserve the empty transpose geometry; multiplier validation uses the logical shape.
    const int64_t cols = x.numel() == 0 ? 1 : input_cols;
    const int64_t rows = x.numel() == 0 ? 0 : input_rows;
    TORCH_CHECK(cols <= INT_MAX - 63 && rows <= INT_MAX - 31,
                "quantize tensor too large for the tiled grid");
    TORCH_CHECK((rows + 31) / 32 <= device->maxGridSize[1],
                "quantize row extent exceeds CUDA grid limit");
    TORCH_CHECK(inv_scale.defined() && inv_scale.is_cuda() && inv_scale.device() == x.device() &&
                    inv_scale.scalar_type() == at::kFloat,
                "inv_scale must be a CUDA float32 tensor on the input device");
    if (inv_scale.numel() != 1) {
        TORCH_CHECK(inv_scale.dim() == 1 || inv_scale.dim() == 2,
                    "inv_scale must be a scalar or a 1D/2D tensor broadcastable to [rows, cols]");
        const int64_t scale_rows = inv_scale.dim() == 1 ? 1 : inv_scale.size(0);
        const int64_t scale_cols = inv_scale.size(-1);
        TORCH_CHECK((scale_rows == 1 || scale_rows == input_rows) &&
                        (scale_cols == 1 || scale_cols == input_cols),
                    "inv_scale shape must broadcast to flattened input [", input_rows, ", ",
                    input_cols, "]");
        const int64_t row_stride = scale_rows == 1 ? 0 : inv_scale.stride(0);
        const int64_t col_stride = scale_cols == 1 ? 0 : inv_scale.stride(-1);
        TORCH_CHECK(row_stride >= 0 && row_stride <= INT_MAX && col_stride >= 0 &&
                        col_stride <= INT_MAX,
                    "inv_scale strides must fit nonnegative native int fields");
        p.inv_scale_row_stride = static_cast<int>(row_stride);
        p.inv_scale_col_stride = static_cast<int>(col_stride);
    }
    p.inv_scale = inv_scale.data_ptr<float>();
    p.rows = static_cast<int>(rows);
    p.cols = static_cast<int>(cols);
    p.out_row_stride = p.cols;
    p.out_col_stride = 1;
}

/* Materialize only validated requests; the returned Tensor owns any input copy. */
torch::Tensor prepare_outputs(const torch::Tensor& x,
                              c10::optional<at::ScalarType> dtype,
                              c10::optional<at::ScalarType> tdtype,
                              QuantParams& p,
                              QuantizeOutputs& outs) {
    auto input = x.contiguous();
    p.input_ptr = input.data_ptr();
    if (dtype.has_value()) {
        outs.out = torch::Tensor(
            at::detail::empty_cuda(input.sizes(), *dtype, input.device(), std::nullopt));
        p.output_ptr = outs.out.data_ptr();
    }
    if (tdtype.has_value()) {
        outs.out_t = torch::Tensor(
            at::detail::empty_cuda({p.cols, p.rows}, *tdtype, input.device(), std::nullopt));
        p.output_transposed_ptr = outs.out_t.data_ptr();
    }
    return input;
}

void launch(const torch::Tensor& input,
            const QuantParams& p,
            c10::optional<at::ScalarType> dtype,
            c10::optional<at::ScalarType> tdtype) {
    // Missing orientations use the requested format without allocating a tensor.
    const auto a = dtype.has_value() ? *dtype : *tdtype;
    const auto b = tdtype.has_value() ? *tdtype : a;
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto output_error = [](at::ScalarType dt) {
        return unsupported_dtype_message<QuantOutputTypes>("quantize output", dt);
    };
    visit_dtype<QuantInputTypes>(
        input.scalar_type(),
        [&](auto input_tag) {
            using InT = typename decltype(input_tag)::Type;
            visit_dtype<QuantOutputTypes>(
                a,
                [&](auto output_tag) {
                    using OutT = typename decltype(output_tag)::Type;
                    visit_dtype<QuantOutputTypes>(
                        b,
                        [&](auto transposed_tag) {
                            using TransposedOutT = typename decltype(transposed_tag)::Type;
                            launch_quantize<InT, OutT, TransposedOutT>(p, stream);
                        },
                        output_error);
                },
                output_error);
        },
        [](at::ScalarType dt) {
            return unsupported_dtype_message<QuantInputTypes>("quantize", dt);
        });
}
} // namespace

QuantizeOutputs quantize(const torch::Tensor& x,
                         const torch::Tensor& inv_scale,
                         c10::optional<at::ScalarType> dtype,
                         c10::optional<at::ScalarType> tdtype) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    QuantParams p{};
    QuantizeOutputs outs;
    check_inputs(x, inv_scale, dtype, tdtype, p);
    const auto input = prepare_outputs(x, dtype, tdtype, p, outs);
    launch(input, p, dtype, tdtype);
    return outs;
}

QuantizeOutputs quantize_scaled(const torch::Tensor& x,
                                const torch::Tensor& inv_scale,
                                const torch::Tensor& ring_state,
                                int64_t hist_len,
                                int64_t hist_idx,
                                c10::optional<at::ScalarType> dtype,
                                c10::optional<at::ScalarType> tdtype,
                                int64_t margin,
                                double quant_max,
                                const torch::Tensor& next_scale,
                                const torch::Tensor& next_inv_scale) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    QuantParams p{};
    QuantizeOutputs outs;
    check_inputs(x, inv_scale, dtype, tdtype, p);
    TORCH_CHECK(inv_scale.numel() == 1, "quantize_scaled requires a scalar inv_scale");
    TORCH_CHECK(ring_state.defined() && ring_state.is_cuda() && ring_state.device() == x.device() &&
                    ring_state.dim() == 1 && ring_state.scalar_type() == at::kFloat &&
                    ring_state.is_contiguous(),
                "ring_state must be a contiguous 1D CUDA float32 tensor on the input device");
    TORCH_CHECK(hist_len > 0 && hist_len <= INT_MAX - kFoldSlots - 6,
                "hist_len must be positive and fit the native ring layout");
    TORCH_CHECK(hist_idx >= 0 && hist_idx < hist_len, "hist_idx out of range");
    const RingLayout layout{static_cast<int>(hist_len)};
    TORCH_CHECK(ring_state.numel() >= layout.size(1), "ring_state too small for hist_len");
    TORCH_CHECK(next_scale.defined() == next_inv_scale.defined(),
                "next_scale and next_inv_scale must be provided together");
    if (next_scale.defined()) {
        check_scalar(next_scale, x.device(), "next_scale");
        check_scalar(next_inv_scale, x.device(), "next_inv_scale");
    }
    const float pow2 = static_cast<float>(std::pow(2.0, static_cast<double>(margin)));
    const float format_max = static_cast<float>(quant_max);
    TORCH_CHECK(std::isfinite(pow2) && pow2 > 0, "margin produces an invalid scale factor");
    TORCH_CHECK(std::isfinite(format_max) && format_max > 0,
                "quant_max must be finite and positive");
    float* base = ring_state.data_ptr<float>();
    outs.amax = ring_state.narrow(0, layout.amax(), 1);
    p.ring_state = base;
    p.hist_len = layout.hist_len;
    p.hist_idx = static_cast<int>(hist_idx);
    p.next_scale = next_scale.defined() ? next_scale.data_ptr<float>() : base + layout.scale();
    p.next_inv_scale =
        next_inv_scale.defined() ? next_inv_scale.data_ptr<float>() : base + layout.recip();
    p.quant_max = format_max;
    p.pow2_margin = pow2;
    const auto input = prepare_outputs(x, dtype, tdtype, p, outs);
    launch(input, p, dtype, tdtype);
    return outs;
}
} // namespace quant
} // namespace astrai
