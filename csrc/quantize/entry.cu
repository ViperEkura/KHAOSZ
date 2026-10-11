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
using QuantOutputPairs = TypeList<DTypePair<fp8_e4m3, fp8_e4m3>,
                                  DTypePair<fp8_e4m3, fp8_e5m2>,
                                  DTypePair<fp8_e5m2, fp8_e4m3>,
                                  DTypePair<fp8_e5m2, fp8_e5m2>>;

void check_scalar(const torch::Tensor& t, c10::Device device, const char* name) {
    TORCH_CHECK(t.defined() && t.is_cuda() && t.device() == device &&
                    t.scalar_type() == at::kFloat && t.numel() == 1,
                name, " must be a CUDA float32 scalar on the input device");
}

/* Initialize the same flat launch parameters for both public operations. */
torch::Tensor prepare(const torch::Tensor& x,
                      const torch::Tensor& inv_scale,
                      c10::optional<at::ScalarType> dtype,
                      c10::optional<at::ScalarType> tdtype,
                      QuantParams& p,
                      QuantizeOutputs& outs) {
    TORCH_CHECK(x.is_cuda(), "quantize requires a CUDA tensor");
    TORCH_CHECK(x.dim() >= 1, "quantize requires a 1D+ input");
    TORCH_CHECK(dtype_supported<QuantInputTypes>(x.scalar_type()),
                unsupported_dtype_message<QuantInputTypes>("quantize", x.scalar_type()));
    TORCH_CHECK(dtype.has_value() || tdtype.has_value(), "quantize requires dtype or tdtype");
    for (const auto dt : {dtype, tdtype})
        TORCH_CHECK(!dt.has_value() || is_fp8(*dt),
                    "quantize output dtype must be float8_e4m3fn or float8_e5m2");
    TORCH_CHECK(!tdtype.has_value() || x.dim() >= 2,
                "transposed quantize output needs a 2D+ tensor");
    check_scalar(inv_scale, x.device(), "inv_scale");
    const auto* device = at::cuda::getDeviceProperties(x.device().index());
    TORCH_CHECK(device->major >= 8, "quantize requires compute capability 8.0+");
    const int64_t cols = x.numel() == 0 ? 1 : x.size(-1);
    const int64_t rows = x.numel() / cols;
    TORCH_CHECK(cols <= INT_MAX - 63 && rows <= INT_MAX - 31,
                "quantize tensor too large for the tiled grid");
    TORCH_CHECK((rows + 31) / 32 <= device->maxGridSize[1],
                "quantize row extent exceeds CUDA grid limit");
    auto input = x.contiguous();
    p.input_ptr = input.data_ptr();
    p.inv_scale = inv_scale.data_ptr<float>();
    p.rows = static_cast<int>(rows);
    p.cols = static_cast<int>(cols);
    p.out_row_stride = p.cols;
    p.out_col_stride = 1;
    if (dtype.has_value()) {
        outs.out = torch::Tensor(
            at::detail::empty_cuda(input.sizes(), *dtype, input.device(), std::nullopt));
        p.output_ptr = outs.out.data_ptr();
    }
    if (tdtype.has_value()) {
        outs.out_t = torch::Tensor(
            at::detail::empty_cuda({cols, rows}, *tdtype, input.device(), std::nullopt));
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
    visit_dtype_pair<QuantOutputPairs>(
        a, b,
        [&](auto pair) {
            using Pair = decltype(pair);
            visit_dtype<QuantInputTypes>(
                input.scalar_type(),
                [&](auto tag) {
                    using T = typename decltype(tag)::Type;
                    launch_fp8_quantize<T, typename Pair::A, typename Pair::B>(p, stream);
                },
                [](at::ScalarType dt) {
                    return unsupported_dtype_message<QuantInputTypes>("quantize", dt);
                });
        },
        [](at::ScalarType a, at::ScalarType b) {
            return c10::str("unsupported quantize output pair ", c10::toString(a), " x ",
                            c10::toString(b));
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
    const auto input = prepare(x, inv_scale, dtype, tdtype, p, outs);
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
                                double fp8_max,
                                const torch::Tensor& next_scale,
                                const torch::Tensor& next_inv_scale) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    QuantParams p{};
    QuantizeOutputs outs;
    const auto input = prepare(x, inv_scale, dtype, tdtype, p, outs);
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
    const float format_max = static_cast<float>(fp8_max);
    TORCH_CHECK(std::isfinite(pow2) && pow2 > 0, "margin produces an invalid scale factor");
    TORCH_CHECK(std::isfinite(format_max) && format_max > 0, "fp8_max must be finite and positive");
    float* base = ring_state.data_ptr<float>();
    outs.amax = ring_state.narrow(0, layout.amax(), 1);
    p.ring_state = base;
    p.hist_len = layout.hist_len;
    p.hist_idx = static_cast<int>(hist_idx);
    p.next_scale = next_scale.defined() ? next_scale.data_ptr<float>() : base + layout.scale();
    p.next_inv_scale =
        next_inv_scale.defined() ? next_inv_scale.data_ptr<float>() : base + layout.recip();
    p.fp8_max = format_max;
    p.pow2_margin = pow2;
    launch(input, p, dtype, tdtype);
    return outs;
}
} // namespace quant
} // namespace astrai
