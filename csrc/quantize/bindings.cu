/*
 * CUDA bindings for the stateless FP8 quantize primitives. The launcher,
 * the ring binding and the composed pass live in ``entry.cu`` (beside this
 * file, declared in api/torch/quantize.h) so the fp8-linear composition
 * (``gemm/fp8_linear.cu``, compiled into the gemm module where the GEMM
 * dispatch state lives) shares them instead of re-deriving them; this TU is
 * only the pybind surface.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <torch/extension.h>

#include <api/torch/dtype.h>
#include <api/torch/quantize.h>

using namespace astrai::quant;

namespace {

/*
 * Shared body for quantize() (RowMajor/Transposed, one output) and
 * quantize_dual() (Dual: both orientations from one read). With ring the
 * kernel runs the delayed-scaling fold (RingView, launch.cuh) and the
 * returned amax is the ring's self-cleaned persistent slot — its only
 * reducer; without ring it is a pure scale+cast and amax is None (dynamic
 * scaling measures its own). ``transposed_dtype`` (default: the row-major
 * format) recasts the transposed side independently (hybrid fwd/bwd pair).
 */
py::object quantize_impl(torch::Tensor x,
                         torch::Tensor scale,
                         at::ScalarType out_dtype,
                         QuantLayout layout,
                         py::object transposed_dtype,
                         py::object ring,
                         int64_t hist_idx,
                         py::object hist_len,
                         py::object fp8_max,
                         double pow2_margin) {
    c10::optional<at::ScalarType> t_dtype = c10::nullopt;
    if (!transposed_dtype.is_none())
        t_dtype = transposed_dtype.cast<at::ScalarType>();
    c10::optional<int64_t> ring_hist_len = c10::nullopt;
    if (!hist_len.is_none())
        ring_hist_len = hist_len.cast<int64_t>();
    c10::optional<torch::Tensor> ring_state = c10::nullopt;
    if (!ring.is_none()) {
        torch::Tensor t = ring.cast<torch::Tensor>();
        TORCH_CHECK(t.defined(), "ring_state must be a defined tensor");
        ring_state = t;
    }
    const double format_max = out_dtype == astrai::scalar_type_v<astrai::fp8_e5m2>
                                  ? astrai::ElemTrait<astrai::fp8_e5m2>::kFiniteMax
                                  : astrai::ElemTrait<astrai::fp8_e4m3>::kFiniteMax;
    const double max = fp8_max.is_none() ? format_max : fp8_max.cast<double>();
    const QuantizeOutputs outs =
        run_quantize(x, scale, layout, out_dtype, t_dtype, ring_state, hist_idx, max, pow2_margin,
                     c10::nullopt, c10::nullopt, ring_hist_len);

    if (layout == QuantLayout::Dual)
        return py::make_tuple(outs.out, outs.out_t, outs.amax);
    else if (layout == QuantLayout::Transposed)
        return py::make_tuple(outs.out_t, outs.amax);
    else
        return py::make_tuple(outs.out, outs.amax);
}

/*
 * Single-orientation quantize binding: row-major x8, or its [cols][rows]
 * transpose when transposed is set — the K-contiguous operand orientation
 * NT GEMMs want. Returns (x8|x8T, amax); amax is the fold's raw-domain amax
 * of the round when ring_state is given, else None (pure scale+cast).
 */
py::object quantize(torch::Tensor x,
                    torch::Tensor scale,
                    at::ScalarType dtype,
                    bool transposed,
                    py::object ring,
                    int64_t hist_idx,
                    py::object hist_len,
                    py::object fp8_max,
                    double pow2_margin) {
    const QuantLayout layout = transposed ? QuantLayout::Transposed : QuantLayout::RowMajor;
    return quantize_impl(x, scale, dtype, layout, py::none(), ring, hist_idx, hist_len, fp8_max,
                         pow2_margin);
}

/*
 * Dual-orientation quantize binding: one read of x produces both the
 * row-major x8 and its transpose (plus amax), for tensors consumed by GEMMs
 * in both orientations (backward g). ``transposed_dtype`` casts the
 * transposed side in a different fp8 format from one read (hybrid training:
 * E4M3 forward operand, E5M2 backward operand). Returns (x8, x8T, amax),
 * amax as above.
 */
py::object quantize_dual(torch::Tensor x,
                         torch::Tensor scale,
                         at::ScalarType dtype,
                         py::object transposed_dtype,
                         py::object ring,
                         int64_t hist_idx,
                         py::object hist_len,
                         py::object fp8_max,
                         double pow2_margin) {
    return quantize_impl(x, scale, dtype, QuantLayout::Dual, transposed_dtype, ring, hist_idx,
                         hist_len, fp8_max, pow2_margin);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    /*
     * The fold-scratch extent, for policy/tests that size the ring buffer
     * (the full layout is RingLayout, api/native/quantize.h).
     */
    m.attr("K_FOLD_SLOTS") = kFoldSlots;
    m.def("quantize", &quantize, py::arg("x"), py::arg("scale"), py::arg("dtype"),
          py::arg("transposed") = false, py::arg("ring") = py::none(), py::arg("hist_idx") = 0,
          py::arg("hist_len") = py::none(), py::arg("fp8_max") = py::none(),
          py::arg("pow2_margin") = 1.0,
          "Scaled cast to fp8/int8; returns (x8|x8T, amax) — amax None without a ring");
    m.def("quantize_dual", &quantize_dual, py::arg("x"), py::arg("scale"), py::arg("dtype"),
          py::arg("transposed_dtype") = py::none(), py::arg("ring") = py::none(),
          py::arg("hist_idx") = 0, py::arg("hist_len") = py::none(),
          py::arg("fp8_max") = py::none(), py::arg("pow2_margin") = 1.0,
          "One read produces both orientations: returns (x8, x8T, amax)");
}
