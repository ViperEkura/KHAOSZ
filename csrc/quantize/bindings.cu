/* Python defaults and tuple conversion; execution lives in the Torch entry. */
#include <api/native/quantize.h>
#include <api/torch/dtype.h>
#include <api/torch/quantize.h>
#include <torch/extension.h>

namespace {
using torch::Tensor;
using DType = c10::optional<at::ScalarType>;

py::tuple result(const astrai::quant::QuantizeOutputs& outs) {
    return py::make_tuple(outs.out, outs.out_t, outs.amax);
}

py::tuple quantize(const Tensor& x, const Tensor& inv_scale, DType dtype, DType tdtype) {
    return result(astrai::quant::quantize(x, inv_scale, dtype, tdtype));
}

py::tuple quantize_scaled(const Tensor& x,
                          const Tensor& inv_scale,
                          const Tensor& ring_state,
                          int64_t hist_len,
                          int64_t hist_idx,
                          DType dtype,
                          DType tdtype,
                          int64_t margin,
                          c10::optional<double> quant_max,
                          const c10::optional<Tensor>& next_scale,
                          const c10::optional<Tensor>& next_inv_scale) {
    TORCH_CHECK(dtype.has_value() || tdtype.has_value(),
                "quantize_scaled requires dtype or tdtype");
    const double max = quant_max.has_value()
                           ? *quant_max
                           : astrai::quant_max_of(dtype.has_value() ? *dtype : *tdtype);
    return result(astrai::quant::quantize_scaled(
        x, inv_scale, ring_state, hist_len, hist_idx, dtype, tdtype, margin, max,
        next_scale.value_or(Tensor()), next_inv_scale.value_or(Tensor())));
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("K_FOLD_SLOTS") = astrai::quant::kFoldSlots;
    m.def("quantize", &quantize, py::arg("x"), py::arg("inv_scale"), py::kw_only(),
          py::arg("dtype") = py::none(), py::arg("tdtype") = py::none(),
          "Scale and cast requested orientations; returns (out, out_t, None)");
    m.def("quantize_scaled", &quantize_scaled, py::arg("x"), py::arg("inv_scale"),
          py::arg("ring_state"), py::kw_only(), py::arg("hist_len"), py::arg("hist_idx"),
          py::arg("dtype") = py::none(), py::arg("tdtype") = py::none(), py::arg("margin") = 0,
          py::arg("quant_max") = py::none(), py::arg("next_scale") = py::none(),
          py::arg("next_inv_scale") = py::none(),
          "Quantize and publish the next scale; returns (out, out_t, amax)");
}
