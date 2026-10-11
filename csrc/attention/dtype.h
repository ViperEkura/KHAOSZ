#pragma once
/* Attention entry dtype dispatch, shared by its four translation units. */
#include <api/torch/dispatch.cuh>
#include <cuda_runtime.h>

namespace astrai {
namespace attention {

#define ASTRAI_ATTN_DTYPES(X) X(bf16) X(fp16)

inline std::string unsupported_attention_dtype(at::ScalarType dtype) {
    return "attention has no kernel for " + std::string(c10::toString(dtype)) +
           " (instantiated: " + dtype_names({ASTRAI_ATTN_DTYPES(ASTRAI_SCALAR_TYPE)}) + ")";
}

ASTRAI_DEFINE_DTYPE_VISITOR(visit_attention_dtype, ASTRAI_ATTN_DTYPES, unsupported_attention_dtype)

template <template <typename> class DispatchFn, typename ParamsT>
inline void attn_dtype_dispatch(at::ScalarType dtype, ParamsT& params, cudaStream_t stream) {
    visit_attention_dtype(dtype, [&](auto tag) {
        using T = typename decltype(tag)::Type;
        DispatchFn<T>::run(params, stream);
    });
}

} // namespace attention
} // namespace astrai
