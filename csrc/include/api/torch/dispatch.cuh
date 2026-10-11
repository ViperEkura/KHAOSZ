#pragma once
/* CUDA element/ScalarType mapping and reusable typed visitor generators. */
#include <cstdint>
#include <stdexcept>

#include <api/torch/dtype.h>
#include <datatype/element.cuh>

namespace astrai {

template <typename T> struct ScalarTypeOf;
template <> struct ScalarTypeOf<bf16> {
    static constexpr at::ScalarType value = at::kBFloat16;
};
template <> struct ScalarTypeOf<fp16> {
    static constexpr at::ScalarType value = at::kHalf;
};
template <> struct ScalarTypeOf<float> {
    static constexpr at::ScalarType value = at::kFloat;
};
template <> struct ScalarTypeOf<int8_t> {
    static constexpr at::ScalarType value = at::kChar;
};
template <> struct ScalarTypeOf<fp8_e4m3> {
    static constexpr at::ScalarType value = at::kFloat8_e4m3fn;
};
template <> struct ScalarTypeOf<fp8_e5m2> {
    static constexpr at::ScalarType value = at::kFloat8_e5m2;
};

template <typename T> inline constexpr at::ScalarType scalar_type_v = ScalarTypeOf<T>::value;

template <typename T> struct DTypeTag {
    using Type = T;
};

template <typename A_, typename B_> struct DTypePairTag {
    using A = A_;
    using B = B_;
};

constexpr uint16_t scalar_pair_key(at::ScalarType a, at::ScalarType b) {
    return static_cast<uint16_t>(static_cast<uint8_t>(a)) << 8 | static_cast<uint8_t>(b);
}

} // namespace astrai

/* X-macro lists live in the calling family. The generators provide a typed
 * visitor named NAME with callback fn, plus that family's error formatter.
 * Supported-type diagnostics expand the very same list as the switch. */
#define ASTRAI_SCALAR_TYPE(T) ::astrai::scalar_type_v<T>,
#define ASTRAI_SCALAR_TYPE_PAIR(A, B) {::astrai::scalar_type_v<A>, ::astrai::scalar_type_v<B>},
#define ASTRAI_DTYPE_CASE(T)                                                                       \
    case ::astrai::scalar_type_v<T>:                                                               \
        return fn(::astrai::DTypeTag<T>{});
#define ASTRAI_DTYPE_PAIR_CASE(A, B)                                                               \
    case ::astrai::scalar_pair_key(::astrai::scalar_type_v<A>, ::astrai::scalar_type_v<B>):        \
        return fn(::astrai::DTypePairTag<A, B>{});
#define ASTRAI_DEFINE_DTYPE_VISITOR(NAME, LIST, ERROR)                                             \
    template <typename F> auto NAME(at::ScalarType dtype, F&& fn) {                                \
        switch (dtype) { LIST(ASTRAI_DTYPE_CASE) }                                                 \
        throw std::runtime_error(ERROR(dtype));                                                    \
    }
#define ASTRAI_DEFINE_DTYPE_PAIR_VISITOR(NAME, LIST, ERROR)                                        \
    template <typename F> auto NAME(at::ScalarType a, at::ScalarType b, F&& fn) {                  \
        switch (::astrai::scalar_pair_key(a, b)) { LIST(ASTRAI_DTYPE_PAIR_CASE) }                  \
        throw std::runtime_error(ERROR(a, b));                                                     \
    }
