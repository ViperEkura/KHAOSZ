#pragma once
/* CUDA element/ScalarType mapping and type-list visitors. */
#include <array>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

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

template <typename... Ts> struct TypeList {};

template <typename T> struct DTypeTag {
    using Type = T;
};

template <typename A_, typename B_> struct DTypePair {
    using A = A_;
    using B = B_;
};

template <typename Left, typename Right> struct ConcatTypeLists;
template <typename... Left, typename... Right>
struct ConcatTypeLists<TypeList<Left...>, TypeList<Right...>> {
    using Type = TypeList<Left..., Right...>;
};
template <typename Left, typename Right>
using concat_type_lists_t = typename ConcatTypeLists<Left, Right>::Type;

constexpr uint16_t scalar_pair_key(at::ScalarType a, at::ScalarType b) {
    return static_cast<uint16_t>(static_cast<uint8_t>(a)) << 8 | static_cast<uint8_t>(b);
}

namespace detail {

template <typename T> struct ScalarEntry {
    using Tag = DTypeTag<T>;
    static constexpr uint16_t key = static_cast<uint16_t>(scalar_type_v<T>);
};
template <typename Pair> struct PairEntry;
template <typename A, typename B> struct PairEntry<DTypePair<A, B>> {
    using Tag = DTypePair<A, B>;
    static constexpr uint16_t key = scalar_pair_key(scalar_type_v<A>, scalar_type_v<B>);
};

/* One traversal implements scalar and legal-pair visitors. No Cartesian product. */
template <template <typename> class Entry, typename List> struct DTypeListDispatch;
template <template <typename> class Entry, typename... Ts>
struct DTypeListDispatch<Entry, TypeList<Ts...>> {
    static constexpr std::array<uint16_t, sizeof...(Ts)> keys{Entry<Ts>::key...};

    static constexpr bool unique() {
        for (size_t i = 0; i < keys.size(); ++i)
            for (size_t j = i + 1; j < keys.size(); ++j)
                if (keys[i] == keys[j])
                    return false;
        return true;
    }
    static_assert(unique(), "duplicate dtype dispatch key");

    static constexpr bool supports(uint16_t key) {
        for (auto candidate : keys)
            if (candidate == key)
                return true;
        return false;
    }

    template <size_t I, typename R, typename F, typename Miss>
    static R invoke(uint16_t key, F&& fn, Miss&& miss) {
        if constexpr (I == sizeof...(Ts)) {
            throw std::runtime_error(std::forward<Miss>(miss)());
        } else {
            using T = std::tuple_element_t<I, std::tuple<Ts...>>;
            if (key == keys[I])
                return std::forward<F>(fn)(typename Entry<T>::Tag{});
            return invoke<I + 1, R>(key, std::forward<F>(fn), std::forward<Miss>(miss));
        }
    }

    template <typename F, typename Miss>
    static decltype(auto) visit(uint16_t key, F&& fn, Miss&& miss) {
        if constexpr (sizeof...(Ts) == 0) {
            static_assert(sizeof...(Ts) > 0, "dtype visitor requires a nonempty type list");
        } else {
            using First = std::tuple_element_t<0, std::tuple<Ts...>>;
            using R = std::invoke_result_t<F&&, typename Entry<First>::Tag>;
            static_assert(
                (std::is_same_v<R, std::invoke_result_t<F&&, typename Entry<Ts>::Tag>> && ...),
                "dtype visitor callbacks must have the same return type");
            return invoke<0, R>(key, std::forward<F>(fn), std::forward<Miss>(miss));
        }
    }
};

template <typename List> struct DTypeListNames;
template <typename... Ts> struct DTypeListNames<TypeList<Ts...>> {
    static std::string scalars() {
        using Dispatch = DTypeListDispatch<ScalarEntry, TypeList<Ts...>>;
        static_assert(Dispatch::unique(), "duplicate dtype dispatch key");
        return dtype_names({scalar_type_v<Ts>...});
    }
    static std::string pairs() {
        using Dispatch = DTypeListDispatch<PairEntry, TypeList<Ts...>>;
        static_assert(Dispatch::unique(), "duplicate dtype dispatch key");
        return dtype_pair_names(
            {ScalarTypePair{scalar_type_v<typename Ts::A>, scalar_type_v<typename Ts::B>}...});
    }
};

} // namespace detail

template <typename List> constexpr bool dtype_supported(at::ScalarType dtype) {
    return detail::DTypeListDispatch<detail::ScalarEntry, List>::supports(
        static_cast<uint16_t>(dtype));
}
template <typename List> constexpr bool dtype_pair_supported(at::ScalarType a, at::ScalarType b) {
    return detail::DTypeListDispatch<detail::PairEntry, List>::supports(scalar_pair_key(a, b));
}
template <typename List> std::string dtype_names() {
    return detail::DTypeListNames<List>::scalars();
}
/* Shared scalar-kernel diagnostic, evaluated only when requested. */
template <typename List>
std::string unsupported_dtype_message(const char* operation, at::ScalarType dtype) {
    return c10::str(operation, " has no kernel for ", c10::toString(dtype),
                    " (instantiated: ", dtype_names<List>(), ")");
}

template <typename List> std::string dtype_pair_names() {
    return detail::DTypeListNames<List>::pairs();
}

/* Error formatters run only on a miss; callback exceptions propagate unchanged. */
template <typename List, typename F, typename Error>
decltype(auto) visit_dtype(at::ScalarType dtype, F&& fn, Error&& error) {
    return detail::DTypeListDispatch<detail::ScalarEntry, List>::visit(
        static_cast<uint16_t>(dtype), std::forward<F>(fn),
        [&]() { return std::forward<Error>(error)(dtype); });
}
template <typename List, typename F, typename Error>
decltype(auto) visit_dtype_pair(at::ScalarType a, at::ScalarType b, F&& fn, Error&& error) {
    return detail::DTypeListDispatch<detail::PairEntry, List>::visit(
        scalar_pair_key(a, b), std::forward<F>(fn),
        [&]() { return std::forward<Error>(error)(a, b); });
}

} // namespace astrai
