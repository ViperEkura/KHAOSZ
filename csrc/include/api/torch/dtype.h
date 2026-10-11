#pragma once
/* Torch dtype metadata; usable by ordinary C++ without CUDA headers. */

#include <c10/core/ScalarType.h>
#include <c10/util/StringUtil.h>
#include <initializer_list>
#include <string>

namespace astrai {

/*
 * Use Python's ``str(torch.dtype)`` spelling;
 * matching also accepts c10 names used by existing checkpoint readers.
 */
inline std::string torch_dtype_str(at::ScalarType t) {
    switch (t) {
    case at::kBFloat16:
        return "torch.bfloat16";
    case at::kHalf:
        return "torch.float16";
    case at::kFloat:
        return "torch.float32";
    case at::kDouble:
        return "torch.float64";
    case at::kChar:
        return "torch.int8";
    case at::kByte:
        return "torch.uint8";
    case at::kShort:
        return "torch.int16";
    case at::kInt:
        return "torch.int32";
    case at::kLong:
        return "torch.int64";
    case at::kBool:
        return "torch.bool";
    case at::kFloat8_e4m3fn:
        return "torch.float8_e4m3fn";
    case at::kFloat8_e5m2:
        return "torch.float8_e5m2";
    default:
        return c10::toString(t);
    }
}

inline bool dtype_str_matches(const std::string& s, at::ScalarType t) {
    return s == torch_dtype_str(t) || s == std::string(c10::toString(t));
}

constexpr bool is_fp8(at::ScalarType dtype) {
    return dtype == at::kFloat8_e4m3fn || dtype == at::kFloat8_e5m2;
}

inline bool dtype_supported(at::ScalarType dtype, std::initializer_list<at::ScalarType> types) {
    for (const auto type : types)
        if (dtype == type)
            return true;
    return false;
}

inline std::string dtype_names(std::initializer_list<at::ScalarType> types) {
    std::string names;
    for (const auto type : types)
        names += c10::str(names.empty() ? "" : ", ", c10::toString(type));
    return names;
}

struct ScalarTypePair {
    at::ScalarType a, b;
};

inline std::string dtype_pair_names(std::initializer_list<ScalarTypePair> pairs) {
    std::string names;
    for (const auto pair : pairs)
        names += c10::str(names.empty() ? "" : ", ", c10::toString(pair.a), " x ",
                          c10::toString(pair.b));
    return names;
}

} // namespace astrai
