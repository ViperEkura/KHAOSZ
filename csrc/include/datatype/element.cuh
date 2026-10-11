#pragma once

#include <cmath>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <utils/define.cuh>

namespace astrai {

using bf16 = __nv_bfloat16;
using fp16 = __half;
using fp8_e4m3 = __nv_fp8_e4m3;
using fp8_e5m2 = __nv_fp8_e5m2;

template <typename T> struct ElemTrait;

template <> struct ElemTrait<bf16> {
    using Pair = __nv_bfloat162;
    static constexpr int kBytes = 2, kPerCell = 2;

    static DEVICE_FORCEINLINE float to_float(bf16 x) { return __bfloat162float(x); }
    static DEVICE_FORCEINLINE bf16 from_float(float x) { return __float2bfloat16(x); }
    static DEVICE_FORCEINLINE Pair pack_pair(float a, float b) {
        return __floats2bfloat162_rn(a, b);
    }
    static DEVICE_FORCEINLINE float2 unpack_pair(Pair x) { return __bfloat1622float2(x); }
    static DEVICE_FORCEINLINE unsigned pack2(float a, float b) {
        Pair x = pack_pair(a, b);
        return *reinterpret_cast<unsigned*>(&x);
    }
    static DEVICE_FORCEINLINE float2 unpack2(unsigned bits) {
        Pair x = *reinterpret_cast<Pair*>(&bits);
        return unpack_pair(x);
    }
};

template <> struct ElemTrait<fp16> {
    using Pair = __half2;
    static constexpr int kBytes = 2, kPerCell = 2;

    static DEVICE_FORCEINLINE float to_float(fp16 x) { return __half2float(x); }
    static DEVICE_FORCEINLINE fp16 from_float(float x) { return __float2half(x); }
    static DEVICE_FORCEINLINE Pair pack_pair(float a, float b) { return __floats2half2_rn(a, b); }
    static DEVICE_FORCEINLINE float2 unpack_pair(Pair x) { return __half22float2(x); }
    static DEVICE_FORCEINLINE unsigned pack2(float a, float b) {
        Pair x = pack_pair(a, b);
        return *reinterpret_cast<unsigned*>(&x);
    }
    static DEVICE_FORCEINLINE float2 unpack2(unsigned bits) {
        Pair x = *reinterpret_cast<Pair*>(&bits);
        return unpack_pair(x);
    }
};

template <> struct ElemTrait<float> {
    using Pair = float2;
    static constexpr int kBytes = 4;

    static DEVICE_FORCEINLINE float to_float(float x) { return x; }
    static DEVICE_FORCEINLINE float from_float(float x) { return x; }
    static DEVICE_FORCEINLINE Pair pack_pair(float a, float b) { return make_float2(a, b); }
    static DEVICE_FORCEINLINE float2 unpack_pair(Pair x) { return x; }
};

template <> struct ElemTrait<fp8_e4m3> {
    static constexpr int kBytes = 1;
    static DEVICE_FORCEINLINE uint8_t cvt_byte(float x) { return fp8_e4m3(x).__x; }
};

template <> struct ElemTrait<fp8_e5m2> {
    static constexpr int kBytes = 1;
    static DEVICE_FORCEINLINE uint8_t cvt_byte(float x) { return fp8_e5m2(x).__x; }
};

template <> struct ElemTrait<int8_t> {
    static constexpr int kBytes = 1;

    /* Symmetric saturation, round-to-nearest-even, with NaN mapped to zero. */
    static DEVICE_FORCEINLINE uint8_t cvt_byte(float x) {
        if (x != x)
            return 0;
        const float bounded = x < -127.0f ? -127.0f : x > 127.0f ? 127.0f : x;
        return static_cast<uint8_t>(static_cast<int>(nearbyintf(bounded)));
    }
};

template <typename T> DEVICE_FORCEINLINE void load8(const T* src, float* out) {
    static_assert(ElemTrait<T>::kBytes == 2, "load8 requires 16-bit elements");
    const uint4 raw = *reinterpret_cast<const uint4*>(src);
    const float2 a = ElemTrait<T>::unpack2(raw.x), b = ElemTrait<T>::unpack2(raw.y);
    const float2 c = ElemTrait<T>::unpack2(raw.z), d = ElemTrait<T>::unpack2(raw.w);
    out[0] = a.x;
    out[1] = a.y;
    out[2] = b.x;
    out[3] = b.y;
    out[4] = c.x;
    out[5] = c.y;
    out[6] = d.x;
    out[7] = d.y;
}

template <typename T> DEVICE_FORCEINLINE void store2(T* dst, float a, float b) {
    static_assert(ElemTrait<T>::kBytes == 2, "store2 requires 16-bit elements");
    *reinterpret_cast<unsigned*>(dst) = ElemTrait<T>::pack2(a, b);
}

template <typename T> DEVICE_FORCEINLINE void load_pair(const T* src, float* out) {
    using Pair = typename ElemTrait<T>::Pair;
    const float2 x = ElemTrait<T>::unpack_pair(*reinterpret_cast<const Pair*>(src));
    out[0] = x.x;
    out[1] = x.y;
}

} // namespace astrai
