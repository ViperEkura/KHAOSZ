#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include <utils/launch.cuh>

namespace {

void check_inputs(const torch::Tensor& up, const torch::Tensor& gate) {
    TORCH_CHECK(up.is_cuda() && gate.is_cuda(), "SwiGLU inputs must be CUDA tensors");
    TORCH_CHECK(up.device() == gate.device(), "SwiGLU inputs must share a CUDA device");
    TORCH_CHECK(up.sizes() == gate.sizes(), "SwiGLU inputs must have the same shape");
    TORCH_CHECK(up.scalar_type() == gate.scalar_type(), "SwiGLU inputs must have the same dtype");
    TORCH_CHECK(up.is_contiguous() && gate.is_contiguous(), "SwiGLU inputs must be contiguous");
    TORCH_CHECK(up.scalar_type() == at::kHalf || up.scalar_type() == at::kBFloat16 ||
                    up.scalar_type() == at::kFloat,
                "SwiGLU supports FP16, BF16, and FP32");
}

template <typename scalar_t, int width>
struct alignas(sizeof(scalar_t) * width) Pack {
    scalar_t values[width];
};

template <typename scalar_t, int width>
__global__ void swiglu_forward_kernel(const scalar_t* __restrict__ up,
                                      const scalar_t* __restrict__ gate,
                                      scalar_t* __restrict__ out,
                                      int64_t n) {
    using Packed = Pack<scalar_t, width>;
    for (int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * width;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x * width) {
        const Packed u = *reinterpret_cast<const Packed*>(up + i);
        const Packed g = *reinterpret_cast<const Packed*>(gate + i);
        Packed result;
#pragma unroll
        for (int j = 0; j < width; ++j) {
            const float gate_value = static_cast<float>(g.values[j]);
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            result.values[j] =
                static_cast<scalar_t>(static_cast<float>(u.values[j]) * gate_value * sigmoid);
        }
        *reinterpret_cast<Packed*>(out + i) = result;
    }
}

template <typename scalar_t, int width>
__global__ void swiglu_backward_kernel(const scalar_t* __restrict__ grad,
                                       const scalar_t* __restrict__ up,
                                       const scalar_t* __restrict__ gate,
                                       scalar_t* __restrict__ grad_up,
                                       scalar_t* __restrict__ grad_gate,
                                       int64_t n) {
    using Packed = Pack<scalar_t, width>;
    for (int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * width;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x * width) {
        const Packed dy = *reinterpret_cast<const Packed*>(grad + i);
        const Packed u = *reinterpret_cast<const Packed*>(up + i);
        const Packed g = *reinterpret_cast<const Packed*>(gate + i);
        Packed du, dg;
#pragma unroll
        for (int j = 0; j < width; ++j) {
            const float gate_value = static_cast<float>(g.values[j]);
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            const float upstream = static_cast<float>(dy.values[j]);
            du.values[j] = static_cast<scalar_t>(upstream * gate_value * sigmoid);
            dg.values[j] = static_cast<scalar_t>(
                upstream * static_cast<float>(u.values[j]) * sigmoid *
                (1.0f + gate_value * (1.0f - sigmoid)));
        }
        *reinterpret_cast<Packed*>(grad_up + i) = du;
        *reinterpret_cast<Packed*>(grad_gate + i) = dg;
    }
}

int grid_size(int64_t n, int width) {
    return static_cast<int>(
        std::min<int64_t>((n + 256 * width - 1) / (256 * width), 4096));
}

template <typename scalar_t>
bool vector_aligned(const torch::Tensor& a, const torch::Tensor& b,
                    const torch::Tensor& c) {
    constexpr uintptr_t mask = alignof(Pack<scalar_t, 4>) - 1;
    return ((reinterpret_cast<uintptr_t>(a.data_ptr()) |
             reinterpret_cast<uintptr_t>(b.data_ptr()) |
             reinterpret_cast<uintptr_t>(c.data_ptr())) & mask) == 0;
}

}  // namespace

torch::Tensor swiglu(torch::Tensor up, torch::Tensor gate) {
    check_inputs(up, gate);
    const at::cuda::OptionalCUDAGuard guard(device_of(up));
    auto out = torch::empty_like(up);
    const int64_t n = up.numel();
    if (n == 0)
        return out;

    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, up.scalar_type(),
                                    "swiglu", [&] {
        if (n >= 1024 && n % 4 == 0 && vector_aligned<scalar_t>(up, gate, out)) {
            swiglu_forward_kernel<scalar_t, 4><<<grid_size(n, 4), 256, 0, stream>>>(
                up.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), n);
        } else {
            swiglu_forward_kernel<scalar_t, 1><<<grid_size(n, 1), 256, 0, stream>>>(
                up.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), n);
        }
    });
    ASTRAI_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> swiglu_backward(torch::Tensor grad, torch::Tensor up,
                                           torch::Tensor gate) {
    check_inputs(up, gate);
    TORCH_CHECK(grad.is_cuda() && grad.device() == up.device() &&
                    grad.scalar_type() == up.scalar_type() &&
                    grad.sizes() == up.sizes() && grad.is_contiguous(),
                "SwiGLU grad must match the contiguous CUDA inputs");
    const at::cuda::OptionalCUDAGuard guard(device_of(up));
    auto grad_up = torch::empty_like(up);
    auto grad_gate = torch::empty_like(gate);
    const int64_t n = up.numel();
    if (n == 0)
        return {grad_up, grad_gate};

    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, up.scalar_type(),
                                    "swiglu_backward", [&] {
        if (n >= 1024 && n % 4 == 0 && vector_aligned<scalar_t>(grad, up, gate) &&
            vector_aligned<scalar_t>(grad_up, grad_gate, up)) {
            swiglu_backward_kernel<scalar_t, 4><<<grid_size(n, 4), 256, 0, stream>>>(
                grad.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(),
                gate.data_ptr<scalar_t>(), grad_up.data_ptr<scalar_t>(),
                grad_gate.data_ptr<scalar_t>(), n);
        } else {
            swiglu_backward_kernel<scalar_t, 1><<<grid_size(n, 1), 256, 0, stream>>>(
                grad.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(),
                gate.data_ptr<scalar_t>(), grad_up.data_ptr<scalar_t>(),
                grad_gate.data_ptr<scalar_t>(), n);
        }
    });
    ASTRAI_LAUNCH_CHECK();
    return {grad_up, grad_gate};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu", &swiglu, "Fused SwiGLU forward");
    m.def("swiglu_backward", &swiglu_backward, "Fused SwiGLU backward");
}
