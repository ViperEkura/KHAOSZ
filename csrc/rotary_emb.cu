#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include <utils/launch.cuh>

namespace {

struct Geometry {
    int tokens;
    int heads;
    int dim;
};

Geometry check_inputs(const torch::Tensor& x, const torch::Tensor& freqs_cis) {
    TORCH_CHECK(x.is_cuda() && freqs_cis.is_cuda(), "x and freqs_cis must be on CUDA");
    TORCH_CHECK(x.device() == freqs_cis.device(),
                "x and freqs_cis must be on the same CUDA device");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(x.dim() == 3 || x.dim() == 4, "x must be [tokens, n_heads, head_dim] or "
                                              "[batch, seq_len, n_heads, head_dim]");
    TORCH_CHECK(x.stride(-1) == 1, "x must have a contiguous head dimension");
    TORCH_CHECK(freqs_cis.dim() == x.dim(), "freqs_cis rank must match x rank");
    TORCH_CHECK(freqs_cis.is_contiguous(), "freqs_cis must be contiguous");
    TORCH_CHECK(freqs_cis.scalar_type() == torch::kFloat32, "freqs_cis must be f32");

    TORCH_CHECK(x.size(0) <= std::numeric_limits<int>::max() &&
                    (x.dim() == 3 || x.size(1) <= std::numeric_limits<int>::max()),
                "rotary token dimensions exceed int32 kernel indexing");
    const int64_t tokens = x.dim() == 3 ? x.size(0) : x.size(0) * x.size(1);
    const int64_t heads = x.size(x.dim() - 2);
    const int64_t dim = x.size(x.dim() - 1);
    TORCH_CHECK(dim > 0 && dim % 4 == 0, "head_dim must be a positive multiple of 4");
    TORCH_CHECK(tokens <= std::numeric_limits<int>::max() &&
                    heads <= std::numeric_limits<int>::max() &&
                    dim <= std::numeric_limits<int>::max() &&
                    x.numel() <= std::numeric_limits<int>::max(),
                "rotary input exceeds int32 kernel indexing");
    TORCH_CHECK(freqs_cis.size(-2) == dim / 2, "freqs_cis dim/2 mismatch");
    TORCH_CHECK(freqs_cis.size(-1) == 2, "freqs_cis last dim must be 2 [cos, sin]");
    TORCH_CHECK(freqs_cis.size(0) == x.size(0) &&
                    (x.dim() == 3 || freqs_cis.size(1) == x.size(1)),
                "freqs_cis token dimensions must match x");
    TORCH_CHECK(freqs_cis.numel() == tokens * dim,
                "freqs_cis token or rotary dimension mismatch");
    return {static_cast<int>(tokens), static_cast<int>(heads), static_cast<int>(dim)};
}

template <bool inverse>
__global__ void rotary_emb_kernel(const __nv_bfloat16* __restrict__ x,
                                  const float* __restrict__ freqs_cis,
                                  __nv_bfloat16* __restrict__ out,
                                  int tokens, int heads, int dim,
                                  int seq_len, int64_t stride0, int64_t stride1,
                                  int64_t head_stride) {
    const int chunks = dim >> 2;
    const int total = tokens * heads * chunks;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < total;
         c += gridDim.x * blockDim.x) {
        const int chunk = c % chunks;
        const int tmp = c / chunks;
        const int token = tmp / heads;
        const int64_t x_off =
            (seq_len ? (token / seq_len) * stride0 + (token % seq_len) * stride1
                     : token * stride0) +
            (tmp % heads) * head_stride + (chunk << 2);
        const int out_off = tmp * dim + (chunk << 2);
        const int f_off = ((token * chunks) + chunk) << 2;
        const float4 f = *reinterpret_cast<const float4*>(freqs_cis + f_off);
        const float e0 = __bfloat162float(x[x_off]);
        const float o0 = __bfloat162float(x[x_off + 1]);
        const float e1 = __bfloat162float(x[x_off + 2]);
        const float o1 = __bfloat162float(x[x_off + 3]);
        if constexpr (inverse) {
            out[out_off] = __float2bfloat16_rn(e0 * f.x + o0 * f.y);
            out[out_off + 1] = __float2bfloat16_rn(o0 * f.x - e0 * f.y);
            out[out_off + 2] = __float2bfloat16_rn(e1 * f.z + o1 * f.w);
            out[out_off + 3] = __float2bfloat16_rn(o1 * f.z - e1 * f.w);
        } else {
            out[out_off] = __float2bfloat16_rn(e0 * f.x - o0 * f.y);
            out[out_off + 1] = __float2bfloat16_rn(e0 * f.y + o0 * f.x);
            out[out_off + 2] = __float2bfloat16_rn(e1 * f.z - o1 * f.w);
            out[out_off + 3] = __float2bfloat16_rn(e1 * f.w + o1 * f.z);
        }
    }
}

}  // namespace

torch::Tensor rotary_emb(torch::Tensor x, torch::Tensor freqs_cis) {
    const auto geometry = check_inputs(x, freqs_cis);
    const at::cuda::OptionalCUDAGuard guard(device_of(x));
    auto out = torch::empty(x.sizes(), x.options());
    if (x.numel() == 0)
        return out;

    const int work = geometry.tokens * geometry.heads * (geometry.dim / 4);
    const int grid = std::min((work + 255) / 256, 2048);
    const int seq_len = x.dim() == 4 ? static_cast<int>(x.size(1)) : 0;
    const auto stream = at::cuda::getCurrentCUDAStream();
    rotary_emb_kernel<false><<<grid, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        freqs_cis.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        geometry.tokens, geometry.heads, geometry.dim, seq_len,
        x.stride(0), x.stride(1), x.stride(-2));
    ASTRAI_LAUNCH_CHECK();
    return out;
}

torch::Tensor rotary_emb_backward(torch::Tensor grad, torch::Tensor freqs_cis) {
    const auto geometry = check_inputs(grad, freqs_cis);
    const at::cuda::OptionalCUDAGuard guard(device_of(grad));
    auto grad_x = torch::empty(grad.sizes(), grad.options());
    if (grad.numel() == 0)
        return grad_x;

    const int work = geometry.tokens * geometry.heads * (geometry.dim / 4);
    const int grid = std::min((work + 255) / 256, 2048);
    const int seq_len = grad.dim() == 4 ? static_cast<int>(grad.size(1)) : 0;
    const auto stream = at::cuda::getCurrentCUDAStream();
    rotary_emb_kernel<true><<<grid, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr()),
        freqs_cis.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(grad_x.data_ptr()),
        geometry.tokens, geometry.heads, geometry.dim, seq_len,
        grad.stride(0), grad.stride(1), grad.stride(-2));
    ASTRAI_LAUNCH_CHECK();
    return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotary_emb", &rotary_emb, py::arg("x"), py::arg("freqs_cis"),
          "Fused rotary embedding for packed 3D or dense 4D tensors");
    m.def("rotary_emb_backward", &rotary_emb_backward, py::arg("grad"), py::arg("freqs_cis"),
          "Inverse rotary embedding gradient with respect to x");
}
