#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <algorithm>
#include <limits>

#include <utils/launch.cuh>

__global__ void rotary_emb_kernel(const __nv_bfloat16* __restrict__ x,
                                  const float* __restrict__ freqs_cis,
                                  __nv_bfloat16* __restrict__ out,
                                  int n_tokens,
                                  int n_heads,
                                  int head_dim) {
    /*
     * Each head tiles into exact 2-pair chunks: one 8B x access and one 16B
     * cos/sin access per chunk (head_dim % 4 == 0 is enforced on the host).
     */
    const int chunks = head_dim >> 2;
    const int total = n_tokens * n_heads * chunks;

    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < total; c += gridDim.x * blockDim.x) {
        const int chunk = c % chunks;
        const int tmp = c / chunks;
        const int head = tmp % n_heads;
        const int token = tmp / n_heads;

        const int x_off = (tmp * head_dim) + (chunk << 2);
        const int f_off = ((token * chunks) + chunk) << 2;

        const float4 f = *reinterpret_cast<const float4*>(freqs_cis + f_off);
        const uint2 xr = *reinterpret_cast<const uint2*>(x + x_off);
        __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162*>(&xr.x);
        __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162*>(&xr.y);

        const float e0 = __bfloat162float(__low2bfloat16(p0));
        const float o0 = __bfloat162float(__high2bfloat16(p0));
        const float e1 = __bfloat162float(__low2bfloat16(p1));
        const float o1 = __bfloat162float(__high2bfloat16(p1));

        __nv_bfloat162 r0 = __floats2bfloat162_rn(e0 * f.x - o0 * f.y, e0 * f.y + o0 * f.x);
        __nv_bfloat162 r1 = __floats2bfloat162_rn(e1 * f.z - o1 * f.w, e1 * f.w + o1 * f.z);

        uint2 oraw;
        *reinterpret_cast<__nv_bfloat162*>(&oraw.x) = r0;
        *reinterpret_cast<__nv_bfloat162*>(&oraw.y) = r1;
        *reinterpret_cast<uint2*>(out + x_off) = oraw;
    }
}

torch::Tensor rotary_emb(torch::Tensor x, torch::Tensor freqs_cis) {
    TORCH_CHECK(x.is_cuda() && freqs_cis.is_cuda(), "x and freqs_cis must be on CUDA");
    TORCH_CHECK(x.device() == freqs_cis.device(),
                "x and freqs_cis must be on the same CUDA device");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(x.dim() == 3 || x.dim() == 4, "x must be [tokens, n_heads, head_dim] or "
                                              "[batch, seq_len, n_heads, head_dim]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
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

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto out = torch::empty_like(x);
    if (x.numel() == 0)
        return out;

    const int n_tokens = static_cast<int>(tokens);
    const int n_heads = static_cast<int>(heads);
    const int head_dim = static_cast<int>(dim);
    const int work = static_cast<int>(tokens * heads * (dim / 4));
    const int block = 256;
    const int grid = std::min((work - 1) / block + 1, 2048);
    auto stream = at::cuda::getCurrentCUDAStream();
    rotary_emb_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), freqs_cis.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n_tokens, n_heads, head_dim);
    ASTRAI_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotary_emb", &rotary_emb, py::arg("x"), py::arg("freqs_cis"),
          "Fused rotary embedding for packed 3D or dense 4D tensors");
}
