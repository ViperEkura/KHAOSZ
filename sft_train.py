import torch
from torch import Tensor

def get_rotary_emb(dim, max_len, base=10000, device='cuda', dtype=torch.bfloat16) -> tuple[Tensor, Tensor]:
    freqs = torch.pow(torch.tensor(base), -(torch.arange(0, dim, 2) / dim))
    freqs = freqs.repeat_interleave(2)

    t = torch.arange(0, max_len).float()
    rads = torch.outer(t, freqs)
    cos_emb = torch.cos(rads).to(device=device, dtype=dtype)
    sin_emb = torch.sin(rads).to(device=device, dtype=dtype)

    return cos_emb, sin_emb

def rotary_emb(x: Tensor, freqs_cis: tuple[Tensor, Tensor]) -> Tensor:
    _, L, _ = x.size()
    cos, sin = freqs_cis
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_half_rotated = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    rot = x * cos[:L] + x_half_rotated * sin[:L]
    return rot


xin = torch.randn(1, 10, 128).to('cuda')
freqs_cis = get_rotary_emb(128, 10)
xout = rotary_emb(xin, freqs_cis)
print(xin)