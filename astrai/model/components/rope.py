from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor


def get_rotary_emb(
    dim: int,
    max_len: int,
    base: float = 10000,
    device: Optional[torch.device] = None,
) -> Tensor:
    theta = base ** (-torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim)
    t = torch.arange(0, max_len, dtype=torch.float64, device=device)
    freqs = torch.outer(t, theta).float()
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return torch.complex(cos, sin)


def ntk_base(base: float, dim: int, factor: float) -> float:
    return base * (factor ** (dim / (dim - 2)))


def apply_rotary_emb(x: torch.Tensor, freqs_cis: Tensor) -> Tensor:
    dtype = x.dtype
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_)
    freqs_cis = freqs_cis.unsqueeze(2)
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_len: int,
        base: float = 10000,
        rope_scaling: Optional[Dict] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        self.rope_scaling = rope_scaling

        if rope_scaling is not None:
            scaling_type = rope_scaling.get("type", "ntk")
            factor = rope_scaling.get("factor", 1.0)
            if scaling_type == "ntk":
                self.base = ntk_base(base, dim, factor)

        self._set_rotary_buffer(self.max_len)

    def _set_rotary_buffer(self, max_len: int):
        rotary_emb = get_rotary_emb(self.dim, max_len, self.base)
        freqs_cis = torch.view_as_real(rotary_emb)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: Tensor, position_ids: Optional[Tensor] = None) -> Tensor:
        if position_ids is None:
            position_ids = (
                torch.arange(x.size(1), device=x.device)
                .unsqueeze(0)
                .expand(x.size(0), -1)
            )
        position_freq_cis = self.freqs_cis[position_ids].float()
        return torch.view_as_complex(position_freq_cis)
