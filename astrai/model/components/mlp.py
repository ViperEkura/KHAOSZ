import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from astrai.model.components.linear import Linear


class MLP(nn.Module):
    def __init__(self, dim: int, dim_feed_forward: int):
        super().__init__()
        self.up = Linear(dim, dim_feed_forward)
        self.gate = Linear(dim, dim_feed_forward)
        self.down = Linear(dim_feed_forward, dim)

    def forward(self, x: Tensor) -> Tensor:
        gated = self.up(x) * F.silu(self.gate(x))
        out = self.down(gated)
        return out
