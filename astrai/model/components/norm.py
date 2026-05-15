import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim, norm_eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.normalized_shape = (dim,)
        self.norm_eps = norm_eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight, self.norm_eps)
