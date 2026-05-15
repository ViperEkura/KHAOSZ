import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)
