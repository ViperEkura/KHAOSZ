import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)
