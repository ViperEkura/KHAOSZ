import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, embedding_dim)))

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return F.embedding(x, self.weight)
