import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch import Tensor
from typing import Dict


class ParallelModel(nn.Module):
    def __init__(self, process_group: dist.ProcessGroup):
        super().__init__()
        self.process_group = process_group
        self.rank = dist.get_rank(self.process_group)
        self.world_size = dist.get_world_size(self.process_group)


class RowParallelLinear(ParallelModel):
    def __init__(
        self, 
        process_group: dist.ProcessGroup,
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        reduce_results: bool = True
    ):
        super().__init__(process_group)
        
        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_rank = in_features // self.world_size
        self.reduce_results = reduce_results
        
        if in_features % self.world_size != 0:
            raise ValueError(f"in_features must be divisible by world_size. Got {in_features} and {self.world_size}")
        
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_rank))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
    def forward(self, input: Tensor) -> Tensor:
        output = F.linear(input, self.weight)
        
        if self.reduce_results:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.process_group)
        
        if self.bias is not None:
            output += self.bias
        
        return output
    
    def load_state_dict(self, state_dict: Dict[str, Tensor]):        
        full_weight = state_dict.get('weight')
        full_bias = state_dict.get('bias')
        
        start_idx = self.rank * self.in_features_per_rank
        end_idx = start_idx + self.in_features_per_rank
        weight_slice = full_weight[:, start_idx:end_idx]
        self.weight.data.copy_(weight_slice)
        
        if self.bias is not None:
            self.bias.data.copy_(full_bias)


class ColumnParallelLinear(ParallelModel):
    def __init__(
        self, 
        process_group: dist.ProcessGroup,
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        gather_results: bool = True
    ):
        super().__init__(process_group)
        
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_rank = out_features // self.world_size
        self.gather_results = gather_results

        if out_features % self.world_size != 0:
            raise ValueError(f"out_features must be divisible by world_size. Got {out_features} and {self.world_size}")

        self.weight = nn.Parameter(torch.empty(self.out_features_per_rank, self.in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features_per_rank)) if bias else None
        
    def forward(self, input: Tensor) -> Tensor:
        output = F.linear(input, self.weight, self.bias)
        
        if self.gather_results:
            output_list = [torch.empty_like(output) for _ in range(self.world_size)]
            dist.all_gather(output_list, output, group=self.process_group)
            output = torch.cat(output_list, dim=-1)
        
        return output
    
    def load_state_dict(self, state_dict: Dict[str, Tensor]):        
        full_weight = state_dict.get('weight')
        full_bias = state_dict.get('bias')
        
        start_idx = self.rank * self.out_features_per_rank
        end_idx = start_idx + self.out_features_per_rank
        weight_slice = full_weight[start_idx:end_idx, :]
        self.weight.data.copy_(weight_slice)
        
        if self.bias is not None:
            bias_slice = full_bias[start_idx:end_idx]
            self.bias.data.copy_(bias_slice)
