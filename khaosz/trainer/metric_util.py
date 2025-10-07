import torch.nn as nn
from typing import Dict

def grad_norm(model: nn.Module, norm_type: int = 2) -> Dict[str, float]:
    """ Compute gradient norm for each parameter in the model. """
    norms = {}
    for name, param in model.named_parameters():
        norms[name] = 0.0
        if param.grad:
            norm = param.grad.data.norm(norm_type).item()
            norms[name] = norm
    return norms

def grad_std(model: nn.Module) -> Dict[str, float]:
    """ Compute standard deviation of gradients for each parameter. """
    stds = {}
    for name, param in model.named_parameters():
        stds[name] = 0.0
        if param.grad:
            std = param.grad.data.std().item()
            stds[name] = std
    return stds

def grad_max(model: nn.Module) -> Dict[str, float]:
    """ Find the maximum absolute gradient value for each parameter. """
    max_vals = {}
    for name, param in model.named_parameters():
        max_vals[name] = -float('inf')
        if param.grad:
            max_val = param.grad.data.max().item()
            max_vals[name] = max_val
            
    return max_vals

def grad_min(model: nn.Module) -> Dict[str, float]:
    """ Find the minimum absolute gradient value for each parameter. """
    min_vals = {}
    for name, param in model.named_parameters():
        min_vals[name] = float('inf')
        if param.grad:
            min_val = param.grad.data.min().item()
            min_vals[name] = min_val
            
    return min_vals

def grad_mean(model: nn.Module) -> Dict[str, float]:
    """ Compute mean of gradients for each parameter. """
    means = {}
    for name, param in model.named_parameters():
        means[name] = 0.0
        if param.grad:
            mean = param.grad.data.mean().item()
            means[name] = mean
            
    return means

def grad_nan_num(model: nn.Module) -> Dict[str, int]:
    """ Count the number of NaNs in gradients for each parameter. """
    nan_nums = {}
    for name, param in model.named_parameters():
        nan_nums[name] = 0
        if param.grad:
            nan_num = param.grad.isnan().sum().item()
            nan_nums[name] = nan_num
    return nan_nums