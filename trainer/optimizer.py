# TODO 完成混合精度的训练方法

import torch
from torch import Tensor
from torch.optim import Optimizer, AdamW, SGD

from typing import (
    Any, 
    List, 
    Tuple, 
    cast
)

def adamw(
    params: List[Tensor], 
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    lr: float=1e-3, 
    betas: Tuple[float, float]=(0.9, 0.999), 
    eps: float =1e-5, 
    weight_decay: float=0.01
):
    beta1, beta2 = betas
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs]
    )
    for params_, grads_, exp_avgs_, exp_avg_sqs_ in grouped_tensors.values():
        device_params = cast(List[Tensor], params_), 
        device_grads = cast(List[Tensor], grads_), 
        device_exp_avgs = cast(List[Tensor], exp_avgs_), 
        device_exp_avg_sqs = cast(List[Tensor], exp_avg_sqs_),
        
        
        if weight_decay != 0:
            torch._foreach_mul_(device_params, 1 - lr * weight_decay)
        
        # m = beta1 * m + (1 - beta1) * grad
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)
        
        # v = beta2 * v + (1 - beta2) * grad^2
        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1 - beta2)
        
        del device_grads
        
        # p = p - lr * m / (sqrt(v) + eps)
        
    pass


class MixedPrecisionOptimizer(Optimizer):
    def __init__(self, 
        params, 
        lr=1e-3, 
        betas=(0.9, 0.999), 
        eps=1e-5, 
        weight_decay=0.01
    ):        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def _init_group(
        self, 
        group:List[dict, Any], 
        grads:List[Tensor]
    ):
        pass
    def step(self):
        pass
    