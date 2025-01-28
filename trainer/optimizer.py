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
    
    
class CustomAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(CustomAdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.bfloat16)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                denom = (exp_avg_sq.to(torch.bfloat16).sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(exp_avg.to(torch.bfloat16), denom, value=-step_size)

                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

        return loss