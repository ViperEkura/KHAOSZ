import torch
from typing import Any, List
from torch import Tensor
from torch.optim import Optimizer, AdamW, SGD

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
        print(self.param_groups[-1]["params"])
        
    def _init_group(
        self, 
        group:List[dict, Any], 
        params:List[Tensor], 
        grads:List[Tensor]
    ):
        for p in group["params"]:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)
            else:
                params.append(None)
                grads.append(None)
        
    def step(self):
        loss = None
        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            
        return loss    