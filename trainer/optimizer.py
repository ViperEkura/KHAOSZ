#TODO 混合精度梯度累积方法
from torch.optim import Optimizer
from torch.optim.adamw import AdamW



class MixedPrecisionOptimizer(Optimizer):
    def __init__(self, 
        parameter, 
        lr=1e-3, 
        betas=(0.9, 0.99), 
        eps=1e-5, 
        weight_decay=0.01
    ):
        self.param_groups = [
            {
                'params': parameter,
                'lr': lr,
                'betas': betas,
                'eps': eps,
                'weight_decay': weight_decay
            }
        ]

    def step(self, closure=None):
        loss =  None
        return loss