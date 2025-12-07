import math
from abc import abstractmethod, ABC
from typing import Any, Dict, List
from torch.optim.lr_scheduler import LRScheduler
from khaosz.config.schedule_config import ScheduleConfig


class BaseScheduler(LRScheduler, ABC):
    """
    Base scheduler class for all other schedulers.
    """
    
    def __init__(self, optimizer, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
    
    @abstractmethod
    def get_lr(self) -> List[float]:
        raise NotImplementedError
    
    def state_dict(self) -> Dict[str, Any]:
        return super().state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        super().load_state_dict(state_dict)


class CosineScheduler(BaseScheduler):
    """
    Cosine decay scheduler with warmup, implemented as PyTorch LRScheduler.
    """
    
    def __init__(
        self, 
        optimizer, 
        warmup_steps: int, 
        lr_decay_steps: int, 
        min_rate: float = 0.05, 
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.min_rate = min_rate
        self.total_steps = warmup_steps + lr_decay_steps
        super().__init__(optimizer, last_epoch)
    
    
    def get_lr(self) -> List[float]:
        # warmup
        if self.last_epoch < self.warmup_steps:
            warmup_factor = max(self.min_rate, self.last_epoch / self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # cosine decay
        decay_progress = (self.last_epoch - self.warmup_steps) / self.lr_decay_steps
        decay_progress = min(decay_progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        decay_factor = max(self.min_rate, cosine_decay)
        return [base_lr * decay_factor for base_lr in self.base_lrs]
    
    def state_dict(self):
        state = super().state_dict()
        state.update({
            'warmup_steps': self.warmup_steps,
            'lr_decay_steps': self.lr_decay_steps,
            'min_rate': self.min_rate,
            'total_steps': self.total_steps,
        })
        return state
    
    def load_state_dict(self, state_dict):
        self.warmup_steps = state_dict.pop('warmup_steps')
        self.lr_decay_steps = state_dict.pop('lr_decay_steps')
        self.min_rate = state_dict.pop('min_rate')
        self.total_steps = state_dict.pop('total_steps')
        super().load_state_dict(state_dict)


class SGDRScheduler(BaseScheduler):
    """
    SGDR (Stochastic Gradient Descent with Warm Restarts) scheduler, 
    """
    
    def __init__(
        self, 
        optimizer, 
        warmup_steps: int, 
        cycle_length: int, 
        min_rate: float = 0.05, 
        t_mult: int = 2, 
        last_epoch: int = -1, 
    ):
        self.warmup_steps = warmup_steps
        self.cycle_length = cycle_length
        self.min_rate = min_rate
        self.t_mult = t_mult
        
        super().__init__(optimizer, last_epoch)

    
    def get_lr(self):
        # warmup
        if self.last_epoch < self.warmup_steps:
            warmup_factor = max(self.min_rate, self.last_epoch / self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # SGDR
        steps_since_warmup = self.last_epoch - self.warmup_steps
        
        # 1. Calculate current cycle and position within cycle
        current_cycle_length = self.cycle_length
        total_cycles_length = 0
        cycle_num = 0
        
        while total_cycles_length + current_cycle_length <= steps_since_warmup:
            total_cycles_length += current_cycle_length
            current_cycle_length *= self.t_mult
            cycle_num += 1
        
        steps_in_cycle = steps_since_warmup - total_cycles_length
        
        # 2. Cosine annealing within the current cycle
        cosine_factor = 0.5 * (1 + math.cos(math.pi * steps_in_cycle / current_cycle_length))
        learning_rate_factor = self.min_rate + (1 - self.min_rate) * cosine_factor
        
        return [base_lr * learning_rate_factor for base_lr in self.base_lrs]
    
    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        state = super().state_dict()
        state.update({
            'warmup_steps': self.warmup_steps,
            'cycle_length': self.cycle_length,
            'min_rate': self.min_rate,
            't_mult': self.t_mult
        })
        return state
    
    def load_state_dict(self, state_dict):
        """Loads the scheduler's state."""
        self.warmup_steps = state_dict.pop('warmup_steps')
        self.cycle_length = state_dict.pop('cycle_length')
        self.min_rate = state_dict.pop('min_rate')
        self.t_mult = state_dict.pop('t_mult')
        super().load_state_dict(state_dict)


    
class SchedulerFactory:
    """
    Factory class for creating learning rate schedulers.
    """

    @staticmethod
    def load(optimizer, scedule_config: ScheduleConfig) -> BaseScheduler:
        kwargs = scedule_config.get_kwargs()
        schedule_type = kwargs.pop("schedule_type")
        
        if schedule_type == "cosine":
            return CosineScheduler(optimizer, **kwargs)
        elif schedule_type == "sgdr":
            return SGDRScheduler(optimizer, **kwargs)
        else:
            raise ValueError(f"Unsupported schedule type: {schedule_type}")
        