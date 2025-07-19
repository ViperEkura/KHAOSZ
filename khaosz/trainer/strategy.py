import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import Dataset
from typing import Any, Literal, Tuple, Callable, Dict
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

def get_logprobs(model:nn.Module, input_ids: Tensor, mask: Tensor, pad_token_id):
    input_mask =  input_ids.ne(pad_token_id)
    logits = model(input_ids, input_mask)
    log_probs = torch.log_softmax(logits, dim=-1)
    
    shifted_log_probs = log_probs[:, :-1, :] 
    shifted_input_ids = input_ids[:, 1:]
    shifted_response_mask = mask[:, 1:]
    
    token_logprobs = torch.gather(
        shifted_log_probs, 
        dim=-1, 
        index=shifted_input_ids.unsqueeze(-1)
    ).squeeze(-1)
    
    prompt_mask = input_mask[:, 1:]
    valid_mask = (prompt_mask & shifted_response_mask).float()
    
    return (token_logprobs * valid_mask).sum(dim=-1)


class BaseStrategy(ABC):
    def __init__(self, model: nn.Module):
        self.model = model
    
    @abstractmethod
    def compute_loss(self, batch: Tuple[Tensor, ...]) -> Tensor:
        raise NotImplementedError
    
    def __call__(self, batch: Tuple[Tensor, ...]) -> Tensor:
        return self.compute_loss(batch)


class SeqStrategy(BaseStrategy):
    def __init__(self, model):
        super().__init__(model)
    
    def compute_loss(self, batch: Tuple[Tensor, ...]) -> Tensor:
        x, y = batch
        B, L = x.size()
        logits: Tensor = self.model(x)
        
        loss = F.cross_entropy(
            logits.view(B * L, -1), y.flatten()
        )
        return loss
    

class SftStrategy(BaseStrategy):
    def __init__(self, model):
        super().__init__(model)
    
    def compute_loss(self, batch: Tuple[Tensor, ...]) -> Tensor:
        x, y, loss_mask = batch
        B, L = x.size()
        ignore_idx = -1
        
        logits: Tensor = self.model(x)
        masked_y = y.masked_fill(loss_mask == 0, ignore_idx)
        
        loss = F.cross_entropy(
            logits.view(B * L, -1),
            masked_y.flatten(), 
            ignore_index=ignore_idx
        )

        return loss

class DpoStrategy(BaseStrategy):
    def __init__(self, model, ref_model, pad_token_id, beta):
        super().__init__(model)
        self.ref_model = ref_model
        self.pad_token_id = pad_token_id
        self.beta = beta
        
    def compute_loss(self, batch: Tuple[Tensor, ...]) -> Tensor:
        good_ids, bad_ids, good_mask, bad_mask = batch
        
        log_pi_good = get_logprobs(self.model, good_ids, good_mask, self.pad_token_id)
        log_pi_bad = get_logprobs(self.model, bad_ids, bad_mask, self.pad_token_id)
        
        with torch.no_grad():
            log_ref_good = get_logprobs(self.ref_model, good_ids, good_mask, self.pad_token_id)
            log_ref_bad = get_logprobs(self.ref_model, bad_ids, bad_mask, self.pad_token_id)
        
        pi_log_ratio = log_pi_good - log_pi_bad
        ref_log_ratio = log_ref_good - log_ref_bad

        ratio_diff = pi_log_ratio - ref_log_ratio
        
        dpo_loss = -F.logsigmoid(self.beta * ratio_diff).mean()
        return dpo_loss


class PpoStrategy(BaseStrategy):
    def __init__(self, model, ref_model, pad_token_id, epsilon):
        super().__init__(model)
        self.ref_model = ref_model
        self.pad_token_id = pad_token_id
        self.epsilon = epsilon
        
    def ppo_clip_loss_masked(
        self,
        log_probs, old_log_probs, advantages, values, returns,
        mask: torch.BoolTensor, 
        clip_eps=0.2, vf_coef=0.5, entropy_coef=0.01,
    ):
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).masked_select(mask).mean()

        value_loss = F.mse_loss(values.masked_select(mask),
                                returns.masked_select(mask))

        entropy = -(log_probs.exp() * log_probs).masked_select(mask).mean()
        entropy_loss = -entropy
        return policy_loss, value_loss, entropy_loss



class StrategyFactory:
    
    def load(model, train_type, pad_token_id, dpo_beta):
        train_strategy: Dict[str, Callable[[], BaseStrategy]] = {
            "seq": lambda: SeqStrategy(model),
            "sft": lambda: SftStrategy(model),
            "dpo": lambda: DpoStrategy(model, pad_token_id, dpo_beta)
        }
        strategy = train_strategy[train_type]()
        return strategy
    

@dataclass
class TrainConfig:
    train_type: Literal["seq", "sft", "dpo"]
    dataset: Dataset
    optimizer: Optimizer
    ckpt_dir: str
    n_epoch: int = 1
    batch_size: int = 4
    n_iter_ckpt: int = 5000
    n_iter_step: int = 1
    max_grad_norm: float = 1.0
    warning_step: int = 1000
    random_seed: int = 3306
    dpo_beta: float = 0.1

    def get_kargs(self)-> Dict[str, Any]:
        config_dict = asdict(self)
        return {k: v for k, v in config_dict.items() if v is not None}
    

class ScheduleConfig:
    schedule_type: str
    schedule_kargs: dict
    
    @abstractmethod
    def get_kargs(self)-> Dict[str, Any]:
        raise NotImplementedError


@dataclass   
class CosineScheduleConfig(ScheduleConfig):
    total_iters: int
    min_rate: float = 0.05
    schedule_type: Literal["cosine"] = "cosine"
    
    def get_kargs(self) -> Dict[str, Any]:
        return {
            "schedule_type": self.schedule_type,
            "total_iters": self.total_iters,
            "min_rate": self.min_rate
        }

@dataclass
class SgdrScheduleConfig(ScheduleConfig):
    cycle_length: int
    min_rate: float = 0.05
    T_mult: int = 2
    schedule_type: Literal["sgdr"] = "sgdr"
    
    def get_kargs(self) -> Dict[str, Any]:
        return {
            "schedule_type": self.schedule_type,
            "cycle_length": self.cycle_length,
            "min_rate": self.min_rate,
            "T_mult": self.T_mult
        }


class SchedulerFactory:

    @staticmethod
    def get_sgdr_schedule(
        warning_step: int, 
        cycle_length: int, 
        min_rate: float = 0.1, 
        T_mult: int = 2
    ) -> Callable[[int], float]:

        def sgdr_schedule(now_iter: int) -> float:
            if now_iter < warning_step:
                return max(min_rate, now_iter / warning_step)
                
            adjusted_iter = now_iter - warning_step
            total_cycles, current_cycle = 0, 0
            while adjusted_iter >= cycle_length * (T_mult ** total_cycles):
                current_cycle += 1
                total_cycles += 1
            
            cycle_start = sum(cycle_length * (T_mult ** i) for i in range(current_cycle))
            cycle_pos = adjusted_iter - cycle_start
            
            cycle_length_current = cycle_length * (T_mult ** current_cycle)
            return max(min_rate, 0.5 * (1 + math.cos(math.pi * cycle_pos / cycle_length_current)))
        
        return sgdr_schedule

    @staticmethod
    def get_cosine_warmup_schedule(
        warning_step: int, 
        lr_decay_iters: int, 
        min_rate: float = 0.1
    ) -> Callable[[int], float]:

        def cosine_warmup_schedule(now_iter: int) -> float:
            if now_iter <= warning_step:
                return max(min_rate, now_iter / warning_step)
            else:
                rate = (now_iter - warning_step) / (lr_decay_iters - warning_step)
                return max(min_rate, 0.5 * (1.0 + math.cos(math.pi * rate)))
        
        return cosine_warmup_schedule
    
    def load_schedule_fn(self, strategy: str, *kargs):
        if strategy == "cosine":
            return self.get_cosine_warmup_schedule(*kargs)
        elif strategy == "sgdr":
            return self.get_sgdr_schedule(*kargs)
        