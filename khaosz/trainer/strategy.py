import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import Dataset
from typing import Any, Literal, Tuple, Callable, Dict
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field


def get_logprobs(model:nn.Module, input_ids: Tensor, mask: Tensor, pad_token_id: int):
    input_mask =  input_ids.ne(pad_token_id)
    logits = model(input_ids, input_mask)["logits"]
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

def build_loss_mask(input_ids: Tensor, bos_token_id: int, eos_token_id: int) -> Tensor:
    token_markers = torch.zeros_like(input_ids, dtype=torch.int8)
    
    is_bos_token = input_ids.eq(bos_token_id)
    is_eos_token = input_ids.eq(eos_token_id)
    
    token_markers[is_bos_token] = 1
    token_markers[is_eos_token] = -1 
    
    cumulative_markers = torch.cumsum(token_markers, dim=-1)
    min_cumulative = cumulative_markers.min(dim=-1, keepdim=True).values
    loss_mask = cumulative_markers - min_cumulative

    return loss_mask.to(dtype=torch.bool)

def build_attention_mask(input_ids: Tensor, user_token_id: int, multi_turn: bool) -> Tensor:
    bsz, seq_len = input_ids.size()
    is_user_token = input_ids.eq(user_token_id)
    turn_id = is_user_token.cumsum(dim=-1)
    
    iq = turn_id.view(bsz, seq_len, 1)
    ik = turn_id.view(bsz, 1, seq_len)
    
    seq_mask = (iq <= ik) if multi_turn else (iq == ik)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).bool()
    attention_mask = seq_mask & causal_mask
    
    return attention_mask


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
        logits: Tensor = self.model(x)["logits"]
        
        loss = F.cross_entropy(
            logits.view(B * L, -1), y.flatten()
        )
        return loss
    

class SftStrategy(BaseStrategy):
    def __init__(
            self, 
            model: nn.Module, 
            bos_id: int, 
            eos_id: int, 
            user_token_id: int,
            multi_turn: bool
        ):
        super().__init__(model)

        self.loss_mask_builder = lambda x: build_loss_mask(x, bos_id, eos_id)
        self.attn_mask_builder = lambda x: build_attention_mask(x, user_token_id, multi_turn)
    
    def compute_loss(self, batch: Tuple[Tensor, ...]) -> Tensor:
        x, y, loss_mask = batch
        B, L = x.size()
        ignore_idx = -1
        
        logits: Tensor = self.model(x)["logits"]
        masked_y = y.masked_fill(loss_mask == 0, ignore_idx)
        
        loss = F.cross_entropy(
            logits.view(B * L, -1),
            masked_y.flatten(), 
            ignore_index=ignore_idx
        )

        return loss

class DpoStrategy(BaseStrategy):
    def __init__(self, model, pad_token_id, beta):
        super().__init__(model)
        ref_model = copy.deepcopy(self.model)
        ref_model.requires_grad_(False)
        ref_model.eval()
        
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
    def __init__(self, model, pad_token_id, epsilon):
        super().__init__(model)
        ref_model = copy.deepcopy(self.model)
        ref_model.requires_grad_(False)
        ref_model.eval()
        
        self.ref_model = ref_model
        self.pad_token_id = pad_token_id
        self.epsilon = epsilon
        
    def ppo_clip_loss_masked(
        self,
        log_probs: Tensor, 
        old_log_probs: Tensor, 
        advantages: Tensor, 
        values: Tensor, 
        returns: Tensor,
        mask: Tensor, 
        clip_eps: float=0.2, 
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
    
    def load(model, train_type, **kwargs):
        train_strategy: Dict[str, Callable[[], BaseStrategy]] = {
            "seq": lambda: SeqStrategy(model),
            "sft": lambda: SftStrategy(model, kwargs.pop("bos_token_id"), kwargs.pop("eos_token_id"), kwargs.pop("multi_turn")),
            "dpo": lambda: DpoStrategy(model, kwargs.pop("pad_token_id") , kwargs.pop("dpo_beta"))
        }
        strategy = train_strategy[train_type]()
        return strategy
    

@dataclass
class TrainConfig:
    train_type: str = field(
        default_factory=["seq", "sft", "dpo"],
        metadata={"help": "Type of training."}
    )
    dataset: Dataset = field(
        default=None,
        metadata={"help": "Dataset for training."}
    )
    optimizer: Optimizer = field(
        default=None,
        metadata={"help": "Optimizer for training."}
    )
    ckpt_dir: str = field(
        default="./checkpoint",
        metadata={"help": "Checkpoint directory."}
    )
    n_epoch: int = field(
        default=1,
        metadata={"help": "Number of epochs for training."}
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for training."}
    )
    n_iter_ckpt: int = field(
        default=5000,
        metadata={"help": "Number of iterations between checkpoints."}
    )
    n_iter_step: int = field(
        default=1,
        metadata={"help": "Number of iterations between steps."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm."}
    )
    random_seed: int = field(
        default=3407,
        metadata={"help": "Random seed."}
    )
    dpo_beta: float = field(
        default=0.1,
        metadata={"help": "DPO beta."}
    )

    def get_kwargs(self)-> Dict[str, Any]:
        config_dict = asdict(self)
        return {k: v for k, v in config_dict.items() if v is not None}
    

@dataclass
class ScheduleConfig:
    schedule_type: str = field(
        default_factory=["cosine", "sgdr"],
        metadata={"help": "Type of learning rate schedule."}
    )
    warning_step: int = field(
        default=1000,
        metadata= {"help": "Warning up step."}
    )
    @abstractmethod
    def get_kwargs(self)-> Dict[str, Any]:
        raise NotImplementedError


@dataclass   
class CosineScheduleConfig(ScheduleConfig):
    total_iters: int = field(
        default=None,
        metadata={"help": "Total iterations for cosine schedule."}
    )
    min_rate: float = field(
        default=0.05,
        metadata={"help": "Minimum rate for cosine schedule."}
    )
    schedule_type: Literal["cosine"] = "cosine"
    
    def get_kwargs(self) -> Dict[str, Any]:
        return {
            "schedule_type": self.schedule_type,
            "warning_step": self.warning_step,
            "lr_decay_iters": self.total_iters - self.warning_step,
            "min_rate": self.min_rate
        }

@dataclass
class SgdrScheduleConfig(ScheduleConfig):
    cycle_length: int = field(
        default=1000,
        metadata={"help": "Cycle length for sgdr schedule."}
    )
    min_rate: float = field(
        default=0.05,
        metadata={"help": "Minimum rate for sgdr schedule."}
    )
    T_mult: int = field(
        default=2,
        metadata={"help": "T_mult for sgdr schedule."}
    )
    schedule_type: Literal["sgdr"] = "sgdr"

    def get_kwargs(self) -> Dict[str, Any]:
        return {
            "schedule_type": self.schedule_type,
            "warning_step": self.warning_step,
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
    
    @staticmethod
    def load_schedule_fn(**kwargs):
        strategy = kwargs.pop("schedule_type")
        if strategy == "cosine":
            return SchedulerFactory.get_cosine_warmup_schedule(**kwargs)
        elif strategy == "sgdr":
            return SchedulerFactory.get_sgdr_schedule(**kwargs)
        else:
            raise ValueError(f"Invalid schedule type: {strategy}")
        