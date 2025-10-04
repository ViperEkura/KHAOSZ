import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import Dataset
from typing import Any, Literal, Optional, Tuple, Callable, Dict
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

def move_to_device(batch:Dict[str, Tensor], device: str) -> Any:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


class BaseStrategy(ABC):
    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device
    
    @abstractmethod
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError
    
    def __call__(self, batch: Tuple[Tensor, ...]) -> Tensor:
        return self.compute_loss(batch)


class SeqStrategy(BaseStrategy):
    def __init__(self, model, device):
        super().__init__(model, device)
    
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        batch = move_to_device(batch, self.device)
        input_ids, target_ids = batch["input_ids"], batch["target_ids"]
        B, L = input_ids.size()
        logits: Tensor = self.model(input_ids=input_ids)["logits"]
        
        loss = F.cross_entropy(
            input=logits.view(B * L, -1),
            target=target_ids.flatten()
        )
        return loss
    

class SftStrategy(BaseStrategy):
    def __init__(self, model, device):
        super().__init__(model, device)
    
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        batch = move_to_device(batch, self.device)
        input_ids, target_ids = batch["input_ids"], batch["target_ids"]
        loss_mask, attn_mask = batch["loss_mask"], batch["attn_mask"]
        
        ignore_index = -100
        B, L = input_ids.size()
        
        logits: Tensor = self.model(
            input_ids=input_ids, 
            input_mask=attn_mask
        )["logits"]
        
        target_ids = target_ids.masked_fill(loss_mask == 0, ignore_index)
        
        loss = F.cross_entropy(
            input=logits.view(B * L, -1),
            target=target_ids.flatten(),
            ignore_index=ignore_index
        )
        
        return loss


class DpoStrategy(BaseStrategy):
    def __init__(self, model, device, pad_token_id, beta):
        super().__init__(model, device)
        ref_model = copy.deepcopy(self.model)
        ref_model.requires_grad_(False)
        ref_model.eval()
        
        self.ref_model = ref_model
        self.pad_token_id = pad_token_id
        self.beta = beta
        
    def compute_loss(self, batch: Tuple[Tensor, ...]) -> Tensor:
        batch = move_to_device(batch, self.device)
        good_ids, bad_ids = batch["chosen"], batch["rejected"]
        good_mask, bad_mask = batch["chosen_mask"], batch["rejected_mask"]
        
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
    
    def load(model, train_type, device, **kwargs):
        train_strategy: Dict[str, Callable[[], BaseStrategy]] = {
            "seq": lambda: SeqStrategy(model, device),
            "sft": lambda: SftStrategy(model, device),
            "dpo": lambda: DpoStrategy(
                model, 
                device,
                kwargs.get("pad_token_id"), 
                kwargs.get("dpo_beta")
            )
        }
        strategy = train_strategy[train_type]()
        return strategy


@dataclass
class ScheduleConfig(ABC):
    schedule_type: str = field(
        default="cosine",
        metadata={
            "help": "Type of learning rate schedule.", 
            "choices": ["cosine", "sgdr"]
        }
    )
    warmup_steps: int = field(
        default=1000,
        metadata={"help": "Number of warmup steps."}
    )
    min_rate: float = field(
        default=0.05,
        metadata={"help": "Minimum learning rate multiplier."}
    )
    
    @abstractmethod
    def get_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if not 0 <= self.min_rate <= 1:
            raise ValueError(f"min_rate must be between 0 and 1, got {self.min_rate}")


@dataclass
class CosineScheduleConfig(ScheduleConfig):
    total_steps: int = field(
        default=None,
        metadata={"help": "Total training steps for cosine schedule."}
    )
    schedule_type: Literal["cosine"] = "cosine"
    
    def get_kwargs(self) -> Dict[str, Any]:
        if self.total_steps is None:
            raise ValueError("total_steps must be specified for cosine schedule")
            
        return {
            "schedule_type": self.schedule_type,
            "warmup_steps": self.warmup_steps,
            "lr_decay_steps": self.total_steps - self.warmup_steps,
            "min_rate": self.min_rate
        }
    
    def validate(self) -> None:
        super().validate()
        if self.total_steps is not None and self.total_steps <= self.warmup_steps:
            raise ValueError(f"total_steps ({self.total_steps}) must be greater than warmup_steps ({self.warmup_steps})")


@dataclass
class SgdrScheduleConfig(ScheduleConfig):
    cycle_length: int = field(
        default=1000,
        metadata={"help": "Length of the first cycle in steps."}
    )
    t_mult: int = field( 
        default=2,
        metadata={"help": "Multiplier for cycle length growth."}
    )
    schedule_type: Literal["sgdr"] = "sgdr"

    def get_kwargs(self) -> Dict[str, Any]:
        return {
            "schedule_type": self.schedule_type,
            "warmup_steps": self.warmup_steps,
            "cycle_length": self.cycle_length,
            "min_rate": self.min_rate,
            "t_mult": self.t_mult
        }
    
    def validate(self) -> None:
        super().validate()
        if self.cycle_length <= 0:
            raise ValueError(f"cycle_length must be positive, got {self.cycle_length}")
        if self.t_mult < 1:
            raise ValueError(f"t_mult must be >= 1, got {self.t_mult}")


class SchedulerFactory:
    """Factory for creating learning rate schedule functions."""
    
    @staticmethod
    def get_sgdr_schedule(
        warmup_steps: int, 
        cycle_length: int, 
        min_rate: float = 0.05, 
        t_mult: int = 2
    ) -> Callable[[int], float]:
        """
        Create SGDR (Stochastic Gradient Descent with Warm Restarts) schedule.
        
        Args:
            warmup_steps: Number of warmup steps
            cycle_length: Length of the first cycle
            min_rate: Minimum learning rate multiplier
            t_mult: Cycle length multiplier
            
        Returns:
            Schedule function that takes current step and returns LR multiplier
        """
        
        def sgdr_schedule(current_step: int) -> float:
            # Warmup phase
            if current_step < warmup_steps:
                return max(min_rate, current_step / warmup_steps)
            
            # SGDR phase
            steps_since_warmup = current_step - warmup_steps
            
            # Find current cycle and position within cycle
            cycle_start = 0
            current_cycle_length = cycle_length
            cycle_index = 0
            
            while steps_since_warmup >= cycle_start + current_cycle_length:
                cycle_start += current_cycle_length
                current_cycle_length *= t_mult
                cycle_index += 1
            
            position_in_cycle = steps_since_warmup - cycle_start
            progress = position_in_cycle / current_cycle_length
            
            # Cosine annealing within cycle
            return max(min_rate, 0.5 * (1 + math.cos(math.pi * progress)))
        
        return sgdr_schedule

    @staticmethod
    def get_cosine_schedule(
        warmup_steps: int, 
        lr_decay_steps: int, 
        min_rate: float = 0.05
    ) -> Callable[[int], float]:
        """
        Create cosine decay schedule with warmup.
        
        Args:
            warmup_steps: Number of warmup steps
            lr_decay_steps: Number of steps for cosine decay after warmup
            min_rate: Minimum learning rate multiplier
            
        Returns:
            Schedule function that takes current step and returns LR multiplier
        """
        
        def cosine_schedule(current_step: int) -> float:
            if current_step < warmup_steps:
                # Linear warmup
                return max(min_rate, current_step / warmup_steps)
            else:
                # Cosine decay
                decay_progress = (current_step - warmup_steps) / lr_decay_steps
                decay_progress = min(decay_progress, 1.0)  # Clamp at 1.0
                return max(min_rate, 0.5 * (1.0 + math.cos(math.pi * decay_progress)))
        
        return cosine_schedule

    @staticmethod
    def load_schedule_fn(scedule_config: ScheduleConfig) -> Callable[[int], float]:
        kwargs = scedule_config.get_kwargs()
        schedule_type = kwargs.pop("schedule_type")
        
        if schedule_type == "cosine":
            return SchedulerFactory.get_cosine_schedule(**kwargs)
        elif schedule_type == "sgdr":
            return SchedulerFactory.get_sgdr_schedule(**kwargs)
        else:
            raise ValueError(f"Unsupported schedule type: {schedule_type}")
        