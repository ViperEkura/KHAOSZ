import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any, Callable, Dict, Union
from abc import ABC, abstractmethod


def move_to_device(batch:Dict[str, Tensor], device: str) -> Any:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}

def get_logprobs(
    model: Union[nn.Module, Callable[..., Dict[str, Tensor]]], 
    input_ids: Tensor, 
    mask: Tensor, 
    pad_token_id: int,
    reduction: str,
):
    allowed_reductions = ["mean", "sum", "none"]
    if reduction not in allowed_reductions:
        raise ValueError(f"reduction must be one of {allowed_reductions}, got '{reduction}'")
    
    pad_mask =  input_ids.ne(pad_token_id)
    logits = model(input_ids, pad_mask)["logits"]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    
    shifted_log_probs = log_probs[:, :-1, :] 
    shifted_input_ids = input_ids[:, 1:]
    shifted_mask = mask[:, 1:]
    prompt_mask = pad_mask[:, 1:]
    
    token_logprobs = torch.gather(
        shifted_log_probs, 
        dim=-1, 
        index=shifted_input_ids.unsqueeze(-1)
    ).squeeze(-1)
    valid_mask = (prompt_mask & shifted_mask)
    
    if reduction == "mean":
        return (token_logprobs * valid_mask).mean(dim=-1)
    elif reduction == "sum":
        return (token_logprobs * valid_mask).sum(dim=-1)
    else:
        return token_logprobs


class BaseStrategy(ABC):
    def __init__(self, model: Union[nn.Module, Callable[..., Dict[str, Tensor]]], device: str):
        self.model = model
        self.device = device
    
    @abstractmethod
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError
    
    def __call__(self, batch: Dict[str, Tensor]) -> Tensor:
        return self.compute_loss(batch)


class SeqStrategy(BaseStrategy):
    def __init__(self, model, device, label_smoothing):
        super().__init__(model, device)
        self.label_smoothing = label_smoothing
    
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        batch = move_to_device(batch, self.device)
        input_ids, target_ids = batch["input_ids"], batch["target_ids"]
        logits = self.model(input_ids=input_ids)["logits"]
        
        loss = F.cross_entropy(
           input=logits.flatten(0, 1).float(),
            target=target_ids.flatten()
        )
        
        return loss
    

class SftStrategy(BaseStrategy):
    def __init__(self, model, device, label_smoothing):
        super().__init__(model, device)
        self.label_smoothing = label_smoothing
    
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        batch = move_to_device(batch, self.device)
        input_ids, target_ids, loss_mask = batch["input_ids"], batch["target_ids"], batch["loss_mask"]
        
        ignore_index = -100
        logits = self.model(input_ids=input_ids)["logits"]
        target_ids = target_ids.masked_fill(loss_mask == 0, ignore_index)
        
        loss = F.cross_entropy(
            input=logits.flatten(0, 1).float(),
            target=target_ids.flatten(),
            ignore_index=ignore_index
        )
        
        return loss


class DpoStrategy(BaseStrategy):
    def __init__(
            self, 
            model, 
            device, 
            pad_token_id: int, 
            beta: float,
            reduction: str,
            
        ):
        super().__init__(model, device)
        ref_model = copy.deepcopy(self.model)
        ref_model.requires_grad_(False)
        ref_model.eval()
        
        self.ref_model = ref_model
        self.pad_token_id = pad_token_id
        self.beta = beta
        self.reduction = reduction
        
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        batch = move_to_device(batch, self.device)
        good_ids, bad_ids = batch["chosen"], batch["rejected"]
        good_mask, bad_mask = batch["chosen_mask"], batch["rejected_mask"]
        
        log_pi_good = get_logprobs(self.model, good_ids, good_mask, self.pad_token_id, self.reduction)
        log_pi_bad = get_logprobs(self.model, bad_ids, bad_mask, self.pad_token_id, self.reduction)
        
        with torch.no_grad():
            log_ref_good = get_logprobs(self.ref_model, good_ids, good_mask, self.pad_token_id, self.reduction)
            log_ref_bad = get_logprobs(self.ref_model, bad_ids, bad_mask, self.pad_token_id, self.reduction)
        
        pi_log_ratio = log_pi_good - log_pi_bad
        ref_log_ratio = log_ref_good - log_ref_bad
        
        ratio_diff = pi_log_ratio - ref_log_ratio
        
        dpo_loss = -F.logsigmoid(self.beta * ratio_diff).mean()
        return dpo_loss


class GrpoStrategy(BaseStrategy):
    
    def __init__(
        self, 
        model, 
        device, 
        pad_token_id: int, 
        clip_eps: float,
        kl_coef: float,
        group_size: int,
        reduction: str,
    ):

        super().__init__(model, device)
        ref_model = copy.deepcopy(self.model)
        ref_model.requires_grad_(False)
        ref_model.eval()
        
        self.ref_model = ref_model
        self.pad_token_id = pad_token_id
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.group_size = group_size
        self.reduction = reduction
    
    def compute_advantages(self, rewards: Tensor, eps=1e-8) -> Tensor:
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True)
        advantages = (rewards - mean) / (std + eps)
        
        return advantages
    
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        batch = move_to_device(batch, self.device)
        input_ids = batch["input_ids"]
        responses = batch["responses"]
        response_masks = batch["response_masks"]
        rewards = batch["rewards"]
        
        batch_size, group_size, response_len = responses.shape
        
        # Shape: (batch_size * group_size, response_len)
        responses_flat = responses.view(-1, response_len)
        masks_flat = response_masks.view(-1, response_len)
        
        # Shape: (batch_size * group_size, seq_len)
        input_ids_expanded = input_ids.unsqueeze(1).repeat(1, group_size, 1).flatten(0, 1)
        
        # Shape: (batch_size * group_size, seq_len + response_len)
        full_sequences = torch.cat([input_ids_expanded, responses_flat], dim=-1)
        full_masks = torch.cat([torch.ones_like(input_ids_expanded), masks_flat], dim=-1)
        
        # Get log probabilities from policy model
        log_probs_policy = get_logprobs(self.model, full_sequences, 
                                        full_masks, self.pad_token_id, self.reduction)
        # Reshape to (batch_size, group_size)
        log_probs_policy = log_probs_policy.view(batch_size, group_size)
        
        # Get log probabilities from reference model (no grad)
        with torch.no_grad():
            log_probs_ref = get_logprobs(self.ref_model, full_sequences, 
                                         full_masks, self.pad_token_id, self.reduction)
            log_probs_ref = log_probs_ref.view(batch_size, group_size)
        
        # Compute advantages from rewards
        advantages = self.compute_advantages(rewards)
        
        # Compute importance sampling ratio
        # Since we're re-generating responses, we assume old policy = reference policy
        log_ratio = log_probs_policy - log_probs_ref
        ratio = torch.exp(log_ratio)
        
        # Advantages shape: (batch_size, group_size)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        kl_penalty = self.kl_coef * (log_probs_policy - log_probs_ref).square().mean()
        total_loss = policy_loss + kl_penalty
        
        return total_loss


class StrategyFactory:
    
    def load(model, train_type, device, **kwargs):
        train_strategy: Dict[str, Callable[[], BaseStrategy]] = {
            "seq": lambda: SeqStrategy(
                model, 
                device,
                kwargs.get("label_smoothing", 0.0)
            ),
            "sft": lambda: SftStrategy(
                model, 
                device,
                kwargs.get("label_smoothing", 0.0)
            ),
            "dpo": lambda: DpoStrategy(
                model,
                device,
                kwargs.get("pad_token_id"), 
                kwargs.get("dpo_beta"),
                kwargs.get("reduction", "mean")
            ),
            "grpo": lambda: GrpoStrategy(
                model,
                device,
                kwargs.get("pad_token_id"),
                kwargs.get("grpo_clip_eps", 0.2),
                kwargs.get("grpo_kl_coef", 0.04),
                kwargs.get("grpo_group_size", 4),
                kwargs.get("reduction", "mean")
            )
        }
        strategy = train_strategy[train_type]()
        return strategy