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
    reduction: str,
):
    # reduction on seq_len dim
    allowed_reductions = ["mean", "sum", "none"]
    if reduction not in allowed_reductions:
        raise ValueError(f"reduction must be one of {allowed_reductions}, got '{reduction}'")

    shifted_input_ids = input_ids[:, 1:]
    shifted_mask = mask[:, 1:]

    logits = model(input_ids[:, :-1, :], mask[:, :-1, :])["logits"]
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    # [batch_size, seq_len - 1]
    token_logprobs = torch.gather(
        log_probs, 
        dim=-1, 
        index=shifted_input_ids.unsqueeze(-1)
    ).squeeze(-1)
    
    if reduction == "mean":
        return (token_logprobs * shifted_mask).sum(dim=-1) / shifted_mask.sum(dim=-1).clamp(min=1.0)
    elif reduction == "sum":
        return (token_logprobs * shifted_mask).sum(dim=-1)
    else:
        return token_logprobs * shifted_mask


class BaseStrategy(ABC):
    def __init__(self, model: Union[nn.Module, Callable[..., Dict[str, Tensor]]], device: str):
        self.model = model
        self.device = device
    
    @abstractmethod
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError
    
    def __call__(self, batch: Dict[str, Tensor]) -> Tensor:
        return self.compute_loss(batch)


class SEQStrategy(BaseStrategy):
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
    

class SFTStrategy(BaseStrategy):
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


class DPOStrategy(BaseStrategy):
    def __init__(
            self, 
            model, 
            device, 
            beta: float,
            reduction: str,
            
        ):
        super().__init__(model, device)
        ref_model = copy.deepcopy(self.model)
        ref_model.requires_grad_(False)
        ref_model.eval()

        self.ref_model = ref_model
        self.beta = beta
        self.reduction = reduction
        
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        batch = move_to_device(batch, self.device)
        chosen_ids, rejected_ids = batch["chosen"], batch["rejected"]
        chosen_mask, rejected_mask = batch["chosen_mask"], batch["rejected_mask"]

        contact_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
        contact_mask = torch.cat([chosen_mask, rejected_mask], dim=0)
        
        log_pi = get_logprobs(self.model, contact_ids, contact_mask, self.reduction)

        with torch.no_grad():
            log_ref = get_logprobs(self.ref_model, contact_ids, contact_mask, self.reduction)
        
        log_pi_chosen = log_pi[:chosen_ids.shape[0]]
        log_pi_rejected = log_pi[chosen_ids.shape[0]:]
        log_ref_chosen = log_ref[:chosen_ids.shape[0]]
        log_ref_rejected = log_ref[chosen_ids.shape[0]:]
        
        pi_log_ratio = log_pi_chosen - log_pi_rejected
        ref_log_ratio = log_ref_chosen - log_ref_rejected

        ratio_diff = pi_log_ratio - ref_log_ratio
        dpo_loss = -F.logsigmoid(self.beta * ratio_diff).mean()

        return dpo_loss


class GRPOStrategy(BaseStrategy):
    
    def __init__(
        self, 
        model, 
        device, 
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
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.group_size = group_size
        self.reduction = reduction
    
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        batch = move_to_device(batch, self.device)
        prompts = batch["prompts"]
        responses = batch["responses"]
        masks = batch["masks"]
        rewards = batch["rewards"]
        
        batch_size, group_size, response_len = responses.shape
        responses_flat = responses.view(-1, response_len)
        masks_flat = masks.view(-1, response_len)
        prompt_expanded = prompts.unsqueeze(1).repeat(1, group_size, 1).flatten(0, 1)
        
        # Shape: (batch_size * group_size, seq_len + response_len)
        full_sequences = torch.cat([prompt_expanded, responses_flat], dim=-1)
        full_masks = torch.cat([torch.ones_like(prompt_expanded), masks_flat], dim=-1)
        
        log_probs_policy = get_logprobs(self.model, full_sequences, full_masks, self.reduction)
        log_probs_policy = log_probs_policy.view(batch_size, group_size)
        
        with torch.no_grad():
            log_probs_ref = get_logprobs(self.ref_model, full_sequences, full_masks, self.reduction)
            log_probs_ref = log_probs_ref.view(batch_size, group_size)
        
        # Compute advantages from rewards
        eps = torch.finfo(log_probs_policy.dtype).eps
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True)
        advantages = (rewards - mean) / (std + eps)
        
        # log_ratio = log_probs_policy - log_probs_old
        # ratio = torch.exp(log_ratio)
        # off policy: policy_model = old_model, then ratio = 1
        ratio = torch.exp(0)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        kl_penalty = self.kl_coef * (log_probs_policy - log_probs_ref).square().mean()
        total_loss = policy_loss + kl_penalty
        
        return total_loss


class StrategyFactory:
    
    def load(model, train_type, device, **kwargs):
        train_strategy: Dict[str, Callable[[], BaseStrategy]] = {
            "seq": lambda: SEQStrategy(
                model, 
                device,
                kwargs.get("label_smoothing", 0.0)
            ),
            "sft": lambda: SFTStrategy(
                model, 
                device,
                kwargs.get("label_smoothing", 0.0)
            ),
            "dpo": lambda: DPOStrategy(
                model,
                device,
                kwargs.get("dpo_beta"),
                kwargs.get("reduction", "mean")
            ),
            "grpo": lambda: GRPOStrategy(
                model,
                device,
                kwargs.get("grpo_clip_eps"),
                kwargs.get("grpo_kl_coef"),
                kwargs.get("grpo_group_size"),
                kwargs.get("reduction", "mean")
            )
        }
        strategy = train_strategy[train_type]()
        return strategy