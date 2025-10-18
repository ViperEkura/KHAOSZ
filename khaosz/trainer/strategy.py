import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any, Tuple, Callable, Dict, Union
from abc import ABC, abstractmethod


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
    def __init__(self, model: Union[nn.Module, Callable[..., Dict[str, Tensor]]], device: str):
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
        logits = self.model(input_ids=input_ids)["logits"]
        
        loss = F.cross_entropy(
            input=logits.flatten(0, 1),
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
        logits = self.model(input_ids=input_ids, input_mask=attn_mask)["logits"]
        target_ids = target_ids.masked_fill(loss_mask == 0, ignore_index)
        
        loss = F.cross_entropy(
            input=logits.flatten(0, 1),
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