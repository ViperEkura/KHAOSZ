import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple
from abc import ABC, abstractmethod
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