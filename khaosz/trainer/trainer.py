import os
import copy
import math
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.optim import Optimizer
from abc import ABC, abstractmethod
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm
from typing import Tuple, Dict,  Callable

from khaosz.module import ModelParameter
from .checkpoint import TrainCheckPoint

def get_lambda_lr(warning_step, lr_decay_iters, min_rate=0.1):
    def get_lr(now_iter):
        if now_iter <= warning_step:
            return max(min_rate, now_iter / warning_step)
        else:
            rate = (now_iter - warning_step) / (lr_decay_iters - warning_step)
            return max(min_rate, 0.5 * (1.0 + math.cos(math.pi * rate)))
    
    return get_lr

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


class Trainer:
    def __init__(
        self,
        parameter: ModelParameter,
        log_path: str="./train_log.log"
    ):
        logger = logging.getLogger()
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(log_path)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
        logger.addHandler(handler)
        logger.info("initializing trainer ...")
        
        self.logger = logger
        self.model = parameter.model
        self.tokenizer = parameter.tokenizer
        self.config = parameter.config

    def train(
        self,
        train_type: str,
        dataset: Dataset,
        optimizer: Optimizer,
        ckpt_dir: str,
        n_epoch: int = 1,
        batch_size: int = 4,
        n_iter_ckpt: int = 5000,
        n_iter_step: int = 1,
        max_grad_norm: float = 1.0,
        warning_step: int = 1000,
        min_rate: float = 0.1,
        dpo_beta: float = 0.1,
        random_seed: int = 3306
    ):
        assert train_type in ["seq", "sft", "dpo"]
        
        n_iter = 0
        loss_list = []
        last_ckpt_iter = 0
        
        total_iters = len(dataset) // batch_size * n_epoch
        lambda_scheduler_fn = get_lambda_lr(warning_step, total_iters, min_rate)
        pad_token_id = self.tokenizer.pad_id
        
        ref_model = None
        if train_type == "dpo":
            ref_model = copy.deepcopy(self.model)
            ref_model.requires_grad_(False)
            ref_model.eval()
            
        train_strategy: Dict[str, Callable[[], BaseStrategy]] = {
            "seq": lambda: SeqStrategy(self.model),
            "sft": lambda: SftStrategy(self.model),
            "dpo": lambda: DpoStrategy(self.model, pad_token_id, dpo_beta)
        }
        strategy = train_strategy[train_type]()
        
        scheduler = LambdaLR(optimizer, lambda_scheduler_fn)
        sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(random_seed))

        ckpt_saver = lambda current_iter: TrainCheckPoint(
            self.model, self.tokenizer, self.config, loss_list, current_iter
        ).save_ckpt(os.path.join(ckpt_dir, f"iter_{current_iter}"))

        self.logger.info(f"Starting {train_type.upper()} training for {n_epoch} epochs")
        self.logger.info(f"Checkpoint interval: {n_iter_ckpt} iterations")

        for epoch in range(n_epoch):
            self.model.train()
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epoch}", dynamic_ncols=True)

            for batch in progress_bar:
                #forward
                loss = strategy(batch)
                loss_list.append(loss.item())
                #backward
                loss.backward()
                #step
                if n_iter % n_iter_step == 0:
                    clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                n_iter += 1
                scheduler.step()
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                #save checkpotint
                if n_iter - last_ckpt_iter >= n_iter_ckpt:
                    ckpt_saver(n_iter)
                    diff_iter = n_iter - last_ckpt_iter
                    avg_loss = sum(loss_list[last_ckpt_iter:n_iter]) / diff_iter
                    self.logger.info(f"iter: {n_iter} loss: {avg_loss}")
                    last_ckpt_iter = n_iter

        if n_iter != last_ckpt_iter:
            ckpt_saver(n_iter)
            diff_iter = n_iter - last_ckpt_iter
            avg_loss = sum(loss_list[last_ckpt_iter:n_iter]) / diff_iter
            self.logger.info(f"iter: {n_iter} loss: {avg_loss}")

        self.logger.info("Training completed")
