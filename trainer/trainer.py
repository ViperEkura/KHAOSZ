import os
import copy
import math
import torch
import logging
import pickle as pkl

import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch as st
import matplotlib.pyplot as plt

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm
from typing import Tuple
from module.transformer import Config
from module.tokenizer import BpeTokenizer


def get_lambda_lr(warning_step, lr_decay_iters, min_rate=0.1):
    def get_lr(now_iter):
        if now_iter <= warning_step:
            return max(min_rate, now_iter / warning_step)
        else:
            rate = (now_iter - warning_step) / (lr_decay_iters - warning_step)
            return max(min_rate, 0.5 * (1.0 + math.cos(math.pi * rate)))
    
    return get_lr

def seq_train_block(in_args: Tuple[Tensor, Tensor], model: nn.Module):
    x, y = in_args
    B, L = x.size()
    logits: Tensor = model(x)
    
    loss = F.cross_entropy(
        logits.view(B * L, -1), 
        y.flatten()
    )
    return loss

def sft_train_block(
    in_args: Tuple[Tensor, Tensor, Tensor, Tensor], 
    model: nn.Module
):
    x, y, loss_mask = in_args
    B, L = x.size()
    ignore_idx = -1
    
    logits: Tensor = model(x)
    masked_y = y.masked_fill(loss_mask == 0, ignore_idx)
    
    loss = F.cross_entropy(
        logits.view(B * L, -1),
        masked_y.flatten(), 
        ignore_index=ignore_idx
    )

    return loss

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

def dpo_train_block(
    in_args: Tuple[Tensor, Tensor, Tensor, Tensor],
    pi_model: nn.Module, 
    ref_model: nn.Module,
    pad_token_id: int,
    beta: float,
):
    # 输入应包含：good_ids, good_mask, bad_ids, bad_mask
    good_ids, bad_ids, good_mask, bad_mask = in_args
    
    log_pi_good = get_logprobs(pi_model, good_ids, good_mask, pad_token_id)
    log_pi_bad = get_logprobs(pi_model, bad_ids, bad_mask, pad_token_id)
    
    with torch.no_grad():
        log_ref_good = get_logprobs(ref_model, good_ids, good_mask, pad_token_id)
        log_ref_bad = get_logprobs(ref_model, bad_ids, bad_mask, pad_token_id)
    
    pi_log_ratio = log_pi_good - log_pi_bad
    ref_log_ratio = log_ref_good - log_ref_bad

    ratio_diff = pi_log_ratio - ref_log_ratio
    
    dpo_loss = -F.logsigmoid(beta * ratio_diff).mean()
    return dpo_loss

def ppo_block(
    in_args: Tuple[Tensor, Tensor, Tensor, Tensor],
    pi_model: nn.Module, 
    ref_model: nn.Module, 
    pad_token_id: int,
    epslion:float = 0.1
):
    # 输入应包含：good_ids, good_mask, bad_ids, bad_mask
    good_ids, bad_ids, good_mask, bad_mask = in_args
    pi_log_probs = get_logprobs(pi_model, good_ids, good_mask,pad_token_id)
    ref_log_probs = get_logprobs(ref_model, good_ids, good_mask, pad_token_id)
    pass


class CheckPoint:
    def __init__(
        self, 
        model: nn.Module, 
        tokenizer: BpeTokenizer,  
        config: Config,
        loss_list: list,
        n_iter: int 
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.loss_list = loss_list
        self.n_iter = n_iter

    def save_ckpt(self, save_dir):
        model_path = os.path.join(save_dir, "model.safetensors")
        config_path = os.path.join(save_dir, "config.json")
        lossfig_path = os.path.join(save_dir, "loss.png")
        loss_path = os.path.join(save_dir, "loss.pkl")
        tokenizer_path = os.path.join(save_dir, "tokenizer.json")
        
        plt.figure()
        plt.plot(self.loss_list)
        plt.title(f"Training Loss - iter {self.n_iter }")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        
        try:    
            os.makedirs(save_dir, exist_ok=True)
            st.save_file(self.model.state_dict(), model_path)
            self.config.save(config_path)
            self.tokenizer.save(tokenizer_path)
            plt.savefig(lossfig_path)
            pkl.dump(self.loss_list, open(loss_path, "wb"))
            
        except Exception as e:
            raise e
        
        finally:
            plt.close()


class Trainer:
    def __init__(
        self,
        model: nn.Module, 
        tokenizer: BpeTokenizer,  
        config: Config,
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
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
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
        generator = torch.Generator().manual_seed(random_seed)
        sampler = RandomSampler(dataset, generator=generator)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        
        total_iters = len(dataloader) * n_epoch
        labmbda_scheduler_fn = get_lambda_lr(warning_step, total_iters, min_rate)
        scheduler = LambdaLR(optimizer, labmbda_scheduler_fn)
        
        ref_model = None
        pad_token_id = self.tokenizer.pad_id
        if train_type == "dpo":
            ref_model = copy.deepcopy(self.model)
            ref_model.eval()
            ref_model.requires_grad_(False)

        train_block = {
            "seq": lambda x: seq_train_block(x, self.model),
            "sft": lambda x: sft_train_block(x, self.model),
            "dpo": lambda x: dpo_train_block(x, self.model, ref_model, pad_token_id, dpo_beta)
        }[train_type]
        
        ckpt_saver = lambda current_iter: CheckPoint(
            self.model, self.tokenizer, self.config, loss_list, current_iter
        ).save_ckpt(os.path.join(ckpt_dir, f"iter_{current_iter}"))

        self.logger.info(f"Starting {train_type.upper()} training for {n_epoch} epochs")
        self.logger.info(f"Checkpoint interval: {n_iter_ckpt} iterations")

        for epoch in range(n_epoch):
            self.model.train()
            progress_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{n_epoch}",
                dynamic_ncols=True
            )

            for batch in progress_bar:
                #forward
                loss = train_block(batch)
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