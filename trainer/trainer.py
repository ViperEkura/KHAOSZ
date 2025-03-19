import os
import copy
import math
import torch
import pickle as pkl
import logging

import safetensors.torch as st
import matplotlib.pyplot as plt

from module.transformer import Config
from module.tokenizer import BpeTokenizer

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm
from typing import Tuple


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
    loss = F.cross_entropy(logits.view(B * L, -1), y.flatten())
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
    loss = F.cross_entropy(logits.view(B * L, -1), masked_y.flatten(), ignore_index=ignore_idx)

    return loss

def get_logprobs(model: nn.Module, input_ids: Tensor, pad_token_id: int):
    logits = model(input_ids)
    log_probs = torch.log_softmax(logits, dim=-1)
    
    shifted_log_probs = log_probs[:, :-1, :]
    shifted_input_ids = input_ids[:, 1:]
    
    token_logprobs = torch.gather(
        input=shifted_log_probs, 
        index=shifted_input_ids.unsqueeze(-1),
        dim=-1
    ).squeeze(-1)
    
    mask = (shifted_input_ids != pad_token_id).type_as(token_logprobs)
    token_logprobs = token_logprobs * mask
    
    return token_logprobs

def dpo_train_block(
    in_args: Tuple[Tensor, Tensor], 
    model: nn.Module, 
    ref_model: nn.Module,
    pad_token_id: int,
    beta: float
):
    good_response_ids, bad_response_ids = in_args
    log_policy_good = get_logprobs(model, good_response_ids, pad_token_id)
    log_policy_bad = get_logprobs(model, bad_response_ids, pad_token_id)
    
    with torch.no_grad():
        log_ref_good = get_logprobs(ref_model, good_response_ids, pad_token_id)
        log_ref_bad = get_logprobs(ref_model, bad_response_ids, pad_token_id)
        
    log_ratio_good = log_policy_good - log_ref_good
    log_ratio_bad = log_policy_bad - log_ref_bad

    ratio_diff = log_ratio_good - log_ratio_bad
    dpo_loss = torch.mean(-F.log_sigmoid(beta * ratio_diff))
    return dpo_loss


class CheckPoint:
    def __init__(
        self, 
        model: nn.Module, 
        tokenizer: BpeTokenizer,  
        config: Config,
        losses: list,
        n_iter: int 
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.losses = losses
        self.n_iter = n_iter

    def save_ckpt(self, save_dir):
        model_path = os.path.join(save_dir, "model.safetensors")
        config_path = os.path.join(save_dir, "config.json")
        lossfig_path = os.path.join(save_dir, "loss.png")
        loss_path = os.path.join(save_dir, "loss.pkl")
        tokenizer_path = os.path.join(save_dir, "tokenizer.json")
        
        plt.figure()
        plt.plot(self.losses)
        plt.title(f"Training Loss - iter {self.n_iter }")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        
        try:    
            os.makedirs(save_dir, exist_ok=True)
            st.save_file(self.model.state_dict(), model_path)
            self.config.save(config_path)
            self.tokenizer.save(tokenizer_path)
            plt.savefig(lossfig_path)
            pkl.dump(self.losses, open(loss_path, "wb"))
            
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
        handler.setFormatter(logging.Formatter('%(asctime)s -- %(message)s'))
        logger.addHandler(handler)
        logger.info("initializing trainer ...")
        
        self.logger = logger
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    def train(
        self, 
        train_type: str,
        dataloader: DataLoader, 
        optimizer: Optimizer,
        ckpt_dir: str,
        n_epoch: int=1,
        n_iter_ckpt: int=5000,
        n_iter_step: int=1,
        max_grad_norm: float=1.0,
        warning_step: int=1000,
        min_rate: float=0.1,
        dpo_beta: float=0.1,
        
    ):
        if train_type not in ["seq", "sft", "dpo"]:
            raise ValueError("train_type must be one of ['seq', 'sft', 'dpo']")
        
        n_iter, start_iter  = 0, 0
        losses = list()
        
        total_iters = len(dataloader) * n_epoch
        
        schdulder = LambdaLR(
            optimizer, 
            get_lambda_lr(
                warning_step=warning_step,   
                lr_decay_iters=total_iters, 
                min_rate=min_rate
            )
        )
        self.logger.info(f"training mode: {train_type}")
        dpo_ref_model = None
        if train_type == "dpo":
            dpo_ref_model = copy.deepcopy(self.model)
            dpo_ref_model.eval()
    
        
        self.logger.info("start training ...")  
        for epoch in range(n_epoch):
            self.model.train()
            tqdm_laoder = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epoch}")
            for input_param in tqdm_laoder:
                
                if train_type == "seq":
                    loss = seq_train_block(
                        input_param,
                        self.model
                    )
                elif train_type == "sft":
                    loss = sft_train_block(
                        input_param,
                        self.model
                    )
                else:
                    loss = dpo_train_block(
                        input_param, 
                        self.model, 
                        dpo_ref_model,
                        pad_token_id=self.tokenizer.pad_id,
                        beta=dpo_beta
                    )
                    
                losses.append(loss.item())
                loss.backward()
                if n_iter % n_iter_step == 0:
                    clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                schdulder.step()
                n_iter += 1
                tqdm_laoder.set_postfix(
                    loss=loss.item(),
                    lr=optimizer.param_groups[-1]["lr"]
                )
            
                if n_iter % n_iter_ckpt == 0:
                    avg_loss = sum(losses[start_iter:]) / (n_iter - start_iter)
                    start_iter = n_iter
                    self.logger.info(f"Epoch {epoch + 1}/{n_epoch} Loss: {avg_loss}")
                    
                    ckpt_epoch_dir = os.path.join(ckpt_dir, f"epoch_{epoch+1:02d}_iter_{n_iter}")
                    checkpoint = CheckPoint(self.model, self.tokenizer, self.config, losses, epoch + 1)
                    checkpoint.save_ckpt(ckpt_epoch_dir)
                    self.logger.info(f"Saved checkpoint to {ckpt_epoch_dir}")
                    