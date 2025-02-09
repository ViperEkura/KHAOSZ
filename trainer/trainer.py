import os
import math
import pickle
import logging

import torch.nn as nn
import safetensors.torch as st
import matplotlib.pyplot as plt
import torch.nn.functional as F

from module.transformer import Config
from module.tokenizer import BpeTokenizer
from .dataset import DpoDataset

from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm
from typing import Tuple


def get_lambda_lr(warmup_iters, lr_decay_iters, min_rate=0.1):
    def get_lr(now_iter):
        if now_iter <= warmup_iters:
            return max(min_rate, now_iter / warmup_iters)
        else:
            rate = (now_iter - warmup_iters) / (lr_decay_iters - warmup_iters)
            return max(min_rate, 0.5 * (1.0 + math.cos(math.pi * rate)))
    
    return get_lr

def train_loss(in_args: Tuple[Tensor, Tensor], model: nn.Module):
    x, y = in_args
    B, L = x.size()
    p: Tensor = model(x)
    return F.cross_entropy(p.view(B * L, -1), y.flatten())

def dpo_train_loss(in_args: Tuple[Tensor, Tensor, Tensor], model: nn.Module, beta=0.1):
    x, y1, y2 = in_args
    _, _, D = x.size()

    # 获取模型的 logits (batch_size, seq_len, vocab_size)
    logits: Tensor = model(x)

    # 计算 chosen 和 rejected 的 log probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)

    # 获取 chosen 和 rejected 的 log probabilities
    chosen_log_probs = log_probs.gather(-1, y1.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)
    rejected_log_probs = log_probs.gather(-1, y2.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)

    # 对序列长度取平均（假设每个 token 的权重相同）
    chosen_log_probs = chosen_log_probs.mean(dim=-1)  # (batch_size,)
    rejected_log_probs = rejected_log_probs.mean(dim=-1)  # (batch_size,)

    # 计算 DPO 损失
    log_ratios = chosen_log_probs - rejected_log_probs  # (batch_size,)
    losses = -F.logsigmoid(beta * log_ratios).mean()  # 平均损失

    return losses



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
            pickle.dump(self.losses, open(loss_path, "wb"))
            
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
    ):
        logger = logging.getLogger()
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler("./train_log.txt")
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
        dataloader: DataLoader, 
        optimizer: Optimizer,
        ckpt_dir: str,
        n_epoch: int=1,
        n_iter_ckpt: int=5000,
        n_iter_step: int=1,
        max_grad_norm: float=1.0,
        warmup_iters: int=5000,
    ):
        n_iter, start_iter  = 0, 0
        losses = list()
        
        total_iters = len(dataloader) * n_epoch
        schdulder = LambdaLR(
            optimizer, 
            get_lambda_lr(warmup_iters=warmup_iters, lr_decay_iters=total_iters)
        )
        is_dpo_train = isinstance(dataloader.dataset, DpoDataset)
        
        self.logger.info("start training ...")  
        for epoch in range(n_epoch):
            self.model.train()
            tqdm_laoder = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epoch}")
            for input_param in tqdm_laoder:
                
                if is_dpo_train:
                    loss = dpo_train_loss(input_param, self.model)
                else:
                    loss = train_loss(input_param, self.model)
                    
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
                
