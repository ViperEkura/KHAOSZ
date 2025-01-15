import os
import torch
import pickle
import logging
import torch.nn as nn

from matplotlib import pyplot as plt
from module.transfomer import Config
from module.tokenizer import BpeTokenizer
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

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
        model_path = os.path.join(save_dir, "model.pt")
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
            torch.save(self.model.state_dict(), model_path)
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
        criterion: _Loss,
        ckpt_dir: str,
        n_epoch: int=1,
        n_iter_ckpt: int=5000,
        max_grad_norm: float=1.0
    ):
        vocab_size = self.config.vocab_size
        n_iter, start_iter  = 0, 0
        losses = list()
        
        self.logger.info("start training ...")  
        for epoch in range(n_epoch):
            self.model.train()
            tqdm_laoder = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epoch}")
            for (x, y) in tqdm_laoder:
                p = self.model(x)
                optimizer.zero_grad()
                loss: torch.Tensor = criterion(p.view(-1, vocab_size), y.view(-1))
                losses.append(loss.item())
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                
                n_iter += 1
                tqdm_laoder.set_postfix(loss=loss.item())
            
                if n_iter % n_iter_ckpt == 0:
                    avg_loss = sum(losses[start_iter:]) / (n_iter - start_iter)
                    start_iter = n_iter
                    self.logger.info(f"Epoch {epoch + 1}/{n_epoch} Loss: {avg_loss}")
                    ckpt_epoch_dir = os.path.join(ckpt_dir, f"epoch_{epoch+1:02d}_iter_{n_iter}")
                    checkpoint = CheckPoint(self.model, self.tokenizer, self.config, losses, epoch + 1)
                    checkpoint.save_ckpt(ckpt_epoch_dir)
                    self.logger.info(f"Saved checkpoint to {ckpt_epoch_dir}")
                
