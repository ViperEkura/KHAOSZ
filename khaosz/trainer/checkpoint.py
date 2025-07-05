import os
import pickle as pkl
import torch.nn as nn
import matplotlib.pyplot as plt

from module.parameter import ModelParameter
from module.tokenizer import BpeTokenizer
from module.transformer import Config

    
class TrainCheckPoint(ModelParameter):
    def __init__(
            self, 
            model: nn.Module, 
            tokenizer: BpeTokenizer, 
            config: Config, 
            loss_list: list, 
            n_iter: int
        ):
            super().__init__(model, tokenizer, config)
            self.loss_list = loss_list
            self.n_iter = n_iter

    def save_ckpt(self, save_dir: str):
        super().save(save_dir)
        paths = {
            "loss_list": os.path.join(save_dir, "loss.pkl"),
            "lossfig": os.path.join(save_dir, "loss.png")
        }
        
        pkl.dump(self.loss_list, open(paths["loss_list"], "wb"))
        plt.figure()
        plt.plot(self.loss_list)
        plt.title(f"Training Loss - iter {self.n_iter}")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.savefig(paths["lossfig"])
        plt.close()
