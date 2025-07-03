import os
import pickle as pkl
import torch.nn as nn
import safetensors.torch as st
import matplotlib.pyplot as plt

from module.tokenizer import BpeTokenizer
from module.transformer import Config

class ParamCheckPoint:
    def __init__(self, model: nn.Module, tokenizer: BpeTokenizer, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def _get_paths(self, save_dir):
        return {
            "model": os.path.join(save_dir, "model.safetensors"),
            "config": os.path.join(save_dir, "config.json"),
            "tokenizer": os.path.join(save_dir, "tokenizer.json")
        }

    def save(self, save_dir):
        paths = self._get_paths(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        st.save_file(self.model.state_dict(), paths["model"])
        self.config.save(paths["config"])
        self.tokenizer.save(paths["tokenizer"])

    def load(self, load_dir):
        paths = self._get_paths(load_dir)
        
        state_dict = st.load_file(paths["model"])
        self.model.load_state_dict(state_dict)
        self.config.load(paths["config"])
        self.tokenizer.load(paths["tokenizer"])
        

class TrainCheckPoint(ParamCheckPoint):
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

    def save_ckpt(self, save_dir):
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
