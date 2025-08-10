import os
import torch.nn as nn
import pickle as pkl
import safetensors.torch as st
import matplotlib.pyplot as plt

from typing import Self, Union
from dataclasses import dataclass, field
from khaosz.core.tokenizer import BpeTokenizer
from khaosz.core.transformer import TransformerConfig, Transformer


@dataclass
class ModelParameter:
    model: nn.Module = field(
        default=None,
        metadata={"help": "Transformer model."}
    )
    tokenizer: BpeTokenizer = field(
        default_factory=BpeTokenizer,
        metadata={"help": "Tokenizer for the model."}
    )
    config: TransformerConfig = field(
        default_factory=TransformerConfig,
        metadata={"help": "Transformer model configuration."}
    )

    def _get_paths(self, save_dir: str):
        return {
            "model": os.path.join(save_dir, "model.safetensors"),
            "config": os.path.join(save_dir, "config.json"),
            "tokenizer": os.path.join(save_dir, "tokenizer.json")
        }

    def save(self, save_dir: str):
        paths = self._get_paths(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        st.save_file(self.model.state_dict(), paths["model"])
        self.config.save(paths["config"])
        self.tokenizer.save(paths["tokenizer"])

    def load(self, load_dir) -> Self:
        paths = self._get_paths(load_dir)
        
        state_dict = st.load_file(paths["model"])
        self.model.load_state_dict(state_dict)
        self.config.load(paths["config"])
        self.tokenizer.load(paths["tokenizer"])
        
        return self
    
    def to(self, *args, **kwargs) -> Self:
        self.model.to(*args, **kwargs)
        return self


@dataclass
class CheckPoint(ModelParameter):
    loss_list: list = field(default_factory=list)
    current_iter: int = field(default=0)

    def save(self, save_dir: str):
        super().save(save_dir)
        paths = {
            "loss_list": os.path.join(save_dir, "loss.pkl"),
            "lossfig": os.path.join(save_dir, "loss.png")
        }
        plt.figure()
        plt.plot(self.loss_list)
        plt.title(f"Training Loss - iter {self.current_iter}")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.savefig(paths["lossfig"])
        plt.close()
        
        with  open(paths["loss_list"], "wb") as f:
            pkl.dump(self.loss_list, f)
        
    def load(self, save_dir: str) -> Self:
        super().load(save_dir)
        with open(os.path.join(save_dir, "loss.pkl"), "rb") as f:
            self.loss_list = pkl.load(f)
            self.current_iter = len(self.loss_list)
        return self


class ParameterLoader:
    @staticmethod
    def load(load_dir: str) -> Union[ModelParameter, CheckPoint]:
        model_path = os.path.join(load_dir, "model.safetensors")
        config_path = os.path.join(load_dir, "config.json")
        tokenizer_path = os.path.join(load_dir, "tokenizer.json")

        has_model_state_dict = os.path.exists(model_path)
        has_config = os.path.exists(config_path)
        assert has_config, "No config.json found in the load directory"

        config = TransformerConfig(config_path)
        tokenizer = BpeTokenizer(tokenizer_path)
        model = Transformer(config)

        if has_model_state_dict:
            state_dict = st.load_file(model_path)
            model.load_state_dict(state_dict)

        model_parameter = ModelParameter(model, tokenizer, config)

        loss_path = os.path.join(load_dir, "loss.pkl")
        if os.path.exists(loss_path):
            with open(loss_path, "rb") as f:
                loss_list = pkl.load(f)
            current_iter = len(loss_list)
            return CheckPoint(
                model=model_parameter.model,
                tokenizer=model_parameter.tokenizer,
                config=model_parameter.config,
                loss_list=loss_list,
                current_iter=current_iter
            )
        else:
            return model_parameter
