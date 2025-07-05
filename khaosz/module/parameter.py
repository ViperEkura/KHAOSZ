import os
import torch.nn as nn
import safetensors.torch as st

from typing import Union
from .retriever import Retriever
from .tokenizer import BpeTokenizer
from .transformer import Config, Transformer

class ModelParameter:
    def __init__(
        self, 
        model: nn.Module, 
        tokenizer: BpeTokenizer, 
        config: Config
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

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

    def load(self, load_dir):
        paths = self._get_paths(load_dir)
        
        state_dict = st.load_file(paths["model"])
        self.model.load_state_dict(state_dict)
        self.config.load(paths["config"])
        self.tokenizer.load(paths["tokenizer"])
    
    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self


class RetrieverParameter(ModelParameter):
    def __init__(
        self, 
        model: nn.Module, 
        tokenizer: BpeTokenizer, 
        config: Config,
        retriver: Retriever
    ):
        super().__init__(model, tokenizer, config)
        self.retriver = retriver if retriver is not None else Retriever()
    
    def _get_path(self, save_dir: str):
        return os.path.join(save_dir, "vectors.db")
        
    def load(self, load_dir):
        super().load(load_dir)
        self.retriver.load(self._get_path(load_dir))
        
    def save(self, save_dir):
        super().save(save_dir)
        self.retriver.save(self._get_path(save_dir))


class ParameterLoader:
    @staticmethod
    def load(load_dir: str) -> Union[ModelParameter, RetrieverParameter]:
        model_path = os.path.join(load_dir, "model.safetensors")
        retriever_path = os.path.join(load_dir, "vectors.db")
        config_path = os.path.join(load_dir, "config.json")
        tokenizer_path = os.path.join(load_dir, "tokenizer.json")
        
        has_retriever = os.path.exists(retriever_path)
        has_model_state_dict = os.path.exists(model_path)
        has_config = os.path.exists(config_path)
        assert has_config, "No config.json found in the load directory"
        
        config = Config(config_path)
        tokenizer = BpeTokenizer(tokenizer_path)
        model = Transformer(config)
        
        if has_model_state_dict:
            state_dict = st.load_file(model_path)
            model.load_state_dict(state_dict)
        
        if has_retriever:
            retriever = Retriever(retriever_path)
            return RetrieverParameter(model, tokenizer, config, retriever)
        else:
            return ModelParameter(model, tokenizer, config)