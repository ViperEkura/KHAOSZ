from torch import nn
from torch import Tensor
from contextlib import contextmanager
from typing import List, Tuple, Generator, Union

from khaosz.inference.generator import (
    GenerationRequest,
    LoopGenerator,
    StreamGenerator, 
    BatchGenerator, 
    EmbeddingEncoder
)
from khaosz.config.param_config import ModelParameter

@contextmanager
def disable_random_init():
    init_functions = [
        'xavier_normal_', 'xavier_uniform_',
        'kaiming_normal_', 'kaiming_uniform_',
        'zeros_', 'ones_', 'constant_',
        'normal_', 'uniform_'
    ]
    original_funcs = {}
    for name in init_functions:
        if hasattr(nn.init, name):
            original_funcs[name] = getattr(nn.init, name)
            setattr(nn.init, name, lambda *args, **kwargs: None)
    try:
        yield
    finally:
        for name, orig_func in original_funcs.items():
            setattr(nn.init, name, orig_func)


class Khaosz:
    def __init__(self, model_dir: str):
        with disable_random_init():
            self.parameter = ModelParameter()
            self.parameter.load(model_dir)
    
    def to(self, *args, **kwargs):
        self.parameter.to(*args, **kwargs)
        return self

    def generate(
            self, 
            query: str, 
            history: List[Tuple[str, str]]=None,
            temperature: float=0.8,
            top_k: int=50,
            top_p: float=0.95,
        ) -> str:
            generator = LoopGenerator(self.parameter)
            return generator.generate(
                GenerationRequest(
                top_k, top_p, temperature, 
                self.parameter.config.max_len,
                query=query, 
                history=history,
                build_prompt=True
            ))
    
    def batch_generate(
            self, 
            query: List[str],
            history: List[Tuple[str, str]]=None,
            temperature: float=0.8,
            top_k: int=50,
            top_p: float=0.95,
        ) -> List[str]:
            generator = BatchGenerator(self.parameter)
            return generator.generate(
                GenerationRequest(
                top_k, top_p, temperature, 
                self.parameter.config.max_len,
                query=query, 
                history=history,
                build_prompt=True
            ))
    
    def stream_generate(
            self, 
            query: str, 
            history: List[Tuple[str, str]]=None,
            temperature: float=0.8,
            top_k: int=50,
            top_p: float=0.95,
        ) -> Generator[Tuple[str, List[Tuple[str, str]]], None, None]:
            stream_generator = StreamGenerator(self.parameter)
            return stream_generator.generate(
                GenerationRequest(
                top_k, top_p, temperature, 
                self.parameter.config.max_len,
                query=query, 
                history=history,
                build_prompt=True
            ))
    
    def retrieve_generate(
            self,
            retrieved,
            query: str, 
            history: List[Tuple[str, str]] = None,
            temperature: float=0.8,
            top_k: int=50,
            top_p: float=0.95,
        ) -> str:
            generator = LoopGenerator(self.parameter)
            return generator.generate(
                GenerationRequest(
                top_k, top_p, temperature, 
                self.parameter.config.max_len,
                query=query, 
                history=history,
                system_prompt=retrieved,
                build_prompt=True
            ))
    
    def text_generate(            
            self, 
            query: str, 
            temperature: float=0.8,
            top_k: int=50,
            top_p: float=0.95,
        ) -> str:
            generator = LoopGenerator(self.parameter)
            return generator.generate(
                GenerationRequest(
                top_k, top_p, temperature, 
                self.parameter.config.max_len,
                query=query, 
                build_prompt=False
            ))

    
    def encode(self, sentence: Union[str, List[str]]) -> Union[Tensor, List[Tensor]]:
        encoder = EmbeddingEncoder(self.parameter)
        return encoder.encode(sentence)