from torch import Tensor
from typing import List, Tuple, Generator, Union

from khaosz.inference.generator import (
    TextGenerator,
    ChatGenerator, 
    StreamGenerator, 
    BatchGenerator, 
    RetrievalGenerator, 
    EmbeddingEncoder
)
from khaosz.config.param_config import ParameterLoader


class Khaosz:
    def __init__(self, model_dir: str):
        self.parameter = ParameterLoader.load(model_dir)
    
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
            generator = ChatGenerator(self.parameter)
            return generator.generate(
                query, 
                history=history,
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p,
            )
    
    def batch_generate(
            self, 
            queries: List[str],
            histories: List[Tuple[str, str]]=None,
            temperature: float=0.8,
            top_k: int=50,
            top_p: float=0.95,
        ) -> List[str]:
            generator = BatchGenerator(self.parameter)
            return generator.generate(
                queries, 
                histories=histories,
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p,
            )
        
    
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
                    query, 
                    history=history,
                    temperature=temperature, 
                    top_k=top_k, 
                    top_p=top_p,
                )
    
    def retrieve_generate(
            self,
            retrieved,
            query: str, 
            history: List[Tuple[str, str]] = None,
            temperature: float=0.8,
            top_k: int=50,
            top_p: float=0.95,
        ) -> str:
            generator = RetrievalGenerator(self.parameter)
            return generator.generate(
                retrieved,
                query,
                history=history,
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p,
            )
    
    def text_generate(            
            self, 
            query: str, 
            temperature: float=0.8,
            top_k: int=50,
            top_p: float=0.95,
        ) -> str:
        generator = TextGenerator(self.parameter)
        
        return generator.generate(
                query,
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p,
            )
    
    def encode(self, sentence: Union[str, List[str]]) -> Union[Tensor, List[Tensor]]:
        encoder = EmbeddingEncoder(self.parameter)
        return encoder.encode(sentence)