__version__ = "1.1.0"
__author__ = "ViperEkura"


from khaosz.module.parameter import ParameterLoader
from khaosz.module.generator import ChatGenerator, StreamGenerator, BatchGenerator, RetrievalGenerator
from typing import List, Tuple, Generator


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
            top_k: int=0,
            top_p: float=0.8,
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
            history: List[Tuple[str, str]]=None,
            temperature: float=0.8,
            top_k: int=0,
            top_p: float=0.8,
        ) -> List[str]:
            generator = BatchGenerator(self.parameter)
            return generator.generate(
                queries, 
                history=history,
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p,
            )
        
    
    def stream_generate(
            self, 
            query: str, 
            history: List[Tuple[str, str]]=None,
            temperature: float=0.8,
            top_k: int=0,
            top_p: float=0.8,
        ) -> Generator[Tuple[str, List[Tuple[str, str]]], None, None]:
            stream_generator = StreamGenerator(self.parameter)
            yield stream_generator.generate(
                    query, 
                    history=history,
                    temperature=temperature, 
                    top_k=top_k, 
                    top_p=top_p,
                )
    
    def retrieve_generate(
            self,
            query: str, 
            history: List[Tuple[str, str]] = None,
            temperature: float = 0.8,
            top_k: int = 0,
            top_p: float = 0.8,
            retrive_top_k: int = 5,
        ) -> str:
            generator = RetrievalGenerator(self.parameter)
            return generator.generate(
                query,
                history=history,
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p,
                retrive_top_k=retrive_top_k
            )
    
