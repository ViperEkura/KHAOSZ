import os
import torch

from .transfomer import Config, Transfomer
from .tokenizer import BpeTokenizer

def build_prompt(query, history) -> str:
    ret_prompt = str()
    if len(history) > 0:
        for query, response in history:
            ret_prompt += f"<|user|>: {query}\n\n <|system|>: {response}\n\n"
    ret_prompt += f"<|user|>: {query}\n\n <|system|>: <sog>"
    return ret_prompt

class Khaosz:
    def __init__(self, path=None):
        self.tokenizer : BpeTokenizer = None
        self.config : Config = None
        self.model : Transfomer = None
        
        if path is not None:
            self.load(path)
            
    def load(self, model_dir):
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        config_path = os.path.join(model_dir, "config.json")
        weight_path = os.path.join(model_dir, "model.pt")
        
        self.tokenizer = BpeTokenizer(tokenizer_path)
        self.config = Config(config_path)
        self.model = Transfomer(self.config)
        self.model.load_state_dict(torch.load(weight_path))
    
    def stream_generate(
            self, 
            query: str, 
            history: list[tuple[str, str]]=None,
            temperature: float=1.0,
            top_k: int=10,
            top_p: float=0.8,
        ):
        if history is None:
            history = list()
        
        tokens = build_prompt(query, history)
        ids = self.tokenizer.encode(tokens)
        response = str()
        
        self.model.eval()
        while len(ids) < self.config.m_len:
            input_tensor = torch.tensor(ids).unsqueeze(0)
            prob = self.model(input_tensor)[-1, -1, :]
            prob = torch.softmax(prob / temperature, dim=-1)
            next_token_id = torch.multinomial(prob, num_samples=1, replacement=True).item()
            next_token = self.tokenizer.id_to_token(next_token_id)
            
            if next_token == "<eog>":
                break
            response += next_token
            
            ids.append(next_token_id)
            yield query, response ,history

    def generate(
            self, 
            query: str, 
            history: list[tuple[str, str]]=None,
            temperature: float=0.8,
            top_k: int=10,
            top_p: float=0.8,
        ):
            pass
    
    
    def to():
        pass