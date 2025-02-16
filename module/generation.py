import os
import torch
import safetensors.torch as st

from torch import Tensor 
from typing import List, Tuple, Union, Optional
from .transformer import Config, Transformer
from .tokenizer import BpeTokenizer


def build_prompt(query, history) -> str:
    ret_prompt = str()
    if len(history) > 0:
        for his_query, his_response in history:
            ret_prompt += f"<|user|>: {his_query} <|system|>: <s> {his_response} </s>"
    ret_prompt += f"<|user|>: {query} <|system|>: <s> "
    return ret_prompt


class Khaosz:
    def __init__(self, path=None):
        self.tokenizer : BpeTokenizer = None
        self.config : Config = None
        self.model : Transformer = None
        
        if path is not None:
            self.load(path)
            
    def load(self, model_dir):
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        config_path = os.path.join(model_dir, "config.json")
        weight_path = os.path.join(model_dir, "model.safetensors")
        
        self.tokenizer = BpeTokenizer(tokenizer_path)
        self.config = Config(config_path)
        self.model = Transformer(self.config)
        state_dict = st.load_file(weight_path)
        self.model.load_state_dict(state_dict)
    
    def to(self, *args, **kargs):
        self.model.to(*args, **kargs)
        return self
        
    def sample_next_token(
        self, 
        ids, 
        temperature: float, 
        top_k: int, 
        top_p: float,
        with_batch: bool=False,
        attn_mask: Optional[Tensor] = None,
        filter_value: float=-float('inf')
    ) -> Union[int, list[int]]:
        device = next(self.model.parameters()).device
        input_tensor = torch.tensor(ids, device=device)
        input_tensor = input_tensor.unsqueeze(0) if not with_batch else input_tensor
        logits: torch.Tensor = self.model(input_tensor, attn_mask)[:, -1, :] / temperature
        
        top_k = min(top_k, logits.size(-1)) if top_k > 0 else 0
        
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k).values[:, -1, None]
            logits[indices_to_remove] = filter_value
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 0] = False

            batch_indices, sorted_pos = torch.where(sorted_indices_to_remove)
            original_col_indices = sorted_indices[batch_indices, sorted_pos]
            logits[batch_indices, original_col_indices] = filter_value
        
        probabilities = torch.softmax(logits, dim=-1)
        next_token_ids = torch.multinomial(probabilities, num_samples=1)
        
        return next_token_ids.item() if not with_batch else next_token_ids.flatten().tolist()
    
    
    def stream_generate(
            self, 
            query: str, 
            history: List[Tuple[str, str]]=None,
            temperature: float=1.0,
            top_k: int=0,
            top_p: float=1.0,
        ):
        
        assert temperature >= 0.0
        assert top_k >= 0
        assert top_p >= 0.0 and top_p <= 1.0
        
        if history is None:
            history = list()
        
        tokens = build_prompt(query, history)
        ids = self.tokenizer.encode(tokens)
        start_id_pos = len(ids)
        response = str()
        
        self.model.eval()
        with torch.no_grad():
            while len(ids) < self.config.m_len:
                next_token_id = self.sample_next_token(
                    ids, temperature, 
                    top_k=top_k, top_p=top_p
                )
                if next_token_id in self.tokenizer.stop_ids:
                    break
                ids.append(next_token_id)
                response = self.tokenizer.decode(ids[start_id_pos:])
                yield response ,history
        
        history.append((query, response))

    def generate(
            self, 
            query: str, 
            history: list[tuple[str, str]]=None,
            temperature: float=0.8,
            top_k: int=10,
            top_p: float=0.8,
        ):
        
        assert temperature >= 0.0
        assert top_k >= 0
        assert top_p >= 0.0 and top_p <= 1.0
        
        if history is None:
            history = list()
        
        tokens = build_prompt(query, history)
        ids = self.tokenizer.encode(tokens)
        start_id_pos = len(ids)
        response = str()
        
        self.model.eval()
        with torch.no_grad():
            while len(ids) < self.config.m_len:
                next_token_id = self.sample_next_token(
                    ids, temperature, 
                    top_k=top_k, top_p=top_p
                )
                if next_token_id in self.tokenizer.stop_ids:
                    break
                ids.append(next_token_id)
            
        response = self.tokenizer.decode(ids[start_id_pos:])
        history.append((query, response))
        
        return response
    