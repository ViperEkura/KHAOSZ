import os
import torch

from .transfomer import Config, Transformer
from .tokenizer import BpeTokenizer


def build_prompt(query, history) -> str:
    ret_prompt = str()
    if len(history) > 0:
        for query, response in history:
            ret_prompt += f"<|user|>: {query} <|system|>: <s> {response} </s>"
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
        weight_path = os.path.join(model_dir, "model.pt")
        
        self.tokenizer = BpeTokenizer(tokenizer_path)
        self.config = Config(config_path)
        self.model = Transformer(self.config)
        self.model.load_state_dict(torch.load(weight_path, weights_only=True))
        
    def sample_next_token(self, ids, temperature=1.0, top_k=0, top_p=0.0, filter_value=-float('Inf')) -> tuple[int, str]:
        device = next(self.model.parameters()).device
        input_tensor = torch.tensor(ids, device=device).unsqueeze(0)
        logits: torch.Tensor = self.model(input_tensor)[-1, -1, :] / temperature
        
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
            
        probabilities = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, num_samples=1).item()
        
        return next_token_id
    
    def stream_generate(
            self, 
            query: str, 
            history: list[tuple[str, str]]=None,
            temperature: float=1.0,
            top_k: int=0,
            top_p: float=0.0,
        ):
        if history is None:
            history = list()
        
        tokens = build_prompt(query, history)
        ids = self.tokenizer.encode(tokens)
        start_id_pos = len(ids)
        response = str()
        
        self.model.eval()
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
        if history is None:
            history = list()
        
        tokens = build_prompt(query, history)
        ids = self.tokenizer.encode(tokens)
        start_id_pos = len(ids)
        stop_id = self.tokenizer.encode("</s>")[0]
        response = str()
        
        self.model.eval()
        while len(ids) < self.config.m_len:
            next_token_id = self.sample_next_token(
                ids, temperature, 
                top_k=top_k, top_p=top_p
            )
            if next_token_id == stop_id:
                break    
            ids.append(next_token_id)
            
        response = self.tokenizer.decode(ids[start_id_pos:])
        history.append((query, response))
        
        return response
    
    def to(self, *args, **kargs):
        self.model.to(*args, **kargs)
        return self