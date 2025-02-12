import os
import torch
import safetensors.torch as st

from typing import List, Tuple
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
        temperature=1.0, 
        top_k=0, 
        top_p=1.0, 
        filter_value=-float('inf')
    ) -> int:
        device = next(self.model.parameters()).device
        input_tensor = torch.tensor(ids, device=device).unsqueeze(0)
        logits: torch.Tensor = self.model(input_tensor)[-1, -1, :] / temperature
        
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
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
    
    def sample_next_token_batch(
        self,
        ids_batch, 
        temperature=1.0, 
        top_k=0, 
        top_p=1.0, 
        filter_value=-float('inf')
    ) -> List[int]:
        device = next(self.model.parameters()).device
        input_tensor = torch.tensor(ids_batch, device=device)
        logits: torch.Tensor = self.model(input_tensor)[:, -1, :] / temperature

        # 处理 top_k 过滤
        top_k = min(top_k, logits.size(-1)) if top_k > 0 else 0
        if top_k > 0:
            topk_values = torch.topk(logits, top_k, dim=1).values
            thresholds = topk_values[:, -1].unsqueeze(1)
            logits = torch.where(logits < thresholds, torch.full_like(logits, filter_value), logits)

        # 处理 top_p 过滤
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=1), dim=1)
            
            # 创建需要移除位置的掩码
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            
            # 将排序后的索引展开并应用到原始logits
            scatter_dim = 1
            range_tensor = torch.arange(sorted_indices.shape[scatter_dim], device=device) \
                .expand(*sorted_indices.shape)
            mask = sorted_indices_to_remove.gather(scatter_dim, range_tensor)
            mask = mask.scatter_(scatter_dim, sorted_indices, sorted_indices_to_remove)
            
            logits = torch.where(mask, torch.full_like(logits, filter_value), logits)

        # 采样下一个token
        probabilities = torch.softmax(logits, dim=1)
        next_token_ids = torch.multinomial(probabilities, num_samples=1).squeeze(1).tolist()
        
        return next_token_ids
    
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
    
    
    def generate_batch(
        self, 
        queries: List[str], 
        histories: List[List[Tuple[str, str]]]=None,
        temperature: float=0.8,
        top_k: int=10,
        top_p: float=0.8,
    ):
        assert temperature >= 0.0
        assert top_k >= 0
        assert top_p >= 0.0 and top_p <= 1.0
        
        if histories is None:
            histories = [list() for _ in queries]
        
        batch_size = len(queries)
        responses = []
        
        tokens_batch = [build_prompt(query, hist) for query, hist in zip(queries, histories)]
        ids_batch = [self.tokenizer.encode(tokens) for tokens in tokens_batch]
        start_id_pos_batch = [len(ids) for ids in ids_batch]
        
        self.model.eval()
        end_items = 0
        
        for _ in range(self.config.m_len):
            next_token_ids = self.sample_next_token_batch(
                ids_batch, temperature, 
                top_k=top_k, top_p=top_p
            )
            
            for i, next_token_id in enumerate(next_token_ids):
                if next_token_id == None:
                    continue
                
                if next_token_id in self.tokenizer.stop_ids:
                    ids_batch[i].append(None)
                    end_items += 1
                else:
                    ids_batch[i].append(next_token_id)
                    
                if end_items == batch_size:
                    break
        
        for i, ids in enumerate(ids_batch):
            ids = [id for id in ids if id is not None]
            response = self.tokenizer.decode(ids[start_id_pos_batch[i]:])
            histories[i].append((queries[i], response))
            responses.append(response)
        
        return responses