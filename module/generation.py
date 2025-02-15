import os
import torch
import safetensors.torch as st

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
        filter_value: float=-float('inf')
    ) -> Union[int, list[int]]:
        device = next(self.model.parameters()).device
        input_tensor = torch.tensor(ids, device=device)
        input_tensor = input_tensor.unsqueeze(0) if not with_batch else input_tensor
        logits: torch.Tensor = self.model(input_tensor)[:, -1, :] / temperature
        
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
    
    def batch_generate(
        self, 
        queries: List[str], 
        histories: Optional[List[List[Tuple[str, str]]]] = None,
        temperature: float = 0.8,
        top_k: int = 10,
        top_p: float = 0.8,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        
        if histories is None:
            histories = [[] for _ in queries]
        
        batch_input_ids: List[List[int]] = []
        start_id_pos: List[int] = []
        
        for query, history in zip(queries, histories):
            prompt = build_prompt(query, history)
            ids = self.tokenizer.encode(prompt)
            start_id_pos.append(len(ids))
            batch_input_ids.append(ids)
        
        batch_size = len(queries)
        is_completed = [False] * batch_size
        responses = [''] * batch_size
        device = next(self.model.parameters()).device
        pad_token_id = self.tokenizer.encode("</s>")[0]
        
        self.model.eval()
        with torch.no_grad():
            while sum(is_completed) < batch_size and len(batch_input_ids[0]) < self.config.m_len:
                # 收集未完成样本的当前ids
                active_indices = [i for i, completed in enumerate(is_completed) if not completed]
                current_batch = [batch_input_ids[i] for i in active_indices]
                
                # 转换为张量并填充
                max_len = max(len(ids) for ids in current_batch)
                padded_ids = [ids + [pad_token_id] * (max_len - len(ids)) for ids in current_batch]
                input_tensor = torch.tensor(padded_ids, device=device)
                
                # 采样下一个token
                next_token_ids = self.sample_next_token(
                    input_tensor, 
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    with_batch=True
                )
                
                # 更新每个样本的ids并检查停止条件
                for pos, idx in enumerate(active_indices):
                    next_token_id = next_token_ids[pos]
                    batch_input_ids[idx].append(next_token_id)
                    if next_token_id in self.tokenizer.stop_ids:
                        is_completed[idx] = True
                    if len(batch_input_ids[idx]) >= self.config.m_len:
                        is_completed[idx] = True
            
            # 解码响应并更新历史
            for i in range(batch_size):
                response_ids = batch_input_ids[i][start_id_pos[i]:]
                responses[i] = self.tokenizer.decode(response_ids)
                histories[i].append((queries[i], responses[i]))
        
        return responses