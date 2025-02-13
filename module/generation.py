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
            histories: List[List[Tuple[str, str]]]=None,
            temperature: float=0.8,
            top_k: int=10,
            top_p: float=0.8,
            batch_size: int=None,
        ):
            assert temperature >= 0.0
            assert top_k >= 0
            assert top_p >= 0.0 and top_p <= 1.0
            
            if histories is None:
                histories = [list() for _ in queries]
            
            # 设置默认batch_size为全部样本
            if batch_size is None:
                batch_size = len(queries)
            
            responses = []
            
            # 分批处理
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i+batch_size]
                batch_histories = histories[i:i+batch_size]
                
                # 处理单个子批次
                batch_responses = self._process_single_batch(
                    batch_queries, batch_histories,
                    temperature, top_k, top_p
                )
                responses.extend(batch_responses)
            
            return responses

    def _process_single_batch(
            self,
            queries: List[str],
            histories: List[List[Tuple[str, str]]],
            temperature: float,
            top_k: int,
            top_p: float,
        ) -> List[str]:
            """处理单个子批次的生成逻辑"""
            batch_size = len(queries)
            batch_data = []
            start_positions = []
            
            device = next(self.model.parameters()).device
            pad_id = self.tokenizer.encode("</s>")[0]
            
            for query, history in zip(queries, histories):
                prompt = build_prompt(query, history)
                encoded = self.tokenizer.encode(prompt)
                batch_data.append(encoded)
                start_positions.append(len(encoded))
            
            # 转换为张量并进行填充
            padded_sequences = [torch.tensor(seq, dtype=torch.long) for seq in batch_data]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                padded_sequences, 
                batch_first=True, 
                padding_value=pad_id
            ).to(device)
            
            # 生成状态跟踪
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            stop_ids_tensor = torch.tensor(list(self.tokenizer.stop_ids), device=device)
            
            # 生成循环
            self.model.eval()
            with torch.no_grad():
                while input_ids.size(1) < self.config.m_len and active_mask.any():
                    # 采样下一个token
                    next_token_ids = self.sample_next_token_batch(
                        input_ids[active_mask].tolist(), 
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    
                    # 更新序列
                    new_tokens = torch.tensor(
                        next_token_ids, 
                        device=device
                    ).unsqueeze(1)
                    input_ids = torch.cat(
                        [input_ids, torch.full((batch_size, 1), pad_id, device=device)],
                        dim=1
                    )
                    input_ids[active_mask, -1] = new_tokens.squeeze()
                    
                    # 检查停止条件
                    is_stop = torch.isin(new_tokens.flatten(), stop_ids_tensor)
                    active_mask[active_mask.clone()] &= ~is_stop
            
            # 解码结果
            responses = []
            for seq, start_pos in zip(input_ids.tolist(), start_positions):
                # 截取生成的response部分
                generated = seq[start_pos:]
                # 去除停止符之后的内容
                stop_positions = [idx for idx, tid in enumerate(generated) 
                                if tid in self.tokenizer.stop_ids]
                cutoff = stop_positions[0] if stop_positions else len(generated)
                response = self.tokenizer.decode(generated[:cutoff])
                responses.append(response)
            
            return responses