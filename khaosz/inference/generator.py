import torch
from torch import Tensor 
from typing import List, Tuple, Union, Optional, Generator
from khaosz.inference.core import GeneratorCore, EmbeddingEncoderCore, KVCacheManager
from khaosz.config.param_config import ModelParameter


def build_prompt(
    query: str, 
    init_prompt: Optional[str] = None,
    history: Optional[List[Tuple[str, str]]] = None
    ) -> str:
    """ 
    Build prompt in ChatML format for query and history
    
    Args:
        query(str): query string
        history(Optional[List[Tuple[str, str]]]): history list of query and response
        
    Returns:
        str: prompt string in ChatML format
        
    """
    prompt = f"<|im_start|>system\n{init_prompt}<|im_end|>\n" if init_prompt else ""
    
    # (convert tuple format to ChatML)
    if history:
        for user_msg, assistant_msg in history:
            prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{assistant_msg}<|im_end|>\n"
    
    prompt += f"<|im_start|>user\n{query}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    
    return prompt

def pad_sequence(ids_list: List[List[int]], max_ids_len: int, pad_id: int) -> List[List[int]]:
    """ 
    Pad a list of sequences to a fixed length.
    
    Args:
        ids_list (List[List[int]]): A list of sequences.
        max_ids_len (int): The maximum length of sequences.
        pad_id (int): The id to pad sequences.
        
    Returns:
        List[List[int]]: A list of padded sequences.
        
    """
    new_ids_list = []
    for ids in ids_list:
        pad_len = max_ids_len - len(ids)
        padded_seq = [pad_id] * pad_len + ids
        new_ids_list.append(padded_seq)
    
    return new_ids_list


class TextGenerator(GeneratorCore):
    def __init__(self, parameter: ModelParameter):
        super().__init__(parameter)
    
    def generate(
            self, 
            query: str, 
            temperature: float,
            top_k: int,
            top_p: float,
        ) -> str:
        assert temperature >= 0.0
        assert top_k >= 0
        assert top_p >= 0.0 and top_p <= 1.0
        
        device = next(self.model.parameters()).device
        cache_manager = KVCacheManager(self.config, 1, device=device)
        
        ids = self.tokenizer.encode(query)
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        
        start_cache_pos = len(ids)
        cur_cache_pos = 0
        self.model.eval()
        kv_caches = cache_manager.get_kvcache()
        
        ids = self.generate_loop(
            input_ids, ids, temperature, top_k, top_p, 
            kv_caches=kv_caches, 
            start_pos=cur_cache_pos
        )
        
        response = self.tokenizer.decode(ids[start_cache_pos:])
        
        return response


class ChatGenerator(GeneratorCore):
    def __init__(self, parameter: ModelParameter):
        super().__init__(parameter)
        
    def generate(
            self, 
            query: str, 
            history: List[Tuple[str, str]],
            temperature: float,
            top_k: int,
            top_p: float,
        ) -> str:
        
        assert temperature >= 0.0
        assert top_k >= 0
        assert top_p >= 0.0 and top_p <= 1.0

        if history is None:
            history = []
        
        device = next(self.model.parameters()).device
        cache_manager = KVCacheManager(self.config, 1, device=device)
        
        ids = self.tokenizer.encode(build_prompt(query, history))
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        
        start_cache_pos = len(ids)
        cur_cache_pos = 0
        self.model.eval()
        kv_caches = cache_manager.get_kvcache()
        
        ids = self.generate_loop(
            input_ids, ids, temperature, top_k, top_p, 
            kv_caches=kv_caches, 
            start_pos=cur_cache_pos
        )
        
        response = self.tokenizer.decode(ids[start_cache_pos:])
        
        return response
    

class StreamGenerator(GeneratorCore):
    def __init__(self, parameter: ModelParameter):
        super().__init__(parameter)
    
    def generate(
            self, 
            query: str, 
            history: List[Tuple[str, str]],
            temperature: float,
            top_k: int,
            top_p: float,
        ) -> Generator[Tuple[str, List[Tuple[str, str]]], None, None]:
        
        assert temperature >= 0.0
        assert top_k >= 0
        assert top_p >= 0.0 and top_p <= 1.0

        if history is None:
            history = []
        
        device = next(self.model.parameters()).device
        cache_manager = KVCacheManager(self.config, 1, device=device)
                
        ids = self.tokenizer.encode(build_prompt(query, history))
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        cpy_history = history.copy()
        
        start_cache_pos = len(ids)
        cur_cache_pos = 0
        self.model.eval()
        kv_caches = cache_manager.get_kvcache()
        
        for _ in range(len(ids), self.config.max_len):
            next_token_id, cache_increase = self.generate_iterator(
                input_ids, temperature, top_k, top_p, kv_caches=kv_caches, start_pos=cur_cache_pos)
            
            input_ids = next_token_id
            ids.append(next_token_id.item())
            cur_cache_pos += cache_increase
            
            response = self.tokenizer.decode(ids[start_cache_pos:])
            yield response, cpy_history + [(query, response)]
            
            if next_token_id.item() in self.tokenizer.stop_ids:
                yield response + "\n", cpy_history + [(query, response)]
                break
    

class BatchGenerator(GeneratorCore):
    def __init__(self, parameter: ModelParameter):
        super().__init__(parameter)
    
    def generate(
        self, 
        queries: List[str],
        histories: List[List[Tuple[str, str]]],
        temperature: float, 
        top_k: int, 
        top_p: float 
    ) -> List[str]:
        
        assert temperature >= 0.0
        assert top_k >= 0
        assert top_p >= 0.0 and top_p <= 1.0
        
        batch_size = len(queries)
        if histories is None:
            histories = [[] for _ in range(batch_size)]
        
        prompts = [build_prompt(query, history) for query, history in zip(queries, histories)]
        ids_list = [self.tokenizer.encode(prompt) for prompt in prompts]
        max_ids_len = max(len(ids) for ids in ids_list)
        ids_list = pad_sequence(ids_list, max_ids_len, self.tokenizer.pad_id)
        
        device = next(self.model.parameters()).device
        cache_manager = KVCacheManager(self.config, batch_size, device=device)
        
        input_tensor = torch.tensor(ids_list, device=device, dtype=torch.long)
        cache_manager.set_seq_mask(input_tensor, self.tokenizer.pad_id)
        activate_task_mask = [True] * batch_size
        
        start_cache_pos = max_ids_len
        cur_cache_pos = 0
        
        while max_ids_len < self.config.max_len and sum(activate_task_mask) != 0:
            kv_caches = cache_manager.get_kvcache()
            attn_mask =cache_manager.get_seq_mask()
            
            next_token_id, cache_increase = self.generate_iterator(
                input_tensor, temperature, top_k, top_p, attn_mask=attn_mask, kv_caches=kv_caches, start_pos=cur_cache_pos)
            
            cur_cache_pos += cache_increase
            active_mask = []
            c_ids = 0
            
            for i in range(batch_size):
                if activate_task_mask[i]:
                    token = next_token_id[c_ids, :].item()
                    ids_list[i].append(token)
                    c_ids += 1
                    
                    is_active = not token in self.tokenizer.stop_ids
                    activate_task_mask[i] = is_active 
                    active_mask.append(is_active)
            
            active_mask = torch.tensor(active_mask, device=device, dtype=torch.bool)
            cache_manager.update(active_mask)
            input_tensor = next_token_id[active_mask, :]

            max_ids_len += 1
        
        
        responses = [str()] * batch_size
        for i in range(batch_size):
            responses[i] = self.tokenizer.decode(ids_list[i][start_cache_pos:])
            histories[i].append((queries[i], responses[i]))
        
        return responses


class RetrievalGenerator(GeneratorCore):
    def __init__(self, retriever_parameter: ModelParameter):
        super().__init__(retriever_parameter)
    
    def generate(
        self,
        retrieved: List[str],
        query: str, 
        history: List[Tuple[str, str]],
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> str:
        assert temperature >= 0.0
        assert top_k >= 0
        assert top_p >= 0.0 and top_p <= 1.0
        
        if history is None:
            history = []
            
        retrieved = "\n".join([f"{idx + 1}. {key}" for idx, key in enumerate(retrieved)]) if retrieved else ""
        retrieved_query = f"{retrieved}\n\n{query}" if retrieved else query
        parameter = ModelParameter(self.model, self.tokenizer, self.config)
        
        return ChatGenerator(parameter).generate(
            retrieved_query, 
            history,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

class EmbeddingEncoder(EmbeddingEncoderCore):
    def __init__(self, parameter: ModelParameter):
        super().__init__(parameter)
        
    def encode(self, sentence: Union[str, List[str]]) -> Union[Tensor, List[Tensor]]:
        return super().encode(sentence)
        