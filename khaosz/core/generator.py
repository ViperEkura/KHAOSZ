import torch
from torch import Tensor 
from typing import List, Tuple, Union, Optional, Generator, Self
from khaosz.core.parameter import ModelParameter


def build_prompt(query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
    """ 
    Build prompt for query and history
    
    Args:
        query(str): query string
        history(Optional[List[Tuple[str, str]]]): history list of query and response
        
    Returns:
        str: prompt string
        
    """
    prompt_parts = []
    
    if history is None:
        history = []
    
    for his_query, his_response in history:
        prompt_parts.append(f"<|user|> {his_query} <|system|> <bos>{his_response}<eos>")
        
    if query is not None:
        prompt_parts.append(f"<|user|> {query} <|system|> <bos>")
    
    return "\n".join(prompt_parts)

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

def apply_sampling_strategies(
    logits: Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    filter_value: float = -float("inf")
) -> Tensor:
    """ 
    Apply sampling strategies to the logits tensor.
    
    Args:
        logits (Tensor): The logits tensor.
        temperature (float): The temperature parameter.
        top_k (int): The top-k parameter.
        top_p (float): The top-p parameter.
        filter_value (float, optional): The filter value. Defaults to -float("inf").
        
    Returns:
        Tensor: The sampled logits tensor.
        
    """
    
    if temperature != 1.0:
        logits = logits / temperature
    
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(
            dim=1,
            index=sorted_indices,
            src=sorted_indices_to_remove
        )
        
        logits[indices_to_remove] = filter_value
    
    return logits


class KVCacheManager:
    def __init__(
        self, 
        num_layers: int, 
        batch_size: int,
        max_len: int, 
        num_heads: int, 
        head_dim: int, 
        device: torch.device = "cuda", 
        dtype: torch.dtype = torch.bfloat16
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        
        self._kv_cache: List[Tuple[Tensor, Tensor]] = None
        self._seq_mask: Tensor = None
        self._initialize()

    def _initialize(self):
        self._kv_cache = []
        for _ in range(self.num_layers):
            k_cache = torch.zeros(
                (self.batch_size, self.max_len, self.num_heads, self.head_dim),
                device=self.device, dtype=self.dtype
            )
            v_cache = torch.zeros(
                (self.batch_size, self.max_len, self.num_heads, self.head_dim),
                device=self.device, dtype=self.dtype
            )
            self._kv_cache.append((k_cache, v_cache))
        
        self._seq_mask = torch.ones(
            (self.batch_size, self.max_len),
            device=self.device, dtype=torch.bool
        )

    def update(self, active_mask: Tensor):
        for i in range(self.num_layers):
            k_cache, v_cache = self._kv_cache[i]
            new_k_cache, new_v_cache = k_cache[active_mask], v_cache[active_mask]
            self._kv_cache[i] = (new_k_cache, new_v_cache)
        
        self._seq_mask = self._seq_mask[active_mask]

    def reset(self, full_reset=False):
        if full_reset:
            self._kv_cache = None
            self._seq_mask = None
        else:
            self._initialize()
    
    def set_seq_mask(self, input_ids: Tensor, pad_id: int):
        batch_size, seq_len = input_ids.shape
        bool_mask = (input_ids != pad_id)
        self._seq_mask[: batch_size, : seq_len] = bool_mask

    def get_kvcache(self) -> List[Tuple[Tensor, Tensor]]:
        return self._kv_cache
    
    def get_seq_mask(self) -> Tensor:        
        return self._seq_mask
    

class GeneratorCore:
    def __init__(self, parameter: ModelParameter):
        self.model = parameter.model
        self.tokenizer = parameter.tokenizer
        self.config = parameter.config

    def compute_logits(
        self,
        input_ids: Tensor,
        attn_mask: Optional[Tensor] = None,
        kv_caches: Optional[List[Tuple[Tensor, Tensor]]] = None,
        start_pos: int = 0
    ) -> Tuple[Tensor, int]:
        with torch.inference_mode():
            outputs = self.model(input_ids, attn_mask, kv_caches, start_pos)
            logits = outputs["logits"][:, -1, :]
            cache_increase = input_ids.size(-1)   
        
        return logits, cache_increase

    def to(self, *args, **kargs) -> Self:
        self.model.to(*args, **kargs)
        return self


class EmbeddingEncoderCore:
    def __init__(self, parameter: ModelParameter):
        self.model = parameter.model
        self.tokenizer = parameter.tokenizer
        self.config = parameter.config
    
    def encode(self, sentence: Union[str, List[str]]) -> Union[Tensor, List[Tensor]]:
        with_batch = isinstance(sentence, list)
        ids = self.tokenizer.encode(sentence)
        batch_ids = ids if with_batch else [ids]
        max_model_len = self.config.m_len
        
        all_fragments = []
        fragment_origin_idx = []
        
        for i, seq in enumerate(batch_ids):
            if len(seq) > max_model_len:
                fragments = [seq[j:j+max_model_len] for j in range(0, len(seq), max_model_len)]
                all_fragments.extend(fragments)
                fragment_origin_idx.extend([i] * len(fragments))
            else:
                all_fragments.append(seq)
                fragment_origin_idx.append(i)
        
        #if empty fragments
        if not all_fragments or not ids:
            return [] if with_batch else torch.tensor([])
        
        device = next(self.model.parameters()).device
        max_len = min(max(len(seq) for seq in all_fragments), max_model_len)
        
        padded_ids = []
        masks = []
        for seq in all_fragments:
            pad_len = max_len - len(seq)
            padded_seq = seq + [self.tokenizer.pad_id] * pad_len
            mask = [token_id != self.tokenizer.pad_id for token_id in padded_seq]
            padded_ids.append(padded_seq)
            masks.append(mask)
        
        input_tensor = torch.tensor(padded_ids, device=device, dtype=torch.long)
        seq_mask = torch.tensor(masks, device=device, dtype=torch.bool)
        
        with torch.inference_mode():
            outputs = self.model(input_tensor, seq_mask)["hidden_states"]
            # [num_fragments, seq_len, hidden_size]
            fragment_embs = torch.mul(outputs, seq_mask.unsqueeze(-1))  
        
        sentence_embs: List[Tensor] = []
        for i in range(len(batch_ids)):
            indices = [idx for idx, orig_idx in enumerate(fragment_origin_idx) if orig_idx == i]
            if indices is not None:
                sum_frags = torch.sum(fragment_embs[indices, :, :], dim=1)      # [frags, hidden_size]
                length = torch.sum(seq_mask[indices, :], dim=1).unsqueeze(1)    # [frags, 1]
                emb = torch.sum(sum_frags / length, dim=0)                      # [frags, hidden_size]
                sentence_embs.append(emb.flatten())
        
        if with_batch:
            return [emb.flatten() for emb in sentence_embs]
        else:
            return sentence_embs[0].flatten()

    def to(self, *args, **kargs) -> Self:
        self.model.to(*args, **kargs)
        return self


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
        cache_manager = KVCacheManager(
            num_layers=self.config.n_layer, 
            batch_size=1, 
            max_len=self.config.m_len, 
            num_heads=self.config.n_kvhead,
            head_dim=self.config.n_dim // self.config.n_head,
            device=device,
        )
        
        ids = self.tokenizer.encode(query)
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        
        start_cache_pos = len(ids)
        cur_cache_pos = 0
        self.model.eval()
        
        while len(ids) < self.config.m_len:
            kv_caches = cache_manager.get_kvcache()
            logits, cache_increase = self.compute_logits(
                input_ids,
                kv_caches=kv_caches, 
                start_pos=cur_cache_pos
            )
            logits = apply_sampling_strategies(logits, temperature, top_k, top_p)
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            input_ids = next_token_id
            ids.append(next_token_id.item())
            cur_cache_pos += cache_increase
            
            if next_token_id.item() in self.tokenizer.stop_ids:
                break
        
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
        cache_manager = KVCacheManager(
            num_layers=self.config.n_layer, 
            batch_size=1, 
            max_len=self.config.m_len, 
            num_heads=self.config.n_kvhead,
            head_dim=self.config.n_dim // self.config.n_head,
            device=device,
        )
        ids = self.tokenizer.encode(build_prompt(query, history))
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        cpy_history = history.copy()
        
        start_cache_pos = len(ids)
        cur_cache_pos = 0
        self.model.eval()
        
        
        while len(ids) < self.config.m_len:
            kv_caches = cache_manager.get_kvcache()
            logits, cache_increase = self.compute_logits(
                input_ids,
                kv_caches=kv_caches, 
                start_pos=cur_cache_pos
            )
            logits = apply_sampling_strategies(logits, temperature, top_k, top_p)
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            input_ids = next_token_id
            ids.append(next_token_id.item())
            cur_cache_pos += cache_increase
            
            if next_token_id.item() in self.tokenizer.stop_ids:
                break
        
        response = self.tokenizer.decode(ids[start_cache_pos:])
        cpy_history.append((query, response))
        
        return response, cpy_history
    

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
        cache_manager = KVCacheManager(
            num_layers=self.config.n_layer, 
            batch_size=1, 
            max_len=self.config.m_len, 
            num_heads=self.config.n_kvhead,
            head_dim=self.config.n_dim // self.config.n_head,
            device=device,
        )
        ids = self.tokenizer.encode(build_prompt(query, history))
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        cpy_history = history.copy()
        
        start_cache_pos = len(ids)
        cur_cache_pos = 0
        self.model.eval()
        
        
        while len(ids) < self.config.m_len:
            kv_caches = cache_manager.get_kvcache()
            logits, cache_increase = self.compute_logits(
                input_ids,
                kv_caches=kv_caches, 
                start_pos=cur_cache_pos
            )
            logits = apply_sampling_strategies(logits, temperature, top_k, top_p)
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
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
        cache_manager = KVCacheManager(
            num_layers=self.config.n_layer, 
            batch_size=batch_size, 
            max_len=self.config.m_len, 
            num_heads=self.config.n_kvhead,
            head_dim=self.config.n_dim // self.config.n_head,
            device=device,
        )
        
        input_tensor = torch.tensor(ids_list, device=device, dtype=torch.long)
        cache_manager.set_seq_mask(input_tensor, self.tokenizer.pad_id)
        activate_task_mask = [True] * batch_size
        
        start_cache_pos = max_ids_len
        cur_cache_pos = 0
        
        while max_ids_len < self.config.m_len and sum(activate_task_mask) != 0:
            kv_caches = cache_manager.get_kvcache()
            attn_mask =cache_manager.get_seq_mask()
            
            logits, cache_increase = self.compute_logits(
                input_tensor,
                attn_mask=attn_mask,
                kv_caches=kv_caches, 
                start_pos=cur_cache_pos
            )
            
            cur_cache_pos += cache_increase
            logits = apply_sampling_strategies(logits, temperature, top_k, top_p)
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
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
        retrieved_query = f"{retrieved}<eos>\n\n根据以上内容回答: {query}" if retrieved else query
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
        