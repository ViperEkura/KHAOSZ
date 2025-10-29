import torch
from torch import Tensor 
from typing import List, Tuple, Union, Optional, Self
from khaosz.config.param_config import ModelParameter


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


class GeneratorCore:
    def __init__(self, parameter: ModelParameter):
        self.model = parameter.model
        self.tokenizer = parameter.tokenizer
        self.config = parameter.config
        
    def generate_iterator(
        self,
        input_ids: Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        attn_mask: Optional[Tensor] = None,
        kv_caches: Optional[List[Tuple[Tensor, Tensor]]] = None,
        start_pos: int = 0
    )-> Tuple[Tensor, int]:
        
        with torch.inference_mode():
            outputs = self.model(input_ids, attn_mask, kv_caches, start_pos)
            logits = outputs["logits"][:, -1, :]
            cache_increase = input_ids.size(-1)   
        
        logits = apply_sampling_strategies(logits, temperature, top_k, top_p)
        probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
    
        return next_token_id, cache_increase

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
        
        self._kv_cache: Tuple[Tensor, Tensor] = None
        self._seq_mask: Tensor = None
        self._initialize()

    def _initialize(self):
        k_cache = torch.zeros(
            (self.batch_size, self.num_layers, self.max_len, self.num_heads, self.head_dim),
            device=self.device, dtype=self.dtype
        )
        v_cache = torch.zeros(
            (self.batch_size, self.num_layers, self.max_len, self.num_heads, self.head_dim),
            device=self.device, dtype=self.dtype
        )
        self._kv_cache = (k_cache, v_cache)
        self._seq_mask = torch.ones((self.batch_size, self.max_len), device=self.device, dtype=torch.bool)

    def update(self, active_mask: Tensor):        
        k_cache, v_cache = self._kv_cache
        self._kv_cache = (k_cache[active_mask], v_cache[active_mask])
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

    def get_kvcache(self) -> Tuple[Tensor, Tensor]:
        return self._kv_cache
    
    def get_seq_mask(self) -> Tensor:        
        return self._seq_mask