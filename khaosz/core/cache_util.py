import torch
from torch import Tensor
from typing import List, Tuple, Optional


class KVCacher:
    def __init__(
        self, 
        max_len: int, 
        num_layers: int, 
        num_heads: int, 
        head_dim: int, 
        device: torch.device = "cuda", 
        dtype: torch.dtype = torch.bfloat16
    ):
        self.max_len = max_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        
        # cache
        self.kv_cache: List[Tuple[Tensor, Tensor]] = None
        self.cache_pos = 0
        
    
    def reset(self):
        """重置KV缓存"""
        self.kv_cache = None
        self.cache_pos = 0
    
    
    def _initialize_cache(self, batch_size: int):
        """初始化KV缓存"""
        self.kv_cache = []
        for _ in range(self.num_layers):
            # 初始化key和value缓存，形状为 [batch_size, num_heads, max_len, head_dim]
            k_cache = torch.zeros(
                (batch_size, self.num_heads, self.max_len, self.head_dim),
                device=self.device, dtype=self.dtype
            )
            v_cache = torch.zeros(
                (batch_size, self.num_heads, self.max_len, self.head_dim),
                device=self.device, dtype=self.dtype
            )
            self.kv_cache.append((k_cache, v_cache))
    
    
    def prefill(self, kv_pairs: List[Tuple[Tensor, Tensor]], batch_indices: Optional[Tensor] = None):
        """预填充KV缓存（处理初始的prompt/context）"""
        batch_size = kv_pairs[0][0].size(0)
        seq_len = kv_pairs[0][0].size(2)
        
        if self.kv_cache is None:
            self._initialize_cache(batch_size)
        
        assert self.cache_pos + seq_len <= self.max_len   
        
        for layer_idx, (k_new, v_new) in enumerate(kv_pairs):
            k_cache, v_cache = self.kv_cache[layer_idx]
            
            if batch_indices is not None:
                k_cache[batch_indices, :, self.cache_pos:self.cache_pos+seq_len, :] = k_new[batch_indices]
                v_cache[batch_indices, :, self.cache_pos:self.cache_pos+seq_len, :] = v_new[batch_indices]
            else:
                k_cache[:, :, self.cache_pos:self.cache_pos+seq_len, :] = k_new
                v_cache[:, :, self.cache_pos:self.cache_pos+seq_len, :] = v_new
        
        self.cache_pos += seq_len
    
    
    def decoding(self, kv_pairs: List[Tuple[Tensor, Tensor]], batch_indices: Optional[Tensor] = None):
        """解码阶段（自回归生成），每次处理一个token"""
        assert self.kv_cache is not None
        assert self.cache_pos < self.max_len
        
        for layer_idx, (k_new, v_new) in enumerate(kv_pairs):
            k_cache, v_cache = self.kv_cache[layer_idx]
            
            if batch_indices is not None:
                k_cache[batch_indices, :, self.cache_pos, :] = k_new[batch_indices].squeeze(2)
                v_cache[batch_indices, :, self.cache_pos, :] = v_new[batch_indices].squeeze(2)
            else:
                k_cache[:, :, self.cache_pos, :] = k_new.squeeze(2)
                v_cache[:, :, self.cache_pos, :] = v_new.squeeze(2)
        
        self.cache_pos += 1
    
    
    def get_cache(self, layer_idx: int, start_pos: int = 0, end_pos: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """获取指定层的缓存"""
        assert self.kv_cache is not None
        
        if end_pos is None:
            end_pos = self.cache_pos
        
        k_cache, v_cache = self.kv_cache[layer_idx]
        return k_cache[:, :, start_pos:end_pos, :], v_cache[:, :, start_pos:end_pos, :]
    
    
    def get_current_length(self) -> int:
        """获取当前缓存的有效长度"""
        return self.cache_pos
    
    
    def get_max_remaining_length(self) -> int:
        """获取缓存剩余的最大长度"""
        return self.max_len - self.cache_pos

