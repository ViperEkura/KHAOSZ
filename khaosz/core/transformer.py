import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init
from dataclasses import asdict, dataclass
from typing import List, Optional, Self, Tuple


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """ 
    Repeat k times along the dimension for attention heads.
    Args:
        x (Tensor): The input tensor.
        n_rep (int): The number of repetitions.
    Returns:
        Tensor: The repeated tensor.
    """
    
    bs, slen, n_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_heads, n_rep, head_dim)
        .reshape(bs, slen, n_heads * n_rep, head_dim)
    )

def get_rotary_emb(
        dim: int, 
        max_len: int, 
        base: float = 10000, 
        device: torch.device = "cuda",
    ) -> torch.Tensor:
    """ 
    Get the rotary embedding for the given dimension and maximum length.
    Args:
        dim (int): The dimension of the input.
        max_len (int): The maximum length of the input.
        base (float, optional): The base for the frequency. Defaults to 10000.
        device (torch.device, optional): The device to use. Defaults to "cuda".
    Returns:
        Tensor: The rotary embedding tensor.
    """

    theta = base ** (-torch.arange(0, dim, 2, device=device).float() / dim)
    t = torch.arange(0, max_len, device=device).float()
    freqs = torch.outer(t, theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis

def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """
    Apply rotary embedding to the input tensor.
    Args:
        x (Tensor): The input tensor.
        freqs_cis (Tensor): The rotary embedding tensor.
    Returns:
        Tensor: The output tensor.
    """
    
    dtype = x.dtype
    seq_len = x.size(1)

    x_complex = torch.view_as_complex(x.view(*x.shape[:-1], -1, 2).float())
    freqs_cis = freqs_cis.reshape(1, seq_len, 1, -1)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    
    return x_out.to(dtype)

def create_attention_mask(
        seq_mask: Tensor, 
        start_pos: int = 0,
        seq_len: int = 0,
        is_causal: bool = False,
        device: torch.device = "cuda", 
        dtype: torch.dtype = torch.float32
    ) -> Tensor:
    """
    Create attention mask for GQA
    Args:
        seq_mask (Tensor): A tensor indicating whether each position is valid or not.
        start_pos (int): The starting position of the sequence.
        seq_len (int): The length of the sequence.
        is_causal (bool): Whether the attention is causal or not.
        device (torch.device): The device to use.
    Returns:
        Tensor: The attention mask tensor.
    """
    
    if start_pos != 0 and seq_mask is None:
        # for single prompt chat
        seq_mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
    
    if seq_mask is None:
        return None
    
    batch_size = seq_mask.size(0)
    seq_mask = seq_mask[:, :start_pos + seq_len].to(device=device, dtype=torch.bool)
    # (bsz, start_pos + seq_len)
    expanded_mask = seq_mask.unsqueeze(1).expand(batch_size, seq_len, start_pos + seq_len)
    # (bsz, seq_len, start_pos + seq_len)
    
    if is_causal:
        causal_mask = torch.tril(
            torch.ones((seq_len, start_pos + seq_len), dtype=torch.bool, device=device), 
            diagonal=start_pos
        )
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, seq_len, start_pos + seq_len)
        expanded_mask = expanded_mask & causal_mask
    
    attention_mask = torch.zeros_like(expanded_mask, dtype=dtype, device=device)
    attention_mask = attention_mask.masked_fill_(~expanded_mask, -torch.finfo(dtype).max / 2).unsqueeze(1)
    # (bsz, 1, seq_len, seq_len + start_pos)
    
    return attention_mask


@dataclass
class TransformerConfig:
    # basic config
    vocab_size: Optional[int] = None
    n_dim: Optional[int] = None
    n_head: Optional[int] = None
    n_layer: Optional[int] = None
    m_len: Optional[int] = None
    norm_eps: Optional[float] = None
    d_ffn: Optional[int] = None
    
    # GQA
    n_kvhead: Optional[int] = None
    
    
    def load(self, config_path: str) -> Self:
        with open(config_path, 'r') as f:
            config: dict = json.load(f)
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
        return self
                    
    def save(self, config_path: str) -> None:
        config_dict = asdict(self)
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
            

class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        init.normal_(self.weight, mean=0, std=0.006)
        
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)
        

class RMSNorm(nn.Module):
    def __init__(self, n_dim, norm_eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))
        self.norm_eps = norm_eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        mean_square = torch.mean(torch.pow(x, 2), dim=-1, keepdim=True)
        norm = x * torch.rsqrt(mean_square + self.norm_eps)
        norm = norm.to(dtype)
        out = norm * self.weight
        return out
    
    
class MLP(nn.Module):
    def __init__(self, n_dim: int, d_ffn: int):
        super().__init__()
        self.up = Linear(n_dim, d_ffn)
        self.gate = Linear(n_dim, d_ffn)
        self.down = Linear(d_ffn, n_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        gated = self.up(x) * F.silu(self.gate(x))
        out = self.down(gated)
        return out


class GQA(nn.Module):
    def __init__(
        self, 
        n_dim: int, 
        n_head: int, 
        n_kvhead: int, 
    ):
        super().__init__()
        assert n_dim % n_head == 0
        assert n_head % n_kvhead == 0
        
        self.head_dim = n_dim // n_head
        self.n_dim = n_dim
        self.n_heads = n_head
        self.n_kvheads = n_kvhead
        self.n_rep = n_head // n_kvhead
        
        self.q_proj = Linear(n_dim, n_head * self.head_dim)
        self.k_proj = Linear(n_dim, n_kvhead * self.head_dim)
        self.v_proj = Linear(n_dim, n_kvhead * self.head_dim)
        self.o_proj = Linear(n_dim, n_dim)
    
    def forward(
        self,
        x: Tensor, 
        freqs_cis: Tensor, 
        mask: Tensor = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        start_pos: int = 0
    ) -> Tensor:
        bsz, seq_len, _ = x.size()
        # x(bsz, seq_len, n_heads * head_dim) -> (bsz, seq_len, n_heads, head_dim)
        q = self._split_heads(self.q_proj(x), self.n_heads)
        k = self._split_heads(self.k_proj(x), self.n_kvheads)
        v = self._split_heads(self.v_proj(x), self.n_kvheads)
        q, k = apply_rotary_emb(q, freqs_cis), apply_rotary_emb(k, freqs_cis)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache

            # copy to cache
            k_cache[:bsz, start_pos:start_pos + seq_len] = k
            v_cache[:bsz, start_pos:start_pos + seq_len] = v
            
            # get cache
            k = k_cache[:bsz, :start_pos + seq_len]
            v = v_cache[:bsz, :start_pos + seq_len]
        
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        
        # (bsz, seq_len, n_heads, head_dim) -> (bsz, n_heads, seq_len, head_dim)
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        sdqa_out = F.scaled_dot_product_attention(q, k, v, mask, is_causal=(mask == None)).permute(0, 2, 1, 3)
        out = self.o_proj(sdqa_out.contiguous().view(bsz, seq_len, -1))

        return out
    
    def _split_heads(self, x: Tensor, n_heads) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, n_heads, self.head_dim)
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, n_dim, n_head, d_ffn, n_kvhead, norm_eps):
        super().__init__()
        self.attention = GQA(n_dim, n_head, n_kvhead)
        self.norm_attn = RMSNorm(n_dim, norm_eps)
        self.ffn = MLP(n_dim, d_ffn)
        self.norm_ffn = RMSNorm(n_dim, norm_eps)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        start_pos: int = 0
    ) -> Tensor:
        # attention
        attn_output = self.attention(
            self.norm_attn(x), 
            freqs_cis, 
            attention_mask, 
            kv_cache, 
            start_pos
        )
        x = attn_output + x
        
        # feed forward
        x = self.ffn(self.norm_ffn(x)) + x
        
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(config.vocab_size, config.n_dim))
        self.layers = nn.ModuleList([
            DecoderBlock(
                config.n_dim, 
                config.n_head, 
                config.d_ffn, 
                config.n_kvhead, 
                config.norm_eps
            )
            for _ in range(config.n_layer)
        ])
        self.norm = RMSNorm(config.n_dim, config.norm_eps)
        self.freq_cis = get_rotary_emb(config.n_dim // config.n_head, config.m_len)
        init.normal_(self.embedding, mean=0, std=0.02)
    
    def forward(
        self, 
        input_ids: Tensor, 
        seq_mask: Optional[Tensor]=None,
        persistent_key_values: Optional[List[Tuple[Tensor, Tensor]]]=None,
        start_pos: int = 0
    ) -> Tensor:
        assert input_ids.ndim == 2
        seq_len = input_ids.size(-1)
        x = F.embedding(input_ids, self.embedding)
        
        self.freq_cis = self.freq_cis.to(x.device)
        freqs_cis = self.freq_cis[start_pos:start_pos+seq_len]
        has_kvcache = persistent_key_values is not None
        
        attn_mask = create_attention_mask(
            seq_mask, 
            start_pos=start_pos,
            seq_len=seq_len,
            is_causal=has_kvcache,
            device=x.device,
            dtype=x.dtype
        )
        
        for i, layer in enumerate(self.layers):
            kv_cache = persistent_key_values[i] if persistent_key_values else None
            x = layer(x, freqs_cis, attn_mask, kv_cache, start_pos)
        
        hidden_states = self.norm(x)        
        logits = F.linear(hidden_states,  self.embedding)
        
        return {
            "logits": logits,
            "hidden_states": hidden_states
        }
        