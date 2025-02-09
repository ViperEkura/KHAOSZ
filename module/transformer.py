import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init
from typing import Tuple

def create_mask(L: int, device) -> Tensor:
    return torch.ones(
        L, L, dtype=torch.bool, device=device
    ).triu(diagonal=1)

def get_rotary_emb(
        dim: int, 
        max_len: int, 
        base: float = 10000, 
        device: torch.device = torch.device('cuda'),
    ) -> torch.Tensor:
    
    theta = base ** (-torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
    t = torch.arange(0, max_len, device=device).float()
    freqs = torch.outer(t, theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis

def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
    dtype = xq.dtype
    ndim = xq.ndim

    xq = torch.view_as_complex(xq.view(*xq.shape[:-1], -1, 2).float())
    xk = torch.view_as_complex(xk.view(*xk.shape[:-1], -1, 2).float())
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq.shape)]
    freqs_cis =  freqs_cis.view(*shape)
    
    xq_out = torch.view_as_real(xq * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk * freqs_cis).flatten(3)
    
    return xq_out.to(dtype), xk_out.to(dtype)

    
def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Config:
    def __init__(self, cfg_path=None):
            self.vocab_size = None
            self.n_dim = None
            self.n_head = None
            self.n_kvhead = None
            self.d_ffn = None
            self.m_len = None
            self.n_layer = None
            self.norm_eps = None
            self.flash_attn = None

            if cfg_path is not None:
                self.load(cfg_path)
    
    def load(self, config_path):
        with open(config_path, 'r') as f:
            config: dict = json.load(f)
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
    def save(self, config_path):
        config_dict ={
            "vocab_size": self.vocab_size,
            "n_dim": self.n_dim,
            "n_head": self.n_head,
            "n_kvhead": self.n_kvhead,
            "d_ffn": self.d_ffn,
            "m_len": self.m_len,
            "n_layer": self.n_layer,
            "norm_eps": self.norm_eps,
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
               

class RMSNorm(nn.Module):
    def __init__(self, n_dim, norm_eps, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
    
    
class FeedForward(nn.Module):
    def __init__(self, n_dim, d_ffn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.up = nn.Linear(n_dim, d_ffn, bias=False)
        self.gate = nn.Linear(n_dim, d_ffn, bias=False)
        self.down = nn.Linear(d_ffn, n_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gated = self.up(x) * F.silu(self.gate(x))
        out = self.down(gated)
        return out


class Attention(nn.Module):
    def __init__(self, n_dim, n_head, n_kvhead, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert n_dim % n_head == 0
        assert n_head % n_kvhead == 0
        
        self.head_dim = n_dim // n_head
        self.n_dim = n_dim
        self.n_heads = n_head
        self.n_kvheads = n_kvhead
        self.n_rep = n_head // n_kvhead
    
        self.q_proj = nn.Linear(n_dim, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_dim, n_kvhead * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_dim, n_kvhead * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_dim, n_dim, bias=False)
    def forward(self, x: Tensor, freqs_cis, mask=None) -> Tensor:
        B, L, _ = x.size()
        # x(B, L, D)
        q: Tensor = self.q_proj(x)
        k: Tensor = self.k_proj(x)
        v: Tensor = self.v_proj(x)
        
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_kvheads, self.head_dim)
        v = v.view(B, L, self.n_kvheads, self.head_dim)
        
        q, k = apply_rotary_emb(q,k, freqs_cis)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        is_causal =  (mask == None)
        attn_out = F.scaled_dot_product_attention(q, k, v, mask, is_causal=is_causal)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        out = self.o_proj(attn_out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, n_dim, n_head, d_ffn, n_kvhead, norm_eps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = Attention(n_dim, n_head, n_kvhead)
        self.norm_attn = RMSNorm(n_dim, norm_eps)
        self.ffn = FeedForward(n_dim, d_ffn)
        self.norm_ffn = RMSNorm(n_dim, norm_eps)

    def forward(self, x, freqs_cis, mask=None) -> torch.Tensor:
        x = self.attention(self.norm_attn(x), freqs_cis, mask) + x
        x = self.ffn(self.norm_ffn(x)) + x
        return x
    
  
class Transformer(nn.Module):
    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_dim = config.n_dim // config.n_head
        self.embedding = nn.Parameter(torch.empty(config.vocab_size, config.n_dim))
        self.layers = nn.ModuleList([
            DecoderBlock(
                config.n_dim, config.n_head, config.d_ffn, config.n_kvhead, config.norm_eps
            )for _ in range(config.n_layer)
        ])
        self.norm = RMSNorm(config.n_dim, config.norm_eps)
        self.freq_cis = get_rotary_emb(self.head_dim, config.m_len)
        init.normal_(self.embedding, mean=0, std=0.02)
    
    def parameter_size(self):
        parameter_size = 0
        for p in self.parameters():
            parameter_size += p.numel()
        return parameter_size
    
    def forward(self, x: Tensor):
        assert x.ndim == 2
        x = F.embedding(x, self.embedding)
        self.freq_cis = self.freq_cis.to(x.device)
        
        for layer in self.layers:
            x = layer(x, self.freq_cis)
            
        x = self.norm(x)
        x = F.linear(x, self.embedding)
        
        return x