import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init


def create_mask(L: int, device) -> Tensor:
    mask = torch.ones(L, L, dtype=torch.bool).triu(diagonal=1)
    mask = mask.to(device)
    return mask

def get_rotary_emb(dim, max_len, base=10000, device='cuda', dtype=torch.bfloat16) -> tuple[Tensor, Tensor]:
    freqs = torch.pow(torch.tensor(base), -(torch.arange(0, dim, 2) / dim))
    freqs = freqs.repeat_interleave(2)

    t = torch.arange(0, max_len).float()
    rads = torch.outer(t, freqs)
    cos_emb = torch.cos(rads).to(device=device, dtype=dtype)
    sin_emb = torch.sin(rads).to(device=device, dtype=dtype)

    return (cos_emb, sin_emb)


def rotary_emb(x: Tensor, freqs_cis: tuple[Tensor, Tensor]) -> Tensor:
    _, L, _ = x.size()
    cos, sin = freqs_cis
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_half_rotated = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    rot = x * cos[:L] + x_half_rotated * sin[:L]
    return rot

def self_attention(
    q: Tensor, k: Tensor, v: Tensor, 
    n_heads: int,
    n_dim: int,
    scale: float,
    mask=None
) -> Tensor:
    head_dim = n_dim // n_heads
    B, L, _ = q.size()
    q = q.view(B, L, n_heads, head_dim).permute(0, 2, 1, 3).reshape(-1, L, head_dim)
    kT = k.view(B, L, n_heads, head_dim).permute(0, 2, 3, 1).reshape(-1, head_dim, L)
    v = v.view(B, L, n_heads, head_dim).permute(0, 2, 1, 3).reshape(-1, L, head_dim)
    # q, v : (B, L, D) -> (B, L, H, D/H) -> (B, H, L, D/H) -> (B*H, L, D/H)
    # kT   : (B, L, D) -> (B, L, H, D/H) -> (B, H, D/H, L) -> (B*H, D/H, L)
    attn_weight = torch.bmm(q, kT) * scale
    attn_weight = attn_weight.masked_fill(mask, -float('inf'))
    attn_weight = F.softmax(attn_weight, dim=-1)
    # attn_weight : (B*H, L, L)
    attn_out = torch.bmm(attn_weight, v)
    attn_out = attn_out.view(B, n_heads, L, head_dim)
    attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
    # attn_out : (B*H, L, D/H) -> (B, L, H, D/H) -> (B, L, D)
    return attn_out



class Config:
    def __init__(self, cfg_path=None):
            self.vocab_size = None
            self.n_dim = None
            self.n_head = None
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
            "d_ffn": self.d_ffn,
            "m_len": self.m_len,
            "n_layer": self.n_layer,
            "norm_eps": self.norm_eps,
            "flash_attn": self.flash_attn
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
        self.up = nn.Parameter(torch.empty(2 * d_ffn, n_dim))
        self.down = nn.Parameter(torch.empty(n_dim, d_ffn))

        init.kaiming_uniform_(self.up, a=math.sqrt(5))
        init.kaiming_uniform_(self.down, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        x = F.linear(x, self.up)
        x, gate = x.chunk(2, dim=-1)
        gate = F.silu(gate)
        gated = x * gate
        out = F.linear(gated, self.down)
        
        return out
    

class Attention(nn.Module):
    def __init__(self, n_dim, n_heads, flash_attn=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert n_dim % n_heads == 0, "n_dim must be divisible by n_heads"
        self.Wqkv = nn.Parameter(torch.empty(3 * n_dim, n_dim))
        self.Wo = nn.Parameter(torch.empty(n_dim, n_dim))
        self.mask = None
        
        self.scale = 1 / n_dim ** 0.5
        self.head_dim = n_dim // n_heads
        self.flash_attn = flash_attn
        self.n_dim = n_dim
        self.n_heads = n_heads
        
        init.kaiming_uniform_(self.Wqkv, a=math.sqrt(5))
        init.kaiming_uniform_(self.Wo, a=math.sqrt(5))

    def forward(self, x: Tensor, freqs_cis, mask=None) -> Tensor:
        B, L, _ = x.size()
        # x(B, L, D)
        qkv = F.linear(x, self.Wqkv) 
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = rotary_emb(q, freqs_cis)
        k = rotary_emb(k, freqs_cis)
        
        if not self.flash_attn:
            if self.mask is None:
                self.mask = create_mask(L, x.device)
            if mask is None:
                mask = self.mask
            attn_out = self_attention(q, k, v, self.n_heads, self.n_dim, self.scale, mask)
        else:
            q = q.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            # use torch.scaled_dot_product_attention impl Flash Attention
            is_causal =  (mask == None)
            attn_out = F.scaled_dot_product_attention(q, k, v, mask, is_causal=is_causal)
            attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
            
        out = F.linear(attn_out, self.Wo)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, n_dim, n_head, d_ffn, flash_attn, norm_eps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = Attention(n_dim, n_head, flash_attn)
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
        self.n_dim = config.n_dim
        self.embedding = nn.Embedding(config.vocab_size, config.n_dim)        
        self.layers = nn.ModuleList([
            DecoderBlock(config.n_dim, config.n_head, config.d_ffn, config.flash_attn, config.norm_eps)
            for _ in range(config.n_layer)
        ])
        
        self.norm = RMSNorm(config.n_dim, config.norm_eps)
        self.out = nn.Linear(config.n_dim, config.vocab_size, bias=False)
        
    
    def parameter_size(self):
        parameter_size = 0
        for p in self.parameters():
            parameter_size += p.numel()
        return parameter_size
    
    def forward(self, x: Tensor):
        L = x.size(-1)
        assert x.ndim == 2, "Input must be a 2D tensor"
        
        freqs_cis = get_rotary_emb(self.n_dim, L, device=x.device, dtype=x.dtype)
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x, freqs_cis)
            
        x = self.norm(x)
        x = self.out(x)
        
        return x