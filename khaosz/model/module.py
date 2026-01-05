import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, Tuple


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
    ) -> Tuple[Tensor, Tensor]:
    """ 
    Get the rotary embedding for the given dimension and maximum length.
    Args:
        dim (int): The dimension of the input.
        max_len (int): The maximum length of the input.
        base (float, optional): The base for the frequency. Defaults to 10000.
    Returns:
        Tensor: The rotary embedding tensor.
    """

    theta = base ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    t = torch.arange(0, max_len, dtype=torch.float64)
    freqs = torch.outer(t, theta)

    return torch.cos(freqs).float(), torch.sin(freqs).float()

def apply_rotary_emb(x: torch.Tensor, rotary_emb: Tuple[Tensor, Tensor]) -> Tensor:
    """
    Apply rotary embedding to the input tensor using cos/sin form.
    Args:
        x (Tensor): The input tensor (shape [..., seq_len, dim]).
        rotary_emb (Tuple[Tensor, Tensor]): The rotary embedding (shape [seq_len, dim//2]).
    Returns:
        Tensor: The output tensor (rotated, same shape as input).
    """
    
    dtype = x.dtype
    cos, sin = rotary_emb
    
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim//2]
    sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim//2]
    
    x_real = x[..., 0::2]  # [batch, seq_len, dim//2]
    x_imag = x[..., 1::2]  # [batch, seq_len, dim//2]
    
    x_real_rot = x_real * cos - x_imag * sin
    x_imag_rot = x_real * sin + x_imag * cos
    
    x_out = torch.stack([x_real_rot, x_imag_rot], dim=-1)  # [batch, seq_len, dim//2, 2]
    x_out = x_out.view(*x_out.shape[:-2], -1)              # [batch, seq_len, dim]
    
    return x_out.to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int, base: int=10000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        self.max_len_cached = None
        self._set_rotary_buffer(self.max_len)
    
    def _set_rotary_buffer(self, max_len: int):
        cos_cached, sin_cached = get_rotary_emb(self.dim, max_len, self.base)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        self.max_len_cached = max_len
    
    def forward(self, x: Tensor, start_pos: int=0) -> Tuple[Tensor, Tensor]:
        seq_len = x.size(1)
        
        if self.max_len_cached < seq_len + start_pos:
            self._set_rotary_buffer(seq_len)
        
        cos = self.cos_cached[start_pos : start_pos + seq_len]
        sin = self.sin_cached[start_pos : start_pos + seq_len]
        
        return (cos, sin)


class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, dim, norm_eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.normalized_shape = (dim, )
        self.norm_eps = norm_eps

    def forward(self, x: Tensor) -> Tensor:
        rms =  F.rms_norm(x.float(), self.normalized_shape, self.weight, self.norm_eps)
        return rms.to(x.dtype)
    
    
class MLP(nn.Module):
    def __init__(self, dim: int, dim_feed_forward: int):
        super().__init__()
        self.up = Linear(dim, dim_feed_forward)
        self.gate = Linear(dim, dim_feed_forward)
        self.down = Linear(dim_feed_forward, dim)
    
    def forward(self, x: Tensor) -> Tensor:
        gated = self.up(x) * F.silu(self.gate(x))
        out = self.down(gated)
        return out

class Attention(nn.Module):
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, is_causal: bool= False):
        # (bsz, seq_len, n_heads, head_dim) -> (bsz, n_heads, seq_len, head_dim)
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        # (bsz, n_heads, seq_len, head_dim) - > (bsz, seq_len, n_heads*head_dim)
        sdqa_out = F.scaled_dot_product_attention(q, k, v, mask, is_causal=is_causal).permute(0, 2, 1, 3).contiguous().flatten(2)
        
        return sdqa_out


class GQA(nn.Module):
    def __init__(
        self, 
        dim: int, 
        n_heads: int, 
        n_kv_heads: int, 
        use_qk_norm: bool,
        norm_eps: float,
        use_gated_attention: bool,
        layer_id: int
    ):
        super().__init__()
        assert dim % n_heads == 0
        assert n_heads % n_kv_heads == 0
        
        self.head_dim = dim // n_heads
        self.layer_id = layer_id
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.use_qk_norm = use_qk_norm
        self.use_gated_attention = use_gated_attention
        
        self.attention = Attention()

        self.q_proj = Linear(dim, n_heads * self.head_dim)
        self.k_proj = Linear(dim, n_kv_heads * self.head_dim)
        self.v_proj = Linear(dim, n_kv_heads * self.head_dim)
        self.o_proj = Linear(dim, dim)
        
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, norm_eps)
            self.k_norm = RMSNorm(self.head_dim, norm_eps)
        
        if self.use_gated_attention:
            self.gate = Linear(dim, dim)

    def _split_heads(self, x: Tensor, n_heads) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, n_heads, self.head_dim)
        return x
    
    def forward(
        self,
        x: Tensor, 
        rotary_emb: Tuple[Tensor, Tensor], 
        mask: Tensor = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        start_pos: int = 0
    ) -> Tensor:
        bsz, seq_len, _ = x.size()
        # x(bsz, seq_len, n_heads * head_dim) -> (bsz, seq_len, n_heads, head_dim)
        q = self._split_heads(self.q_proj(x), self.n_heads)
        k = self._split_heads(self.k_proj(x), self.n_kv_heads)
        v = self._split_heads(self.v_proj(x), self.n_kv_heads)
        q, k = apply_rotary_emb(q, rotary_emb), apply_rotary_emb(k, rotary_emb)
        
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            
            # copy to cache
            k_cache[:bsz, start_pos:start_pos + seq_len, self.layer_id] = k
            v_cache[:bsz, start_pos:start_pos + seq_len, self.layer_id] = v
            
            # get cache
            k = k_cache[:bsz, :start_pos + seq_len, self.layer_id]
            v = v_cache[:bsz, :start_pos + seq_len, self.layer_id]
        
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        sdqa_out = self.attention(q, k, v, mask, is_causal=(mask == None))
        
        if self.use_gated_attention:
            sdqa_out = sdqa_out * F.sigmoid(self.gate(x))
        
        out = self.o_proj(sdqa_out)

        return out


class DecoderBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        n_heads: int, 
        dim_ffn: int, 
        n_kv_heads: int, 
        norm_eps: int, 
        use_qk_norm: bool, 
        use_gated_attention: bool, 
        layer_id: int
    ):
        super().__init__()
        self.attention = GQA(dim, n_heads, n_kv_heads, 
                             use_qk_norm, norm_eps, use_gated_attention, layer_id)
        self.input_norm = RMSNorm(dim, norm_eps)
        self.mlp = MLP(dim, dim_ffn)
        self.post_attention_norm = RMSNorm(dim, norm_eps)

    def forward(
        self,
        x: Tensor,
        rotary_emb: Tuple[Tensor, Tensor], 
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        start_pos: int = 0
    ) -> Tensor:
        # attention
        attn_output = self.attention(
            self.input_norm(x), 
            rotary_emb, 
            attention_mask, 
            kv_cache, 
            start_pos
        )
        x = attn_output + x
        
        # feed forward
        x = self.mlp(self.post_attention_norm(x)) + x
        
        return x


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, embedding_dim)))
    
    def forward(self, x: Tensor) -> Tensor:
        return F.embedding(x, self.weight)