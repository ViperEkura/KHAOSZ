import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init
from typing import Optional, Tuple

from khaosz.config.model_config import TransformerConfig
from khaosz.model.module import DecoderBlock, RMSNorm, get_rotary_emb


def process_attention_mask(
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
    
    if seq_mask is None:
        if start_pos != 0:
            # for single prompt chat
            seq_mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
        else:
            return None
    
    if seq_mask.dim() > 2:
        # shape (bsz, seq_len) or (bsz,n_heads, seq_len, seq_len + start_pos)
        # if ndim > 2, it's 4D tensor
        return seq_mask
    
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


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(config.vocab_size, config.n_dim))
        self.layers = nn.ModuleList([
            DecoderBlock(config.n_dim, config.n_head, config.d_ffn, config.n_kvhead, config.norm_eps, layer_id)
            for layer_id in range(config.n_layer)
        ])
        self.norm = RMSNorm(config.n_dim, config.norm_eps)
        self.freq_cis = get_rotary_emb(config.n_dim // config.n_head, config.m_len)
        init.normal_(self.embedding, mean=0, std=0.02)
    
    def forward(
        self, 
        input_ids: Tensor, 
        input_mask: Optional[Tensor]=None,
        persistent_key_values: Optional[Tuple[Tensor, Tensor]]=None,
        start_pos: int = 0
    ) -> Tensor:
        assert input_ids.ndim == 2
        seq_len = input_ids.size(-1)
        x = F.embedding(input_ids, self.embedding)
        
        self.freq_cis = self.freq_cis.to(x.device)
        freqs_cis = self.freq_cis[start_pos:start_pos+seq_len]
        has_kvcache = persistent_key_values is not None
        
        attn_mask = process_attention_mask(
            input_mask, 
            start_pos=start_pos,
            seq_len=seq_len,
            is_causal=has_kvcache,
            device=x.device,
            dtype=x.dtype
        )
        
        for layer in self.layers:
            x = layer(x, freqs_cis, attn_mask, persistent_key_values, start_pos)
        
        hidden_states = self.norm(x)        
        logits = F.linear(hidden_states,  self.embedding)
        
        return {
            "logits": logits,
            "hidden_states": hidden_states
        }
        