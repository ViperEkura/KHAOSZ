import torch
import torch.nn as nn

from torch import Tensor
from typing import Any, Mapping, Optional, Tuple
from khaosz.config.model_config import ModelConfig
from khaosz.model.module import Embedding, DecoderBlock, Linear, RMSNorm, RotaryEmbedding


def process_attention_mask(
        seq_mask: Tensor, 
        input_tensor: Tensor,
        start_pos: int = 0,
        is_causal: bool = False,
    ) -> Tensor:
    """
    Create attention mask for GQA
    Args:
        seq_mask (Tensor): A tensor indicating whether each position is valid or not.
        input_tensor (Tensor): The input tensor.
        start_pos (int): The starting position of the sequence.
        is_causal (bool): Whether the attention is causal or not.
    Returns:
        Tensor: The attention mask tensor.
    """
    device = input_tensor.device
    dtype = input_tensor.dtype
    seq_len = input_tensor.size(1)
    
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
        expanded_mask = torch.tril(expanded_mask, diagonal=start_pos)
    
    attention_mask = torch.zeros_like(expanded_mask, dtype=dtype, device=device)
    attention_mask = attention_mask.masked_fill_(~expanded_mask, -torch.finfo(dtype).max / 2).unsqueeze(1)
    # (bsz, 1, seq_len, seq_len + start_pos)
    
    return attention_mask


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.rotary_embeding = RotaryEmbedding(config.dim // config.n_heads, config.max_len)
        self.embed_tokens = Embedding(config.vocab_size, config.dim)
        
        self.layers = nn.ModuleList([
            DecoderBlock(config.dim, config.n_heads, config.dim_ffn, config.n_kv_heads, 
                         config.norm_eps, config.use_qk_norm, config.use_gated_attention, layer_id)
            for layer_id in range(config.n_layers)
        ])
        
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = Linear(config.dim, config.vocab_size)
        
        if self.config.tie_weight == True:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_parameters()
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict=True, assign=False):
        lm_head_key = 'lm_head.weight'
        embed_key = 'embed_tokens.weight'

        if self.config.tie_weight == True:
            # same tensor
            state_dict[lm_head_key] = state_dict[embed_key]
        else:
            if lm_head_key not in state_dict and embed_key in state_dict:
                # use clone to avoid sharing the same tensor
                state_dict[lm_head_key] = torch.clone(state_dict[embed_key])
        
        return super().load_state_dict(state_dict, strict, assign)
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        
        if self.config.tie_weight == True:
            lm_head_key = prefix + 'lm_head.weight'
            if lm_head_key in state_dict:
                del state_dict[lm_head_key]
        
        return state_dict
    
    def _init_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.normal_(param, mean=0.0, std=0.006)
    
    def forward(
        self, 
        input_ids: Tensor, 
        input_mask: Optional[Tensor]=None,
        persistent_key_values: Optional[Tuple[Tensor, Tensor]]=None,
        start_pos: int = 0
    ) -> Tensor:
        assert input_ids.ndim == 2
        
        x = self.embed_tokens(input_ids)
        rotary_emb = self.rotary_embeding(x, start_pos)
        
        attn_mask = process_attention_mask(
            input_mask, x, start_pos, is_causal=True
        )
        
        for layer in self.layers:
            x = layer(x, rotary_emb, attn_mask, persistent_key_values, start_pos)
        
        hidden_states = self.norm(x)        
        logits = self.lm_head(hidden_states)
        
        return {
            "logits": logits,
            "hidden_states": hidden_states
        }
        