from typing import Optional

import torch.nn as nn
from torch import Tensor

from astrai.inference.core.cache import KvcacheView
from astrai.model.components.attention import GQA
from astrai.model.components.mlp import MLP
from astrai.model.components.norm import RMSNorm


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
        layer_id: int,
    ):
        super().__init__()
        self.attention = GQA(
            dim,
            n_heads,
            n_kv_heads,
            use_qk_norm,
            norm_eps,
            use_gated_attention,
            layer_id,
        )
        self.input_norm = RMSNorm(dim, norm_eps)
        self.mlp = MLP(dim, dim_ffn)
        self.post_attention_norm = RMSNorm(dim, norm_eps)

    def forward(
        self,
        x: Tensor,
        rotary_emb: Tensor,
        attention_mask: Optional[Tensor] = None,
        paged_cache: Optional[KvcacheView] = None,
    ) -> Tensor:
        attn_output = self.attention(
            self.input_norm(x),
            rotary_emb,
            attention_mask,
            paged_cache,
        )
        x = attn_output + x
        x = self.mlp(self.post_attention_norm(x)) + x

        return x
