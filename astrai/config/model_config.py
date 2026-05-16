import json
import warnings
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Self

from astrai.config.base import BaseConfig


@dataclass
class BaseModelConfig(BaseConfig):
    """Field-aware JSON from/to file for dataclass configs.

    Subclass with additional fields. The base ``model_type`` field
    enables ``AutoModel`` to pick the correct subclass.
    """

    model_type: Optional[str] = None

    @classmethod
    def from_file(cls, config_path: str) -> Self:
        with open(config_path, "r") as f:
            raw: Dict[str, Any] = json.load(f)

        valid = {fld.name for fld in fields(cls)}
        for key in list(raw):
            if key not in valid:
                warnings.warn(f"Unknown config key '{key}'")
                del raw[key]

        return cls.from_dict(raw)

    def to_file(self, config_path: str):
        d = self.to_dict()
        config_dict = {k: v for k, v in d.items() if v is not None}
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)


@dataclass
class ModelConfig(BaseModelConfig):
    vocab_size: Optional[int] = None
    dim: Optional[int] = None

    n_layers: Optional[int] = None
    norm_eps: Optional[float] = None
    dim_ffn: Optional[int] = None
    tie_weight: Optional[bool] = None

    # RoPE
    max_len: Optional[int] = None
    rope_theta: Optional[float] = None

    # attention
    attn_type: str = "gqa"
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None
    use_qk_norm: Optional[bool] = None
    use_gated_attention: Optional[bool] = None

    # MLA
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None

    # MoE
    ffn_type: str = "mlp"
    n_routed_experts: Optional[int] = None
    n_shared_experts: Optional[int] = None
    n_activated_experts: Optional[int] = None
    moe_topk_method: Optional[str] = None
