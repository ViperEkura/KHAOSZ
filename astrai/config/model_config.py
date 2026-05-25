import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Self

from astrai.config.base import BaseConfig
from astrai.factory import BaseFactory


class ConfigFactory(BaseFactory[BaseConfig]):
    """Factory that dispatches config classes by ``model_type``."""

    @classmethod
    def load(cls, raw: Dict[str, Any]) -> BaseConfig:
        model_type = raw.get("model_type") or "autoregressive_lm"
        config_cls = cls.get_component_class(model_type)
        return config_cls.from_dict(raw)


@dataclass
class BaseModelConfig(BaseConfig):
    """Base config with ``model_type`` dispatch and file I/O."""

    model_type: Optional[str] = None

    @classmethod
    def from_file(cls, config_path: str) -> Self:
        with open(config_path, "r") as f:
            raw: Dict[str, Any] = json.load(f)
        return cls.from_dict(raw)

    def to_file(self, config_path: str):
        d = self.to_dict()
        config_dict = {k: v for k, v in d.items() if v is not None}
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)


@dataclass
@ConfigFactory.register("autoregressive_lm")
class AutoRegressiveLMConfig(BaseModelConfig):
    """Configuration for autoregressive language model."""

    vocab_size: Optional[int] = None
    dim: Optional[int] = None
    n_layers: Optional[int] = None
    norm_eps: Optional[float] = None
    dim_ffn: Optional[int] = None
    tie_weight: Optional[bool] = None

    max_len: Optional[int] = None
    rope_theta: Optional[float] = None
    rope_scaling: Optional[dict] = None

    attn_type: str = "gqa"
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None
    use_qk_norm: Optional[bool] = None
    use_gated_attention: Optional[bool] = None

    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None

    ffn_type: str = "mlp"
    n_routed_experts: Optional[int] = None
    n_shared_experts: Optional[int] = None
    n_activated_experts: Optional[int] = None
    topk_method: Optional[str] = None


@dataclass
@ConfigFactory.register("embedding")
class EncoderConfig(BaseModelConfig):
    """Configuration for embedding encoder model."""

    vocab_size: Optional[int] = None
    dim: Optional[int] = None
    n_layers: Optional[int] = None
    norm_eps: Optional[float] = None
    dim_ffn: Optional[int] = None

    max_len: Optional[int] = None
    rope_theta: Optional[float] = None
    rope_scaling: Optional[dict] = None

    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None
    use_qk_norm: Optional[bool] = None
    use_gated_attention: Optional[bool] = None

    pooling_type: Optional[str] = None
    normalize_embeddings: Optional[bool] = None
