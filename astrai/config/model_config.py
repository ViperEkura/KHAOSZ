import json
import sys
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Self, get_type_hints


@dataclass
class BaseModelConfig:
    """Field-aware JSON load/save for dataclass configs.

    Subclass with additional fields. The base ``model_type`` field
    enables ``AutoModel`` to pick the correct subclass.
    """

    model_type: Optional[str] = None

    def load(self, config_path: str) -> Self:
        raw: Dict[str, Any] = {}
        with open(config_path, "r") as f:
            raw.update(json.load(f))

        hints = get_type_hints(type(self))
        valid = {fld.name for fld in fields(self)}
        for key, value in raw.items():
            if key not in valid:
                sys.stderr.write(f"WARNING: unknown config key '{key}'\n")
                continue

            target_type = self._unwrap_optional(hints.get(key))
            if target_type is None:
                continue

            try:
                value = self._coerce(value, target_type)
            except (TypeError, ValueError):
                sys.stderr.write(
                    f"WARNING: cannot coerce '{key}' = {value!r} to {target_type}\n"
                )
                continue

            setattr(self, key, value)

        return self

    def save(self, config_path: str):
        config_dict: Dict[str, Any] = {}
        for fld in fields(self):
            v = getattr(self, fld.name)
            if v is not None:
                config_dict[fld.name] = v
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    @staticmethod
    def _unwrap_optional(tp: type) -> Optional[type]:
        if tp is None:
            return None
        origin = getattr(tp, "__origin__", None)
        if origin is not None:
            args = getattr(tp, "__args__", ())
            non_none = [a for a in args if a is not type(None)]
            return non_none[0] if non_none else None
        return tp

    @staticmethod
    def _coerce(value: Any, target_type: type) -> Any:
        if target_type is bool and isinstance(value, bool):
            return value
        if (
            target_type is int
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            return int(value)
        if (
            target_type is float
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            return float(value)
        if target_type is str and isinstance(value, str):
            return value
        if isinstance(value, target_type):
            return value
        raise TypeError


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

    # MoE
    ffn_type: str = "mlp"
    n_routed_experts: Optional[int] = None
    n_shared_experts: Optional[int] = None
    n_activated_experts: Optional[int] = None
    moe_topk_method: Optional[str] = None
