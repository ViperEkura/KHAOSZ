import json

from dataclasses import asdict, dataclass
from typing import Optional, Self


@dataclass
class ModelConfig:
    # basic config
    vocab_size: Optional[int] = None
    dim: Optional[int] = None

    n_layers: Optional[int] = None
    norm_eps: Optional[float] = None
    dim_ffn: Optional[int] = None
    tie_weight: Optional[bool] = None
    
    # RoPE
    max_len: Optional[int] = None
    rope_theta: Optional[float] = None
    
    # GQA
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None
    use_qk_norm: Optional[bool] = None
    use_gated_attention: Optional[bool] = None
    
    
    def load(self, config_path: str) -> Self:
        config = {}
        with open(config_path, 'r') as f:
            config.update(json.load(f)) 
            
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
        return self
                    
    def save(self, config_path: str):
        config_dict = {k: v for k, v in asdict(self).items() if v is not None}
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
