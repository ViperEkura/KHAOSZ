import json

from dataclasses import asdict, dataclass
from typing import  Optional, Self

@dataclass
class TransformerConfig:
    # basic config
    vocab_size: Optional[int] = None
    n_dim: Optional[int] = None
    n_head: Optional[int] = None
    n_layer: Optional[int] = None
    m_len: Optional[int] = None
    norm_eps: Optional[float] = None
    d_ffn: Optional[int] = None
    tie_weight: Optional[bool] = None
    
    # GQA
    n_kvhead: Optional[int] = None
    
    
    def load(self, config_path: str) -> Self:
        with open(config_path, 'r') as f:
            config: dict = json.load(f)
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
        return self
                    
    def save(self, config_path: str) -> None:
        config_dict = asdict(self)
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)


