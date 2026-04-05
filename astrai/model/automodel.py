"""
AutoModel base class for model loading and saving.
"""

import torch.nn as nn
import safetensors.torch as st

from pathlib import Path
from contextlib import contextmanager
from typing import Self, Union, Dict, Type

from astrai.config import ModelConfig


@contextmanager
def _disable_random_init(enable: bool = True):
    init_functions = [
        "xavier_normal_",
        "xavier_uniform_",
        "kaiming_normal_",
        "kaiming_uniform_",
        "zeros_",
        "ones_",
        "constant_",
        "normal_",
        "uniform_",
    ]
    original_funcs = {}
    for name in init_functions:
        if enable and hasattr(nn.init, name):
            original_funcs[name] = getattr(nn.init, name)
            setattr(nn.init, name, lambda *args, **kwargs: None)
    try:
        yield
    finally:
        if enable:
            for name, orig_func in original_funcs.items():
                setattr(nn.init, name, orig_func)


class AutoModel(nn.Module):
    """
    Autoregressive language model base class.
    Provides model loading/saving and generation capabilities.
    """

    # Model registry - stored as class attribute
    _registry: Dict[str, Type["AutoModel"]] = {}

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @classmethod
    def register(cls, model_type: str):
        """
        Class method decorator to register model type.

        Usage:
            @AutoModel.register('transformer')
            class Transformer(AutoModel):
                ...
        """

        def decorator(sub_cls: Type["AutoModel"]) -> Type["AutoModel"]:
            cls._registry[model_type.lower()] = sub_cls
            return sub_cls

        return decorator

    @classmethod
    def get_model_class(cls, model_type: str) -> Type["AutoModel"]:
        """Get model class by model_type string."""
        model_type = model_type.lower()
        if model_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown model_type: {model_type}. Available: {available}"
            )
        return cls._registry[model_type]

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        disable_random_init: bool = True,
    ) -> nn.Module:

        model_path = Path(path)

        # Load config
        config = ModelConfig()
        config_path = model_path / "config.json"
        if config_path.exists():
            config.load(str(config_path))
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # If called from base class, use model_type to determine actual model class
        if cls is AutoModel:
            model_type = config.model_type or "transformer"
            actual_cls = cls.get_model_class(model_type)
        else:
            raise ValueError(
                f"Cannot call from_pretrained() on subclass {cls.__name__}"
            )

        with _disable_random_init(enable=disable_random_init):
            model = actual_cls(config)

        # Load weights
        weights_path = model_path / "model.safetensors"
        if weights_path.exists():
            state_dict = st.load_file(str(weights_path))
            model.load_state_dict(state_dict, strict=False)

        return model

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
    ) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(str(save_path / "config.json"))

        # Save weights
        st.save_file(self.state_dict(), str(save_path / "model.safetensors"))

    def to(self, *args, **kwargs) -> Self:
        """Move model to device/dtype."""
        return super().to(*args, **kwargs)
