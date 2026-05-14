"""
AutoModel base class for model loading and saving.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Self, Union

import safetensors.torch as st
import torch.nn as nn

from astrai.config import ModelConfig
from astrai.factory import BaseFactory


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


class AutoModel(BaseFactory["AutoModel"], nn.Module):
    """
    Autoregressive language model base class.
    Provides model loading/saving, registration, and generation.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        disable_random_init: bool = True,
        strict: bool = True,
    ) -> nn.Module:

        model_path = Path(path)

        # Load config
        config = ModelConfig()
        config_path = model_path / "config.json"
        if config_path.exists():
            config.load(str(config_path))
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

        model_type = config.model_type or "transformer"
        actual_cls = AutoModel.get_component_class(model_type)

        with _disable_random_init(enable=disable_random_init):
            model = actual_cls(config)

        # Load weights
        weights_path = model_path / "model.safetensors"
        if weights_path.exists():
            state_dict = st.load_file(str(weights_path))
            model.load_state_dict(state_dict, strict=strict)

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
