import torch.nn as nn
import safetensors.torch as st

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Self, Union
from pathlib import Path

from astrai.data.tokenizer import BpeTokenizer
from astrai.config.model_config import ModelConfig
from astrai.model.transformer import Transformer


@contextmanager
def disable_random_init(enable: bool = True):
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


@dataclass
class BaseModelIO:
    """Base class for model I/O operations."""

    model: nn.Module = field(
        default_factory=nn.Identity, metadata={"help": "Transformer model."}
    )
    tokenizer: BpeTokenizer = field(
        default_factory=BpeTokenizer, metadata={"help": "Tokenizer for the model."}
    )
    config: ModelConfig = field(
        default_factory=ModelConfig,
        metadata={"help": "Transformer model configuration."},
    )

    def _get_file_paths(self, directory: Union[str, Path]) -> dict[str, Path]:
        """Get standardized file paths for model components."""
        dir_path = Path(directory)
        return {
            "model": dir_path / "model.safetensors",
            "config": dir_path / "config.json",
            "tokenizer": dir_path / "tokenizer.json",
        }

    def save_components(self, save_dir: Union[str, Path]):
        """Save core model components."""
        paths = self._get_file_paths(save_dir)
        paths["model"].parent.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            st.save_file(self.model.state_dict(), str(paths["model"]))

        self.config.save(str(paths["config"]))
        self.tokenizer.save(str(paths["tokenizer"]))

    def load_components(
        self, load_dir: Union[str, Path], disable_init: bool = False
    ) -> Self:
        """Load core model components."""
        paths = self._get_file_paths(load_dir)

        self.config.load(str(paths["config"]))
        self.tokenizer.load(str(paths["tokenizer"]))

        if self.model is None:
            with disable_random_init(enable=disable_init):
                self.model = Transformer(self.config)

        if paths["model"].exists():
            state_dict = st.load_file(str(paths["model"]))
            self.model.load_state_dict(state_dict)

        return self

    def to(self, *args, **kwargs) -> "BaseModelIO":
        """Move model to device."""
        if self.model is not None:
            self.model.to(*args, **kwargs)
        return self


@dataclass
class ModelParameter(BaseModelIO):
    """Container for model parameters with serialization capabilities."""

    @classmethod
    def save(cls, instance: "ModelParameter", save_dir: Union[str, Path]):
        instance.save_components(save_dir)

    @classmethod
    def load(
        cls, load_dir: Union[str, Path], disable_init: bool = False
    ) -> "ModelParameter":
        instance = cls()
        return instance.load_components(load_dir, disable_init=disable_init)
