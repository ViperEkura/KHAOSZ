"""Pipeline configuration for JSONL preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from astrai.config.base import BaseConfig


@dataclass
class InputConfig(BaseConfig):
    type: str = "chat"
    messages_key: str = "messages"
    prompt_key: str = "prompt"
    response_key: str = "response"
    text_key: str = "text"


@dataclass
class ProcessingConfig(BaseConfig):
    max_seq_len: int = 2048
    min_chars: int = 50
    max_chars: int = 2_000_000
    deduplicate: bool = True
    max_items: Optional[int] = None


@dataclass
class OutputConfig(BaseConfig):
    domain_key: Optional[str] = None
    storage_format: str = "bin"
    max_tokens_per_shard: int = 100_000_000


@dataclass
class PipelineConfig(BaseConfig):
    version: int = 1
    input: InputConfig = field(default_factory=InputConfig)
    mask: Dict[str, str] = field(default_factory=dict)
    mask_default: str = "mask"
    preprocessing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
