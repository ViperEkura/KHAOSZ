"""Pipeline configuration for JSONL preprocessing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class InputConfig:
    type: str = "chat"
    messages_key: str = "messages"
    prompt_key: str = "prompt"
    response_key: str = "response"
    text_key: str = "text"


@dataclass
class ProcessingConfig:
    max_seq_len: int = 2048
    min_chars: int = 50
    max_chars: int = 2_000_000
    deduplicate: bool = True
    max_items: Optional[int] = None


@dataclass
class OutputConfig:
    domain_key: Optional[str] = None
    storage_format: str = "bin"
    max_tokens_per_shard: int = 100_000_000


@dataclass
class PipelineConfig:
    version: int = 1
    input: InputConfig = field(default_factory=InputConfig)
    mask: Dict[str, str] = field(default_factory=dict)
    mask_default: str = "mask"
    preprocessing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "input": {
                "type": self.input.type,
                "messages_key": self.input.messages_key,
                "prompt_key": self.input.prompt_key,
                "response_key": self.input.response_key,
                "text_key": self.input.text_key,
            },
            "mask": self.mask,
            "mask_default": self.mask_default,
            "preprocessing": {
                "max_seq_len": self.preprocessing.max_seq_len,
                "min_chars": self.preprocessing.min_chars,
                "max_chars": self.preprocessing.max_chars,
                "deduplicate": self.preprocessing.deduplicate,
                "max_items": self.preprocessing.max_items,
            },
            "output": {
                "domain_key": self.output.domain_key,
                "storage_format": self.output.storage_format,
                "max_tokens_per_shard": self.output.max_tokens_per_shard,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> PipelineConfig:
        return PipelineConfig(
            version=data.get("version", 1),
            input=InputConfig(**data.get("input", {})),
            mask=data.get("mask", {}),
            mask_default=data.get("mask_default", "mask"),
            preprocessing=ProcessingConfig(**data.get("preprocessing", {})),
            output=OutputConfig(**data.get("output", {})),
        )

    @classmethod
    def from_json(cls, path: str) -> PipelineConfig:
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
