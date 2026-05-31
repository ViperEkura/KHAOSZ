"""Config-driven JSONL preprocessing pipeline.

Composes a :class:`BaseMaskBuilder` (selected by ``input.type``) with
deduplication, sharding, and flush to ``.h5`` / ``.bin`` storage.
"""

import hashlib
import json
import os
from collections import defaultdict
from itertools import chain
from typing import Optional

import torch
import tqdm

from astrai.config.preprocess_config import PipelineConfig
from astrai.dataset.storage import save_bin, save_h5
from astrai.preprocessing.builder import SectionedMaskBuilder
from astrai.tokenize import AutoTokenizer

_STR_TO_DTYPE: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def filter_by_length(text: str, min_len: int = 50, max_len: int = 2_000_000) -> bool:
    return min_len <= len(text) <= max_len


def dedup_signature(item: dict) -> str:
    raw = json.dumps(item, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw[:200].encode()).hexdigest()


class Pipeline:
    """Tokenization pipeline driven by a declarative :class:`PipelineConfig`.

    Usage::

        config = PipelineConfig.from_json("sft_pipeline.json")
        Pipeline(config, ["data.jsonl"], output_dir="out", tokenizer_path="params").run()
    """

    def __init__(
        self,
        config: PipelineConfig,
        input_paths: list[str],
        output_dir: str,
        tokenizer_path: str,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.config = config
        self.paths = input_paths
        self.output_dir = output_dir
        self.tokenizer_path = tokenizer_path

        self.mask_builder = SectionedMaskBuilder()

    def transform(self, item: dict) -> Optional[dict]:
        return self.mask_builder.build(item, self.config, self._tokenizer)

    def run(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        seen: set = set()
        domains: dict = defaultdict(lambda: defaultdict(list))
        total_tokens = 0
        shard_idx: dict[str, int] = defaultdict(int)
        count = 0

        pp = self.config.preprocessing

        for item in tqdm.tqdm(
            self._iter_items(), desc="Tokenizing", unit="docs", mininterval=0.5
        ):
            if pp.max_items and count >= pp.max_items:
                break

            if pp.deduplicate:
                sig = dedup_signature(item)
                if sig in seen:
                    continue
                seen.add(sig)

            result = self.transform(item)
            if result is None:
                continue

            ids = result["ids"]
            if not ids:
                continue

            domain = result.get("domain", "__default__")
            domains[domain]["sequence"].append(ids)
            if "loss_mask" in result:
                domains[domain]["loss_mask"].append(result["loss_mask"])

            count += 1
            total_tokens += len(ids)

            if total_tokens >= self.config.output.max_tokens_per_shard:
                self._flush(domains, shard_idx)
                domains.clear()
                total_tokens = 0

        if total_tokens > 0:
            self._flush(domains, shard_idx)

        print(f"Done. {count} documents tokenized.")

    def _iter_items(self):
        for path in self.paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)

    def _flush(self, domains, shard_idx):
        for domain, keys in domains.items():
            idx = shard_idx[domain]
            tensors = {}
            for key, ids_list in keys.items():
                dt = _STR_TO_DTYPE.get(
                    self.config.output.dtype.get(key, "int32"), torch.int32
                )
                tensors[key] = [
                    torch.tensor(list(chain.from_iterable(ids_list)), dtype=dt)
                ]
            chunk_dir = os.path.join(self.output_dir, domain)
            fmt = self.config.output.storage_format
            if fmt == "bin":
                save_bin(os.path.join(chunk_dir, f"shard_{idx:04d}"), tensors)
            else:
                save_h5(chunk_dir, f"data_{idx:04d}", tensors)
            shard_idx[domain] = idx + 1
            tqdm.tqdm.write(
                f"  saved {domain}/shard_{idx:04d}  "
                f"({tensors['sequence'][0].numel():,} tokens)"
            )
