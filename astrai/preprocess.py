"""Composable pipeline: raw JSONL → tokenized .h5 / .bin.

Auto-detects JSONL format:
    - ``messages`` → applies chat template, computes loss_mask
    - ``text`` / plain string field → pure tokenize (pretraining)
    - ``prompt`` + ``response`` → explicit loss_mask from field boundaries

Override ``Pipeline.transform()`` to add custom filters or format support.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from typing import List, Optional

import torch
import tqdm

from astrai.dataset.storage import save_bin, save_h5
from astrai.tokenize import AutoTokenizer

TEXT_KEYS = ["text", "content", "document", "body", "article", "passage"]
DOMAIN_KEYS = ["domain", "source", "category", "topic", "lang", "language"]
MESSAGE_KEYS = ["messages", "conversation", "conversations", "dialog"]


def detect_format(paths: List[str]) -> dict:
    """Auto-detect JSONL schema from first non-empty line.

    Returns ``{text_key, domain_key, is_chat}``.
    """
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for k in MESSAGE_KEYS:
                    if k in obj and isinstance(obj[k], list):
                        return {
                            "text_key": k,
                            "domain_key": _find(obj, DOMAIN_KEYS),
                            "is_chat": True,
                        }
                tk = _find(obj, TEXT_KEYS)
                dk = _find(obj, DOMAIN_KEYS)
                return {"text_key": tk or "text", "domain_key": dk, "is_chat": False}
    return {"text_key": "text", "domain_key": None, "is_chat": False}


def _find(obj: dict, candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in obj and isinstance(obj[k], str):
            return k
    for k, v in obj.items():
        if isinstance(v, str) and len(v) > 20:
            return k
    return None


def filter_length(text: str, min_len: int = 50, max_len: int = 2_000_000) -> bool:
    return min_len <= len(text) <= max_len


def dedup_signature(item: dict) -> str:
    raw = json.dumps(item, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw[:200].encode()).hexdigest()


class Pipeline:
    """Tokenization pipeline: JSONL → tokenized → .h5/.bin.

    Formats handled automatically:

    ===============  ============================================
    JSON keys         behaviour
    ===============  ============================================
    ``messages``      apply chat template, auto loss_mask
    ``text``          plain tokenize (sequence only)
    ``prompt``+``response``  explicit loss_mask
    ===============  ============================================

    Usage::

        p = Pipeline(["docs.jsonl"], output_dir="data/train", tokenizer_path="params")
        p.run()
    """

    def __init__(
        self,
        input_paths: List[str],
        output_dir: str,
        tokenizer_path: str,
        text_key: Optional[str] = None,
        domain_key: Optional[str] = None,
        max_len: int = 2048,
        min_text_len: int = 50,
        max_text_len: int = 2_000_000,
        dedup: bool = True,
        max_items: Optional[int] = None,
        max_tokens_per_shard: int = 100_000_000,
        storage_format: str = "bin",
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.paths = input_paths
        self.output_dir = output_dir
        self.tokenizer_path = tokenizer_path
        self.max_len = max_len
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.dedup = dedup
        self.max_items = max_items
        self.max_tokens_per_shard = max_tokens_per_shard
        self.storage_format = storage_format

        if text_key or domain_key:
            self.text_key = text_key or "text"
            self.domain_key = domain_key
            self.is_chat = False
        else:
            fmt = detect_format(input_paths)
            self.text_key = fmt["text_key"]
            self.domain_key = fmt["domain_key"]
            self.is_chat = fmt["is_chat"]

    def transform(self, item: dict) -> Optional[dict]:
        """Process one JSONL line → {ids, loss_mask?, domain}.

        Override to add custom filters or data formats.
        """
        if self.is_chat:
            return self._transform_chat(item)

        if "prompt" in item and "response" in item:
            return self._transform_prompt_response(item)

        return self._transform_text(item)

    def _transform_text(self, item: dict) -> Optional[dict]:
        text = item.get(self.text_key, "")
        if not isinstance(text, str) or not text.strip():
            return None
        if not filter_length(text, self.min_text_len, self.max_text_len):
            return None
        ids = self._tokenizer.encode(text, add_special_tokens=True)
        ids = ids[: self.max_len]
        return {"ids": ids, "domain": self._domain(item)}

    def _transform_chat(self, item: dict) -> Optional[dict]:
        messages = item.get(self.text_key)
        if not isinstance(messages, list) or not messages:
            return None

        def _encode(msgs):
            s = self._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            return s, self._tokenizer.encode(s, add_special_tokens=True)

        full_str, full_ids = _encode(messages)
        if not filter_length(full_str, self.min_text_len, self.max_text_len):
            return None

        prompt_msgs = messages[:-1]
        if prompt_msgs:
            _, prompt_ids = _encode(prompt_msgs)
        else:
            prompt_ids = []

        full_ids = full_ids[: self.max_len]
        loss_mask = [0] * min(len(prompt_ids), len(full_ids))
        loss_mask += [1] * (len(full_ids) - len(loss_mask))

        return {"ids": full_ids, "loss_mask": loss_mask, "domain": self._domain(item)}

    def _transform_prompt_response(self, item: dict) -> Optional[dict]:
        prompt = str(item.get("prompt", ""))
        response = str(item.get("response", ""))
        if not prompt.strip() and not response.strip():
            return None

        p_ids = self._tokenizer.encode(prompt, add_special_tokens=True)
        r_ids = self._tokenizer.encode(response, add_special_tokens=False)
        full_ids = (p_ids + r_ids)[: self.max_len]
        loss_mask = [0] * min(len(p_ids), len(full_ids))
        loss_mask += [1] * (len(full_ids) - len(loss_mask))

        return {"ids": full_ids, "loss_mask": loss_mask, "domain": self._domain(item)}

    def _domain(self, item: dict) -> str:
        if not self.domain_key:
            return "__default__"
        val = item.get(self.domain_key, "__default__")
        return val if isinstance(val, str) else "__default__"

    def run(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        seen = set()
        domains: dict[str, dict[str, list[list[int]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        total_tokens = 0
        shard_idx: dict[str, int] = defaultdict(int)
        count = 0

        for item in tqdm.tqdm(
            self._iter_items(), desc="Tokenizing", unit="docs", mininterval=0.5
        ):
            if self.max_items and count >= self.max_items:
                break

            if self.dedup:
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

            domain = result["domain"]
            domains[domain]["sequence"].append(ids)
            if "loss_mask" in result:
                domains[domain]["loss_mask"].append(result["loss_mask"])
            count += 1
            total_tokens += len(ids)

            if total_tokens >= self.max_tokens_per_shard:
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
                tensors[key] = [torch.tensor(sum(ids_list, []), dtype=torch.long)]
            chunk_dir = os.path.join(self.output_dir, domain)
            if self.storage_format == "bin":
                save_bin(chunk_dir, tensors)
            else:
                save_h5(chunk_dir, f"data_{idx:04d}", tensors)
            shard_idx[domain] = idx + 1
            tqdm.tqdm.write(
                f"  saved {domain}/shard_{idx:04d}  "
                f"({tensors['sequence'][0].numel():,} tokens)"
            )
