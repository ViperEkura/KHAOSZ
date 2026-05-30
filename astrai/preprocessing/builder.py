"""Mask building strategies for preprocessing pipeline.

Each builder knows how to tokenize one input format and construct
the loss_mask according to declarative mask rules from the config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from astrai.factory import BaseFactory


class BaseMaskBuilder(ABC):
    """Convert a JSONL item into token ids and optional loss_mask."""

    @abstractmethod
    def build(self, item: dict, config, tokenizer) -> Optional[dict]:
        """Build ``{ids, loss_mask?, domain}`` from a JSONL record.

        Returns ``None`` to skip the item entirely.
        """
        ...


class MaskBuilderFactory(BaseFactory["BaseMaskBuilder"]):
    @classmethod
    def _validate_component(cls, component_cls: type):
        if not issubclass(component_cls, BaseMaskBuilder):
            raise TypeError(
                f"{component_cls.__name__} must inherit from BaseMaskBuilder"
            )


def _extract_domain(item: dict, domain_key: Optional[str]) -> str:
    if not domain_key:
        return "__default__"
    val = item.get(domain_key, "__default__")
    return val if isinstance(val, str) else "__default__"


@MaskBuilderFactory.register("chat")
class ChatMaskBuilder(BaseMaskBuilder):
    """Mask by role via message-level tokenisation with role-span tracking.

    For each message, renders the chat template for that single message,
    encodes individually, and records its token span + role action.
    The concatenated sequence receives a loss_mask built from span rules.
    """

    def build(self, item: dict, config, tokenizer) -> Optional[dict]:
        messages = item.get(config.input.messages_key)
        if not isinstance(messages, list) or not messages:
            return None

        all_ids: List[int] = []
        spans: List[tuple] = []

        if tokenizer.bos_token_id is not None:
            all_ids.append(tokenizer.bos_token_id)

        for msg in messages:
            role = msg.get("role", "")
            action = config.mask.get(role, config.mask_default)

            rendered = tokenizer.apply_chat_template(
                [msg], tokenize=False, add_generation_prompt=False
            )
            ids = tokenizer.encode(rendered, add_special_tokens=False)

            start = len(all_ids)
            all_ids.extend(ids)
            spans.append((start, len(all_ids), action))

        if len(all_ids) <= 1:
            return None

        max_len = config.preprocessing.max_seq_len
        all_ids = all_ids[:max_len]

        loss_mask = [0] * len(all_ids)
        for start, end, action in spans:
            if start >= len(all_ids):
                break
            e = min(end, len(all_ids))
            if action == "train":
                loss_mask[start:e] = [1] * (e - start)

        return {
            "ids": all_ids,
            "loss_mask": loss_mask,
            "domain": _extract_domain(item, config.output.domain_key),
        }


@MaskBuilderFactory.register("instruction")
class InstructionMaskBuilder(BaseMaskBuilder):
    """Mask by prompt / response field boundary.

    Encodes prompt and response independently, then fills mask
    according to ``prompt`` / ``response`` entries in the mask config.
    """

    def build(self, item: dict, config, tokenizer) -> Optional[dict]:
        prompt = str(item.get(config.input.prompt_key, ""))
        response = str(item.get(config.input.response_key, ""))

        if not prompt.strip() and not response.strip():
            return None

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        response_ids = tokenizer.encode(response, add_special_tokens=False)

        max_len = config.preprocessing.max_seq_len
        full_ids = (prompt_ids + response_ids)[:max_len]

        prompt_action = config.mask.get("prompt", config.mask_default)
        response_action = config.mask.get("response", config.mask_default)

        p_len = min(len(prompt_ids), len(full_ids))
        r_len = len(full_ids) - p_len

        loss_mask = []
        if prompt_action == "train":
            loss_mask += [1] * p_len
        else:
            loss_mask += [0] * p_len

        if response_action == "train":
            loss_mask += [1] * r_len
        else:
            loss_mask += [0] * r_len

        return {
            "ids": full_ids,
            "loss_mask": loss_mask,
            "domain": _extract_domain(item, config.output.domain_key),
        }


@MaskBuilderFactory.register("text")
class TextMaskBuilder(BaseMaskBuilder):
    """Plain tokenisation — no mask, used for pre-training data."""

    def build(self, item: dict, config, tokenizer) -> Optional[dict]:
        text = item.get(config.input.text_key, "")
        if not isinstance(text, str) or not text.strip():
            return None

        pp = config.preprocessing
        if not (pp.min_chars <= len(text) <= pp.max_chars):
            return None

        ids = tokenizer.encode(text, add_special_tokens=True)
        ids = ids[: pp.max_seq_len]

        return {
            "ids": ids,
            "domain": _extract_domain(item, config.output.domain_key),
        }
