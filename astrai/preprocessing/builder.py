"""Mask building strategies for preprocessing pipeline.

The single :class:`SectionedMaskBuilder` handles all input formats
via declarative ``input.sections`` config.
"""

from abc import ABC, abstractmethod
from typing import Optional

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


def _resolve_action(action: str, role: str, config) -> str:
    """Resolve action to "train" or "mask".

    - ``"train"`` / ``"mask"`` → literal
    - ``"$role"`` → look up ``role`` in ``config.mask``, fall back to ``config.mask_default``
    """
    if action == "$role":
        return config.mask.get(role, config.mask_default)
    return action


@MaskBuilderFactory.register("sectioned")
class SectionedMaskBuilder(BaseMaskBuilder):
    """Config-driven builder: iterates over ``input.sections`` in order.

    Each section specifies a JSONL field + mask action.

    Section spec::

        {
            "field":    "messages",   # JSONL key
            "action":   "$role",      # "train" | "mask" | "$role"
            "template": true,         # apply chat_template per message (optional)
            "add_special_tokens": false  # override encode flag (optional)
        }

    Example configs::

        # Chat
        {"input": {"sections": [
            {"field": "messages", "action": "$role", "template": true}
        ]}}

        # Instruction
        {"input": {"sections": [
            {"field": "prompt",   "action": "mask", "add_special_tokens": true},
            {"field": "response", "action": "train"}
        ]}}

        # Text
        {"input": {"sections": [
            {"field": "text", "action": "train"}
        ]}}
    """

    def build(self, item: dict, config, tokenizer) -> Optional[dict]:
        sections = config.input.sections
        if not sections:
            return None

        all_ids: list[int] = []
        loss_mask: list[int] = []

        has_template = any(s.get("template") for s in sections)
        is_text_config = not has_template and all(
            s["action"] == "train" for s in sections
        )

        if has_template and tokenizer.bos_token_id is not None:
            all_ids.append(tokenizer.bos_token_id)
            loss_mask.append(0)

        first_section = True
        for sec in sections:
            field = sec["field"]
            action = sec["action"]
            use_template = sec.get("template", False)
            add_special = sec.get(
                "add_special_tokens", not use_template and first_section
            )

            if use_template:
                messages = item.get(field)
                if not isinstance(messages, list) or not messages:
                    continue
                for msg in messages:
                    role = msg.get("role", "")
                    act = _resolve_action(action, role, config)
                    rendered = tokenizer.apply_chat_template(
                        [msg], tokenize=False, add_generation_prompt=False
                    )
                    ids = tokenizer.encode(rendered, add_special_tokens=False)
                    all_ids.extend(ids)
                    val = 1 if act == "train" else 0
                    loss_mask.extend([val] * len(ids))
            else:
                text = str(item.get(field, ""))
                if not text.strip():
                    continue
                if is_text_config:
                    pp = config.preprocessing
                    if pp.min_chars > 0 and len(text) < pp.min_chars:
                        continue
                    if len(text) > pp.max_chars:
                        continue
                ids = tokenizer.encode(text, add_special_tokens=add_special)
                all_ids.extend(ids)
                val = 1 if action == "train" else 0
                loss_mask.extend([val] * len(ids))

            first_section = False

        max_len = config.preprocessing.max_seq_len
        all_ids = all_ids[:max_len]
        loss_mask = loss_mask[: len(all_ids)]

        if not all_ids:
            return None

        if has_template and len(all_ids) <= 1:
            return None

        result: dict = {
            "ids": all_ids,
            "domain": _extract_domain(item, config.output.domain_key),
        }
        if not all(m == 1 for m in loss_mask):
            result["loss_mask"] = loss_mask
        return result
