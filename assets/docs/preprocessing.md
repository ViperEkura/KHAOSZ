# Preprocessing Pipeline

Declarative JSON-driven data preprocessing. No code needed -- describe your input format and mask rules in a config file, the engine does the rest.

## Philosophy

| Component | Responsibility |
|-----------|---------------|
| `tokenizer_config.json` (`chat_template`) | Formatting -- how roles become tokens |
| `pipeline.json` (`mask`) | Masking -- which roles participate in training |

The two are fully decoupled. A single config file captures the entire pipeline, reusable and version-controllable. Extension is via factory registration (`@MaskBuilderFactory.register`) -- no need to touch existing code.

## Quick Start

### SFT Chat

```json
{
  "version": 1,
  "input": {
    "type": "chat",
    "messages_key": "messages"
  },
  "mask": {
    "system": "mask",
    "user": "mask",
    "assistant": "train"
  },
  "mask_default": "mask",
  "preprocessing": {
    "max_seq_len": 2048,
    "deduplicate": true
  },
  "output": {
    "domain_key": "source",
    "storage_format": "bin",
    "max_tokens_per_shard": 100000000
  }
}
```

Three lines of mask rules cover the most common SFT case: train on assistant turns, mask everything else.

### Instruction Tuning

```json
{
  "version": 1,
  "input": {
    "type": "instruction",
    "prompt_key": "instruction",
    "response_key": "output"
  },
  "mask": {
    "prompt": "mask",
    "response": "train"
  },
  "mask_default": "mask",
  "preprocessing": {
    "max_seq_len": 2048
  },
  "output": {
    "storage_format": "bin"
  }
}
```

Mask splits at the prompt/response field boundary.

### Pretraining

```json
{
  "version": 1,
  "input": {
    "type": "text",
    "text_key": "content"
  },
  "mask": {},
  "preprocessing": {
    "max_seq_len": 2048,
    "min_chars": 50
  },
  "output": {
    "storage_format": "bin"
  }
}
```

No mask -- train on all tokens.

### Run

```bash
python scripts/tools/preprocess.py data/*.jsonl -o output/ -c sft.json
```

## Configuration Reference

### `input`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | string | yes | `"chat"` | Format: `"chat"`, `"instruction"`, or `"text"` |
| `messages_key` | string | no | `"messages"` | JSON key for messages array (chat) |
| `prompt_key` | string | no | `"prompt"` | JSON key for prompt field (instruction) |
| `response_key` | string | no | `"response"` | JSON key for response field (instruction) |
| `text_key` | string | no | `"text"` | JSON key for text field |

### `mask`

A map of `{role_or_field: "mask" | "train"}`. The engine uses this to build `loss_mask`:

- `"mask"` -- tokens in this span are ignored during training (`loss_mask=0`)
- `"train"` -- tokens in this span contribute to the loss (`loss_mask=1`)

For chat mode, keys are role names (`system`, `user`, `assistant`, ...).
For instruction mode, keys are `"prompt"` and `"response"`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mask` | dict | `{}` | Role/field to action mapping |
| `mask_default` | string | `"mask"` | Default action for unlisted roles |

### `preprocessing`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_seq_len` | int | `2048` | Maximum token length; truncated if exceeded |
| `min_chars` | int | `50` | Minimum character length; dropped if shorter (text mode only) |
| `max_chars` | int | `2000000` | Maximum character length; dropped if longer (text mode only) |
| `deduplicate` | bool | `true` | Remove exact duplicates via MD5 of first 200 chars |
| `max_items` | int or null | `null` | Maximum items to process; `null` = unlimited |

### `output`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `domain_key` | string or null | `null` | JSON key for domain grouping; `null` = all output to `__default__` |
| `storage_format` | string | `"bin"` | `"bin"` (mmap, zero-copy) or `"h5"` (HDF5) |
| `max_tokens_per_shard` | int | `100000000` | Max tokens per output shard |

## Mask Algorithm

### Chat Mode (role-span tracking)

For each message in the `messages` array:

1. Prepend BOS token (position 0, always masked)
2. Render through the chat template for that single message
3. Encode the rendered text, record token span `(start, end, role)`
4. Concatenate all spans — special tokens from the chat template naturally prevent BPE merging across message boundaries
5. Fill `loss_mask` from the mask rules

**Multi-turn example**:

```
Data:
  [system: "You are helpful."]
  [user: "What is 2+2?"]
  [assistant: "4"]
  [user: "What is 3+3?"]
  [assistant: "6"]

Config:
  "mask": {"system": "mask", "user": "mask", "assistant": "train"}

Result:
  tokens:  <bos> [system span] [user span] [assistant:4 span] [user span] [assistant:6 span]
  mask:      0       0            0              1               0             1
```

Both assistant turns are trained. All system and user tokens are masked.

### Instruction Mode (field boundary)

Encode the prompt and response fields independently, then split the mask at the field boundary.

- `"prompt": "mask", "response": "train"` -- mask the left half, train the right half
- `"prompt": "train", "response": "mask"` -- the reverse

### Text Mode (no mask)

Pure tokenization. No `loss_mask` is produced. Used for pretraining.

## Output Layout

```
output_dir/
  __default__/              # when domain_key is null
    meta.json               # {"sequence": {"shape": [N], "dtype": "int64"}, ...}
    sequence.bin            # int64 raw bytes, mmap-able for zero-copy reads
    loss_mask.bin           # int64 raw bytes
  wiki/                     # when domain_key="source" and item["source"]="wiki"
    meta.json
    sequence.bin
    loss_mask.bin
```

## Extension

Register a custom builder for new formats:

```python
from astrai.preprocessing.builder import BaseMaskBuilder, MaskBuilderFactory

@MaskBuilderFactory.register("my_format")
class MyFormatBuilder(BaseMaskBuilder):
    def build(self, item: dict, config, tokenizer) -> dict | None:
        # Return {"ids": [...], "loss_mask": [...], "domain": "..."}
        # Return None to skip this item
        ...
```

Then set `"input": {"type": "my_format"}` in your config.

## Compared to Old Pipeline

| Old (`astrai.preprocess.Pipeline`) | New (`astrai.preprocessing.pipeline.Pipeline`) |
|---|---|
| Configured via constructor arguments | Configured via JSON file |
| Hardcoded `_transform_chat` / `_transform_text` | Factory-registered `Builder` with declarative mask rules |
| Auto-detects format via magic key lists | Explicit `input.type` declaration |
| Double-encodes (full + prompt), uses length diff for mask | Single-encode with role-span tracking |
| Only trains the last assistant turn | Configurable: multi-turn, single-turn, or no mask |

> Document Update Time: 2026-05-30
