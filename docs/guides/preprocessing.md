# Preprocessing Pipeline

Declarative JSON-driven data preprocessing. `MaskBuilderFactory` supports three registered builders: `"single"` (single-output via `input.sections`), `"multi"` (multi-output via `input.sources`), and `"sectioned"` (façade dispatching to `single` or `multi` based on config).

## Contents

- [Philosophy](#philosophy)
- [Config Structure](#config-structure)
- [Quick Start](#quick-start) — SFT Chat, SFT Instruction, Pretrain, DPO, GRPO examples
- [Configuration Reference](#configuration-reference) — all fields
- [Mask Algorithm](#mask-algorithm)
- [Output Layout](#output-layout)
- [Training Compatibility](#training-compatibility)
- [CLI](#cli)
- [Python API](#python-api)

## Philosophy

| Component | Responsibility |
|-----------|---------------|
| `tokenizer_config.json` (`chat_template`) | Formatting -- how roles become tokens |
| `pipeline.json` (`mask`) | Masking -- which roles participate in training |

A single config file captures the entire pipeline, reusable and version-controllable.

## Config Structure

```json
{
  "version":       1,
  "input":         {},   // sections (single) or sources (multi)
  "mask":          {},   // role -> "train" | "mask"
  "mask_default":  "mask",
  "preprocessing": {},
  "output":        {}
}
```

### Section Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `field` | str | -- | JSONL key to read |
| `action` | str | -- | `"train"` / `"mask"` / `"$role"` / `"value"`; `"value"` copies raw values without tokenization |
| `template` | bool | `false` | Apply `chat_template` per message |
| `add_special_tokens` | bool | `true` for first non-template section | Add special tokens during encode |

### Source Fields (multi-output mode)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sections` | list[dict] | -- | Same as single-output section list |
| `list_field` | bool | `false` | JSONL field holds a list; tokenise each element |
| `mask_key` | str | `"{key}_mask"` | Explicit output key for loss mask |

---

## Quick Start

### SFT Chat

Input JSONL:

```json
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
```

Config:

```json
{
  "input": {
    "sections": [
      {"field": "messages", "action": "$role", "template": true}
    ]
  },
  "mask": {
    "system": "mask",
    "user": "mask",
    "assistant": "train"
  },
  "mask_default": "mask",
  "preprocessing": {
    "max_seq_len": 2048
  },
  "output": {
    "storage_format": "bin",
    "dtype": {"loss_mask": "bool"}
  }
}
```

Output keys: `sequence` (int32), `loss_mask` (bool), `position_ids` (int32)

### SFT Instruction

Input JSONL:

```json
{"prompt": "Translate to French: Hello", "response": "Bonjour"}
```

Config:

```json
{
  "input": {
    "sections": [
      {"field": "prompt",   "action": "mask", "add_special_tokens": true},
      {"field": "response", "action": "train"}
    ]
  },
  "mask_default": "mask",
  "preprocessing": {
    "max_seq_len": 2048
  }
}
```

Output keys: `sequence`, `loss_mask`, `position_ids`

### Pretrain

Input JSONL:

```json
{"text": "Artificial Intelligence is a field of computer science..."}
```

Config:

```json
{
  "input": {
    "sections": [
      {"field": "text", "action": "train"}
    ]
  },
  "preprocessing": {
    "max_seq_len": 8192,
    "min_chars": 100
  }
}
```

Output keys: `sequence`, `position_ids` (no `loss_mask` — all tokens trained)

### DPO

Input JSONL:

```json
{"chosen": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}], "rejected": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "5"}]}
```

Config:

```json
{
  "input": {
    "sources": {
      "chosen": {
        "sections": [
          {"field": "chosen", "action": "$role", "template": true}
        ]
      },
      "rejected": {
        "sections": [
          {"field": "rejected", "action": "$role", "template": true}
        ]
      }
    }
  },
  "mask": {
    "user": "mask",
    "assistant": "train"
  },
  "mask_default": "mask"
}
```

Output keys: `chosen`, `chosen_mask`, `rejected`, `rejected_mask`

The offline `Pipeline` can construct these keys, but its `.bin` output is not
currently loadable for DPO training because the writer does not preserve
per-record offsets. Train DPO directly from raw JSONL instead; see
[Training Compatibility](#training-compatibility).

### GRPO

Input JSONL:

```json
{"prompt": [{"role": "user", "content": "What is 2+2?"}], "responses": ["4", "Five", "Four"], "rewards": [1.0, 0.3, 0.8]}
```

Config:

```json
{
  "input": {
    "sources": {
      "prompts": {
        "sections": [
          {"field": "prompt", "action": "mask", "template": true}
        ]
      },
      "responses": {
        "sections": [
          {"field": "responses", "action": "train"}
        ],
        "list_field": true,
        "mask_key": "masks"
      },
      "rewards": {
        "sections": [
          {"field": "rewards", "action": "value"}
        ]
      }
    }
  },
  "mask": {
    "user": "mask",
    "assistant": "train"
  },
  "mask_default": "mask"
}
```

Output keys: `prompts`, `prompts_mask`, `responses`, `masks`, `rewards` (float32)

- `action: "value"` — extract raw values from JSONL without tokenisation
- `list_field: true` — tokenise each list element independently, then concatenate
- `mask_key: "masks"` — rename the auto-generated mask key (default: `responses_mask`)
- `prompts_mask` is auto-generated (all masked) and unused by GRPOStrategy

The offline `Pipeline` flattens GRPO response groups for `.bin` output without
preserving their boundaries, and there is no automatic raw-JSONL GRPO processor
in `DatasetFactory`. See
[Training Compatibility](#training-compatibility) for the supported routes.

---

## Configuration Reference

### `input`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sections` | list[dict] or null | `null` | Section specs for single-output mode |
| `sources` | dict[str, dict] or null | `null` | Source specs for multi-output mode (DPO/GRPO) |

When `sources` is set, `sections` is ignored.

### `mask`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mask` | dict | `{}` | `{role: "train" \| "mask"}` |
| `mask_default` | str | `"mask"` | Default action for unlisted roles |

### `preprocessing`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_seq_len` | int | `2048` | Truncate sequences to this length |
| `min_chars` | int | `50` | Skip text-mode items shorter than this |
| `max_chars` | int | `2000000` | Skip text-mode items longer than this |
| `max_items` | int or null | `null` | Stop after N documents |
| `batch_size` | int | `256` | Records per tokenization batch |
| `packing_strategy` | str | `"simple"` | Packing is supported for single-output data with a `sequence` key: `"simple"`, `"bfd"`, or `"bfd_split"`. Multi-output DPO/GRPO data is not record-preserving packed output. |
| `max_packed_len` | int | `8192` | Maximum length of a packed bin |
| `truncation_mode` | str | `"keep_start"` | How to truncate sequences: `"keep_start"` or `"keep_end"` |

### Tokenizer parallelism

Batch inputs passed to AutoTokenizer.encode use the Rust tokenizers backend.
Enable its Rayon thread pool before starting preprocessing with:

```bash
TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=2 \
python scripts/tools/preprocess.py data/part-000.jsonl -o output/ -c configs/pretrain.json
```

RAYON_NUM_THREADS limits threads per process. In a job that also uses
multiple worker processes, each process can create its own Rayon pool;
start with a small thread limit and tune the process-count times
thread-count against available CPU capacity. Set these variables before
launching the job; they do not reconfigure already-running workers.

### `output`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `domain_key` | str or null | `null` | JSONL key for domain grouping |
| `storage_format` | str | `"bin"` | Pipeline output format. Only `"bin"` has a registered writer; `"jsonl"` is accepted by config validation but cannot be emitted by `Pipeline`. |
| `max_tokens_per_shard` | int | `100000000` | Flush threshold counted from each record's primary flat sequence: `sequence` for single-output data, otherwise the first flat source output |
| `dtype` | dict[str, str] | `{}` | Per-key tensor dtype override (e.g. `{"loss_mask": "bool"}`) |
| `position_ids_mode` | str | `"doc_reset"` | How to compute position_ids: `"none"`, `"doc_reset"`, `"continuous"` |

---

## Mask Algorithm

### Template mode (`template: true`)

1. Prepend BOS token (masked)
2. For each message in the field's array:
   1. Render through `chat_template` for that single message
   2. Encode rendered text
   3. Apply mask rule for the message's role

### Non-template mode

Encode the field value as text. Mask value is 1 (train) or 0 (mask) per the section's `action`.

### Text config detection

When no section uses `template` and all sections have `action: "train"`, the builder omits `loss_mask` from the output — all tokens are trained.

---

## Output Layout

### Single-Shard (`bin`)

```
output/
  __default__/
    shard_0000/
      meta.json
      sequence.bin
      loss_mask.bin
      position_ids.bin
  wiki/
    shard_0000/
      meta.json
      sequence.bin
      loss_mask.bin
      position_ids.bin
```

### Multi-Shard (`bin`)

When `max_tokens_per_shard` is exceeded:

```
output/
  __default__/
    shard_0000/
      meta.json
      sequence.bin
      loss_mask.bin
      position_ids.bin
    shard_0001/
      meta.json
      sequence.bin
      loss_mask.bin
      position_ids.bin
```

`MmapStore` discovers binary shards recursively through their `meta.json` files.
Each shard's metadata is a top-level object keyed by tensor name:

```json
{
  "sequence": {"shape": [123456], "dtype": "int32"},
  "loss_mask": {"shape": [123456], "dtype": "bool"},
  "position_ids": {"shape": [123456], "dtype": "int32"}
}
```

An optional `offsets` array may appear for record-oriented binary data written
through `save_bin(..., record_keys=...)`; the preprocessing `BinWriter` does not
currently request those offsets.

---

## Training Compatibility

| Training type | Supported input route |
|---------------|-----------------------|
| `seq` | Offline preprocessed `.bin`, or raw `.jsonl` eagerly transformed by `JsonlStore` using `dataset_config.json` or the built-in `messages` config |
| `sft` | Offline preprocessed `.bin`, or raw `.jsonl` through the same eager transform routes |
| `dpo` | Raw `.jsonl` through the automatic lazy DPO processor selected by `DatasetFactory` when `tokenizer_path` is supplied, or a caller-provided record store |
| `grpo` | A caller-provided, already-loaded `Store` with record-shaped `prompts`, `responses`, `masks`, and `rewards`; no automatic raw-JSONL processor is currently wired |

Offline DPO and GRPO preprocessing configs describe the intended token fields,
but their `.bin` output is not currently loadable for training. DPO binary
shards lack per-record offsets. GRPO response groups are flattened before the
binary writer and their record/group boundaries are not preserved.

---

## CLI

```bash
# SFT
python scripts/tools/preprocess.py data/sft/part-000.jsonl -o output/sft/ -c configs/sft_chat.json --batch_size 128

# DPO
python scripts/tools/preprocess.py data/dpo/part-000.jsonl -o output/dpo/ -c configs/dpo.json --tokenizer_path params

# GRPO
python scripts/tools/preprocess.py data/grpo/part-000.jsonl -o output/grpo/ -c configs/grpo.json
```

Inputs may be `.jsonl` files or `.json` files containing one object or a list of
objects. Each positional path must exist. A wildcard such as `data/*.jsonl`
works only when the invoking shell expands it before Click receives the
arguments; otherwise pass the files explicitly.

---

## Python API

```python
from astrai.preprocessing.pipeline import Pipeline
from astrai.config.preprocess_config import PipelineConfig

config = PipelineConfig.from_file("sft.json")
Pipeline(
    config,
    ["data_part1.jsonl", "data_part2.jsonl"],
    output_dir="output/",
    tokenizer_path="params",
).run()
```

> Document Update Time: 2026-07-09
