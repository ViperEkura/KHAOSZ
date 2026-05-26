# Data Flow

This document describes the data pipeline: from raw text to model input tensors.

## Overview

```
Raw Text → AutoTokenizer → Token IDs → .h5/.json → Dataset → Sampler → DataLoader → Training/Inference
```

## Data Preparation

Raw text is tokenized via `AutoTokenizer.encode()` and saved as HDF5 (`.h5`) or JSON (`.json`/`.jsonl`) files with keyed tensor groups.

Storage format is auto-detected by `detect_format()`; backends are dispatched via registry:

```
StorageFactory.create("h5")   → H5Storage
StorageFactory.create("json") → JSONStorage
```

Both support shared memory via `.share_memory_()`.

## Data Keys by Training Type

| Type | Storage Keys |
|------|-------------|
| `seq` | `sequence` (→ input_ids, target_ids via offset-by-1) |
| `sft` | `sequence`, `loss_mask` |
| `dpo` | `chosen`, `rejected`, `chosen_mask`, `rejected_mask` |
| `grpo` | `prompts`, `responses`, `masks`, `rewards` |

## Dataset Architecture

```
DatasetFactory.load(train_type, path, window_size, stride, storage_type, tokenizer)
  → StorageFactory.create(detect_format(path))
    → MultiSegmentFetcher(BaseSegmentFetcher per key)
      → BaseDataset.__getitem__(idx)
        → sliding window [begin, end) via get_index(idx)
```

`window_size` = max input length, `stride` = step between consecutive samples (defaults to `window_size`).

## Sampler

`ResumableDistributedSampler` supports checkpoint-aware distributed sampling:

- Tracks `start_epoch` / `start_iter` for resume
- Shuffle via `torch.Generator(seed + epoch)`
- Per-replica index slicing for DDP

## DataLoader

Standard PyTorch `DataLoader` with configurable `batch_size`, `num_workers`, `pin_memory`, `prefetch_factor`. Sampler produces indices; dataloader fetches tensor batches via `__getitem__`.

> Document Update Time: 2026-05-17
