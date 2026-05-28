# Data Flow

This document describes the data pipeline: from raw text to model input tensors.

## Overview

```
Raw Text → AutoTokenizer → Token IDs → .h5/.bin → Dataset → Sampler → DataLoader → Training/Inference
```

## Data Preparation

Raw text is tokenized via `AutoTokenizer.encode()` and saved as HDF5 (`.h5`) or binary (`.bin` + `meta.json`) files with keyed tensor groups.

Storage format is auto-detected by `detect_format()`; backends are dispatched via registry:

```
StoreFactory.create("h5")  → H5Store
StoreFactory.create("bin") → MmapStore
```

H5 backend supports shared memory via `.share_memory_()`. Bin (mmap) uses OS page-cache sharing natively.

## Data Keys by Training Type

| Type | Storage Keys |
|------|-------------|
| `seq` | `sequence` (→ input_ids, target_ids via offset-by-1) |
| `sft` | `sequence`, `loss_mask` |
| `dpo` | `chosen`, `rejected`, `chosen_mask`, `rejected_mask` |
| `grpo` | `prompts`, `responses`, `masks`, `rewards` |

## Dataset Architecture

```
DatasetFactory.load(train_type, load_path, window_size, stride, storage_type)
  → StoreFactory.create(detect_format(path))
    → Store._data[Dict[str, List[Tensor]]] + _cum[Dict[str, List[int]]]
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

> Document Update Time: 2026-05-28
