# Parameter Documentation

## Training Parameters

### Basic Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--train_type` | Training type (`seq`, `sft`, `dpo`, `grpo`) | required |
| `--data_root_path` | Dataset root directory | required |
| `--param_path` | Model parameters or checkpoint path | required |
| `--n_epoch` | Total training epochs | 1 |
| `--batch_per_device` | Batch size per device | 1 |
| `--grad_accum_steps` | Gradient accumulation steps between optimizer steps | 1 |

### Learning Rate Scheduling

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--warmup_ratio` | Fraction of total steps used for LR warmup | 0.05 |
| `--max_lr` | Maximum learning rate (cosine decay after warmup) | 3e-4 |
| `--max_grad_norm` | Maximum gradient norm for clipping | 1.0 |

### Optimizer (AdamW)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--adamw_beta1` | AdamW beta1 | 0.9 |
| `--adamw_beta2` | AdamW beta2 | 0.95 |
| `--adamw_weight_decay` | AdamW weight decay | 0.01 |

### Data Loading

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--window_size` | Max input sequence length | model config `max_len` |
| `--stride` | Stride for sliding window over sequences | None |
| `--random_seed` | Random seed for reproducibility | 3407 |
| `--num_workers` | DataLoader worker processes | 4 |
| `--no_pin_memory` | Disable pin_memory (enabled by default) | (flag) |

### Checkpoint & Resume

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--ckpt_interval` | Iterations between checkpoints | 5000 |
| `--ckpt_dir` | Checkpoint save directory | checkpoint |
| `--start_epoch` | Resume from epoch (0 = from scratch) | 0 |
| `--start_batch` | Resume from batch iteration | 0 |

### Distributed Training

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--nprocs` | Number of GPUs / processes | 1 |
| `--parallel_mode` | Parallel strategy (`none`, `ddp`, or `fsdp`) | none |
| `--device_type` | Device type | cuda |
| `--start_method` | Multiprocessing start method (`spawn`, `fork`, `forkserver`) | spawn |

### Strategy-specific

| Parameter | Description | Default | Used by |
|-----------|-------------|---------|---------|
| `--dpo_beta` | DPO beta value | 0.1 | `dpo` |
| `--label_smoothing` | Label smoothing for cross-entropy loss | 0.05 | `seq`, `sft` |
| `--group_size` | GRPO group size | 4 | `grpo` |
| `--grpo_clip_eps` | GRPO clipping epsilon | 0.2 | `grpo` |
| `--grpo_kl_coef` | GRPO KL penalty coefficient | 0.01 | `grpo` |
| `--grpo_sync_interval` | GRPO ref_model sync interval (steps) | 200 | `grpo` |

### Usage Example

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup python scripts/tools/train.py \
    --nprocs=4 \
    --parallel_mode=ddp \
    --train_type=seq \
    --data_root_path=/path/to/dataset \
    --param_path=/path/to/model \
    --batch_per_device=4 \
    --grad_accum_steps=8 \
    --warmup_ratio=0.05 \
    --max_lr=1e-4 \
    --max_grad_norm=1.0 \
    --adamw_beta1=0.9 \
    --adamw_beta2=0.95 \
    --adamw_weight_decay=0.01 \
    --window_size=2048 \
    --ckpt_interval=10000 \
    --ckpt_dir=./checkpoint \
    --random_seed=3407 \
    --label_smoothing=0.05 \
    > out.log 2> err.log &
```

---

> Document Update Time: 2026-05-24