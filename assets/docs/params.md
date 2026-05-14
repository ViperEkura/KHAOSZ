# Parameter Documentation

## Training Parameters

### Basic Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--train_type` | Training type (`seq`, `sft`, `dpo`, `grpo`) | required |
| `--data_root_path` | Dataset root directory | required |
| `--param_path` | Model parameters or checkpoint path | required |
| `--n_epoch` | Total training epochs | 1 |
| `--batch_size` | Batch size | 1 |
| `--accumulation_steps` | Gradient accumulation steps between optimizer steps | 1 |

### Learning Rate Scheduling

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--warmup_steps` | Warmup steps | 1000 |
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
| `--device_type` | Device type | cuda |

### Strategy-specific

| Parameter | Description | Default | Used by |
|-----------|-------------|---------|---------|
| `--dpo_beta` | DPO beta value | 0.1 | `dpo` |
| `--label_smoothing` | Label smoothing for cross-entropy loss | 0.1 | `seq`, `sft` |
| `--group_size` | GRPO group size | 4 | `grpo` |
| `--grpo_clip_eps` | GRPO clipping epsilon | 0.2 | `grpo` |
| `--grpo_kl_coef` | GRPO KL penalty coefficient | 0.01 | `grpo` |
| `--grpo_sync_interval` | GRPO ref_model sync interval (steps) | 200 | `grpo` |

### Usage Example

```bash
python scripts/tools/train.py \
  --train_type seq \
  --data_root_path /path/to/dataset \
  --param_path /path/to/model \
  --n_epoch 3 \
  --batch_size 4 \
  --accumulation_steps 8 \
  --max_lr 3e-4 \
  --warmup_steps 2000 \
  --max_grad_norm 1.0 \
  --ckpt_interval 5000 \
  --ckpt_dir ./checkpoints \
  --num_workers 4 \
  --nprocs 1 \
  --device_type cuda
```

---

## Generation Parameters

### GenerationRequest Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `messages` | List of message dictionaries (role, content) | required |
| `temperature` | Sampling temperature (higher = more random) | 1.0 |
| `top_p` | Nucleus sampling threshold | 1.0 |
| `top_k` | Top-k sampling count | 50 |
| `max_len` | Maximum generation length | 1024 |
| `stream` | Whether to stream output | False |

### Usage Example

```python
import torch
from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer
from astrai.inference import InferenceEngine, GenerationRequest

# Load model using AutoModel
model = AutoModel.from_pretrained("your_model_dir")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("your_model_dir")

# Create engine with separate model and tokenizer
engine = InferenceEngine(
    model=model,
    tokenizer=tokenizer,
)

# Build request with messages format
request = GenerationRequest(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ],
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    max_len=1024,
)

# Generate (streaming)
for token in engine.generate_with_request(request):
    print(token, end="", flush=True)

# Or use simple generate interface
result = engine.generate(
    prompt="Hello",
    stream=False,
    max_tokens=1024,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
)
```

### Generation Modes

| Mode | Description |
|------|-------------|
| `stream=True` | Streaming output, yields token by token |
| `stream=False` | Non-streaming output, returns complete result |

> Document Update Time: 2026-05-14