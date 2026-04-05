# Parameter Documentation

## Training Parameters

### Basic Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--train_type` | Training type (seq, sft, dpo, grpo) | required |
| `--model_type` | Model type for AutoModel loading (e.g., transformer) | transformer |
| `--data_root_path` | Dataset root directory | required |
| `--param_path` | Model parameters or checkpoint path | required |
| `--n_epoch` | Total training epochs | 1 |
| `--batch_size` | Batch size | 4 |
| `--accumulation_steps` | Gradient accumulation steps | 1 |

### Learning Rate Scheduling

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--warmup_steps` | Warmup steps | 1000 |
| `--max_lr` | Maximum learning rate (warmup + cosine decay) | 3e-4 |
| `--max_grad_norm` | Maximum gradient norm | 1.0 |

### Checkpoint

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--ckpt_interval` | Checkpoint save interval (iterations) | 5000 |
| `--ckpt_dir` | Checkpoint save directory | checkpoint |
| `--resume_dir` | Resume training from specified path | - |

### Optimizer Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--adamw_beta1` | AdamW beta1 | 0.9 |
| `--adamw_beta2` | AdamW beta2 | 0.95 |
| `--adamw_weight_decay` | AdamW weight decay | 0.01 |

### Data Loading

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--random_seed` | Random seed | 3407 |
| `--num_workers` | DataLoader workers | 0 |
| `--prefetch_factor` | Prefetch factor for dataloader | None |
| `--pin_memory` | Enable pin_memory | False |
| `--no_pin_memory` | Disable pin_memory | - |

### Distributed Training

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--nprocs` | Number of GPUs | 1 |
| `--device_type` | Device type (cuda/cpu) | cuda |

### Other Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--window_size` | Maximum input sequence length | model config max_len |
| `--stride` | Input sequence stride | - |
| `--dpo_beta` | DPO beta value | 0.1 |
| `--label_smoothing` | Label smoothing parameter | 0.1 |
| `--start_epoch` | Starting epoch | 0 |
| `--start_batch` | Starting batch | 0 |

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
from astrai.tokenize import Tokenizer
from astrai.inference import InferenceEngine, GenerationRequest

# Load model using AutoModel
model = AutoModel.from_pretrained("your_model_dir")

# Load tokenizer
tokenizer = Tokenizer("your_model_dir")

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