# Parameter Documentation

## Training Parameters

### Basic Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--train_type` | Training type (seq, sft, dpo) | required |
| `--data_root_path` | Dataset root directory | required |
| `--param_path` | Model parameters or checkpoint path | required |
| `--n_epoch` | Total training epochs | 1 |
| `--batch_size` | Batch size | 1 |
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
| `--num_workers` | DataLoader workers | 4 |
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
| `query` | Input text or text list | required |
| `history` | Conversation history | None |
| `system_prompt` | System prompt | None |
| `temperature` | Sampling temperature (higher = more random) | required |
| `top_p` | Nucleus sampling threshold | required |
| `top_k` | Top-k sampling count | required |
| `max_len` | Maximum generation length | model config max_len |
| `stream` | Whether to stream output | False |

### Usage Example

```python
from astrai.config import ModelParameter
from astrai.inference import InferenceEngine, GenerationRequest

# Load model
param = ModelParameter.load("your_model_dir")
param.to(device="cuda", dtype=torch.bfloat16)

# Create engine with separate model and tokenizer
engine = InferenceEngine(
    model=param.model,
    tokenizer=param.tokenizer,
    config=param.config,
)

# Build request
request = GenerationRequest(
    query="Hello",
    history=[],
    temperature=0.8,
    top_p=0.95,
    top_k=50,
)

# Generate (streaming)
for token in engine.generate_with_request(request):
    print(token, end="", flush=True)
```

### Generation Modes

| Mode | Description |
|------|-------------|
| `stream=True` | Streaming output, yields token by token |
| `stream=False` | Non-streaming output, returns complete result |