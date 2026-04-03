## 1. Why I Created This Project

There are many large language models on the market today, such as GPT, LLaMA, and others, with tens of billions or even hundreds of billions of parameters. But honestly, these models have extremely high hardware requirements, making them inaccessible for ordinary developers. I thought: **Can we create a model that is both useful and can run on ordinary computers?** This is also what most people currently hope for - a locally deployable AI project that achieves complete privatization while maintaining some level of intelligence.

Thus, the AstrAI project was born - 1B parameters, Chinese-English bilingual, supporting dialogue, text generation, and the training code is open source!

## 2. System Architecture

```mermaid
classDiagram
    %% Configuration Classes
    class ModelConfig {
        +int vocab_size
        +int dim
        +int n_layers
        +float norm_eps
        +int dim_ffn
        +int max_len
        +float rope_theta
        +int n_heads
        +int n_kv_heads
        +bool use_qk_norm
        +bool use_gated_attention
        +load(config_path) ModelConfig
        +save(config_path)
    }

    class TrainConfig {
        +nn.Module model
        +str strategy
        +Dataset dataset
        +Callable optimizer_fn
        +Callable scheduler_fn
        +int n_epoch
        +int batch_size
        +int accumulation_steps
        +float max_grad_norm
        +str ckpt_dir
        +int ckpt_interval
        +int nprocs
        +str backend
        +validate()
    }

    %% Data Classes
    class Dataset {
        +__len__()
        +__getitem__()
    }

    class Checkpoint {
        +dict state_dict
        +int epoch
        +int iteration
    }

    class Tokenizer {
        +encode(text) List[int]
        +decode(ids) str
    }

    %% Model Classes
    class Transformer {
        +forward(input_ids, mask) Dict
    }

    %% Trainer Classes
    class Trainer {
        +TrainConfig train_config
        +List~TrainCallback~ callbacks
        +train()
        +_build_context() TrainContext
    }

    class TrainContext {
        +nn.Module model
        +BaseStrategy strategy
        +DataLoader dataloader
        +Optimizer optimizer
        +LRScheduler scheduler
        +Checkpoint checkpoint
        +int epoch
        +int iteration
    }

    class TrainContextBuilder {
        +TrainConfig config
        +with_checkpoint(Checkpoint) TrainContextBuilder
        +with_dataloader() TrainContextBuilder
        +with_strategy() TrainContextBuilder
        +build() TrainContext
    }

    class BaseStrategy {
        +nn.Module model
        +str device
        +compute_loss(batch) Tensor
    }

    class StrategyFactory {
        +frozenset SUPPORTED_STRATEGIES
        +Dict STRATEGY_MAP
        +register(name) decorator
        +create(model, train_type, device) BaseStrategy
        +available_strategies() list
    }

    class SEQStrategy {
        +float label_smoothing
        +compute_loss(batch) Tensor
    }

    class SFTStrategy {
        +float label_smoothing
        +compute_loss(batch) Tensor
    }

    class DPOStrategy {
        +nn.Module ref_model
        +float beta
        +str reduction
        +compute_loss(batch) Tensor
    }

    class GRPOStrategy {
        +nn.Module ref_model
        +float clip_eps
        +float kl_coef
        +int group_size
        +compute_loss(batch) Tensor
    }

    class TrainCallback {
        +on_train_begin(trainer)
        +on_train_end(trainer)
        +on_epoch_begin(epoch, trainer)
        +on_epoch_end(epoch, trainer)
        +on_batch_begin(batch, trainer)
        +on_batch_end(batch, trainer)
    }

    class Schedule {
        +step()
    }

    %% Inference Classes
    class Generator {
        +generate(prompt, config) str
        +generate_batch(prompts, config) List[str]
        +stream_generate(prompt, config) Generator
    }

    class InferenceCore {
        +forward(input_ids) Dict
        +apply_sampling_strategies()
    }

    class Server {
        +start()
        +predict(request)
    }

    %% Parallel Classes
    class ParallelSetup {
        +spawn_parallel_fn(fn, nprocs)
    }

    %% Relationships
    TrainConfig --> ModelConfig : contains
    TrainConfig --> Dataset : uses
    TrainConfig --> Transformer : uses
    Trainer --> TrainConfig : configures
    Trainer --> TrainContextBuilder : builds
    Trainer --> TrainCallback : manages
    TrainContextBuilder --> TrainContext : creates
    TrainContext --> Checkpoint : manages
    StrategyFactory ..> BaseStrategy : creates
    BaseStrategy <|-- SEQStrategy
    BaseStrategy <|-- SFTStrategy
    BaseStrategy <|-- DPOStrategy
    BaseStrategy <|-- GRPOStrategy
    TrainContext --> BaseStrategy : uses
    Generator --> InferenceCore : uses
    Generator --> Transformer : uses
    Server --> Generator : uses
    ParallelSetup --> Trainer : enables
    TrainConfig --> StrategyFactory : selects
    TrainCallback <|-- CheckpointCallback
    TrainCallback <|-- MetricLoggerCallback
    TrainCallback <|-- SchedulerCallback
    TrainContext --> Schedule : uses
```

### Design Pattern Summary

| Pattern | Classes | Purpose |
|---------|---------|---------|
| **Strategy** | `BaseStrategy`, `SEQStrategy`, `SFTStrategy`, `DPOStrategy`, `GRPOStrategy`, `StrategyFactory` | Flexible training strategy switching, supports SEQ/SFT/DPO/GRPO |
| **Builder** | `TrainContextBuilder` | Chain-building training context, step-by-step initialization of components |
| **Factory** | `StrategyFactory` | Decorator registration mechanism, dynamically create training strategies |
| **Observer** | `TrainCallback` | Callback mechanism for training process monitoring (checkpoint, early stopping, metrics) |
| **Singleton** | `TrainContext` | Training process global state management |

### Core Relationships

1. **Configuration → Training**: `TrainConfig` contains `ModelConfig`, holds model, dataset, optimizer and other references
2. **Training Flow**: `Trainer` → `TrainContextBuilder` → `TrainContext`, uses `BaseStrategy` to compute loss
3. **Strategy Selection**: `StrategyFactory` creates corresponding strategy instance based on `train_type`
4. **Inference Flow**: `Server` → `Generator` → `InferenceCore` → `Transformer`
5. **Distributed Support**: `ParallelSetup` provides multi-process training capability for `Trainer`

## 3. Training Process

The common training process for large language models (LLM) typically includes three stages: **Pre-training (SEQ)**, **Supervised Fine-Tuning (SFT)**, and **Reinforcement Learning from Human Feedback (DPO/GRPO)**. This system is designed to support seamless end-to-end flow, achieving efficient switching and state management of different training stages through modular strategies.

### Core Formulas

**Pre-training (SEQ):**

$$
L_{\text{PT}} = - \sum_{t=1}^{T} \log P(x_t \mid x_{\lt t}; \theta)
$$

**SFT:**

$$
L_{\text{SFT}} = - \sum_{t=P+1}^{P+L} \log P(s_t \mid s_{\lt t}; \theta)
$$

**DPO:**

$$
L_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

Through the above three-stage progressive training, the model completes its evolution from a general language foundation to a specialized, highly-aligned dialogue intelligence.
