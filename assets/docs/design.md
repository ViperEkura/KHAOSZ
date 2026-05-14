## 1. Why I Created This Project

There are many large language models on the market today, such as GPT, LLaMA, and others, with tens of billions or even hundreds of billions of parameters. But honestly, these models have extremely high hardware requirements, making them inaccessible for ordinary developers. I thought: **Can we create a model that is both useful and can run on ordinary computers?** This is also what most people currently hope for - a locally deployable AI project that achieves complete privatization while maintaining some level of intelligence.

Thus, the AstrAI project was born - 1B parameters, Chinese-English bilingual, supporting dialogue, text generation, and the training code is open source!

## 2. System Architecture

```mermaid
classDiagram
    namespace config {
        class ModelConfig {
            +int vocab_size
            +int dim
            +int n_layers
            +float norm_eps
            +int dim_ffn
            +bool tie_weight
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
            +int start_epoch
            +int start_batch
            +str ckpt_dir
            +int ckpt_interval
            +int random_seed
            +int num_workers
            +int prefetch_factor
            +bool pin_memory
            +int nprocs
            +str backend
            +str master_addr
            +str master_port
            +Callable parallel_wrapper
            +Callable state_dict_fn
            +str device_type
            +dict extra_kwargs
            +validate()
        }

    }

    namespace dataset {
        class BaseDataset {
            +int window_size
            +int stride
            +MultiSegmentFetcher fetcher
            +load(load_path)
            +__getitem__(index)
            +__len__()
        }

        class SEQDataset {
            +__getitem__(index) Dict
        }

        class SFTDataset {
            +__getitem__(index) Dict
        }

        class DPODataset {
            +__getitem__(index) Dict
        }

        class GRPODataset {
            +__getitem__(index) Dict
        }

        class BaseSegmentFetcher {
            +List[Tensor] segments
            +List[int] cum_lengths
            +int total_length
            +fetch_data(begin_idx, end_idx) Tensor
        }

        class MultiSegmentFetcher {
            +Dict multi_fetchers
            +List multi_keys
            +key_fetch(begin_idx, end_idx, keys) Dict
            +fetch_data(begin_idx, end_idx) Dict
        }

        class ResumableDistributedSampler {
            +int epoch
            +int iter
        }

        class DatasetFactory {
            +Registry _registry
            +register(name) decorator
            +create(train_type, window_size, stride) BaseDataset
            +load(train_type, load_path, window_size, stride) BaseDataset
        }
    }

    namespace serialization {
        class Checkpoint {
            +dict state_dict
            +int epoch
            +int iteration
            +save(save_dir)
            +load(save_dir) Checkpoint
        }
    }

    namespace model {
        class AutoModel {
            +ModelConfig config
            +Registry _registry
            +register(model_type) decorator
            +get_model_class(model_type) Type
            +from_pretrained(path, disable_random_init) nn.Module
            +save_pretrained(save_directory)
            +to(*args, **kwargs) Self
        }

        class Transformer {
            +ModelConfig config
            +RotaryEmbedding rotary_embedding
            +Embedding embed_tokens
            +ModuleList layers
            +RMSNorm norm
            +Linear lm_head
            +forward(input_ids, input_mask, paged_cache, position_ids) Tensor
            +load_state_dict(state_dict)
            +state_dict()
        }

        class DecoderBlock {
            +GQA attention
            +RMSNorm input_norm
            +MLP mlp
            +RMSNorm post_attention_norm
            +forward(x, rotary_emb, attention_mask, position_ids, paged_cache) Tensor
        }

        class GQA {
            +int n_heads
            +int n_kv_heads
            +int head_dim
            +Linear q_proj, k_proj, v_proj, o_proj
            +RMSNorm q_norm, k_norm
            +forward(x, rotary_emb, attn_mask, position_ids, paged_cache) Tensor
        }

        class MLA {
            +int n_heads
            +int n_kv_heads
            +int head_dim
            +int kv_lora_rank
            +int qk_nope_head_dim
            +int qk_rope_head_dim
            +Linear q_proj, kv_a_proj, kv_b_proj
            +Linear o_proj
            +RMSNorm kv_norm
            +forward(x, rotary_emb, attn_mask, position_ids, paged_cache) Tensor
        }

        class MLP {
            +Linear up, gate, down
            +forward(x) Tensor
        }

        class RMSNorm {
            +Parameter weight
            +float norm_eps
            +forward(x) Tensor
        }

        class Linear {
            +Parameter weight
            +Parameter bias
            +forward(x) Tensor
        }

        class RotaryEmbedding {
            +int dim
            +int max_len
            +float base
            +forward(x, position_ids=None) Tuple[Tensor, Tensor]
        }

        class Embedding {
            +Parameter weight
            +forward(x) Tensor
        }
    }

    namespace tokenize {
        class AutoTokenizer {
            +List[int] stop_ids
            +int bos_id
            +int eos_id
            +int pad_id
            +vocab_size int
            +encode(tokens, out_ids, add_special_tokens) List[int]
            +decode(tokens, skip_special_tokens) str
            +apply_chat_template(messages, tokenize) Union[str, List[int]]
            +set_chat_template(template)
            +load(path)
            +from_pretrained(path) AutoTokenizer
            +save_pretrained(save_path)
        }

        class ChatTemplate {
            +String template_str
            +render(messages, system_prompt, **extra_variables) str
            +from_string(template) ChatTemplate
        }
    }

    namespace factory {
        class Registry {
            +Dict _entries
            +register(name, component_cls, category, priority)
            +get(name) Type
            +list_names() List[str]
        }

        class BaseFactory {
            +Registry _registry
            +register(name, category, priority) decorator
            +create(name, *args, **kwargs) T
            +list_registered() list
        }
    }

    namespace trainer {
        class Trainer {
            +TrainConfig train_config
            +List[TrainCallback] callbacks
            +train(checkpoint)
            +_build_context(checkpoint) TrainContext
            +_get_default_callbacks() List[TrainCallback]
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
            +float loss
            +int world_size
            +int rank
        }

        class TrainContextBuilder {
            +TrainConfig config
            +with_checkpoint(checkpoint) TrainContextBuilder
            +build() TrainContext
        }

        class BaseStrategy {
            +nn.Module model
            +str device
            +compute_loss(batch) Tensor
        }

        class StrategyFactory {
            +Registry _registry
            +register(name) decorator
            +create(model, train_type, device, **kwargs) BaseStrategy
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

        class BaseScheduler {
            +get_lr() List[float]
            +step()
        }

        class SchedulerFactory {
            +Registry _registry
            +register(name) decorator
            +create(optimizer, schedule_type, **kwargs) BaseScheduler
        }

        class CosineScheduler {
            +int warmup_steps
            +int lr_decay_steps
            +float min_rate
        }

        class SGDRScheduler {
            +int warmup_steps
            +int cycle_length
            +float min_rate
            +int t_mult
        }

        class TrainCallback {
            +on_train_begin(context)
            +on_train_end(context)
            +on_epoch_begin(context)
            +on_epoch_end(context)
            +on_step_begin(context)
            +on_step_end(context)
            +on_batch_begin(context)
            +on_batch_end(context)
            +on_error(context)
        }

        class GradientClippingCallback {
            +float max_grad_norm
            +on_step_begin(context)
        }

        class SchedulerCallback {
            +on_train_begin(context)
            +on_batch_end(context)
        }

        class CheckpointCallback {
            +str save_dir
            +int interval
            +_save_checkpoint(context)
            +on_batch_end(context)
            +on_train_end(context)
            +on_error(context)
        }

        class ProgressBarCallback {
            +int num_epoch
            +on_epoch_begin(context)
            +on_batch_end(context)
            +on_epoch_end(context)
        }

        class MetricLoggerCallback {
            +str log_dir
            +int save_interval
            +on_batch_end(context)
            +on_train_end(context)
        }

        class CallbackFactory {
            +Registry _registry
            +register(name) decorator
            +create(name, **kwargs) TrainCallback
        }
    }

    namespace inference {
        class InferenceEngine {
            +nn.Module model
            +AutoTokenizer tokenizer
            +InferenceScheduler scheduler
            +int max_batch_size
            +Optional int max_seq_len
            +generate(prompt, stream, max_tokens, temperature, top_p, top_k) Union[Generator, str, List[str]]
            +generate_with_request(request) Union[Generator, str, List[str]]
            +generate_async(prompt, max_tokens, temperature, top_p, top_k) AsyncGenerator
            +get_stats() Dict
            +shutdown()
        }

        class InferenceScheduler {
            +nn.Module model
            +AutoTokenizer tokenizer
            +PagedCache page_cache
            +int max_batch_size
            +int max_seq_len
            +int max_prompt_len
            +int page_size
            +List waiting_queue
            +List active_tasks
            +add_task(prompt, max_tokens, temperature, top_p, top_k, stream_callback) str
            +remove_task(task_id)
            +start()
            +stop()
            +get_stats() Dict
        }

        class PagedCache {
            +int page_size
            +Tensor k_cache
            +Tensor v_cache
            +alloc_n(n) List[int]
            +free(idx)
            +bind(page_table, total_len) CacheView
            +write(layer_id, page_table, position_ids, k, v)
            +gather(layer_id, page_table, total_len) Tuple[Tensor, Tensor]
        }

        class CacheView {
            +PagedCache _cache
            +Tensor _page_table
            +int _total_len
            +write(layer_id, position_ids, k, v)
            +gather(layer_id) Tuple[Tensor, Tensor]
        }

        class Task {
            +str task_id
            +List prompt_ids
            +int max_tokens
            +float temperature
            +float top_p
            +int top_k
            +TaskStatus status
            +List output_ids
            +int input_tokens
            +int output_tokens
            +List[int] page_table
            +int n_pages
            +float arrival_time
            +float finish_time
            +Callable stream_callback
            +int next_pos
            +is_finished(stop_ids) bool
        }

        class TaskStatus {
            <<enumeration>>
            PENDING
            RUNNING
            FINISHED
            ABORTED
        }

        class GenerationRequest {
            +List[Dict] messages
            +GenerationParams params
            +bool stream
        }

        class GenerationParams {
            <<value object>>
            +int top_k
            +float top_p
            +float temperature
            +int max_tokens
        }

        class BaseSamplingStrategy {
            <<abstract>>
            +apply(logits, filter_value) Tensor
        }

        class TemperatureStrategy {
            +float temperature
            +apply(logits, filter_value) Tensor
        }

        class TopKStrategy {
            +int top_k
            +apply(logits, filter_value) Tensor
        }

        class TopPStrategy {
            +float top_p
            +apply(logits, filter_value) Tensor
        }

        class SamplingPipeline {
            +List strategies
            +apply(logits, filter_value) Tensor
            +sample(logits, filter_value) Tensor
        }

        class GenerateResult {
            +List[str] tokens
            +List[str] results
            +List[bool] _done
            +append(token, idx)
            +get_results() List[str]
            +pop_all() List[str]
            +wait(timeout) bool
            +wait_completion()
        }

        class ChatMessage {
            +str role
            +str content
        }

        class ChatCompletionRequest {
            +List[ChatMessage] messages
            +float temperature
            +float top_p
            +int top_k
            +int max_tokens
            +bool stream
            +Optional[str] stop
            +Optional[int] n
        }
    }

    namespace parallel {
        class ParallelFunctions {
            +spawn_parallel_fn(fn, nprocs)
            +setup_parallel(rank, world_size, backend, master_addr, master_port, device_type)
        }

        class ParallelModel {
            +dist.ProcessGroup process_group
            +int rank
            +int world_size
        }

        class ColumnParallelLinear {
            +forward(x) Tensor
        }

        class RowParallelLinear {
            +forward(x) Tensor
        }
    }

    %% Relationships
    TrainConfig --> ModelConfig : uses
    TrainConfig --> BaseDataset : uses
    TrainConfig --> StrategyFactory : selects
    StrategyFactory ..> BaseStrategy : creates
    BaseStrategy <|-- SEQStrategy
    BaseStrategy <|-- SFTStrategy
    BaseStrategy <|-- DPOStrategy
    BaseStrategy <|-- GRPOStrategy
    DPOStrategy --> Transformer : uses
    GRPOStrategy --> Transformer : uses
    Trainer --> TrainConfig : configures
    Trainer --> TrainContextBuilder : builds
    Trainer --> TrainCallback : manages
    TrainContextBuilder --> TrainContext : creates
    Checkpoint ..> Checkpoint : saves/loads
    TrainContext --> Checkpoint : manages
    TrainContext --> BaseStrategy : uses
    TrainContext --> BaseScheduler : uses
    SchedulerFactory ..> BaseScheduler : creates
    BaseScheduler <|-- CosineScheduler
    BaseScheduler <|-- SGDRScheduler
    CallbackFactory ..> TrainCallback : creates
    TrainCallback <|-- GradientClippingCallback
    TrainCallback <|-- SchedulerCallback
    TrainCallback <|-- CheckpointCallback
    TrainCallback <|-- ProgressBarCallback
    TrainCallback <|-- MetricLoggerCallback
    InferenceEngine --> InferenceScheduler : uses
    InferenceEngine --> GenerationRequest : uses
    GenerationRequest --> GenerationParams : contains
    InferenceScheduler --> Task : manages
    Task --> TaskStatus : uses
    InferenceScheduler --> TaskStatus : uses
    InferenceScheduler --> PagedCache : uses
    InferenceScheduler --> Transformer : uses
    InferenceEngine --> Transformer : uses
    InferenceEngine --> GenerateResult : uses
    BaseSamplingStrategy <|-- TemperatureStrategy
    BaseSamplingStrategy <|-- TopKStrategy
    BaseSamplingStrategy <|-- TopPStrategy
    SamplingPipeline --> BaseSamplingStrategy : composes
    BaseDataset <|-- SEQDataset
    BaseDataset <|-- SFTDataset
    BaseDataset <|-- DPODataset
    BaseDataset <|-- GRPODataset
    DatasetFactory ..> BaseDataset : creates
    MultiSegmentFetcher --> BaseSegmentFetcher : uses
    BaseDataset --> MultiSegmentFetcher : uses
    AutoModel <|-- Transformer
    AutoModel --> ModelConfig : contains
    Transformer --> DecoderBlock : uses
    Transformer --> RotaryEmbedding : uses
    Transformer --> Embedding : uses
    DecoderBlock --> GQA : uses
    DecoderBlock --> MLP : uses
    DecoderBlock --> RMSNorm : uses
    TrainContextBuilder --> ResumableDistributedSampler : creates
    ResumableDistributedSampler --> BaseDataset : samples
    ParallelModel <|-- RowParallelLinear
    ParallelModel <|-- ColumnParallelLinear
    AutoTokenizer --> ChatTemplate : uses
    TrainConfig --> DatasetFactory : selects
    TrainConfig --> SchedulerFactory : selects
    TrainConfig --> CallbackFactory : selects
    AutoModel ..> AutoTokenizer : loads with
    BaseFactory <|-- DatasetFactory
    BaseFactory <|-- StrategyFactory
    BaseFactory <|-- SchedulerFactory
    BaseFactory <|-- CallbackFactory
```

### Module Overview

| Module | Components | Description |
|--------|------------|-------------|
| **astrai.config** | ModelConfig, TrainConfig | Configuration management |
| **astrai.dataset** | BaseDataset, SEQDataset, SFTDataset, DPODataset, GRPODataset, BaseSegmentFetcher, MultiSegmentFetcher, ResumableDistributedSampler, DatasetFactory, save_h5, load_h5 | Dataset loading and management |
| **astrai.serialization** | Checkpoint | Model serialization and checkpoint management |
| **astrai.model** | AutoModel, Transformer, DecoderBlock, GQA, MLA, MLP, RMSNorm, Linear, RotaryEmbedding, Embedding | Neural network model |
| **astrai.tokenize** | AutoTokenizer, ChatTemplate | Tokenizer and chat template |
| **astrai.trainer** | Trainer, TrainContext, TrainContextBuilder, BaseStrategy, StrategyFactory, BaseScheduler, SchedulerFactory, TrainCallback, CallbackFactory | Training workflow management |
| **astrai.inference** | InferenceEngine, InferenceScheduler, PagedCache, CacheView, Task, TaskStatus, GenerationParams, GenerationRequest, BaseSamplingStrategy, TemperatureStrategy, TopKStrategy, TopPStrategy, SamplingPipeline, ChatMessage, ChatCompletionRequest | Inference service with continuous batching and paged KV cache |
| **astrai.parallel** | ParallelFunctions, ParallelModel, ColumnParallelLinear, RowParallelLinear | Distributed parallel |
| **astrai.factory** | Registry, BaseFactory | Generic component registration |

### Design Patterns

| Pattern | Classes | Purpose |
|---------|---------|---------|
| **Strategy** | `BaseStrategy`, `SEQStrategy`, `SFTStrategy`, `DPOStrategy`, `GRPOStrategy`, `StrategyFactory` | Flexible training strategy switching, supports SEQ/SFT/DPO/GRPO |
| **Builder** | `TrainContextBuilder` | Chain-building training context, step-by-step initialization of components |
| **Factory** | `StrategyFactory`, `SchedulerFactory`, `DatasetFactory`, `CallbackFactory`, `BaseFactory` | Decorator registration mechanism, dynamically create training strategies, schedulers, datasets, and callbacks |
| **Observer** | `TrainCallback`, `CallbackFactory` | Callback mechanism for training process monitoring (checkpoint, early stopping, metrics) |
| **Context** | `TrainContext` | Training process state container with model, optimizer, scheduler and checkpoint |
| **Registry** | `BaseFactory`, `Registry` | Generic component registration with category and priority support |
| **Object Pool** | `PagedCache` | Page-based KV cache with O(1) alloc/free via bitmask |
| **Strategy (Sampling)** | `BaseSamplingStrategy`, `TemperatureStrategy`, `TopKStrategy`, `TopPStrategy`, `SamplingPipeline` | Composable logit transformations with temperature, top-k, top-p |
| **Producer-Consumer** | `InferenceScheduler`, `Task`, `waiting_queue`, `active_tasks` | Continuous batching with dynamic task queue management |
| **Event-Driven** | `threading.Event`, `_task_event` | Non-blocking wait mechanism for task scheduling using Python's `threading` module |
| **AutoModel Registry** | `AutoModel`, `Transformer` | Model type registration and dynamic loading via decorator pattern |
| **Generator Pattern** | `GenerateResult`, `GenerationRequest` | Event-based result notification for streaming/non-streaming generation |

### Core Relationships

1. **Configuration → Training**: `TrainConfig` contains `ModelConfig`, holds model, dataset, optimizer and other references
2. **Training Flow**: `Trainer` → `TrainContextBuilder` → `TrainContext`, uses `BaseStrategy` to compute loss
3. **Strategy Selection**: `StrategyFactory` creates corresponding strategy instance based on `train_type`
4. **Inference Flow**: `InferenceEngine` → `InferenceScheduler` → `Transformer`, uses `PagedCache` for paged KV cache management and `SamplingPipeline` for efficient continuous batching with streaming/non-streaming
5. **Distributed Support**: `spawn_parallel_fn` and `setup_parallel` provide multi-process training capability for `Trainer`
6. **Dataset Loading**: `DatasetFactory` creates datasets (SEQDataset, SFTDataset, DPODataset, GRPODataset), supports HDF5 loading via `BaseSegmentFetcher` and `MultiSegmentFetcher`
7. **Checkpoint Management**: `Checkpoint` handles model state serialization/deserialization with safetensors
8. **Scheduler Support**: `SchedulerFactory` creates learning rate schedulers (CosineScheduler, SGDRScheduler)
9. **AutoModel Loading**: `AutoModel.from_pretrained()` dynamically loads model based on `config.json` model_type, uses `Registry` pattern for model type registration

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

**GRPO:**

GRPO (Group Relative Policy Optimization) computes advantages from multiple responses to the same prompt, then optimizes using a PPO-style clipped objective:

$$
\text{Advantage}_i = \frac{r_i - \mu}{\sigma + \epsilon}
$$

Where $r_i$ is the reward for the $i$-th response, $\mu$ and $\sigma$ are the mean and standard deviation of group rewards.

$$
L_{\text{GRPO}} = -\mathbb{E} \left[ \min\left( \frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)} \cdot A, \text{clip}\left(\frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)}, 1-\epsilon, 1+\epsilon\right) \cdot A \right) \right] + \lambda \cdot D_{KL}
$$

The KL divergence term uses mean squared error approximation:

$$
L_{KL} = \lambda \cdot \mathbb{E} \left[ (\log \pi_\theta - \log \pi_{\text{ref}})^2 \right]
$$

The final loss is the sum of both: $L = L_{\text{policy}} + L_{KL}$

Through the above three-stage progressive training, the model completes its evolution from a general language foundation to a specialized, highly-aligned dialogue intelligence.

> Document Update Time: 2026-05-14
