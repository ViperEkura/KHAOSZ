# AstrAI Architecture

## Class Diagram

```mermaid
classDiagram
    namespace config {
        class BaseConfig {
            +to_dict() Dict
            +from_dict(d) Self
        }

        class BaseModelConfig {
            +Optional[str] model_type
            +from_file(config_path) Self
            +to_file(config_path)
        }

        class AutoRegressiveLMConfig {
            +int vocab_size
            +int dim
            +int n_layers
            +float norm_eps
            +int dim_ffn
            +bool tie_weight
            +int max_len
            +float rope_theta
            +str attn_type
            +int n_heads
            +int n_kv_heads
            +bool use_qk_norm
            +bool use_gated_attention
            +Optional[int] kv_lora_rank
            +Optional[int] qk_nope_head_dim
            +Optional[int] qk_rope_head_dim
            +str ffn_type
            +int n_routed_experts
            +int n_shared_experts
            +int n_activated_experts
            +Optional[str] topk_method
        }

        class EncoderConfig {
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
            +Optional[str] pooling_type
            +Optional[bool] normalize_embeddings
        }

        class ConfigFactory {
            +Registry _registry
            +register(name) decorator
            +load(raw) BaseConfig
        }

        class TrainConfig {
            +nn.Module model
            +str strategy
            +Dataset dataset
            +Callable optimizer_fn
            +Callable scheduler_fn
            +int n_epoch
            +int batch_per_device
            +int grad_accum_steps
            +float max_grad_norm
            +list gradient_checkpointing_modules
            +int start_epoch
            +int start_batch
            +str ckpt_dir
            +int ckpt_interval
            +int random_seed
            +int num_workers
            +Optional[int] prefetch_factor
            +bool pin_memory
            +int nprocs
            +str backend
            +str master_addr
            +str master_port
            +Callable parallel_wrapper
            +Callable state_dict_fn
            +str start_method
            +str device_type
            +Optional[Dataset] val_dataset
            +int val_step
            +dict extra_kwargs
            +validate()
        }

    }

    namespace dataset {
        class BaseDataset {
            +int window_size
            +int stride
            +Optional[BaseStorage] storage
            +load(load_path, storage_type, tokenizer)
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

        class BaseStorage {
            +MultiSegmentFetcher _fetcher
            +keys (property)
            +load(load_path, tokenizer)
            +fetch(begin, end, keys)
            +__len__()
        }

        class H5Storage {
            +load(load_path, tokenizer)
            +fetch(begin, end, keys) Dict
            +keys() List
        }

        class JSONStorage {
            +load(load_path, tokenizer)
            +fetch(begin, end, keys) Dict
            +keys() List
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

        class StorageFactory {
            +Registry _registry
            +register(name) decorator
            +create(storage_type) BaseStorage
        }

        class DatasetFactory {
            +Registry _registry
            +register(name) decorator
            +create(train_type, window_size, stride) BaseDataset
            +load(train_type, load_path, window_size, stride, storage_type, tokenizer) BaseDataset
        }
    }

    namespace serialization {
        class Checkpoint {
            +dict state_dict
            +int epoch
            +int iteration
            +dict extra
            +dict meta
            +save(save_dir)
            +load(save_dir) Checkpoint
        }
    }

    namespace model {
        class AutoModel {
            +BaseModelConfig config
            +Registry _registry
            +register(model_type) decorator
            +get_component_class(model_type) Type
            +from_pretrained(path, disable_random_init, strict) nn.Module
            +save_pretrained(save_directory)
            +to(*args, **kwargs) Self
        }

        class AutoRegressiveLM {
            +AutoRegressiveLMConfig config
            +RotaryEmbedding rotary_embedding
            +Embedding embed_tokens
            +ModuleList layers
            +RMSNorm norm
            +Linear lm_head
            +forward(input_ids, input_mask, paged_cache, position_ids) Dict[str, Tensor]
            +load_state_dict(state_dict)
            +state_dict()
        }

        class EmbeddingEncoder {
            +EncoderConfig config
            +RotaryEmbedding rotary_embedding
            +Embedding embed_tokens
            +ModuleList layers
            +RMSNorm norm
            +str pooling_type
            +bool normalize_embeddings
            +forward(input_ids, input_mask, position_ids) Tensor
            +load_state_dict(state_dict)
        }

        class DecoderBlock {
            +nn.Module attention  # GQA or MLA via AttnFactory
            +RMSNorm input_norm
            +nn.Module mlp        # MLP or DeepSeekMoE via FFNFactory
            +RMSNorm post_attention_norm
            +forward(x, rotary_emb, attention_mask, paged_cache) Tensor
        }

        class GQA {
            +int n_heads
            +int n_kv_heads
            +int head_dim
            +int n_rep
            +int layer_id
            +bool use_qk_norm
            +bool use_gated_attention
            +Linear q_proj, k_proj, v_proj, o_proj
            +Linear gate  # only if use_gated_attention
            +RMSNorm q_norm, k_norm  # only if use_qk_norm
            +forward(x, rotary_emb, attn_mask, paged_cache) Tensor
        }

        class MLA {
            +int n_heads
            +int n_kv_heads
            +int head_dim
            +int kv_lora_rank
            +int qk_nope_head_dim
            +int qk_rope_head_dim
            +int n_rep
            +int layer_id
            +bool use_gated_attention
            +Linear q_proj, kv_a_proj, kv_b_proj
            +Linear o_proj
            +Linear gate  # only if use_gated_attention
            +RMSNorm kv_norm
            +forward(x, rotary_emb, attn_mask, paged_cache) Tensor
        }

        class MLP {
            +Linear up, gate, down
            +forward(x) Tensor
        }

        class DeepSeekMoE {
            +int dim
            +int n_routed_experts
            +int n_shared_experts
            +int n_activated_experts
            +str topk_method
            +Linear router
            +ModuleList shared_experts
            +ModuleList routed_experts
            +forward(x) Tensor
        }

        class AttnFactory {
            +create(attn_type, **kwargs) nn.Module
        }

        class FFNFactory {
            +create(ffn_type, dim, dim_ffn, **kwargs) nn.Module
        }

        class RMSNorm {
            +Parameter weight
            +float norm_eps
            +tuple normalized_shape
            +forward(x) Tensor
        }

        class Linear {
            +Parameter weight
            +Optional[Parameter] bias  # only if bias=True
            +forward(x) Tensor
        }

        class RotaryEmbedding {
            +int dim
            +int max_len
            +float base
            +forward(x, position_ids=None) Tensor
        }

        class Embedding {
            +Parameter weight
            +forward(x) Tensor
        }
    }

    namespace tokenize {
        class AutoTokenizer {
            +vocab_size int
            +encode(tokens, out_ids, add_special_tokens) List[int]
            +decode(tokens, skip_special_tokens) str
            +__getattr__(name) Any (bos_id, eos_id, pad_id, stop_ids)
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
            +_get_default_callbacks() List[TrainCallback]
        }

        class TrainContext {
            +nn.Module model
            +BaseStrategy strategy
            +DataLoader dataloader
            +Optimizer optimizer
            +LRScheduler scheduler
            +Checkpoint checkpoint
            +TrainConfig config
            +int epoch
            +int iteration
            +float loss
            +DataLoader val_dataloader
            +float val_loss
            +int world_size
            +int rank
            +dict kwargs
        }

        class TrainContextBuilder {
            +TrainConfig config
            +with_checkpoint(checkpoint) TrainContextBuilder
            +build() TrainContext
        }

        class BaseStrategy {
            +Union[Callable, nn.Module] model
            +str device
            +compute_loss(batch) Tensor
        }

        class StrategyFactory {
            +Registry _registry
            +register(name) decorator
            +create(train_type, model, device, **kwargs) BaseStrategy
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
            +str reduction
            +int sync_interval
            +compute_loss(batch) Tensor
            +sync_ref_model()
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
            <<protocol>>
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

        class GradientCheckpointingCallback {
            +tuple modules
            +on_train_begin(context)
            +on_train_end(context)
        }

        class CheckpointCallback {
            +str save_dir
            +int interval
            +_save_checkpoint(context)
            +on_train_begin(context)
            +on_batch_end(context)
            +on_train_end(context)
            +on_error(context)
            +save_extra(context)$
            +load_extra(extra, context)$
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
            +on_error(context)
        }

        class ValidationCallback {
            +_run_validation(context)
            +on_step_end(context)
        }

        class CallbackFactory {
            +Registry _registry
            +register(name) decorator
            +create(name, **kwargs) TrainCallback
        }

        class Muon {
            +float lr
            +float momentum
            +float weight_decay
            +int ns_steps
            +step(closure) Optional[float]
        }
    }

    namespace inference {
        class InferenceEngine {
            +nn.Module model
            +AutoTokenizer tokenizer
            +InferenceScheduler scheduler
            +generate(prompt, stream, max_tokens, temperature, top_p, top_k) Union[Generator, str, List[str]]
            +generate_with_request(request) Union[Generator, str, List[str]]
            +generate_async(prompt, max_tokens, temperature, top_p, top_k) AsyncGenerator
            +get_stats() Dict
            +shutdown()
        }

        class Executor {
            +AutoModel model
            +AutoTokenizer tokenizer
            +KVCache page_cache
            +execute_prefill(tasks, prompt_len, start_pos)
            +execute_decode(tasks) List[int]
        }

        class InferenceScheduler {
            +KVCache _page_cache
            +Executor _executor
            +TaskManager _task_mgr
            +bool _running
            +Thread _loop_thread
            +int max_seq_len
            +add_task(prompt, max_tokens, temperature, top_p, top_k, stream_callback) str
            +remove_task(task_id)
            +start()
            +stop()
            +get_stats() Dict
        }

        class Allocator {
            +int _free_mask
            +List[int] _refs
            +OrderedDict _lru
            +alloc() int
            +free(idx, keep_cached)
            +inc_ref(idx)
            +touch(idx)
            +ref_count(idx) int
        }

        class PrefixCache {
            +int _page_size
            +evict(page_idx)
            +has_page(idx) bool
            +lookup(token_ids) List[int]
            +record(page_idx, token_ids, logical_page_idx)
        }

        class PagePool {
            -Allocator _alloc
            -PrefixCache _prefix
            +alloc() int
            +free(idx)
            +inc_ref(idx)
            +lookup(token_ids) List[int]
            +record(page_idx, token_ids, logical_page_idx)
        }

        class Storage {
            +int page_size
            +Tensor k_cache
            +Tensor v_cache
            +write(layer_id, page_table, start_pos, k, v)
            +gather(layer_id, page_table, total_len) Tuple[Tensor, Tensor]
        }

        class KVCache {
            -PagePool _pool
            -Storage _storage
            -TaskTable _table
            +int page_size
            +task_alloc(task_id, prompt_ids) bool
            +task_free(task_id)
            +task_extend(task_id, pos) bool
            +task_cached(task_id) int
            +task_record_hashes(task_id, prompt_ids, start_logical_page)
            +make_table_tensor(task_ids, device) Tensor
            +bind(page_table, total_len) KvcacheView
        }

        class KvcacheView {
            -Storage _storage
            +Tensor _page_table
            +int _total_len
            +write(layer_id, k, v)
            +gather(layer_id) Tuple[Tensor, Tensor]
        }

        class TaskTable {
            +set(task_id, page_table, cached)
            +get(task_id) List[int]
            +get_cached(task_id) int
            +get_ref(task_id) List[int]
            +pop(task_id) Tuple[List[int], int]
            +table_tensor(task_ids, device) Tensor
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

        class TaskManager {
            +AutoTokenizer tokenizer
            +Deque waiting_queue
            +List active_tasks
            +add_task(prompt, **kwargs) str
            +remove_task(task_id) List[Task]
            +remove_finished_tasks(stop_ids) List[Task]
            +pull_candidates(n) List[Task]
            +activate(task)
            +return_to_waiting(tasks)
            +get_active_tasks() List[Task]
        }

        class GenerationRequest {
            +List[Dict] messages
            +int top_k
            +float top_p
            +float temperature
            +Optional[int] max_tokens
            +bool stream
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
            +List[Tuple[int, str]] tokens
            +List[str] results
            +List[bool] _done
            +append(token, idx)
            +get_results() List[str]
            +pop_all() List[Tuple[int, str]]
            +wait(timeout) bool
            +wait_completion(timeout)
        }

        class ChatMessage {
            +str role
            +str content
        }

        class ChatCompletionRequest {
            +str model
            +List[ChatMessage] messages
            +float temperature
            +float top_p
            +int top_k
            +int max_tokens
            +bool stream
            +Optional[Union[str, List[str]]] stop
            +Optional[int] n
            +Optional[float] presence_penalty
            +Optional[float] frequency_penalty
            +Optional[Dict] logit_bias
            +Optional[str] user
        }

        class AnthropicMessage {
            +str role
            +Union[str, List[Dict]] content
        }

        class MessagesRequest {
            +str model
            +List[AnthropicMessage] messages
            +Optional[str] system
            +float temperature
            +float top_p
            +int top_k
            +int max_tokens
            +bool stream
            +Optional[List[str]] stop_sequences
        }

        class ProtocolHandler {
            <<abstract>>
            +request
            +engine
            +build_prompt() str
            +create_response_id() str
            +get_stop_sequences() List[str]
            +create_stop_checker() StopChecker
            +on_token(ctx, token, stop_checker) Optional[str]
            +format_stream_start(ctx) List[str]
            +format_stream_token(ctx, token) str
            +format_stream_end(ctx) List[str]
            +format_non_stream_response(ctx, content) Dict
            +handle() Union[StreamingResponse, Dict]
        }

        class OpenAIHandler {
            +build_prompt() str
            +create_response_id() str
        }

        class AnthropicHandler {
            +build_prompt() str
            +create_response_id() str
            +on_token(ctx, token, stop_checker) Optional[str]
        }

        class StopChecker {
            +has_sequences (property) bool
            +check(text) Optional[str]
            +trim(text, matched) str
        }

        class StreamContext {
            +str resp_id
            +int created
            +str model
            +int prompt_tokens
            +int completion_tokens
            +str accumulated
            +Optional[str] stop_matched
            +str last_yield_trimmed
        }

        class app {
            <<singleton>>
            +FastAPI app
        }
    }

    namespace parallel {
        class Functions {
            <<module>>
            +spawn_parallel_fn(func, world_size, backend, master_addr, master_port, device_type, start_method, **kwargs)
            +setup_parallel(rank, world_size, backend, master_addr, master_port, device_type)
            +get_current_device() str
            +get_world_size() int
            +get_rank() int
            +only_on_rank(rank, sync) decorator
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

    %% Relationships — UML notation: <|-- generalization, *-- composition, o-- aggregation, --> association, ..> dependency

    %% --- Generalization (inheritance) ---
    BaseStrategy <|-- SEQStrategy
    BaseStrategy <|-- SFTStrategy
    BaseStrategy <|-- DPOStrategy
    BaseStrategy <|-- GRPOStrategy
    BaseScheduler <|-- CosineScheduler
    BaseScheduler <|-- SGDRScheduler
    TrainCallback <|-- GradientClippingCallback
    TrainCallback <|-- GradientCheckpointingCallback
    TrainCallback <|-- CheckpointCallback
    TrainCallback <|-- ProgressBarCallback
    TrainCallback <|-- MetricLoggerCallback
    BaseDataset <|-- SEQDataset
    BaseDataset <|-- SFTDataset
    BaseDataset <|-- DPODataset
    BaseDataset <|-- GRPODataset
    BaseStorage <|-- H5Storage
    BaseStorage <|-- JSONStorage
    BaseSamplingStrategy <|-- TemperatureStrategy
    BaseSamplingStrategy <|-- TopKStrategy
    BaseSamplingStrategy <|-- TopPStrategy
    BaseSamplingStrategy <|-- SamplingPipeline
    ParallelModel <|-- RowParallelLinear
    ParallelModel <|-- ColumnParallelLinear
    AutoModel <|-- AutoRegressiveLM
    AutoModel <|-- EmbeddingEncoder
    BaseConfig <|-- BaseModelConfig
    BaseConfig <|-- TrainConfig
    BaseModelConfig <|-- AutoRegressiveLMConfig
    BaseModelConfig <|-- EncoderConfig
    BaseFactory <|-- AutoModel
    BaseFactory <|-- AttnFactory
    BaseFactory <|-- FFNFactory
    BaseFactory <|-- DatasetFactory
    BaseFactory <|-- StrategyFactory
    BaseFactory <|-- SchedulerFactory
    BaseFactory <|-- CallbackFactory
    BaseFactory <|-- StorageFactory
    BaseFactory <|-- ConfigFactory
    TrainCallback <|-- ValidationCallback
    ProtocolHandler <|-- OpenAIHandler
    ProtocolHandler <|-- AnthropicHandler

    %% --- Composition (strong ownership, part destroyed with whole) ---
    KVCache *-- PagePool
    KVCache *-- Storage
    KVCache *-- TaskTable
    PagePool *-- Allocator
    PagePool *-- PrefixCache
    InferenceEngine *-- InferenceScheduler
    InferenceScheduler *-- KVCache
    InferenceScheduler *-- Executor
    InferenceScheduler *-- TaskManager
    SamplingPipeline *-- BaseSamplingStrategy
    AutoRegressiveLM *-- DecoderBlock
    AutoRegressiveLM *-- RotaryEmbedding
    AutoRegressiveLM *-- Embedding
    EmbeddingEncoder *-- DecoderBlock
    EmbeddingEncoder *-- RotaryEmbedding
    EmbeddingEncoder *-- Embedding
    DecoderBlock *-- RMSNorm
    BaseDataset o-- BaseStorage
    ChatCompletionRequest *-- ChatMessage
    MessagesRequest *-- AnthropicMessage

    %% --- Aggregation (weak ownership) ---
    AutoModel o-- BaseModelConfig
    Trainer o-- TrainCallback
    TrainContext o-- BaseStrategy
    TrainContext o-- BaseScheduler
    TrainContext o-- Checkpoint
    AutoTokenizer o-- ChatTemplate
    KvcacheView o-- Storage
    BaseFactory o-- Registry

    %% --- Dependency (uses temporarily) ---
    TrainConfig ..> BaseStrategy : selects
    StrategyFactory ..> BaseStrategy : creates
    SchedulerFactory ..> BaseScheduler : creates
    DatasetFactory ..> BaseDataset : creates
    CallbackFactory ..> TrainCallback : creates
    AttnFactory ..> GQA : creates
    AttnFactory ..> MLA : creates
    FFNFactory ..> MLP : creates
    FFNFactory ..> DeepSeekMoE : creates
    DecoderBlock ..> AttnFactory : uses
    DecoderBlock ..> FFNFactory : uses
    StorageFactory ..> H5Storage : creates
    StorageFactory ..> JSONStorage : creates
    ConfigFactory ..> AutoRegressiveLMConfig : creates
    ConfigFactory ..> EncoderConfig : creates
    Trainer ..> TrainContextBuilder : uses
    TrainContextBuilder ..> TrainContext : creates
    Trainer ..> Functions : spawns
    TrainContextBuilder ..> StrategyFactory : uses
    TrainContextBuilder ..> ResumableDistributedSampler : creates
    Checkpoint ..> Checkpoint : serializes
    CheckpointCallback ..> Checkpoint : creates
    KVCache ..> KvcacheView : binds
    InferenceEngine ..> GenerationRequest : uses
    InferenceEngine ..> GenerateResult : creates
    OpenAIHandler ..> ChatCompletionRequest : receives
    AnthropicHandler ..> MessagesRequest : receives
    ProtocolHandler ..> StopChecker : creates
    ProtocolHandler ..> StreamContext : creates

    %% --- Association (general usage) ---
    Trainer --> TrainConfig
    DPOStrategy --> AutoModel
    GRPOStrategy --> AutoModel
    InferenceScheduler --> Task
    InferenceScheduler --> TaskStatus
    Task --> TaskStatus
    InferenceEngine --> AutoModel
    Executor --> AutoModel
    Executor --> AutoTokenizer
    TaskManager --> AutoTokenizer
    MultiSegmentFetcher --> BaseSegmentFetcher
    ResumableDistributedSampler --> BaseDataset

```


## Module Overview

| Module | Components | Description |
|--------|------------|-------------|
| **astrai.config** | BaseConfig, BaseModelConfig, AutoRegressiveLMConfig, EncoderConfig, ConfigFactory, TrainConfig | Configuration management (to_dict/from_dict, to_file/from_file) |
| **astrai.dataset** | BaseDataset–GRPODataset, BaseStorage–JSONStorage, StorageFactory, BaseSegmentFetcher, MultiSegmentFetcher, ResumableDistributedSampler, DatasetFactory | Dataset loading and management |
| **astrai.serialization** | Checkpoint | Model serialization |
| **astrai.model** | AutoModel, AutoRegressiveLM, EmbeddingEncoder, DecoderBlock, GQA, MLA, MLP, DeepSeekMoE, AttnFactory, FFNFactory, RMSNorm, Linear, RotaryEmbedding, Embedding | Neural network model |
| **astrai.tokenize** | AutoTokenizer, ChatTemplate | Tokenizer and chat template |
| **astrai.trainer** | Trainer, TrainContext, TrainContextBuilder, BaseStrategy–GRPOStrategy, StrategyFactory, BaseScheduler–SGDRScheduler, SchedulerFactory, TrainCallback(Protocol)–ValidationCallback, CallbackFactory, Muon | Training workflow |
| **astrai.inference** | InferenceEngine, InferenceScheduler, Executor, KVCache–KvcacheView, Allocator–Storage, Task, TaskManager, TaskStatus, GenerationRequest, BaseSamplingStrategy–SamplingPipeline, ProtocolHandler–AnthropicHandler, ChatMessage–MessagesRequest, app | Inference service |
| **astrai.parallel** | spawn_parallel_fn, setup_parallel, get_rank/get_world_size/get_current_device, only_on_rank, ParallelModel, RowParallelLinear, ColumnParallelLinear | Distributed parallel |
| **astrai.factory** | Registry, BaseFactory[T] | Component registration |

## Design Patterns

| Pattern | Classes | Purpose |
|---------|---------|---------|
| **Factory** | `AttnFactory`, `FFNFactory`, `StrategyFactory`, `DatasetFactory`, `SchedulerFactory`, `CallbackFactory`, `StorageFactory`, `ConfigFactory` | Decorator-based component creation |
| **Registry** | `BaseFactory`, `Registry` | Component registration with category/priority |
| **Strategy** | `SEQStrategy`, `SFTStrategy`, `DPOStrategy`, `GRPOStrategy` | Training strategy switching |
| **Strategy (Sampling)** | `TemperatureStrategy`, `TopKStrategy`, `TopPStrategy`, `SamplingPipeline` | Composable logit transformations |
| **Template Method** | `ProtocolHandler`, `OpenAIHandler`, `AnthropicHandler` | HTTP API handler with format hooks |
| **Builder** | `TrainContextBuilder` | Chain-building training context |
| **Observer** | `TrainCallback`, callback implementations | Training process monitoring |
| **Context** | `TrainContext` | Unified training state bag |
| **Object Pool** | `Allocator`, `PagePool` | Page-based KV cache with LRU eviction |
| **Storage** | `BaseStorage`, `H5Storage`, `JSONStorage` | Format-agnostic data access |
| **Producer-Consumer** | `InferenceScheduler`, `Task`, queues | Continuous batching |
| **AutoModel Registry** | `AutoModel`, `AutoRegressiveLM`, `EmbeddingEncoder` | Model-type dynamic loading |

## Core Relationships

1. **Config → Training**: `TrainConfig` holds model, dataset, optimizer_fn, scheduler_fn
2. **Training Flow**: `Trainer` → `TrainContextBuilder` → `TrainContext`, uses `BaseStrategy` for loss
3. **Strategy Selection**: `StrategyFactory` creates strategy by `train_type`
4. **Inference Flow**: `InferenceEngine` → `InferenceScheduler` → `AutoRegressiveLM`, backed by `KVCache` + `SamplingPipeline`
5. **Distributed**: `spawn_parallel_fn` + `setup_parallel` for multi-process DDP
6. **Dataset Loading**: `DatasetFactory` creates datasets, `BaseStorage` (H5Storage/JSONStorage) loads via `BaseSegmentFetcher` + `MultiSegmentFetcher`
7. **Checkpoint**: `Checkpoint` saves/loads safetensors + metadata (rank-0 only)
8. **Scheduler**: `SchedulerFactory` creates `CosineScheduler`/`SGDRScheduler`
9. **AutoModel**: `from_pretrained()` loads `config.json` + `model.safetensors`, `_disable_random_init` replaces `nn.init.*` with no-ops

> Document Update Time: 2026-05-17
