# AstrAI Architecture

## Class Diagram

```mermaid
classDiagram
    namespace config {
        class BaseConfig {
            +to_dict() Dict
            +from_dict(d) Self
            +from_json(path) Self
            +to_json(path)
        }

        class BaseModelConfig {
            +Optional[str] model_type
            +from_file(config_path) Self
            +to_file(config_path)
        }

        class AutoRegressiveLMConfig {
            +Optional[int] vocab_size
            +Optional[int] dim
            +Optional[int] n_layers
            +Optional[float] norm_eps
            +Optional[int] dim_ffn
            +Optional[bool] tie_weight
            +Optional[dict] rope_scaling
            +Optional[int] max_len
            +Optional[float] rope_theta
            +str attn_type
            +Optional[int] n_heads
            +Optional[int] n_kv_heads
            +Optional[bool] use_qk_norm
            +Optional[bool] use_gated_attention
            +Optional[int] kv_lora_rank
            +Optional[int] qk_nope_head_dim
            +Optional[int] qk_rope_head_dim
            +str ffn_type
            +Optional[int] n_routed_experts
            +Optional[int] n_shared_experts
            +Optional[int] n_activated_experts
            +Optional[str] topk_method
        }

        class EncoderConfig {
            +Optional[int] vocab_size
            +Optional[int] dim
            +Optional[int] n_layers
            +Optional[float] norm_eps
            +Optional[int] dim_ffn
            +Optional[int] max_len
            +Optional[float] rope_theta
            +Optional[int] n_heads
            +Optional[int] n_kv_heads
            +Optional[bool] use_qk_norm
            +Optional[bool] use_gated_attention
            +Optional[dict] rope_scaling
            +Optional[str] pooling_type
            +Optional[bool] normalize_embeddings
        }

        class ConfigFactory {
            +Registry _registry
            +register(name) decorator
            +load(raw) BaseConfig
        }

        class InputConfig {
            +str type
            +str messages_key
            +str prompt_key
            +str response_key
            +str text_key
        }

        class ProcessingConfig {
            +int max_seq_len
            +int min_chars
            +int max_chars
            +bool deduplicate
            +Optional[int] max_items
        }

        class OutputConfig {
            +Optional[str] domain_key
            +str storage_format
            +int max_tokens_per_shard
        }

        class PipelineConfig {
            +int version
            +InputConfig input
            +dict mask
            +str mask_default
            +ProcessingConfig preprocessing
            +OutputConfig output
            +from_dict(d) Self
        }

        class TrainConfig {
            +Callable[[], nn.Module] model_fn
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
            +str log_dir
            +int log_interval
            +List[str] metrics
            +Optional[LoRAConfig] lora
            +int random_seed
            +int num_workers
            +Optional[int] prefetch_factor
            +bool pin_memory
            +int nprocs
            +str backend
            +str master_addr
            +str master_port
            +str start_method
            +str device_type
            +Optional[Dataset] val_dataset
            +int val_step
            +str parallel_mode
            +dict executor_kwargs
            +dict extra_kwargs
            +validate()
        }

    }

    namespace dataset {
        class BaseDataset {
            +int window_size
            +int stride
            +Optional[Store] storage
            +load(load_path, storage_type)
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

        class Store {
            +Dict[str, List[Tensor]] _data
            +Dict[str, List[int]] _cum
            +int _length
            +keys (property)
            +load(path)
            +fetch(begin, end, keys)
            +__len__()
            -_fetch_key(key, begin, end) Tensor
            -_normalize(raw)
        }

        class H5Store {
            +load(path)
        }

        class MmapStore {
            +List _mmap_refs
            +load(path)
        }

        class ResumableDistributedSampler {
            +int epoch
            +int iter
        }

        class StoreFactory {
            +Registry _registry
            +register(name) decorator
            +create(storage_type) Store
        }

        class DatasetFactory {
            +Registry _registry
            +register(name) decorator
            +create(train_type, window_size, stride) BaseDataset
            +load(train_type, load_path, window_size, stride, storage_type) BaseDataset
        }
    }

    namespace serialization {
        class Checkpoint {
            +dict state_dict
            +int epoch
            +int iteration
            +dict extra
            +dict meta
            +dict config
            +save(save_dir)
            +load(save_dir, broadcast) Checkpoint
        }
    }

    namespace model {
        class AutoModel {
            +BaseModelConfig config
            +Registry _registry
            +register(name) decorator
            +get_component_class(name) Type
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
            +load_state_dict(state_dict, strict, assign)
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
            +int dim
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
            +int dim
            +int n_heads
            +int n_kv_heads
            +int head_dim
            +int kv_lora_rank
            +int qk_nope_head_dim
            +int qk_rope_head_dim
            +int n_rep
            +int layer_id
            +bool use_qk_norm
            +bool use_gated_attention
            +Linear q_proj, kv_a_proj, kv_b_proj
            +Linear o_proj
            +Linear gate  # only if use_gated_attention
            +RMSNorm kv_norm
            +RMSNorm q_norm, k_norm  # only if use_qk_norm
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
            +Optional[Dict] rope_scaling
            +forward(x, position_ids=None) Tensor
        }

        class Embedding {
            +Parameter weight
            +forward(x) Tensor
        }
    }

    namespace preprocessing {
        class BaseMaskBuilder {
            <<abstract>>
            +build(item, config, tokenizer) Optional[dict]
        }

        class ChatMaskBuilder {
            +build(item, config, tokenizer) Optional[dict]
        }

        class InstructionMaskBuilder {
            +build(item, config, tokenizer) Optional[dict]
        }

        class TextMaskBuilder {
            +build(item, config, tokenizer) Optional[dict]
        }
    }

    namespace tokenize {
        class AutoTokenizer {
            +vocab_size int
            +encode(tokens, out_ids, is_pretokenized, add_special_tokens) List
            +decode(tokens, skip_special_tokens) str
            +__getattr__(name) Any (bos_id, eos_id, pad_id, stop_ids)
            +apply_chat_template(messages, system_prompt, tokenize, add_generation_prompt) Union[str, List[int]]
            +set_chat_template(template)
            +load(path)
            +from_pretrained(path) AutoTokenizer
            +save_pretrained(save_path)
        }

        class ChatTemplate {
            +str template_str
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

        class MaskBuilderFactory {
            +Registry _registry
            +register(name) decorator
            +create(input_type, config, tokenizer) BaseMaskBuilder
        }
    }

    namespace trainer {
        class Trainer {
            +TrainConfig train_config
            +List[TrainCallback] callbacks
            +train(resume_dir)
            -_get_default_callbacks() List[TrainCallback]
        }

        class TrainContext {
            +nn.Module model
            +BaseStrategy strategy
            +DataLoader dataloader
            +OptimizerProtocol optimizer
            +SchedulerProtocol scheduler
            +Checkpoint checkpoint
            +TrainConfig config
            +dict model_config
            +BaseExecutor executor
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
            +with_resume_dir(resume_dir) TrainContextBuilder
            +build() TrainContext
        }

        class BaseStrategy {
            +Callable model
            +Optional[BaseExecutor] executor
            +Optional[Callable] model_fn
            +dict extra_kwargs
            +str device
            +__call__(batch) Tensor
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
            +state_dict() dict
            +load_state_dict(d)
        }

        class SchedulerFactory {
            +Registry _registry
            +register(name) decorator
            +create(optimizer, schedule_type, **kwargs) BaseScheduler
        }

        class CosineScheduler {
            +int warmup_steps
            +int lr_decay_steps
            +int total_steps
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
            +on_batch_begin(context)
            +on_batch_end(context)
            +on_optimizer_step(context)
            +on_error(context)
        }

        class GradientClippingCallback {
            +float max_grad_norm
            +on_optimizer_step(context)
        }

        class GradientCheckpointingCallback {
            +tuple modules
            +on_train_begin(context)
            +on_train_end(context)
        }

        class CheckpointCallback {
            +str save_dir
            +int interval
            +bool weight_only
            +Callable save_extra_fn
            -_save_checkpoint(context)
            +on_batch_end(context)
            +on_train_end(context)
            +on_error(context)
            +save_extra(context) dict$
        }

        class ProgressBarCallback {
            +int num_epoch
            +int log_interval
            +IO file
            +on_epoch_begin(context)
            +on_batch_end(context)
            +on_epoch_end(context)
        }

        class MetricLoggerCallback {
            +Path log_dir
            +int save_interval
            +int log_interval
            +List[str] metrics
            +on_batch_end(context)
            +on_train_end(context)
            +on_error(context)
        }

        class ValidationCallback {
            -_run_validation(context)
            +on_optimizer_step(context)
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
            +bool nesterov
            +int ns_steps
            +Optional[float] adamw_lr
            +tuple adamw_betas
            +float adamw_eps
            +float adamw_wd
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
            +Optional[str] device
            +Optional[torch.dtype] dtype
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
            +str device
            +torch.dtype dtype
            +add_task(prompt, **kwargs) str
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
            +Optional[int] max_tokens
            +float temperature
            +float top_p
            +int top_k
            +TaskStatus status
            +List output_ids
            +int input_tokens
            +int output_tokens
            +float arrival_time
            +Optional[float] finish_time
            +Optional[Callable] stream_callback
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
            +int max_batch_size
            +int max_seq_len
            +int max_prompt_len
            +Deque waiting_queue
            +List active_tasks
            +add_task(prompt, max_tokens, temperature, top_p, top_k, stream_callback) str
            +remove_task(task_id) List[Task]
            +remove_finished_tasks(stop_ids) List[Task]
            +pull_candidates(n) List[Task]
            +activate(task)
            +return_to_waiting(tasks)
            +get_active_tasks() List[Task]
            +has_work() bool
            +wait_for_tasks(timeout)
            +get_waiting_tasks() List[Task]
            +clear_queues()
            +wake()
            +get_stats() Dict
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
            +List[BaseSamplingStrategy] strategies
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
            +Optional[float] temperature
            +Optional[float] top_p
            +Optional[int] top_k
            +Optional[int] max_tokens
            +Optional[bool] stream
            +Optional[Union[str, List[str]]] stop
            +Optional[int] n
            +Optional[float] presence_penalty
            +Optional[float] frequency_penalty
            +Optional[Dict[int, float]] logit_bias
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
            +Optional[float] temperature
            +Optional[float] top_p
            +Optional[int] top_k
            +int max_tokens
            +Optional[bool] stream
            +Optional[List[str]] stop_sequences
        }

        class ResponseBuilder {
            <<abstract>>
            +prepare(request, tokenizer) Tuple[str, GenContext, List[str]]
            +format_stream_start(ctx) List[str]
            +format_chunk(token) str
            +format_stream_end(ctx, stop) List[str]
            +format_response(ctx, content, stop) Dict
        }

        class OpenAIResponseBuilder {
            +prepare(request, tokenizer) Tuple
            +format_stream_start(ctx) List[str]
            +format_chunk(token) str
            +format_stream_end(ctx, stop) List[str]
            +format_response(ctx, content, stop) Dict
        }

        class AnthropicResponseBuilder {
            +prepare(request, tokenizer) Tuple
            +format_stream_start(ctx) List[str]
            +format_chunk(token) str
            +format_stream_end(ctx, stop) List[str]
            +format_response(ctx, content, stop) Dict
        }

        class ProtocolHandler {
            +request
            +engine
            +builder: ResponseBuilder
            +async handle() Union[StreamingResponse, Dict]
            -_handle_stream(agen, ctx, stop_sequences) StreamingResponse
            -async _handle_non_stream(agen, ctx, stop_sequences) Dict
        }

        class StopChecker {
            +__init__(sequences)
            +check(text) Optional[str]
        }

        class GenContext {
            +str resp_id
            +int created
            +str model
            +int prompt_tokens
            +int completion_tokens
        }

        class StopInfo {
            +Optional[str] matched
            +str body
            +str yielded
        }

        class app {
            <<singleton>>
            +FastAPI app
        }
    }

    namespace protocols {
        class OptimizerProtocol {
            <<protocol>>
            +step(closure)
            +zero_grad()
            +state_dict() dict
            +load_state_dict(d)
        }

        class SchedulerProtocol {
            <<protocol>>
            +step()
            +state_dict() dict
            +load_state_dict(d)
            +get_last_lr()
        }
    }

    namespace parallel {
        class setup {
            <<module>>
            +spawn_parallel_fn(func, world_size, backend, master_addr, master_port, device_type, start_method, **kwargs)
            +setup_parallel(rank, world_size, backend, master_addr, master_port, device_type) contextmanager
            +get_current_device() str
            +get_world_size() int
            +get_rank() int
            +only_on_rank(rank, sync=False) decorator
        }

        class GradientState {
            +int num_steps
            +sync_gradients (property) bool
        }

        class AccumOptimizer {
            +Optimizer optimizer
            +GradientState gradient_state
            +param_groups (property)
            +step(closure)
            +zero_grad()
            +state_dict() dict
            +load_state_dict(d)
        }

        class AccumScheduler {
            +LRScheduler scheduler
            +GradientState gradient_state
            +step()
            +state_dict() dict
            +load_state_dict(d)
            +get_last_lr()
        }

        class BaseExecutor {
            +GradientState gradient_state
            +prepare(model, optimizer, dataloader, scheduler) tuple
            +accumulate(model) context manager
            +backward(loss)
            +unwrap_model(model) dict
            +sync_gradients (property) bool
            +grad_accum_steps (property) int
        }

        class NoneExecutor {
        }

        class DDPExecutor {
            -_prepare_model(model) nn.Module
            -_no_sync(model) context manager
            +unwrap_model(model) dict
        }

        class FSDPExecutor {
            -_prepare_model(model) nn.Module
            +unwrap_model(model) dict
        }

        class ExecutorFactory {
            +Registry _registry
            +register(name) decorator
            +create(parallel_mode, **kwargs) BaseExecutor
        }

        class ParallelModel {
            +dist.ProcessGroup process_group
            +int rank
            +int world_size
        }

        class ColumnParallelLinear {
            +int in_features
            +int out_features
            +int out_features_per_rank
            +bool gather_results
            +Parameter weight
            +Optional[Parameter] bias
            +forward(x) Tensor
            +load_state_dict(state_dict)
        }

        class RowParallelLinear {
            +int in_features
            +int out_features
            +int in_features_per_rank
            +bool reduce_results
            +Parameter weight
            +Optional[Parameter] bias
            +forward(x) Tensor
            +load_state_dict(state_dict)
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
    TrainCallback <|-- ValidationCallback
    BaseDataset <|-- SEQDataset
    BaseDataset <|-- SFTDataset
    BaseDataset <|-- DPODataset
    BaseDataset <|-- GRPODataset
    Store <|-- H5Store
    Store <|-- MmapStore
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
    BaseConfig <|-- InputConfig
    BaseConfig <|-- ProcessingConfig
    BaseConfig <|-- OutputConfig
    BaseConfig <|-- PipelineConfig
    BaseModelConfig <|-- AutoRegressiveLMConfig
    BaseModelConfig <|-- EncoderConfig
    BaseFactory <|-- AutoModel
    BaseFactory <|-- AttnFactory
    BaseFactory <|-- FFNFactory
    BaseFactory <|-- DatasetFactory
    BaseFactory <|-- StrategyFactory
    BaseFactory <|-- SchedulerFactory
    BaseFactory <|-- CallbackFactory
    BaseFactory <|-- StoreFactory
    BaseFactory <|-- ExecutorFactory
    BaseFactory <|-- ConfigFactory
    BaseFactory <|-- MaskBuilderFactory
    BaseExecutor <|-- NoneExecutor
    BaseExecutor <|-- DDPExecutor
    BaseExecutor <|-- FSDPExecutor
    ResponseBuilder <|-- OpenAIResponseBuilder
    ResponseBuilder <|-- AnthropicResponseBuilder
    BaseMaskBuilder <|-- ChatMaskBuilder
    BaseMaskBuilder <|-- InstructionMaskBuilder
    BaseMaskBuilder <|-- TextMaskBuilder

    %% --- Composition (strong ownership, part destroyed with whole) ---
    KVCache *-- PagePool
    KVCache *-- Storage
    KVCache *-- TaskTable
    InferenceEngine *-- InferenceScheduler
    InferenceScheduler *-- KVCache
    InferenceScheduler *-- Executor
    InferenceScheduler *-- TaskManager
    AutoRegressiveLM *-- DecoderBlock
    AutoRegressiveLM *-- RotaryEmbedding
    AutoRegressiveLM *-- Embedding
    EmbeddingEncoder *-- DecoderBlock
    EmbeddingEncoder *-- RotaryEmbedding
    EmbeddingEncoder *-- Embedding
    DecoderBlock *-- RMSNorm
    ChatCompletionRequest *-- ChatMessage
    MessagesRequest *-- AnthropicMessage
    BaseFactory *-- Registry
    BaseExecutor *-- GradientState
    AccumOptimizer o-- GradientState
    AccumScheduler o-- GradientState

    %% --- Aggregation (weak ownership) ---
    AutoModel o-- BaseModelConfig
    AutoTokenizer o-- ChatTemplate
    PagePool o-- Allocator
    PagePool o-- PrefixCache
    Trainer o-- TrainCallback
    TrainContext o-- BaseStrategy
    TrainContext o-- BaseScheduler
    TrainContext o-- Checkpoint
    TrainContext o-- BaseExecutor
    KvcacheView o-- Storage
    SamplingPipeline o-- BaseSamplingStrategy
    BaseDataset o-- Store

    %% --- Dependency (uses temporarily) ---
    TrainConfig ..> BaseStrategy : selects
    PipelineConfig ..> MaskBuilderFactory : selects
    MaskBuilderFactory ..> BaseMaskBuilder : creates
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
    StoreFactory ..> H5Store : creates
    StoreFactory ..> MmapStore : creates
    ConfigFactory ..> AutoRegressiveLMConfig : creates
    ConfigFactory ..> EncoderConfig : creates
    ExecutorFactory ..> NoneExecutor : creates
    ExecutorFactory ..> DDPExecutor : creates
    ExecutorFactory ..> FSDPExecutor : creates
    TrainContextBuilder ..> ExecutorFactory : creates
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
    OpenAIResponseBuilder ..> ChatCompletionRequest : receives
    AnthropicResponseBuilder ..> MessagesRequest : receives
    ProtocolHandler ..> StopChecker : creates
    ProtocolHandler ..> GenContext : creates

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

```


## Module Overview

| Module | Components | Description |
|--------|------------|-------------|
| **astrai.config** | BaseConfig, BaseModelConfig, AutoRegressiveLMConfig, EncoderConfig, ConfigFactory, TrainConfig, PipelineConfig, InputConfig, ProcessingConfig, OutputConfig | Configuration management (to_dict/from_dict, to_file/from_file, from_json/to_json) |
| **astrai.preprocessing** | BaseMaskBuilder, MaskBuilderFactory, ChatMaskBuilder, InstructionMaskBuilder, TextMaskBuilder, Pipeline, filter_by_length, dedup_signature | Declarative JSON-driven data preprocessing |
| **astrai.dataset** | BaseDataset–GRPODataset, Store–MmapStore, StoreFactory, ResumableDistributedSampler, DatasetFactory | Dataset loading and management |
| **astrai.serialization** | Checkpoint | Model serialization |
| **astrai.model** | AutoModel, AutoRegressiveLM, EmbeddingEncoder, DecoderBlock, GQA, MLA, MLP, DeepSeekMoE, AttnFactory, FFNFactory, RMSNorm, Linear, RotaryEmbedding, Embedding | Neural network model |
| **astrai.tokenize** | AutoTokenizer, ChatTemplate | Tokenizer and chat template |
| **astrai.trainer** | Trainer, TrainContext, TrainContextBuilder, BaseStrategy–GRPOStrategy, StrategyFactory, BaseScheduler–SGDRScheduler, SchedulerFactory, TrainCallback(Protocol)–ValidationCallback, CallbackFactory, Muon | Training workflow |
| **astrai.inference** | InferenceEngine, InferenceScheduler, Executor, KVCache–KvcacheView, Allocator–Storage, Task, TaskManager, TaskStatus, GenerationRequest, GenerateResult, BaseSamplingStrategy–SamplingPipeline, ProtocolHandler, ResponseBuilder, OpenAIResponseBuilder, AnthropicResponseBuilder, StopChecker, GenContext, ChatMessage–MessagesRequest, app | Inference service |
| **astrai.parallel** | spawn_parallel_fn, setup_parallel, get_rank/get_world_size/get_current_device, only_on_rank, BaseExecutor, ExecutorFactory, NoneExecutor, DDPExecutor, FSDPExecutor, GradientState, AccumOptimizer, AccumScheduler, ParallelModel, RowParallelLinear, ColumnParallelLinear | Distributed parallel & gradient accumulation |
| **astrai.factory** | Registry, BaseFactory[T] | Component registration |
| **astrai.protocols** | OptimizerProtocol, SchedulerProtocol | Structural subtyping for optimizer/scheduler wrappers |

## Design Patterns

| Pattern | Classes | Purpose |
|---------|---------|---------|
| **Factory** | `AttnFactory`, `FFNFactory`, `StrategyFactory`, `DatasetFactory`, `SchedulerFactory`, `CallbackFactory`, `StoreFactory`, `ConfigFactory`, `ExecutorFactory` | Decorator-based component creation |
| **Registry** | `BaseFactory`, `Registry` | Component registration with category/priority |
| **Strategy** | `SEQStrategy`, `SFTStrategy`, `DPOStrategy`, `GRPOStrategy` | Training strategy switching |
| **Strategy (Sampling)** | `TemperatureStrategy`, `TopKStrategy`, `TopPStrategy`, `SamplingPipeline` | Composable logit transformations |
| **Strategy (API)** | `ResponseBuilder`, `OpenAIResponseBuilder`, `AnthropicResponseBuilder` | HTTP API handler with format hooks |
| **Builder** | `TrainContextBuilder` | Chain-building training context |
| **Observer** | `TrainCallback`, callback implementations | Training process monitoring |
| **Context** | `TrainContext` | Unified training state bag |
| **Object Pool** | `Allocator`, `PagePool` | Page-based KV cache with LRU eviction |
| **Executor** | `BaseExecutor`, `NoneExecutor`, `DDPExecutor`, `FSDPExecutor` | Gradient accumulation & model distribution |
| **Storage** | `Store`, `H5Store`, `MmapStore` | Format-agnostic data access with multi-segment support |
| **Producer-Consumer** | `InferenceScheduler`, `Task`, queues | Continuous batching |
| **AutoModel Registry** | `AutoModel`, `AutoRegressiveLM`, `EmbeddingEncoder` | Model-type dynamic loading |

## Core Relationships

1. **Config → Training**: `TrainConfig` holds `model_fn`, `dataset`, `optimizer_fn`, `scheduler_fn`, `parallel_mode`, `executor_kwargs`
2. **Training Flow**: `Trainer` → `TrainContextBuilder` → `TrainContext`, uses `BaseStrategy` for loss, `BaseExecutor` for gradient accumulation + model distribution
3. **Strategy Selection**: `StrategyFactory` creates strategy by `train_type`
4. **Executor Selection**: `ExecutorFactory.create(cfg.parallel_mode, grad_accum_steps=cfg.grad_accum_steps, **cfg.executor_kwargs)` → `NoneExecutor` / `DDPExecutor` / `FSDPExecutor`
5. **Inference Flow**: `InferenceEngine` → `InferenceScheduler` → `AutoRegressiveLM`, backed by `KVCache` + `SamplingPipeline`
6. **Distributed**: `spawn_parallel_fn` + `setup_parallel` for multi-process DDP
7. **Dataset Loading**: `DatasetFactory` creates datasets, `Store` (H5Store/MmapStore) loads data with explicit `_length` and multi-segment `_data`
8. **Checkpoint**: `Checkpoint` saves/loads safetensors + metadata (rank-0 only), extra state saved as `{key}.pt`
9. **Scheduler**: `SchedulerFactory` creates `CosineScheduler`/`SGDRScheduler`
10. **AutoModel**: `from_pretrained()` loads `config.json` + `model.safetensors`, `_disable_random_init` replaces `nn.init.*` with no-ops
11. **Protocols**: `OptimizerProtocol` / `SchedulerProtocol` — structural subtyping for `AccumOptimizer` / `AccumScheduler` wrappers

> Document Update Time: 2026-05-30
