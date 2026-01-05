import torch
from typing import Dict, Any
from dataclasses import dataclass
from khaosz.model.transformer import ModelConfig, Transformer


@dataclass
class BenchmarkResult:
    total_tokens: int
    total_time: float
    tokens_per_second: float
    metadata: Dict[str, Any]


class GenerationBenchmark:
    def __init__(
        self,
        config: ModelConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.model = Transformer(config).to(device=device, dtype=dtype)
        self.model.eval()
    
    def _initialize_kv_cache(self, batch_size: int) -> list:
        """初始化KV缓存"""
        config = self.config
        shape = (batch_size, config.max_len, config.n_layers, config.n_kv_heads, config.dim // config.n_heads)
        k_cache = torch.zeros(shape, device=self.device, dtype=self.dtype)
        v_cache = torch.zeros(shape, device=self.device, dtype=self.dtype)
        return (k_cache, v_cache)
    
    def _prepare_inputs(self, batch_size: int, prompt_length: int, total_length: int):
        prompt_ids = torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=(batch_size, prompt_length),
            device=self.device,
            dtype=torch.long
        )
        
        gen_ids = torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=(batch_size, total_length - prompt_length),
            device=self.device,
            dtype=torch.long
        )
        
        return prompt_ids, gen_ids
    
    @torch.inference_mode()
    def run_prefill_benchmark(
        self,
        batch_size: int = 1,
        prompt_length: int = 512,
        num_trials: int = 10,
    ) -> BenchmarkResult:
        
        for _ in range(3):
            prompt_ids, _ = self._prepare_inputs(batch_size, prompt_length, prompt_length)
            _ = self.model(prompt_ids)
        
        torch.cuda.synchronize()
        
        total_time = 0.0
        total_tokens = batch_size * prompt_length * num_trials
        
        for trial in range(num_trials):
            prompt_ids, _ = self._prepare_inputs(batch_size, prompt_length, prompt_length)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            _ = self.model(prompt_ids)
            end_event.record()
            torch.cuda.synchronize()
            
            trial_time = start_event.elapsed_time(end_event) / 1000
            total_time += trial_time
            
            print(f"Trial {trial + 1}/{num_trials}: {prompt_length} tokens in {trial_time:.3f}s "
                  f"({prompt_length / trial_time:.1f} tokens/s)")
        
        return BenchmarkResult(
            total_tokens=total_tokens,
            total_time=total_time,
            tokens_per_second=total_tokens / total_time,
            metadata={
                "benchmark_type": "prefill",
                "batch_size": batch_size,
                "prompt_length": prompt_length,
                "dtype": self.dtype,
                "device": self.device,
            }
        )
    
    @torch.inference_mode()
    def run_decoding_benchmark(
        self,
        batch_size: int = 1,
        prompt_length: int = 512,
        gen_length: int = 128,
        num_trials: int = 5,
    ) -> BenchmarkResult:
        
        total_time = 0.0
        total_tokens = batch_size * gen_length * num_trials
        
        for trial in range(num_trials):
            
            prompt_ids, gen_ids = self._prepare_inputs(batch_size, prompt_length, prompt_length + gen_length)
            kv_cache = self._initialize_kv_cache(batch_size)
            _ = self.model(prompt_ids, persistent_key_values=kv_cache, start_pos=0)
            
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            
            current_pos = prompt_length
            for i in range(gen_length):
                input_token = gen_ids[:, i:i+1]
                _ = self.model(input_token, persistent_key_values=kv_cache, start_pos=current_pos)
                current_pos += 1
            
            end_event.record()
            torch.cuda.synchronize()
            
            trial_time = start_event.elapsed_time(end_event) / 1000
            total_time += trial_time
            
            print(f"Trial {trial + 1}/{num_trials}: {gen_length} tokens in {trial_time:.3f}s "
                  f"({gen_length / trial_time:.1f} tokens/s)")

        
        return BenchmarkResult(
            total_tokens=total_tokens,
            total_time=total_time,
            tokens_per_second=total_tokens / total_time,
            metadata={
                "benchmark_type": "decoding",
                "batch_size": batch_size,
                "prompt_length": prompt_length,
                "gen_length": gen_length,
                "dtype": self.dtype,
                "device": self.device,
            }
        )


def print_benchmark_result(result: BenchmarkResult):
    """打印基准测试结果"""
    benchmark_type = result.metadata["benchmark_type"]
    
    print(f"\n{' ' + benchmark_type.upper().replace('_', ' ') + ' Benchmark ':-^80}")
    print(f"Total Tokens Processed: {result.total_tokens:,}")
    print(f"Time Consumed: {result.total_time:.3f}s")
    print(f"Throughput: {result.tokens_per_second:,.1f} tokens/s")
    
    if benchmark_type == "prefill":
        print(f"Batch Size: {result.metadata['batch_size']} | Prompt Length: {result.metadata['prompt_length']}")
    elif benchmark_type == "decoding":
        print(f"Batch Size: {result.metadata['batch_size']} | Gen Length: {result.metadata['gen_length']}")
    
    print(f"Device: {result.metadata['device']} | Dtype: {result.metadata['dtype']}")
    print("-" * 80)


if __name__ == "__main__":
    config = ModelConfig(
        vocab_size=10000,
        dim=1536,
        n_heads=24,
        n_kv_heads=4,
        dim_ffn=6912,
        max_len=2048,
        n_layers=24,
        norm_eps=1e-5,
    )
    
    benchmark = GenerationBenchmark(config)
    
    print("=" * 80)
    print("Running Transformer Generation Benchmark")
    print("=" * 80)
    
    prefill_result = benchmark.run_prefill_benchmark(batch_size=4, prompt_length=512, num_trials=5)
    print_benchmark_result(prefill_result)
    
    gen_result = benchmark.run_decoding_benchmark(batch_size=4, prompt_length=512, gen_length=128, num_trials=5)
    print_benchmark_result(gen_result)
    