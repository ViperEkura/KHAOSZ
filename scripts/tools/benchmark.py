"""Benchmark Transformer with PagedCache (replaces old persistent_key_values)."""

from dataclasses import dataclass
from typing import Any, Dict

import torch

from astrai.config import ModelConfig
from astrai.inference import PagedCache
from astrai.model.transformer import Transformer


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
        dtype: torch.dtype = torch.bfloat16,
        page_size: int = 128,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.model = Transformer(config).to(device=device, dtype=dtype)
        self.model.eval()
        head_dim = config.dim // config.n_heads
        n_pages = (config.max_len * 4 + page_size - 1) // page_size
        self._page_cache = PagedCache(
            config.n_layers,
            n_pages,
            page_size,
            config.n_kv_heads,
            head_dim,
            device,
            dtype,
        )

    def _prepare_inputs(self, batch_size: int, prompt_length: int, total_length: int):
        prompt_ids = torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=(batch_size, prompt_length),
            device=self.device,
            dtype=torch.long,
        )
        gen_ids = torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=(batch_size, total_length - prompt_length),
            device=self.device,
            dtype=torch.long,
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
            prompt_ids, _ = self._prepare_inputs(
                batch_size, prompt_length, prompt_length
            )
            _ = self.model(prompt_ids)
        torch.cuda.synchronize()

        total_time = 0.0
        total_tokens = batch_size * prompt_length * num_trials

        for trial in range(num_trials):
            prompt_ids, _ = self._prepare_inputs(
                batch_size, prompt_length, prompt_length
            )
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = self.model(prompt_ids)
            end.record()
            torch.cuda.synchronize()

            trial_time = start.elapsed_time(end) / 1000
            total_time += trial_time

            print(
                f"  Trial {trial + 1}/{num_trials}: {prompt_length} tokens in {trial_time:.3f}s "
                f"({prompt_length / trial_time:.1f} tok/s)"
            )

        return BenchmarkResult(
            total_tokens=total_tokens,
            total_time=total_time,
            tokens_per_second=total_tokens / total_time,
            metadata={
                "benchmark_type": "prefill",
                "batch_size": batch_size,
                "prompt_length": prompt_length,
                "dtype": str(self.dtype),
                "device": self.device,
            },
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
        page_size = self._page_cache.page_size

        for trial in range(num_trials):
            prompt_ids, gen_ids = self._prepare_inputs(
                batch_size,
                prompt_length,
                prompt_length + gen_length,
            )

            n_pages = (prompt_length + gen_length + page_size - 1) // page_size
            pages = self._page_cache.alloc_n(n_pages * batch_size)
            page_table = torch.tensor(
                [pages[i * n_pages : (i + 1) * n_pages] for i in range(batch_size)],
                dtype=torch.long,
                device=self.device,
            )

            cv = self._page_cache.bind(page_table, total_len=prompt_length)
            _ = self.model(
                prompt_ids,
                paged_cache=cv,
                position_ids=torch.arange(
                    prompt_length, dtype=torch.long, device=self.device
                )
                .unsqueeze(0)
                .expand(batch_size, -1),
            )

            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            current_pos = prompt_length
            for i in range(gen_length):
                input_token = gen_ids[:, i : i + 1]
                cv = self._page_cache.bind(page_table, total_len=current_pos + 1)
                _ = self.model(
                    input_token,
                    paged_cache=cv,
                    position_ids=torch.full(
                        (batch_size, 1),
                        current_pos,
                        dtype=torch.long,
                        device=self.device,
                    ),
                )
                current_pos += 1
            end.record()
            torch.cuda.synchronize()

            trial_time = start.elapsed_time(end) / 1000
            total_time += trial_time

            for idx in pages:
                self._page_cache.free(idx)

            print(
                f"  Trial {trial + 1}/{num_trials}: {gen_length} tokens in {trial_time:.3f}s "
                f"({gen_length / trial_time:.1f} tok/s)"
            )

        return BenchmarkResult(
            total_tokens=total_tokens,
            total_time=total_time,
            tokens_per_second=total_tokens / total_time,
            metadata={
                "benchmark_type": "decoding",
                "batch_size": batch_size,
                "prompt_length": prompt_length,
                "gen_length": gen_length,
                "dtype": str(self.dtype),
                "device": self.device,
            },
        )


def print_benchmark_result(result: BenchmarkResult):
    btype = result.metadata["benchmark_type"]
    print(f"\n{' ' + btype.upper() + ' Benchmark ':-^80}")
    print(f"Total Tokens Processed: {result.total_tokens:,}")
    print(f"Time Consumed: {result.total_time:.3f}s")
    print(f"Throughput: {result.tokens_per_second:,.1f} tok/s")
    for k, v in result.metadata.items():
        if k != "benchmark_type":
            print(f"{k.replace('_', ' ').title()}: {v}")
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
    print("Running Transformer Generation Benchmark (PagedCache)")
    print("=" * 80)

    prefill_result = benchmark.run_prefill_benchmark(
        batch_size=4,
        prompt_length=512,
        num_trials=5,
    )
    print_benchmark_result(prefill_result)

    gen_result = benchmark.run_decoding_benchmark(
        batch_size=4,
        prompt_length=512,
        gen_length=128,
        num_trials=5,
    )
    print_benchmark_result(gen_result)
