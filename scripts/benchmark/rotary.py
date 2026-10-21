"""Compare RoPE training memory and latency on one CUDA GPU.

Example:
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/benchmark/rotary.py \
    --shape 4x2048x16x128
"""

import argparse
import gc
import json
import statistics

import torch

from astrai.extension.backend.rotary import apply_rotary_emb
from astrai.extension.runtime.context import op_backend
from astrai.extension.runtime.loader import is_available
from astrai.performance import measure_operations


def operation(choice, x, freqs, grad):
    def run():
        with op_backend(rotary=choice):
            output = apply_rotary_emb(x, freqs)
        return torch.autograd.grad(output, x, grad)[0]

    return run


def graph_operation(run):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    def replay():
        graph.replay()
        return graph

    return replay


def peak_memory(run):
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = run()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    del result
    return peak / 2**20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="4x2048x16x128")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    try:
        shape = tuple(map(int, args.shape.split("x")))
    except ValueError:
        parser.error("shape must have four positive dimensions separated by x")
    if len(shape) != 4 or min(shape) < 1 or shape[-1] % 4:
        parser.error(
            "shape must be BxSxHxD with positive dimensions and D divisible by 4"
        )
    if min(args.warmup, args.iterations, args.trials) < 1:
        parser.error("warmup, iterations, and trials must be positive")
    if not torch.cuda.is_available() or not is_available("rotary_emb"):
        parser.error("a CUDA GPU and the built RoPE extension are required")

    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    angles = torch.randn((*shape[:2], shape[-1] // 2), device="cuda")
    freqs = torch.stack((angles.cos(), angles.sin()), dim=-1)
    grad = torch.randn_like(x)
    runs = {choice: operation(choice, x, freqs, grad) for choice in ("torch", "cuda")}
    memory = {choice: peak_memory(run) for choice, run in runs.items()}
    report = {
        "gpu": torch.cuda.get_device_name(),
        "shape": shape,
        "dtype": "bfloat16",
        "torch_peak_mib": round(memory["torch"], 2),
        "cuda_peak_mib": round(memory["cuda"], 2),
        "memory_reduction": round(1 - memory["cuda"] / memory["torch"], 3),
    }
    for execution in ("eager", "graph"):
        operations = (
            runs
            if execution == "eager"
            else {name: graph_operation(run) for name, run in runs.items()}
        )
        samples = measure_operations(
            operations,
            warmup=args.warmup,
            iterations=args.iterations,
            trials=args.trials,
        )
        medians = {
            name: statistics.median(values) * 1000 for name, values in samples.items()
        }
        report[execution] = {
            "torch_us": round(medians["torch"], 2),
            "cuda_us": round(medians["cuda"], 2),
            "speedup": round(medians["torch"] / medians["cuda"], 2),
        }
    print(json.dumps(report))


if __name__ == "__main__":
    main()
