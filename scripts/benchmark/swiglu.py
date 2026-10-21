"""Benchmark dispatched Torch and fused CUDA SwiGLU on one GPU.

Example:
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/benchmark/swiglu.py \
    --shapes 128x4096 2048x6912 --dtype bfloat16
"""

import argparse
import json
import statistics

import torch

from astrai.extension.backend.swiglu import swiglu
from astrai.extension.runtime.context import op_backend
from astrai.extension.runtime.loader import is_available
from astrai.performance import measure_operations


def operation(choice, up, gate, grad):
    def run():
        with op_backend(swiglu=choice):
            output = swiglu(up, gate)
        if grad is None:
            return output
        grad_up, grad_gate = torch.autograd.grad(output, (up, gate), grad_outputs=grad)
        return grad_up

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["128x4096", "2048x6912"])
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.trials) < 1:
        parser.error("warmup, iterations, and trials must be positive")
    if not torch.cuda.is_available() or not is_available("swiglu"):
        parser.error("a CUDA GPU and the built SwiGLU extension are required")
    dtype = getattr(torch, args.dtype)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "dtype": args.dtype}))
    for shape_text in args.shapes:
        try:
            shape = tuple(map(int, shape_text.split("x")))
        except ValueError:
            parser.error("shapes must be positive dimensions separated by x")
        if not shape or min(shape) < 1:
            parser.error("shapes must have positive dimensions")
        for mode in ("forward", "forward_backward"):
            up = torch.randn(
                shape, device="cuda", dtype=dtype, requires_grad=mode != "forward"
            )
            gate = torch.randn_like(up, requires_grad=mode != "forward")
            grad = torch.randn_like(up) if mode != "forward" else None
            runs = {
                choice: operation(choice, up, gate, grad)
                for choice in ("torch", "cuda")
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
                    name: statistics.median(values) * 1000
                    for name, values in samples.items()
                }
                print(
                    json.dumps(
                        {
                            "shape": shape,
                            "mode": mode,
                            "execution": execution,
                            "torch_us": round(medians["torch"], 2),
                            "cuda_us": round(medians["cuda"], 2),
                            "speedup": round(medians["torch"] / medians["cuda"], 2),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
