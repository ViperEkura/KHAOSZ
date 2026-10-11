"""Benchmark GEMM dtypes and operand layouts.

``dtypes`` compares the quantized-GEMM family against bf16 ``F.linear``.

Every dtype pairing the gemm dispatch instantiates, as an activation-kind
x weight-kind grid: W16A16 (bf16 x bf16), W8A16 (bf16 activations against
int8 weights, per-channel scales), W8A8 (int8 x int8,
per-row activations), and the symmetric-fp8 training pair (matching
formats, per-tensor activations). All modes run the NT orientation the
linear path uses (activation ``[M][K]``, weight ``[N][K]``); quantize
passes are excluded from the timed GEMM — they price the kernel, not the
policy. Agreement columns report the max error against the dequantized
reference.

``layouts`` compares NT, NN, TT and TN through production dispatch. Operands
are transposed before timing, so the measured work is GEMM alone. Each cell
is checked against the same matrix product before timing. ``--planner-ab``
compares NT and TT with both the hybrid and model planners to separate
storage effects from measured plan-row coverage.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import click
import torch
import torch.nn.functional as F

from astrai.extension import is_available, quantize_act_int8, quantize_weight_int8
from astrai.extension.kernel.gemm import probe, quant_gemm, set_planner
from astrai.performance import measure_operations

# GEMM shapes as (N, K) weight mats; M comes from --m-values.
GEMM_SHAPES = (
    ("astrai_1b_square", 1536, 1536),
    ("astrai_1b_qkv", 1536 * 4, 1536),
    ("llama2_7b_qkv", 4096, 4096),
    ("llama2_7b_up_gate", 11008, 4096),
    ("llama2_7b_down", 4096, 11008),
    ("llama3_70b_up_gate", 28672, 8192),
)

# The fp8 formats and their max finite magnitudes.
FP8_FORMATS = (
    ("f8e4m3", torch.float8_e4m3fn, 448.0),
    ("f8e5m2", torch.float8_e5m2, 57344.0),
)

# Every dtype pairing the gemm dispatch instantiates (find_gemm_dispatch in
# csrc/gemm/entry.cu): each row names the cell, then the activation
# and weight kinds. Asymmetric low-bit mixes — int8 x fp8, mismatched fp8
# formats, quantized acts against bf16 weights — have no kernel and no row.
GEMM_COMBOS = (
    ("w16a16", "bf16", "bf16"),
    ("w8a16", "bf16", "int8"),
    ("w8a8", "int8", "int8"),
    ("f8a8_e4m3", "f8e4m3", "f8e4m3"),
    ("f8a8_e5m2", "f8e5m2", "f8e5m2"),
)
OP_ORDER = ("bf16", *(label for label, _, _ in GEMM_COMBOS))


def parse_positive_ints(value: str) -> tuple[int, ...]:
    """START:END:STEP (end inclusive) or a comma list — same grid format as
    the -m/-n/-k options in the other bench scripts."""
    value = value.strip()
    if ":" in value:
        start, stop, step = (int(x) for x in value.split(":"))
        if step <= 0 or stop < start:
            raise click.BadParameter("want START:END:STEP with positive step")
        values = tuple(range(start, stop + 1, step))
    else:
        try:
            values = tuple(
                dict.fromkeys(int(item.strip()) for item in value.split(","))
            )
        except ValueError as exc:
            raise click.BadParameter("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise click.BadParameter("values must be positive integers")
    return values


def parse_shape(value: str) -> tuple[str, int, int]:
    parts = value.split(":")
    if len(parts) != 3 or not parts[0]:
        raise click.BadParameter("shape must use NAME:ROWS:COLS")
    try:
        rows, cols = (int(item) for item in parts[1:])
    except ValueError as exc:
        raise click.BadParameter("ROWS:COLS must be integers") from exc
    if rows <= 0 or cols <= 0:
        raise click.BadParameter("ROWS and COLS must be positive")
    return parts[0], rows, cols


def quantize_fp8(
    t: torch.Tensor, dtype: torch.dtype, max_val: float, per_channel: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric fp8 quantization onto the format's finite range.

    Weights go per-channel (``[N]`` scales); activations per-tensor (one
    scalar, the F8A8 training pairing). The kernel applies the inverse
    scales in its epilogue.
    """
    amax = t.float().abs().amax(dim=-1 if per_channel else None, keepdim=True)
    scale = amax.clamp_min(1e-12) / max_val
    t_q = (t.float() / scale).clamp(-max_val, max_val).to(dtype)
    return t_q, (scale.squeeze(-1) if per_channel else scale).float()


def make_combo_op(
    a: torch.Tensor,
    a_scale: torch.Tensor | None,
    b: torch.Tensor,
    b_scale: torch.Tensor | None,
) -> Callable[[], torch.Tensor]:
    """Bind one combo cell into a zero-arg op (lambdas in loops bind late)."""
    return lambda: quant_gemm(a, b, a_scale=a_scale, b_scale=b_scale)


def dequantize(t: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    """Undo a quantize pair for the F.linear reference."""
    if scale is None:
        return t
    if scale.ndim == 1:
        scale = scale.unsqueeze(-1)
    return (t.float() * scale).to(torch.bfloat16)


def benchmark_gemm(
    name: str,
    n: int,
    k: int,
    m: int,
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[str, object]:
    x = (torch.randn(m, k, device="cuda") * 0.05).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * 0.05).to(torch.bfloat16)
    acts: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {
        "bf16": (x, None),
        "int8": quantize_act_int8(x),
    }
    weights: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {
        "bf16": (w, None),
        "int8": quantize_weight_int8(w),
    }
    for label, dtype, max_val in FP8_FORMATS:
        acts[label] = quantize_fp8(x, dtype, max_val, per_channel=False)
        weights[label] = quantize_fp8(w, dtype, max_val, per_channel=True)

    operations: dict[str, Callable[[], torch.Tensor]] = {"bf16": lambda: F.linear(x, w)}
    ref = {}
    for label, a_kind, b_kind in GEMM_COMBOS:
        operations[label] = make_combo_op(*acts[a_kind], *weights[b_kind])
        ref[label] = F.linear(
            dequantize(*acts[a_kind]), dequantize(*weights[b_kind])
        ).float()

    samples = measure_operations(
        operations, warmup=warmup, iterations=iterations, trials=trials
    )
    med = {key: statistics.median(vals) for key, vals in samples.items()}
    flops = 2.0 * m * n * k
    tfs = {key: flops / (ms * 1e-3) / 1e12 for key, ms in med.items()}
    err = {
        label: (operations[label]().float() - ref[label]).abs().max().item()
        for label, _, _ in GEMM_COMBOS
    }
    row = [name, f"{m}x{n}x{k}"]
    row += [f"{med[op]:.4f}" for op in OP_ORDER]
    row += [f"{tfs[op]:.1f}" for op in OP_ORDER]
    row += [f"{med['bf16'] / med[op]:.2f}x" for op in OP_ORDER[1:]]
    row += [f"{err[op]:.4f}" for op in OP_ORDER[1:]]
    print(",".join(row))
    return {
        "shape": name,
        "m": m,
        "n": n,
        "k": k,
        "median_ms": med,
        "tflops": tfs,
        "speedup_vs_bf16": {
            key: med["bf16"] / ms for key, ms in med.items() if key != "bf16"
        },
        "max_err": err,
    }


@click.command("dtypes")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional JSON evidence path (kept out of the repository).",
)
@click.option(
    "-m",
    "--m-values",
    default="512,2048,4096",
    show_default=True,
    callback=lambda _c, _p, v: parse_positive_ints(v),
    help="M grid: START:END:STEP (end inclusive) or a comma list.",
)
@click.option(
    "--shape",
    "shape_values",
    multiple=True,
    help="Filter defaults by name or add NAME:N:K.",
)
@click.option("--warmup", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--iterations", type=click.IntRange(min=1), default=50, show_default=True)
@click.option("--trials", type=click.IntRange(min=1), default=3, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def benchmark_command(
    output: Path | None,
    m_values: tuple[int, ...],
    shape_values: tuple[str, ...],
    warmup: int,
    iterations: int,
    trials: int,
    seed: int,
) -> None:
    """Compare GEMM dtype pairings with a bf16 linear baseline."""
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required")
    if not is_available("gemm"):
        raise click.ClickException("the built gemm kernel is required")

    bare_names = {value for value in shape_values if ":" not in value}
    known = {shape[0] for shape in GEMM_SHAPES}
    unknown = sorted(bare_names - known)
    if unknown:
        raise click.BadParameter(f"unknown default shape names: {', '.join(unknown)}")
    specs = [parse_shape(value) for value in shape_values if ":" in value]
    if shape_values:
        by_name = {s[0]: s for s in GEMM_SHAPES if s[0] in bare_names}
        by_name.update({s[0]: s for s in specs})
        shapes = list(by_name.values())
    else:
        shapes = list(GEMM_SHAPES)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    header = (
        ["shape", "mxn_xk"]
        + [f"{op}_ms" for op in OP_ORDER]
        + [f"{op}_tflops" for op in OP_ORDER]
        + [f"{op}_vs_bf16" for op in OP_ORDER[1:]]
        + [f"err_{op}" for op in OP_ORDER[1:]]
    )
    print(",".join(header))
    results = []
    with torch.inference_mode():
        for name, n, k in shapes:
            for m in m_values:
                results.append(
                    benchmark_gemm(
                        name,
                        n,
                        k,
                        m,
                        warmup=warmup,
                        iterations=iterations,
                        trials=trials,
                    )
                )
            torch.cuda.empty_cache()

    if output is not None:
        props = torch.cuda.get_device_properties(0)
        payload = {
            "metadata": {
                "gpu_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            "settings": {
                "warmup": warmup,
                "iterations": iterations,
                "trials": trials,
                "seed": seed,
                "order": "A-B-C-C-B-A",
                "m_values": list(m_values),
            },
            "results": results,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")


# Layout benchmark: NT/NN/TT/TN storage and optional planner A/B.
FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0

# (name, trans_a, trans_b)
LAYOUTS = (
    ("NT", False, True),
    ("NN", False, False),
    ("TT", True, True),
    ("TN", True, False),
)
DTYPES = ("bf16", "fp8")

# The astrai_1b projections (n x k); m comes from -m.
SHAPES = (
    ("square_1536", 1536, 1536),
    ("qkv_6144", 6144, 1536),
    ("mlp_up_6912", 6912, 1536),
    ("mlp_down_6912", 1536, 6912),
)


def dtype_pair(kind: str) -> tuple[torch.dtype, torch.dtype]:
    return (FP8, FP8) if kind == "fp8" else (torch.bfloat16, torch.bfloat16)


def build_cell(a_bf, b_bf, ta, tb, kind, ref):
    """One (layout, dtype) cell: operand storage + the callable to time."""
    if kind == "bf16":
        a, b, a_scale = a_bf, b_bf, None
    else:
        a8, sa = quantize_fp8(a_bf, FP8, FP8_MAX, per_channel=False)
        b8, sb = quantize_fp8(b_bf, FP8, FP8_MAX, per_channel=False)
        a, b, a_scale = a8, b8, (sa * sb).reshape(1)
    a_op = a.t().contiguous() if ta else a
    b_op = b if tb else b.t().contiguous()
    op = lambda: quant_gemm(a_op, b_op, a_scale=a_scale, trans_a=ta, trans_b=tb)
    return op, ref


def report(shape, order, samples, flops):
    for key in order:
        med = statistics.median(samples[key])
        lo, hi = min(samples[key]), max(samples[key])
        print(
            f"{shape:14s} {key[1]:6s} {key[2]:4s} {med:9.4f} "
            f"{flops / (med * 1e-3) / 1e12:9.1f}   [{lo:.4f}..{hi:.4f}]"
        )


@click.command("layouts")
@click.option(
    "-m",
    type=int,
    default=16384,
    show_default=True,
    help="activation rows (pretrain micro-batch 8x window 2048).",
)
@click.option("--shapes", default="", help="comma list of SHAPES names to keep.")
@click.option("--warmup", type=int, default=5, show_default=True)
@click.option("--iterations", type=int, default=30, show_default=True)
@click.option(
    "--trials",
    type=int,
    default=3,
    show_default=True,
    help="round-robin passes; each pass times every cell twice.",
)
@click.option(
    "--planner-ab",
    is_flag=True,
    help="NT x {builtin-wins, model} to split layout from row coverage.",
)
def layout_command(m, shapes, warmup, iterations, trials, planner_ab):
    """Compare GEMM operand layouts or isolate the planner effect."""
    keep = {s.strip() for s in shapes.split(",") if s.strip()}
    print(
        f"device={torch.cuda.get_device_name(0)} m={m} warmup={warmup} "
        f"iters={iterations} trials={trials}"
    )

    if planner_ab:
        n, k = 6144, 1536
        torch.manual_seed(11)
        a_bf = (torch.randn(m, k, device="cuda") * 0.05).to(torch.bfloat16)
        b_bf = (torch.randn(n, k, device="cuda") * 0.05).to(torch.bfloat16)
        a8, sa = quantize_fp8(a_bf, FP8, FP8_MAX, per_channel=False)
        b8, sb = quantize_fp8(b_bf, FP8, FP8_MAX, per_channel=False)
        scale = (sa * sb).reshape(1)
        cells = {}
        for lname, ta, tb in (("NT", False, True), ("TT", True, True)):
            for kind, a, b, sc in (("bf16", a_bf, b_bf, None), ("fp8", a8, b8, scale)):
                a_op = a.t().contiguous() if ta else a
                b_op = b if tb else b.t().contiguous()
                for mode in ("hybrid", "model"):

                    def op(a_op=a_op, b_op=b_op, sc=sc, ta=ta, tb=tb, mode=mode):
                        set_planner(mode)
                        return quant_gemm(
                            a_op, b_op, a_scale=sc, trans_a=ta, trans_b=tb
                        )

                    cells[(lname, kind, mode)] = op
        print(
            "probe:",
            {
                lname: probe(m, n, k, *dtype_pair(kind), trans_a=ta, trans_b=tb)
                for lname, ta, tb in (("NT", False, True), ("TT", True, True))
                for kind in DTYPES
            },
        )
    else:
        cells = {}
        for name, n, k in SHAPES:
            if keep and name not in keep:
                continue
            torch.manual_seed(11)
            a_bf = (torch.randn(m, k, device="cuda") * 0.05).to(torch.bfloat16)
            b_bf = (torch.randn(n, k, device="cuda") * 0.05).to(torch.bfloat16)
            a8, sa = quantize_fp8(a_bf, FP8, FP8_MAX, per_channel=False)
            b8, sb = quantize_fp8(b_bf, FP8, FP8_MAX, per_channel=False)
            refs = {
                "bf16": a_bf.float() @ b_bf.float().t(),
                "fp8": (a8.float() * sa) @ (b8.float() * sb).t(),
            }
            del a8, b8
            for lname, ta, tb in LAYOUTS:
                for kind in DTYPES:
                    op, ref = build_cell(a_bf, b_bf, ta, tb, kind, refs[kind])
                    # Correctness before timing: a layout that changes the
                    # math silently (square shapes hide a flipped operand in
                    # the shape check) must never reach the timing loop.
                    rel = ((op().float() - ref).abs().max() / ref.abs().max()).item()
                    assert rel < 0.02, f"{name} {lname} {kind}: rel={rel}"
                    cells[(name, lname, kind)] = op
            print(
                f"# {name} n={n} k={k}: "
                + ", ".join(
                    f"{l}/{d}={probe(m, n, k, *dtype_pair(d), trans_a=ta, trans_b=tb)['source']}"
                    for l, ta, tb in LAYOUTS
                    for d in DTYPES
                )
            )
        del a_bf, b_bf, refs

    samples = measure_operations(
        cells, warmup=warmup, iterations=iterations, trials=trials
    )
    order = list(cells)

    header = f"{'shape':14s} {'layout':6s} {'dtype':4s} {'ms':>9s} {'TFLOP/s':>9s}"
    print(header + "   [min..max]")
    print("-" * (len(header) + 24))
    if planner_ab:
        n, k = 6144, 1536
        for key in order:
            med = statistics.median(samples[key])
            lo, hi = min(samples[key]), max(samples[key])
            print(
                f"{'qkv_6144':14s} {key[0]:6s} {key[1] + '/' + key[2]:14s} "
                f"{med:9.4f} {2.0 * m * n * k / (med * 1e-3) / 1e12:9.1f}   "
                f"[{lo:.4f}..{hi:.4f}]"
            )
        set_planner("")
    else:
        for name, n, k in SHAPES:
            if keep and name not in keep:
                continue
            keys = [key for key in order if key[0] == name]
            report(name, keys, samples, 2.0 * m * n * k)


@click.group(help=__doc__)
def cli() -> None:
    """GEMM benchmark suites."""


cli.add_command(benchmark_command)
cli.add_command(layout_command)


if __name__ == "__main__":
    cli()
