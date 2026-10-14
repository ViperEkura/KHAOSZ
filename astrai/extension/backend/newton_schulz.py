"""Newton-Schulz matrix primitives, dispatch and recurrence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from astrai.extension.kernel import newton_schulz as cuda
from astrai.extension.policy import newton_schulz as plan
from astrai.extension.runtime.context import selection_cache_token
from astrai.extension.runtime.operation import (
    ExplicitSelectionError,
    OpDispatcher,
    OpSpec,
)


def _torch_syrk(
    x: Tensor,
    output: Tensor,
    *,
    addend: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    tile: Optional[str] = None,
) -> None:
    if beta == 0:
        (torch.mm if x.ndim == 2 else torch.bmm)(x, x.transpose(-2, -1), out=output)
        if alpha != 1:
            output.mul_(alpha)
    else:
        (torch.addmm if x.ndim == 2 else torch.baddbmm)(
            addend, x, x.transpose(-2, -1), alpha=alpha, beta=beta, out=output
        )


def _torch_symm(
    symmetric: Tensor,
    x: Tensor,
    output: Tensor,
    *,
    addend: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    tile: Optional[str] = None,
    raster: int = 1,
) -> None:
    if beta == 0:
        (torch.mm if x.ndim == 2 else torch.bmm)(symmetric, x, out=output)
        if alpha != 1:
            output.mul_(alpha)
    else:
        (torch.addmm if x.ndim == 2 else torch.baddbmm)(
            addend, symmetric, x, alpha=alpha, beta=beta, out=output
        )


def _axes(
    operation: str,
    *matrices: Tensor,
    addend: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    tile: Optional[str] = None,
    raster: int = 1,
) -> Dict[str, Any]:
    x = matrices[-2]
    operands = matrices + ((addend,) if beta != 0 and addend is not None else ())
    return {
        "cuda": all(plan.supports(matrix) for matrix in operands)
        and (
            tile is None
            or any(
                candidate["name"] == tile
                and plan.layout(x) in candidate["input_layouts"]
                for candidate in cuda.tiles(operation)
            )
        ),
        "measured": plan.probe(
            operation, x, output=matrices[-1], addend=beta != 0
        ).backend
        == "cuda",
    }


@dataclass(frozen=True)
class SyrkSpec(OpSpec):
    shape: Tuple[int, ...]
    layout: str
    cuda_eligible: bool
    measured: bool

    @property
    def operation(self) -> str:
        return "syrk"


@dataclass(frozen=True)
class SymmSpec(OpSpec):
    shape: Tuple[int, ...]
    layout: str
    cuda_eligible: bool
    measured: bool

    @property
    def operation(self) -> str:
        return "symm"


def _matrix_spec(operation: str, spec_type, matrices, kwargs):
    x = matrices[-2]
    ax = _axes(operation, *matrices, **kwargs)
    addend = kwargs.get("addend")
    operands = (
        *matrices,
        *((addend,) if kwargs.get("beta", 0) != 0 and addend is not None else ()),
    )
    return spec_type(
        device=x.device,
        dtype=x.dtype,
        phase="train" if torch.is_grad_enabled() else "infer",
        needs_backward=torch.is_grad_enabled()
        and any(bool(t.requires_grad) for t in operands),
        shape=tuple(x.shape),
        layout=plan.layout(x),
        cuda_eligible=ax["cuda"],
        measured=ax["measured"],
    )


def _select_matrix(
    operation: str,
    spec: Union[SyrkSpec, SymmSpec],
    choice: Any,
    origin: str,
) -> Callable[..., None]:
    """SYRK and SYMM share shape planning, then select their own callable."""

    cuda_op = cuda.syrk_out if operation == "syrk" else cuda.symm_out
    torch_op = _torch_syrk if operation == "syrk" else _torch_symm
    if choice == "torch" or choice is torch_op:
        return torch_op
    if (choice == "cuda" or choice is cuda_op) and spec.cuda_eligible:
        return cuda_op
    if choice == "measured" and spec.cuda_eligible and spec.measured:
        return cuda_op
    if choice is not None and origin in ("explicit", "context"):
        raise ExplicitSelectionError(
            f"Explicitly-set {operation} implementation {choice!r} "
            "cannot handle this call"
        )
    return cuda_op if spec.cuda_eligible and spec.measured else torch_op


class SyrkDispatcher(OpDispatcher[SyrkSpec]):
    @property
    def operation(self) -> str:
        return "syrk"

    def build_spec(self, *matrices: Tensor, **kwargs: Any) -> SyrkSpec:
        return _matrix_spec("syrk", SyrkSpec, matrices, kwargs)

    def select(
        self, spec: SyrkSpec, choice: Any, origin: str, *matrices: Tensor, **kwargs: Any
    ) -> Callable[..., None]:
        return _select_matrix("syrk", spec, choice, origin)


class SymmDispatcher(OpDispatcher[SymmSpec]):
    @property
    def operation(self) -> str:
        return "symm"

    def build_spec(self, *matrices: Tensor, **kwargs: Any) -> SymmSpec:
        return _matrix_spec("symm", SymmSpec, matrices, kwargs)

    def select(
        self, spec: SymmSpec, choice: Any, origin: str, *matrices: Tensor, **kwargs: Any
    ) -> Callable[..., None]:
        return _select_matrix("symm", spec, choice, origin)


_SYRK_DISPATCHER = SyrkDispatcher()
_SYMM_DISPATCHER = SymmDispatcher()


_selection_cache: Dict[Tuple[Any, ...], Callable[..., None]] = {}


def _selection_key(operation: str, matrices, kwargs) -> Optional[Tuple[Any, ...]]:
    token = selection_cache_token(operation)
    if token is None:
        return None
    addend = kwargs.get("addend")
    beta = kwargs.get("beta", 0) != 0
    operands = matrices + ((addend,) if beta and addend is not None else ())
    metadata = tuple(
        (
            tuple(x.shape),
            x.stride(),
            x.dtype,
            x.device,
            x.requires_grad,
            x.data_ptr() % 16,
        )
        for x in operands
    )
    return (
        token,
        plan.revision(),
        operation,
        metadata,
        beta,
        kwargs.get("tile"),
        kwargs.get("raster"),
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.are_deterministic_algorithms_enabled(),
    )


def select(
    operation: str, *matrices: Tensor, backend: Optional[str] = None, **kwargs: Any
) -> Callable[..., None]:
    """Resolve once; callers can reuse the selected function inside a recurrence."""
    if operation not in ("syrk", "symm"):
        raise ValueError("operation must be syrk or symm")
    if backend == "torch":
        dispatcher = _SYRK_DISPATCHER if operation == "syrk" else _SYMM_DISPATCHER
        return dispatcher.dispatch(*matrices, explicit="torch", **kwargs)
    key = _selection_key(operation, matrices, kwargs) if backend is None else None
    cached = _selection_cache.get(key) if key is not None else None
    if cached is not None:
        return cached
    dispatcher = _SYRK_DISPATCHER if operation == "syrk" else _SYMM_DISPATCHER
    selected = dispatcher.dispatch(*matrices, explicit=backend, **kwargs)
    decision = plan.probe(
        operation, matrices[-2], output=matrices[-1], addend=kwargs.get("beta", 0) != 0
    )
    options = {"tile": kwargs.get("tile") or decision.tile}
    if operation == "symm":
        options["raster"] = kwargs.get("raster", decision.raster)
    function = partial(selected, **options)
    if key is not None:
        if len(_selection_cache) >= 128:
            _selection_cache.clear()
        _selection_cache[key] = function
    return function


def _validate(
    x: Tensor,
    output: Tensor,
    shape,
    operands,
    addend: Optional[Tensor],
    alpha: float,
    beta: float,
) -> None:
    if any(t.ndim != x.ndim for t in operands) or tuple(output.shape) != tuple(shape):
        raise ValueError("matrix shape mismatch")
    if not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError("coefficients must be finite")
    if beta != 0 and addend is None:
        raise ValueError("nonzero beta requires an addend")
    if beta != 0:
        if addend.shape != output.shape:
            raise ValueError("addend shape mismatch")
        operands = (*operands, addend)
    for tensor in operands:
        if tensor.device != output.device or tensor.dtype != output.dtype:
            raise ValueError("matrix device or dtype mismatch")
        if torch._C._is_alias_of(output, tensor):
            raise ValueError("output must not alias inputs")


def syrk_out(
    x: Tensor,
    output: Tensor,
    *,
    addend: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    backend: Optional[str] = None,
    tile: Optional[str] = None,
) -> None:
    """Compute alpha * X X.T + beta * C into output."""
    if x.ndim not in (2, 3):
        raise ValueError("expected a matrix or matrix batch")
    _validate(
        x, output, (*x.shape[:-2], x.size(-2), x.size(-2)), (x,), addend, alpha, beta
    )
    select(
        "syrk",
        x,
        output,
        backend=backend,
        addend=addend,
        alpha=alpha,
        beta=beta,
        tile=tile,
    )(x, output, addend=addend, alpha=alpha, beta=beta)


def symm_out(
    symmetric: Tensor,
    x: Tensor,
    output: Tensor,
    *,
    addend: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    backend: Optional[str] = None,
    tile: Optional[str] = None,
    raster: Optional[int] = None,
) -> None:
    """Compute alpha * S X + beta * C into output."""
    _validate(x, output, x.shape, (symmetric, x), addend, alpha, beta)
    if x.ndim not in (2, 3) or symmetric.shape != (
        *x.shape[:-2],
        x.size(-2),
        x.size(-2),
    ):
        raise ValueError("symmetric matrix shape mismatch")
    options = {"tile": tile}
    if raster is not None:
        options["raster"] = raster
    select(
        "symm",
        symmetric,
        x,
        output,
        backend=backend,
        addend=addend,
        alpha=alpha,
        beta=beta,
        **options,
    )(symmetric, x, output, addend=addend, alpha=alpha, beta=beta)


def _use_row_work(x: Tensor, steps: int, backend: str) -> bool:
    if backend != "auto" or x.is_contiguous() or steps <= 1:
        return False
    return plan.probe("syrk", x, input_layout="row").backend == "cuda"


def _choice(operation: str, selected) -> Optional[Tuple[bool, str, int]]:
    if not isinstance(selected, partial):
        return None
    kernel_op = cuda.syrk_out if operation == "syrk" else cuda.symm_out
    torch_op = _torch_syrk if operation == "syrk" else _torch_symm
    if selected.func is kernel_op:
        return True, selected.keywords["tile"], selected.keywords.get("raster", 1)
    if selected.func is torch_op:
        return False, "", 1
    return None


def newton_schulz(
    matrix: Tensor,
    coefficients: Tuple[float, float, float],
    steps: int = 5,
    eps: float = 1e-7,
    *,
    backend: str = "torch",
) -> Tensor:
    """Orthogonalize a matrix or independent matrix batch with BF16 NS.

    backend="auto" uses symmetric dispatch plans; "torch" retains
    Torch arithmetic. Coefficients and iteration count are caller supplied.
    A rank-3 input has independent Frobenius normalization per matrix.
    BF16 inputs are normalized in place, matching Torch Muon semantics, but
    subsequent iterations never overwrite the normalized caller storage.
    """
    if matrix.ndim not in (2, 3) or len(coefficients) != 3 or not 0 <= steps < 100:
        raise ValueError("invalid Newton-Schulz matrix, coefficients or steps")
    if backend not in ("torch", "auto"):
        raise ValueError("backend must be torch or auto")
    a, b, c = coefficients
    x = matrix.bfloat16()
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.transpose(-2, -1)
    x.div_(
        (x.norm() if x.ndim == 2 else x.norm(dim=(-2, -1), keepdim=True)).clamp(min=eps)
    )
    if steps == 0:
        return x.transpose(-2, -1) if tall else x
    gram = torch.empty(
        (*x.shape[:-2], x.size(-2), x.size(-2)), dtype=x.dtype, device=x.device
    )
    polynomial = torch.empty_like(gram)
    explicit = "torch" if backend == "torch" else None
    # The first/last BLAS calls change layout directly when the selected Gram
    # plan prefers row-major scratch; external tensors keep their orientation.
    row_work = _use_row_work(x, steps, backend)
    work = (
        torch.empty(x.shape, dtype=x.dtype, device=x.device)
        if row_work
        else torch.empty_like(x)
    )
    spare = torch.empty_like(work) if steps > 1 else work
    final = torch.empty_like(x) if row_work else None
    first_gram = select("syrk", x, gram, backend=explicit)
    gram_op = select("syrk", work, gram, backend=explicit) if row_work else first_gram
    polynomial_op = select(
        "syrk", gram, polynomial, backend=explicit, addend=gram, alpha=c, beta=b
    )
    first_update = select(
        "symm", polynomial, x, work, backend=explicit, addend=x, beta=a
    )
    update_op = (
        select("symm", polynomial, work, spare, backend=explicit, addend=work, beta=a)
        if row_work
        else first_update
    )
    final_update = (
        select("symm", polynomial, work, final, backend=explicit, addend=work, beta=a)
        if final is not None
        else update_op
    )
    if backend == "auto" and cuda.is_available():
        choices = (
            _choice("syrk", first_gram),
            _choice("syrk", gram_op),
            _choice("syrk", polynomial_op),
            _choice("symm", first_update),
            _choice("symm", update_op),
            _choice("symm", final_update),
        )
        if all(choice is not None for choice in choices):
            x = cuda.iterate(
                x, gram, polynomial, work, spare, final, steps, a, b, c, choices
            )
            return x.transpose(-2, -1) if tall else x
    for iteration in range(steps):
        (first_gram if iteration == 0 else gram_op)(x, gram)
        polynomial_op(gram, polynomial, addend=gram, alpha=c, beta=b)
        output = (
            final
            if final is not None and iteration == steps - 1
            else (work if iteration % 2 == 0 else spare)
        )
        operation = (
            first_update
            if iteration == 0
            else final_update
            if output is final
            else update_op
        )
        operation(polynomial, x, output, addend=x, beta=a)
        x = output
    return x.transpose(-2, -1) if tall else x
