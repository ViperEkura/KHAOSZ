"""Symmetric BLAS contracts, selection and compiled tiles."""

from importlib import import_module

import pytest
import torch

from astrai.extension import (
    ExplicitSelectionError,
    op_backend,
)
from astrai.extension.backend.newton_schulz import select, symm_out, syrk_out
from astrai.extension.kernel.newton_schulz import is_available
from astrai.extension.policy import newton_schulz as plan

backend = import_module("astrai.extension.backend.newton_schulz")

CUDA_AVAILABLE = torch.cuda.is_available() and is_available()


@pytest.mark.parametrize("operation", ["syrk", "symm"])
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (-0.3, 0.7), (0.0, -0.5)])
def test_arbitrary_cpu_geometry_and_independent_addend(operation, alpha, beta):
    x = torch.randn((7, 13), dtype=torch.float64)
    symmetric = torch.randn((7, 7), dtype=x.dtype)
    symmetric = (symmetric + symmetric.T) / 2
    output = torch.empty_like(symmetric if operation == "syrk" else x)
    addend = torch.randn_like(output)
    if operation == "syrk":
        addend = (addend + addend.T) / 2
        syrk_out(x, output, alpha=alpha, beta=beta, addend=addend)
        expected = alpha * (x @ x.T) + beta * addend
    else:
        symm_out(symmetric, x, output, alpha=alpha, beta=beta, addend=addend)
        expected = alpha * (symmetric @ x) + beta * addend
    torch.testing.assert_close(output, expected)


def test_invalid_contracts_and_strict_cuda_selection():
    x = torch.randn(7, 13)
    output = torch.empty(7, 7)
    with pytest.raises(ValueError, match="addend"):
        syrk_out(x, output, beta=1)
    with pytest.raises(ValueError, match="finite"):
        syrk_out(x, output, alpha=float("nan"))
    with pytest.raises(ValueError, match="alias"):
        symm_out(torch.eye(7), x, x)
    with pytest.raises(ExplicitSelectionError):
        syrk_out(x, output, backend="cuda")
    with op_backend(syrk="torch"):
        syrk_out(x, output)
    torch.testing.assert_close(output, x @ x.T)


def test_plan_configuration_is_atomic_and_scoped():
    before = plan.configure()
    with plan.override([]):
        assert plan.configure() == []
        with pytest.raises(ValueError, match="duplicate"):
            plan.configure(
                [
                    {
                        "operation": "symm",
                        "cc": 0,
                        "rows": 7,
                        "cols": 13,
                        "backend": "torch",
                    }
                ]
                * 2
            )
        assert plan.configure() == []
    assert plan.configure() == before


@pytest.mark.parametrize("device", ["cpu"])
def test_zero_beta_does_not_read_addend(device):
    x = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
    x.div_(x.norm())
    output = torch.empty(64, 64, device=device, dtype=x.dtype)
    addend = torch.full_like(output, float("nan"))
    syrk_out(
        x,
        output,
        addend=addend,
        beta=0,
        backend="cuda" if device == "cuda" else "torch",
    )
    torch.testing.assert_close(output, x @ x.T, atol=0.0001, rtol=0.02)


def test_matrix_dispatchers_keep_independent_context_choices():
    x = torch.randn(7, 13)
    syrk_output = torch.empty(7, 7)
    symm_output = torch.empty_like(x)
    with op_backend(syrk="torch", symm="torch"):
        assert select("syrk", x, syrk_output).func is backend._torch_syrk
        assert select("symm", torch.eye(7), x, symm_output).func is backend._torch_symm
    assert select("syrk", x, syrk_output).func is backend._torch_syrk
