"""Per-operation selection precedence and capability tests."""

import importlib

import pytest
import torch

from astrai.extension import (
    ExplicitSelectionError,
    op_backend,
    set_op,
    snapshot_op_selection,
    use_op_selection,
)
from astrai.extension.backend.rotary import RotaryDispatcher, _torch_apply
from astrai.extension.runtime import context

attention = importlib.import_module("astrai.extension.backend.attention")


def test_nested_choices_restore_and_operators_are_independent():
    x = torch.randn(1, 2, 2, 8)
    freqs = torch.randn(1, 2, 4, 2)
    with op_backend(rotary="torch", attention="torch_native"):
        assert RotaryDispatcher().dispatch(x, freqs) is _torch_apply
        with op_backend(attention="cuda"):
            assert RotaryDispatcher().dispatch(x, freqs) is _torch_apply
        assert (
            attention.AttentionDispatcher().dispatch(
                x, x, x, None, 0, None, False, None
            )
            is attention.torch.forward
        )


def test_explicit_rotary_cuda_rejects_cpu():
    x = torch.randn(1, 2, 2, 8)
    freqs = torch.randn(1, 2, 4, 2)
    with pytest.raises(ExplicitSelectionError):
        RotaryDispatcher().dispatch(x, freqs, explicit="cuda")


def test_process_choice_is_soft_and_scope_is_strict():
    q = torch.zeros(1, 2, 4, 8)
    set_op("attention", "cuda")
    selected = attention.AttentionDispatcher().dispatch(
        q, q, q, None, 0, None, False, None
    )
    assert selected is attention.torch.forward
    with op_backend(attention="cuda"):
        with pytest.raises(ExplicitSelectionError):
            attention.AttentionDispatcher().dispatch(
                q, q, q, None, 0, None, False, None
            )


def test_snapshot_freezes_process_choice():
    set_op("attention", "torch_native")
    snapshot = snapshot_op_selection()
    set_op("attention", "cuda")
    with use_op_selection(snapshot):
        assert context.op_context.snapshot().process()["attention"] == "torch_native"


def test_env_seed_accepts_only_known_operators(monkeypatch):
    monkeypatch.setenv("ASTR_OPS", "attention=torch,rotary=torch,profile=reference")
    monkeypatch.setattr(context, "_process_choices", None)
    assert context.op_context.snapshot().process() == {
        "attention": "torch",
        "rotary": "torch",
    }
    with pytest.raises(ValueError, match="unknown operator"):
        set_op("profile", "reference")
