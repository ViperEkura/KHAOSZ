"""Rotary operation dispatch and fused gradient on CUDA."""

import pytest
import torch

from astrai.extension.backend import apply_rotary_emb
from astrai.extension.backend.rotary import RotaryDispatcher, _cuda_rotary, _torch_apply
from astrai.extension.runtime.context import op_backend
from astrai.extension.runtime.loader import is_available


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="rotary CUDA path needs a GPU"
)
class TestRotaryDispatch:
    def _input(self):
        torch.manual_seed(0)
        x = torch.randn(1, 5, 3, 16, device="cuda", dtype=torch.bfloat16)
        freqs = torch.randn(1, 5, 8, 2, device="cuda", dtype=torch.float32)
        return x, freqs

    def test_auto_selects_cuda_for_inference_when_available(self):
        x, freqs = self._input()
        with torch.inference_mode():
            selected = RotaryDispatcher().dispatch(x, freqs)
        assert selected is (
            _cuda_rotary if is_available("rotary_emb") else _torch_apply
        )

    def test_grad_selects_cuda_when_available(self):
        x, freqs = self._input()
        x.requires_grad_()
        assert RotaryDispatcher().dispatch(x, freqs) is (
            _cuda_rotary if is_available("rotary_emb") else _torch_apply
        )

    def test_frequency_grad_uses_torch(self):
        x, freqs = self._input()
        freqs.requires_grad_()
        assert RotaryDispatcher().dispatch(x, freqs) is _torch_apply

    def test_context_switch_to_torch(self):
        x, freqs = self._input()
        with torch.inference_mode(), op_backend(rotary="torch"):
            out = apply_rotary_emb(x, freqs)
        assert out.shape == x.shape and out.dtype == torch.bfloat16

    def test_cuda_matches_torch_numerics(self):
        if not is_available("rotary_emb"):
            pytest.skip("rotary kernel not built")
        x, freqs = self._input()
        with torch.inference_mode():
            fast = apply_rotary_emb(x, freqs)
            ref = _torch_apply(x, freqs)
        torch.testing.assert_close(fast.float(), ref.float(), atol=2e-2, rtol=1e-2)

    @pytest.mark.parametrize("sliced", [False, True])
    def test_cuda_backward_matches_torch(self, sliced):
        if not is_available("rotary_emb"):
            pytest.skip("rotary kernel not built")
        x, freqs = self._input()
        if sliced:
            x = torch.cat((x, x), dim=-1)[..., :16]
        x = x.detach().requires_grad_()
        grad = torch.randn_like(x)
        with op_backend(rotary="cuda"):
            output = apply_rotary_emb(x, freqs)
        grad_x = torch.autograd.grad(output, x, grad)[0]
        ref_x = x.detach().float().requires_grad_()
        ref = _torch_apply(ref_x, freqs)
        ref_grad = torch.autograd.grad(ref, ref_x, grad.float())[0]
        torch.testing.assert_close(output.float(), ref.float(), atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(grad_x.float(), ref_grad, atol=2e-2, rtol=1e-2)
