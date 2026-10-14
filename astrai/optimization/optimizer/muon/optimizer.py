"""Muon plus AdamW combined optimizer."""

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn, optim

from astrai.optimization.optimizer.composite import (
    composite_state_dict,
    composite_step,
    composite_zero_grad,
    refresh_param_groups,
)
from astrai.optimization.optimizer.factory import OptimizerFactory
from astrai.optimization.optimizer.muon.sharded_update import ShardedMuon


@OptimizerFactory.register("muon_adamw")
class MuonAdamW(optim.Optimizer):
    """Combined Muon and AdamW optimizer.\n\n    ns_backend selects torch, cuda or auto independently of\n    batch_ns_updates. If batching is unspecified, the legacy\n    use_ns_kernels flag selects both together.\n"""

    optimizer_name = "muon_adamw"

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr_fn: str = "match_rms_adamw",
        reuse_ns_buffers: bool = False,
        use_ns_kernels: bool = False,
        ns_batch_size: int = 4,
        ns_backend: Optional[str] = None,
        batch_ns_updates: Optional[bool] = None,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        params = [param for param in model.parameters() if param.requires_grad]
        super().__init__(params, defaults)

        matrix_params: List[Tensor] = []
        other_params: List[Tensor] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if (
                param.dim() >= 2
                and "norm" not in name
                and "bias" not in name
                and "embed" not in name
                and "lm_head" not in name
            ):
                matrix_params.append(param)
            else:
                other_params.append(param)

        self.muon = ShardedMuon(
            matrix_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
            reuse_ns_buffers=reuse_ns_buffers,
            use_ns_kernels=use_ns_kernels,
            ns_batch_size=ns_batch_size,
            ns_backend=ns_backend,
            batch_ns_updates=batch_ns_updates,
        )
        self.adamw = optim.AdamW(
            [{"params": other_params, "weight_decay": 0.0}],
            lr=lr,
            betas=(0.9, 0.95),
            fused=True,
        )

        self.param_groups = refresh_param_groups([self.muon, self.adamw])

    @torch.no_grad()
    def step(self, closure=None):
        return composite_step([self.muon, self.adamw], closure)

    def zero_grad(self, set_to_none: bool = True):
        composite_zero_grad([self.muon, self.adamw], set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return composite_state_dict({"muon": self.muon, "adamw": self.adamw})

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if "muon" not in state_dict or "adamw" not in state_dict:
            raise ValueError(
                "Checkpoint optimizer state is not compatible with muon_adamw"
            )
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = refresh_param_groups([self.muon, self.adamw])
