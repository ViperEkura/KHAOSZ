"""FSDP executor on the per-module ``fully_shard`` API."""

import logging
from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FSDPModule,
    fully_shard,
)
from torch.distributed.tensor import DTensor

from astrai.parallel.executor.base import (
    _COMPILE_PREFIX,
    BaseExecutor,
    RolloutCapabilities,
)

logger = logging.getLogger(__name__)


class FSDPExecutor(BaseExecutor):
    """FSDP executor using `torch.distributed.fsdp.fully_shard` (per-module API).

    Wraps decoder layers before their parent to overlap parameter gathers
    with compute. Set ``layerwise=False`` to retain coarse child wrapping
    when minimizing FSDP overhead matters more (for example, short sequences).
    Accumulation keeps full parameters between microbatches to avoid
    repeated all-gathers; disable this if memory is constrained.
    Skips the root model because ``ABC + Generic[T]`` in the MRO makes
    ``fully_shard``'s dynamic ``__class__`` assignment fail at the CPython level.
    Original ``Parameter`` objects are preserved (as DTensors) — no
    ``FlatParameter``, no ``use_orig_params=True`` hack.
    """

    def __init__(
        self,
        grad_accum_steps: int = 1,
        mesh: Optional[Any] = None,
        mp_policy: Optional[Any] = None,
        reshard_after_forward: bool = False,
        layerwise: bool = True,
        keep_unsharded_during_accumulation: bool = True,
    ):
        super().__init__(grad_accum_steps=grad_accum_steps)
        if not isinstance(layerwise, bool):
            raise ValueError("layerwise must be a boolean")
        if not isinstance(keep_unsharded_during_accumulation, bool):
            raise ValueError("keep_unsharded_during_accumulation must be a boolean")
        self._mesh = mesh
        self._mp_policy = mp_policy
        self._reshard_after_forward = reshard_after_forward
        self._layerwise = layerwise
        self._keep_unsharded_during_accumulation = keep_unsharded_during_accumulation

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        if not self.use_distributed:
            logger.warning("FSDP backend selected but world_size=1, model not wrapped")
            return model

        kwargs = dict(
            mesh=self._mesh,
            mp_policy=self._mp_policy,
            reshard_after_forward=self._reshard_after_forward,
        )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        layer_count = 0
        for child in model.children():
            if self._layerwise:
                layers = getattr(child, "layers", None)
                if isinstance(layers, nn.ModuleList):
                    for layer in layers:
                        fully_shard(layer, **kwargs)
                        layer_count += 1
            if isinstance(child, nn.ModuleList):
                for sub in child:
                    fully_shard(sub, **kwargs)
            else:
                fully_shard(child, **kwargs)

        logger.info(
            "FSDP wrapping applied to %d layers and %d direct children",
            layer_count,
            len(list(model.children())),
        )
        return model

    def rollout_capabilities(self):
        if self.use_distributed:
            return RolloutCapabilities(
                supports_in_process=False,
                reason=(
                    "Distributed FSDP online rollout is not supported: inference "
                    "requires a replicated model view, but parameters are sharded"
                ),
            )
        return super().rollout_capabilities()

    @contextmanager
    def _no_sync(self, model: nn.Module):
        fsdp_modules = [m for m in model.modules() if isinstance(m, FSDPModule)]
        if not fsdp_modules:
            yield
            return
        for module in fsdp_modules:
            if self._keep_unsharded_during_accumulation:
                module.set_reshard_after_backward(False, recurse=False)
            module.set_requires_gradient_sync(False, recurse=False)
        try:
            yield
        finally:
            for module in fsdp_modules:
                module.set_requires_gradient_sync(True, recurse=False)
                if self._keep_unsharded_during_accumulation:
                    module.set_reshard_after_backward(True, recurse=False)

    def clip_grad_norm(self, model: nn.Module, max_norm: float) -> float:
        if not self.use_distributed:
            return super().clip_grad_norm(model, max_norm)

        # FSDP params are DTensors (sharded across ranks), and
        # get_total_norm reduces them to a globally-reduced replicated
        # DTensor — no extra all-reduce is needed. Manually summing the
        # local slice again would inflate the norm by sqrt(world_size)
        # and over-clip every step.
        total_norm = torch.nn.utils.get_total_norm(
            [p.grad for p in model.parameters() if p.grad is not None],
        )
        if isinstance(total_norm, DTensor):
            total_norm = total_norm.full_tensor()

        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(clip_coef_clamped)

        return total_norm.item()

    def model_state_dict(self, model: nn.Module):
        if not self.use_distributed:
            return model.state_dict()

        # unshard() and full_tensor() are collective ops — all ranks must
        # participate. Non-rank-0 ranks still call them but discard results.
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.unshard()

        state_dict = model.state_dict()
        result = {}
        for k, v in state_dict.items():
            k = k.removeprefix(_COMPILE_PREFIX)
            if isinstance(v, DTensor):
                full = v.full_tensor()
                if self.writes_checkpoints:
                    result[k] = full
            elif self.writes_checkpoints:
                result[k] = v

        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

        if not self.writes_checkpoints:
            return None

        return result
