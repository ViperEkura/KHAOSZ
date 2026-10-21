"""Training optimizer implementations and factory registration."""

from astrai.optimization.optimizer.composite import (
    composite_state_dict,
    composite_step,
    composite_zero_grad,
    refresh_param_groups,
)
from astrai.optimization.optimizer.factory import OptimizerFactory
from astrai.optimization.optimizer.mano_adamw import Mano, ManoAdamW
from astrai.optimization.optimizer.muon import MuonAdamW
from astrai.optimization.optimizer.nora_nadamw import (
    NAdamW,
    Nora,
    NoraNAdamW,
    OptimizerParameterGroups,
    nora_direction,
    nora_lr_scale,
    partition_optimizer_parameters,
)

__all__ = [
    "Mano",
    "ManoAdamW",
    "MuonAdamW",
    "NAdamW",
    "Nora",
    "NoraNAdamW",
    "OptimizerFactory",
    "OptimizerParameterGroups",
    "composite_state_dict",
    "composite_step",
    "composite_zero_grad",
    "nora_direction",
    "nora_lr_scale",
    "partition_optimizer_parameters",
    "refresh_param_groups",
]
