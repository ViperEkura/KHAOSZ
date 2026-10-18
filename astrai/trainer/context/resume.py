"""Resolve the persisted state a run starts from.

Everything that reads a checkpoint lives here, so the builder stays an
assembler: what a stored run *means* (cursor, step count, component states,
the refusal rules) is decided in one module.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

from torch import nn

from astrai.config.model_config import ConfigFactory
from astrai.config.train_config import TrainConfig
from astrai.parallel.executor import strip_compile_prefix
from astrai.parallel.topology import ParallelTopology
from astrai.serialization import (
    Checkpoint,
    adapt_config,
    convert_hf_weights,
    load_json,
    looks_like_hf_state_dict,
)
from astrai.trainer.context.state import samples_per_step
from astrai.trainer.optional_extras import CONTEXT_EXTRAS

logger = logging.getLogger(__name__)


@dataclass
class PreloadedState:
    """What the checkpoint contributes to a context that is about to be built."""

    model_config: Dict = field(default_factory=dict)
    state_dict: Optional[Dict] = None
    epoch: int = 0
    consumed_samples: int = 0
    optimizer_steps_completed: int = 0
    checkpoint: Optional[Checkpoint] = None
    component_states: Dict = field(default_factory=dict)


class ResumeLoader:
    """Read the checkpoint (or launch weights) a run starts from.

    A resume restores a whole training state or none of it: a component whose
    state is missing would silently come back fresh, which is a different run
    wearing the same step number.  The same reasoning refuses a cursor that
    cannot describe a resumable state.
    """

    def __init__(
        self,
        config: TrainConfig,
        param_path: Optional[str],
        resume: bool,
        topology: Optional[ParallelTopology],
        model_factory: Callable[[Dict], nn.Module],
    ):
        self.config = config
        self.param_path = param_path
        self.resume = resume
        self.topology = topology
        self.model_factory = model_factory

    def load(self) -> PreloadedState:
        cfg = self.config
        state = PreloadedState(
            epoch=cfg.start_epoch,
            consumed_samples=cfg.start_samples * self.topology.dp_size,
        )
        if self.param_path:
            self._load_from_param_path(state)
        if not state.model_config:
            model = self.model_factory(state.model_config)
            if hasattr(model, "config"):
                state.model_config = model.config.to_dict()
        state.optimizer_steps_completed = self._step_count(state)
        return state

    def _load_from_param_path(self, state: PreloadedState) -> None:
        cfg = self.config
        config_path = Path(self.param_path) / "config.json"
        if config_path.exists():
            state.model_config = adapt_config(load_json(config_path), self.param_path)

        options = cfg.state_dict_load_options(self.resume)
        context_keys = [name for name, _ in CONTEXT_EXTRAS]
        extra_keys = [name for name in context_keys if options[name]]
        if self.resume:
            extra_keys.extend(
                path.stem
                for path in Path(self.param_path).glob("*.pt")
                if path.stem not in context_keys
            )
        checkpoint = Checkpoint.load_any(
            self.param_path, extra_keys=extra_keys, load_weights=options["model"]
        )
        if checkpoint is None:
            return

        if checkpoint.config:
            checkpoint.config = adapt_config(checkpoint.config, self.param_path)
        if checkpoint.state_dict and looks_like_hf_state_dict(checkpoint.state_dict):
            checkpoint.state_dict = convert_hf_weights(
                checkpoint.state_dict,
                ConfigFactory.load(checkpoint.config or state.model_config),
            )
        checkpoint.state_dict = strip_compile_prefix(checkpoint.state_dict)
        state.state_dict = checkpoint.state_dict if options["model"] else None
        state.model_config = checkpoint.config or state.model_config

        if self.resume:
            self._take_cursor(state, checkpoint, options)
        for name in context_keys:
            if name not in checkpoint.extra:
                continue
            self._check_component_type(name, checkpoint)
            state.component_states[name] = checkpoint.extra[name]

    def _take_cursor(
        self, state: PreloadedState, checkpoint: Checkpoint, options: Dict
    ) -> None:
        """Adopt the checkpoint's epoch and data cursor, refusing stale layouts."""
        cfg = self.config
        context_keys = [name for name, _ in CONTEXT_EXTRAS]
        missing = [
            name
            for name in context_keys
            if options[name] and name not in checkpoint.extra
        ]
        if missing:
            raise ValueError(
                f"Cannot resume from {self.param_path}: no "
                f"{', '.join(missing)} state in the checkpoint; set "
                f"load_state_dict.{missing[0]}=False to accept a fresh one"
            )
        state.epoch = checkpoint.epoch
        per_step = samples_per_step(cfg, self.topology.dp_size)
        # The cursor only advances at commits, so a new checkpoint is always on
        # this boundary; one that is not came from an older run and cannot say
        # which state it describes.  Async GRPO is exempt: it advances by the
        # prompts of a round, which is not a whole number of steps.
        if cfg.rollout_mode != "async_round" and (
            checkpoint.consumed_samples % per_step
        ):
            raise ValueError(
                f"Checkpoint cursor {checkpoint.consumed_samples} is not on an "
                f"optimizer-step boundary ({per_step} samples); "
                "do not resume it unchanged"
            )
        state.consumed_samples = checkpoint.consumed_samples
        state.checkpoint = checkpoint

    def _check_component_type(self, name: str, checkpoint: Checkpoint) -> None:
        """Refuse to pour one optimizer's state into another implementation."""
        saved_spec = checkpoint.meta.get(name)
        saved_type = (
            saved_spec.get("type")
            if isinstance(saved_spec, dict)
            else checkpoint.meta.get("optimizer_name")
            if name == "optimizer"
            else None
        )
        current_type = getattr(self.config, name).type
        if saved_type and saved_type != current_type:
            raise ValueError(
                f"Cannot load {name} state from {saved_type!r} into "
                f"{current_type!r}; set load_state_dict.{name}=False when "
                "changing components"
            )

    def _step_count(self, state: PreloadedState) -> int:
        """The committed step count, which is not the cursor divided by a step.

        A replayed batch yields several steps for one data position, so the
        checkpoint's own count wins and the cursor is only a fallback for
        checkpoints that never wrote one.
        """
        if state.checkpoint is not None and state.checkpoint.meta:
            saved = state.checkpoint.meta.get("optimizer_step")
            if saved is not None:
                return int(saved)
        return state.consumed_samples // samples_per_step(
            self.config, self.topology.dp_size
        )
