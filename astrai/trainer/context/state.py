"""The mutable state one training run carries through its lifecycle."""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from torch import nn
from torch.utils.data import DataLoader, Dataset

from astrai.config.train_config import TrainConfig
from astrai.optimization.protocols import OptimizerProtocol, SchedulerProtocol
from astrai.parallel.executor import BaseExecutor
from astrai.parallel.topology import ParallelTopology
from astrai.serialization import Checkpoint
from astrai.trainer.callbacks.metric_util import GradSNRTracker
from astrai.trainer.rollout import RolloutEvaluator, RolloutRunner
from astrai.trainer.strategy import BaseStrategy

logger = logging.getLogger(__name__)


def samples_per_step(config: TrainConfig, dp_size: int) -> int:
    """Samples one optimizer step consumes; the unit of the data cursor.

    The cursor, the saved ``consumed_samples`` and the resume check all read
    this one definition, so "one step" is the same number of samples wherever
    it is counted.
    """
    return config.batch_per_device * dp_size * config.grad_accum_steps


@dataclass
class TrainContext:
    model: nn.Module = field(default=None)
    strategy: BaseStrategy = field(default=None)
    dataset: Optional[Dataset] = field(default=None)
    val_dataset: Optional[Dataset] = field(default=None)
    dataloader: DataLoader = field(default=None)
    #: Iterator currently being drained; owned by the trainer so shutdown can
    #: stop its workers before anything is written.
    dataloader_iter: Any = field(default=None)
    optimizer: OptimizerProtocol = field(default=None)
    scheduler: SchedulerProtocol = field(default=None)
    checkpoint: Checkpoint = field(default=None)
    config: TrainConfig = field(default=None)
    model_config: Dict = field(default_factory=dict)
    executor: BaseExecutor = field(default=None)
    epoch: int = field(default=0)
    consumed_samples: int = field(default=0)
    loss: float = field(default=0.0)
    metrics: Dict[str, float] = field(default_factory=dict)
    grad_norm: Optional[float] = field(default=None)
    grad_snr_tracker: GradSNRTracker = field(default_factory=GradSNRTracker)
    grad_snr_value: Optional[float] = field(default=None)
    val_dataloader: Optional[DataLoader] = field(default=None)
    val_loss: Optional[float] = field(default=None)
    #: Online-strategy validation: reward-statistics evaluator under its
    #: own sampling params. ``None`` keeps the legacy validate_online path.
    val_evaluator: Optional[RolloutEvaluator] = field(default=None)
    async_rollout: Optional[RolloutRunner] = field(default=None)
    #: Committed optimizer steps. Advanced once per commit in every rollout
    #: mode, so a checkpoint's step number never has to be derived from the
    #: data cursor (a batch that is replayed yields several steps).
    optimizer_steps_completed: int = field(default=0)
    checkpoint_safe: bool = field(default=True)
    #: A stop request that every rank has agreed on, so leaving the loop cannot
    #: strand a peer inside a collective.
    stop_agreed: bool = field(default=False)

    world_size: int = field(default=1)
    rank: int = field(default=0)
    topology: Optional[ParallelTopology] = field(default=None)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    param_path: Optional[str] = field(default=None)

    _stop_event: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self):
        # A topology is always answerable: contexts built outside the
        # builder (tests, tools) still get rank-layout math from their
        # world_size, with a trivial layout by definition.
        if self.topology is None:
            self.topology = ParallelTopology(self.world_size)

    @property
    def dp_size(self) -> int:
        """Data-parallel degree — the unit for data sharding and step math.

        Ranks sharing a dp_rank consume the same batch (cp peers hold
        different sequence slices of it), so sample and step counts derived
        from world size must use this instead.
        """
        return self.topology.dp_size

    @property
    def dataset_len(self) -> int:
        if self.dataset is None:
            raise RuntimeError("Training dataset has not been initialized")
        return len(self.dataset)

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def request_stop(self) -> None:
        self._stop_event.set()

    def close_dataloader(self) -> None:
        """Stop DataLoader workers so their death cannot abort the shutdown save."""
        train_iterator = self.dataloader_iter
        self.dataloader_iter = None
        self._stop_iterator(train_iterator)
        self._stop_iterator(getattr(self.val_dataloader, "_iterator", None))

    @staticmethod
    def _stop_iterator(iterator) -> None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is None:
            return
        try:
            shutdown()
        except Exception:
            logger.debug("DataLoader worker shutdown failed", exc_info=True)

    @property
    def samples_per_step(self) -> int:
        """Samples one optimizer step consumes; the unit of the data cursor."""
        return samples_per_step(self.config, self.dp_size)

    @property
    def optimizer_step(self) -> int:
        """Committed optimizer steps; the checkpoint's ``meta.optimizer_step``."""
        return self.optimizer_steps_completed
