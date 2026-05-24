"""Unified training backend — parallel strategy + gradient accumulation."""

import contextlib
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from astrai.factory import BaseFactory
from astrai.parallel.setup import get_rank, get_world_size

logger = logging.getLogger(__name__)


class AccumOptimizer:
    def __init__(self, optimizer: Optimizer, backend: "BaseTrainingBackend"):
        self.optimizer = optimizer
        self._backend = backend

    def step(self, closure=None):
        if self._backend._sync_gradients:
            self.optimizer.step(closure)

    def zero_grad(self):
        if self._backend._sync_gradients:
            self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, d):
        self.optimizer.load_state_dict(d)


class AccumScheduler:
    def __init__(self, scheduler: LRScheduler, backend: "BaseTrainingBackend"):
        self.scheduler = scheduler
        self._backend = backend

    def step(self):
        if self._backend._sync_gradients:
            self.scheduler.step()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, d):
        self.scheduler.load_state_dict(d)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()


class BaseTrainingBackend(ABC):
    def __init__(self, grad_accum_steps: int = 1):
        self.grad_accum_steps = max(grad_accum_steps, 1)
        self._step: int = 0
        self._sync_gradients: bool = True

    def prepare(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        dataloader: Optional[DataLoader] = None,
        scheduler: Optional[LRScheduler] = None,
    ) -> Tuple[
        nn.Module, Optional[Optimizer], Optional[DataLoader], Optional[LRScheduler]
    ]:
        model = self._prepare_model(model)
        if optimizer is not None:
            optimizer = AccumOptimizer(optimizer, self)
        if scheduler is not None:
            scheduler = AccumScheduler(scheduler, self)
        return model, optimizer, dataloader, scheduler

    @abstractmethod
    def _prepare_model(self, model: nn.Module) -> nn.Module:
        pass

    def _no_sync(self, model: nn.Module):
        return contextlib.nullcontext()

    @contextmanager
    def accumulate(self, model: nn.Module):
        self._step += 1
        self._sync_gradients = self._step % self.grad_accum_steps == 0
        if not self._sync_gradients:
            with self._no_sync(model):
                yield
        else:
            yield

    def backward(self, loss: torch.Tensor):
        loss.backward()

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model

    @property
    def use_distributed(self) -> bool:
        return get_world_size() > 1


class BackendFactory(BaseFactory[BaseTrainingBackend]):
    pass


@BackendFactory.register("single")
class SingleDeviceBackend(BaseTrainingBackend):
    def _prepare_model(self, model: nn.Module) -> nn.Module:
        return model


@BackendFactory.register("ddp")
class DDPTrainingBackend(BaseTrainingBackend):
    def __init__(
        self,
        grad_accum_steps: int = 1,
        dim: int = 0,
        broadcast_buffers: bool = True,
        init_sync: bool = True,
        process_group=None,
        bucket_cap_mb: int = 25,
        find_unused_parameters: bool = False,
        check_reduction: bool = False,
        gradient_as_bucket_view: bool = False,
        static_graph: bool = False,
        delay_all_reduce_named_params=None,
        param_to_hook_all_reduce=None,
        mixed_precision=None,
        device_mesh=None,
    ):
        super().__init__(grad_accum_steps=grad_accum_steps)
        self._ddp_kwargs = dict(
            dim=dim,
            broadcast_buffers=broadcast_buffers,
            init_sync=init_sync,
            process_group=process_group,
            bucket_cap_mb=bucket_cap_mb,
            find_unused_parameters=find_unused_parameters,
            check_reduction=check_reduction,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
            delay_all_reduce_named_params=delay_all_reduce_named_params,
            param_to_hook_all_reduce=param_to_hook_all_reduce,
            mixed_precision=mixed_precision,
            device_mesh=device_mesh,
        )

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        if not self.use_distributed:
            logger.warning("DDP backend selected but world_size=1, model not wrapped")
            return model
        local_rank = get_rank()
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            **self._ddp_kwargs,
        )
        logger.info("Model wrapped with DDP (world_size=%d)", get_world_size())
        return model

    def _no_sync(self, model: nn.Module):
        if isinstance(model, DDP):
            return model.no_sync()
        return contextlib.nullcontext()

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        if isinstance(model, DDP):
            return model.module
        return model
