import logging
import random
from typing import List, Optional

import numpy as np
import torch

from astrai.config import TrainConfig
from astrai.parallel.setup import agree_stop, get_rank, spawn_parallel_fn
from astrai.trainer.callbacks import (
    CallbackFactory,
    TrainCallback,
)
from astrai.trainer.session import TrainSession
from astrai.trainer.train_context import TrainContext, TrainContextBuilder

logger = logging.getLogger(__name__)


def seed_everything(seed: int, rank: int = 0) -> None:
    """Seed the global generators; resume restores the checkpoint's RNG states."""
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(
        self, train_config: TrainConfig, callbacks: Optional[List[TrainCallback]] = None
    ):
        self.train_config = train_config
        default_callbacks = self._get_default_callbacks()
        self.callbacks = (
            default_callbacks + callbacks if callbacks else default_callbacks
        )

    def _get_default_callbacks(self) -> List[TrainCallback]:
        cfg = self.train_config
        callbacks = [
            CallbackFactory.create_checked(
                "gradient_checkpointing",
                modules=cfg.gradient_checkpointing_modules,
            ),
            CallbackFactory.create_checked(
                "checkpoint",
                cfg.ckpt_dir,
                cfg.ckpt_interval,
            ),
            CallbackFactory.create_checked(
                "metric",
                ckpt_dir=cfg.ckpt_dir,
                save_interval=cfg.ckpt_interval,
                metrics=cfg.metrics,
                val_step=cfg.val_step,
                grad_snr_interval=cfg.grad_snr_interval,
            ),
            CallbackFactory.create_checked("progress_bar", cfg.n_epoch),
            CallbackFactory.create_checked("gradient_clipping", cfg.max_grad_norm),
        ]
        return callbacks

    def _call_callbacks(self, method_name: str, context: TrainContext):
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method:
                method(context)

    def _commit_update(self, context: TrainContext, batch, new_data: bool = True):
        if context.async_rollout is not None:
            self._commit_online_update(context, batch, new_data)
        else:
            self._commit_offline_update(context, new_data)
        self._call_callbacks("after_optimizer_step", context)
        if context.async_rollout is None:
            self._agree_stop(context)

    def _commit_offline_update(self, context: TrainContext, new_data: bool) -> None:
        """Apply one step; both counters roll back if the step itself fails.

        The step advances before the observers run, so the hooks that log or
        validate record the step they belong to; advancing it after would
        label every logged step with the previous one.
        """
        context.optimizer_steps_completed += 1
        if new_data:
            context.consumed_samples += context.samples_per_step
        try:
            self._call_callbacks("before_optimizer_step", context)
            context.strategy.optimizer_step(context.optimizer)
            context.optimizer.zero_grad()
            if context.scheduler:
                context.scheduler.step()
        except BaseException:
            context.optimizer_steps_completed -= 1
            if new_data:
                context.consumed_samples -= context.samples_per_step
            raise

    def _commit_online_update(
        self, context: TrainContext, batch, new_data: bool
    ) -> None:
        """Publish one policy version; the cursor follows the version, not the call."""
        self._call_callbacks("before_optimizer_step", context)
        coordinator = context.async_rollout
        context.checkpoint_safe = False
        version_before = coordinator.policy_version
        try:
            context.strategy.optimizer_step(context.optimizer)
        finally:
            # A version advance marks a committed step, even if later
            # publication fails and an error checkpoint is needed.
            if coordinator.policy_version > version_before:
                context.optimizer.zero_grad()
                if context.scheduler:
                    context.scheduler.step()
                context.optimizer_steps_completed += 1
                if new_data:
                    context.consumed_samples += batch.prompts.shape[0] * context.dp_size
                context.checkpoint_safe = True

    def _train_batch(self, context: TrainContext, batch) -> None:
        executor = context.executor
        with executor.accumulate(context.model):
            self._call_callbacks("on_batch_begin", context)
            microbatch_prompts = (
                context.config.async_train_microbatch_prompts
                if context.async_rollout is not None
                else None
            )
            # A batch can produce several updates (replayed rollout rounds) or
            # none (mid-window). Steps follow every commit; the data cursor
            # moves once per batch, because a replayed batch is one position.
            committed = False
            for update in context.strategy.training_updates(batch, microbatch_prompts):
                update_loss = None
                update_metrics = {}
                for output in update:
                    loss = output["loss"]
                    executor.backward(loss / executor.grad_accum_steps)
                    detached = loss.detach()
                    update_loss = (
                        detached if update_loss is None else update_loss + detached
                    )
                    for name, value in output["metrics"].items():
                        update_metrics[name] = update_metrics.get(name, 0.0) + float(
                            value
                        )
                if update_loss is None:
                    raise RuntimeError("training update has no valid loss")
                context.loss = float(update_loss.item())
                context.metrics = update_metrics
                if executor.sync_gradients:
                    self._commit_update(context, batch, new_data=not committed)
                    committed = True
            self._call_callbacks("on_batch_end", context)

    def _should_stop(self, context: TrainContext) -> bool:
        """Online training stops between rounds; sync training at an agreed step."""
        if context.async_rollout is not None:
            return context.stop_requested
        return context.stop_agreed

    def _agree_stop(self, context: TrainContext) -> bool:
        """Agree on a stop with every rank, so none leaves a peer mid-collective."""
        if context.async_rollout is not None:
            return context.stop_requested
        device = next(context.model.parameters()).device
        if agree_stop(context.stop_requested, device):
            context.request_stop()
            context.stop_agreed = True
        return context.stop_agreed

    def _train_epoch(self, context: TrainContext, epoch: int) -> None:
        context.epoch = epoch
        self._call_callbacks("on_epoch_begin", context)
        batches = context.dataloader
        if context.async_rollout is not None:
            batches = context.async_rollout.iter_rounds(
                batches, lambda: context.stop_requested
            )
        # Finish the current window first: a partial window has no valid cursor.
        if not context.stop_agreed:
            self._agree_stop(context)
        iterator = iter(batches)
        context.dataloader_iter = iterator
        while not self._should_stop(context):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            self._train_batch(context, batch)
        self._call_callbacks("on_epoch_end", context)

    def _trainer_loop(self, param_path: Optional[str] = None, resume: bool = False):
        # Seed before the context is built so initialisation and data drawing start
        # from a defined state; a resume overwrites this with the saved states.
        seed_everything(self.train_config.random_seed, get_rank())
        with TrainSession(
            lambda: (
                TrainContextBuilder(self.train_config)
                .with_param_path(param_path, resume=resume)
                .build()
            ),
            self._call_callbacks,
        ) as context:
            context.model.train()
            for epoch in range(context.epoch, context.config.n_epoch):
                if context.stop_agreed or self._agree_stop(context):
                    break
                self._train_epoch(context, epoch)

            if context.stop_requested:
                logger.warning(
                    "Training interrupted by signal, saving emergency checkpoint..."
                )

    def train(self, param_path: Optional[str] = None, resume: bool = False):
        cfg = self.train_config
        spawn_parallel_fn(
            self._trainer_loop,
            backend=cfg.backend,
            world_size=cfg.nprocs,
            master_addr=cfg.master_addr,
            master_port=cfg.master_port,
            device_type=cfg.device_type,
            start_method=cfg.start_method,
            process_group_timeout=cfg.process_group_timeout,
            param_path=param_path,
            resume=resume,
        )
