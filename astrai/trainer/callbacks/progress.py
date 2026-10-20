"""Progress display callback."""

from __future__ import annotations

import sys
from typing import IO, TYPE_CHECKING, Optional

from tqdm import tqdm

from astrai.parallel import only_on_rank
from astrai.trainer.callbacks.base import CallbackFactory, TrainCallback

if TYPE_CHECKING:
    from astrai.trainer.context import TrainContext


@CallbackFactory.register("progress_bar")
class ProgressBarCallback(TrainCallback):
    """
    Progress bar callback for trainer.
    """

    def __init__(
        self, num_epoch: int, log_interval: int = 100, file: Optional[IO[str]] = None
    ):
        self.num_epoch = num_epoch
        self.log_interval = log_interval
        self.file = file
        self.progress_bar: tqdm = None

    @only_on_rank(0)
    def on_epoch_begin(self, context: TrainContext):
        if context.global_cursor:
            config = context.config
            global_batch = context.samples_per_step
            full, tail = divmod(
                context.dataset_len - context.epoch_sample_offset, global_batch
            )
            flush_tail = tail > 0 and config.epoch_tail_policy == "flush"
            if context.strategy.policy_version is not None:
                group_size = config.rl_minibatch_prompts or global_batch
                full_groups = (global_batch + group_size - 1) // group_size
                tail_groups = (tail + group_size - 1) // group_size if flush_tail else 0
                total_steps = config.rl_update_epochs * (
                    full * full_groups + tail_groups
                )
            else:
                total_steps = full + int(flush_tail)
        elif (
            context.async_rollout is None and context.config.epoch_tail_policy == "drop"
        ):
            full_batches = (
                len(context.dataloader.sampler) // context.config.batch_per_device
            )
            total_steps = full_batches // context.executor.grad_accum_steps
        else:
            total_steps = (
                len(context.dataloader) + context.executor.grad_accum_steps - 1
            ) // context.executor.grad_accum_steps
        self.progress_bar = tqdm(
            total=total_steps,
            desc=f"Epoch {context.epoch + 1}/{self.num_epoch}",
            dynamic_ncols=True,
            file=self.file or sys.stdout,
        )

    @only_on_rank(0)
    def before_optimizer_step(self, context: TrainContext):
        postfix = {
            "step": f"{context.optimizer_step:d}",
            "loss": f"{context.loss:.4f}",
            "lr": f"{context.optimizer.param_groups[-1]['lr']:.2e}",
        }
        if context.grad_norm is not None:
            postfix["grad_norm"] = f"{context.grad_norm:.2f}"
        if context.val_loss is not None:
            postfix["val_loss"] = f"{context.val_loss:.4f}"
        self.progress_bar.set_postfix(postfix)
        self.progress_bar.update(1)

    @only_on_rank(0)
    def on_epoch_end(self, context: TrainContext):
        _ = context
        if self.progress_bar:
            self.progress_bar.close()
