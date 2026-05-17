import logging
from typing import List, Optional

from astrai.config import TrainConfig
from astrai.parallel.setup import spawn_parallel_fn
from astrai.serialization import Checkpoint
from astrai.trainer.train_callback import (
    CallbackFactory,
    TrainCallback,
)
from astrai.trainer.train_context import TrainContext, TrainContextBuilder

logger = logging.getLogger(__name__)


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
        return [
            CallbackFactory.create(
                "checkpoint",
                cfg.ckpt_dir,
                cfg.ckpt_interval,
                state_dict_fn=cfg.state_dict_fn,
            ),
            CallbackFactory.create("progress_bar", cfg.n_epoch),
            CallbackFactory.create("metric_logger", cfg.ckpt_dir, cfg.ckpt_interval),
            CallbackFactory.create("gradient_clipping", cfg.max_grad_norm),
        ]

    def _call_callbacks(self, method_name: str, context: TrainContext):
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method:
                method(context)

    def train(self, checkpoint: Optional[Checkpoint] = None):
        cfg = self.train_config
        spawn_parallel_fn(
            self._train_impl,
            backend=cfg.backend,
            world_size=cfg.nprocs,
            master_addr=cfg.master_addr,
            master_port=cfg.master_port,
            device_type=cfg.device_type,
            start_method=cfg.start_method,
            checkpoint=checkpoint,
        )

    def _train_impl(self, checkpoint: Optional[Checkpoint] = None):
        cfg = self.train_config
        context = TrainContextBuilder(cfg).with_checkpoint(checkpoint).build()
        self._call_callbacks("on_train_begin", context)

        try:
            context.model.train()
            grad_accum_steps = cfg.grad_accum_steps

            for epoch in range(context.epoch, cfg.n_epoch):
                context.epoch = epoch
                self._call_callbacks("on_epoch_begin", context)

                for batch in context.dataloader:
                    self._call_callbacks("on_batch_begin", context)
                    loss = context.strategy(batch)
                    context.loss = loss.item()
                    stand_loss = loss / grad_accum_steps
                    stand_loss.backward()
                    context.iteration += 1
                    self._call_callbacks("on_batch_end", context)

                    if context.iteration % grad_accum_steps == 0:
                        self._call_callbacks("on_step_begin", context)
                        context.optimizer.step()
                        context.optimizer.zero_grad()
                        self._call_callbacks("on_step_end", context)

                        if context.scheduler:
                            context.scheduler.step()

                self._call_callbacks("on_epoch_end", context)

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            self._call_callbacks("on_error", context)
            raise
        finally:
            self._call_callbacks("on_train_end", context)
