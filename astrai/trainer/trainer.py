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
            CallbackFactory.create("progress_bar", cfg.n_epoch),
            CallbackFactory.create("checkpoint", cfg.ckpt_dir, cfg.ckpt_interval),
            CallbackFactory.create("metric_logger", cfg.ckpt_dir, cfg.ckpt_interval),
            CallbackFactory.create("gradient_clipping", cfg.max_grad_norm),
            CallbackFactory.create("scheduler"),
        ]

    def _build_context(self, checkpoint: Optional[Checkpoint]) -> TrainContext:
        return (
            TrainContextBuilder(self.train_config)
            .with_checkpoint(checkpoint)
            .with_dataloader()
            .with_strategy()
            .build()
        )

    def _call_callbacks(self, method_name: str, context: TrainContext):
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method:
                method(context)

    def train(self, checkpoint: Optional[Checkpoint] = None):
        config = self.train_config
        spawn_parallel_fn(
            self._train_impl,
            backend=config.backend,
            world_size=config.nprocs,
            master_addr=config.master_addr,
            master_port=config.master_port,
            device_type=config.device_type,
            device_ids=config.device_ids,
            checkpoint=checkpoint,
        )

    def _train_impl(self, checkpoint: Optional[Checkpoint] = None) -> Checkpoint:
        context = self._build_context(checkpoint)
        self._call_callbacks("on_train_begin", context)

        try:
            context.model.train()
            # 1.epoch
            for epoch in range(context.epoch, self.train_config.n_epoch):
                context.epoch = epoch
                self._call_callbacks("on_epoch_begin", context)

                for batch in context.dataloader:
                    if context.iteration % self.train_config.accumulation_steps == 0:
                        # 2. step
                        self._call_callbacks("on_step_begin", context)
                        context.optimizer.step()
                        context.optimizer.zero_grad()
                        self._call_callbacks("on_step_end", context)

                    # 3. batch
                    self._call_callbacks("on_batch_begin", context)
                    loss = context.strategy(batch)
                    context.loss = loss.item()
                    context.iteration += 1

                    # to make the loss normalized by accumulation steps
                    stand_loss = loss / self.train_config.accumulation_steps
                    stand_loss.backward()

                    self._call_callbacks("on_batch_end", context)

                self._call_callbacks("on_epoch_end", context)

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            self._call_callbacks("on_error", context)
            raise
        finally:
            self._call_callbacks("on_train_end", context)
