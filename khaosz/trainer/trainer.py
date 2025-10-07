import logging
from typing import Optional, List

from khaosz.core import ModelParameter, Checkpoint
from khaosz.trainer.strategy import ScheduleConfig
from khaosz.trainer.train_config import TrainConfig
from khaosz.trainer.train_callback import (
    TrainCallback, 
    ProgressBarCallback, 
    CheckpointCallback, 
    GradientClippingCallback,
    SchedulerCallback
)
from khaosz.trainer.train_context import TrainContext, TrainContextBuilder

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        parameter: ModelParameter,
        train_config: TrainConfig,
        schedule_config: ScheduleConfig,
        callbacks: Optional[List[TrainCallback]] = None
    ):
        self.parameter = parameter
        self.train_config = train_config
        self.schedule_config = schedule_config
        self.callbacks = callbacks or self._get_default_callbacks()

    def _get_default_callbacks(self) -> List[TrainCallback]:
        return [
            ProgressBarCallback(),
            CheckpointCallback(self.train_config.checkpoint_interval),
            GradientClippingCallback(),
            SchedulerCallback(self.schedule_config),
        ]
        
    def _build_train_context(self, checkpoint: Optional[Checkpoint]) -> TrainContext:
        return (TrainContextBuilder(self)
                .with_checkpoint(checkpoint)
                .with_sampler()
                .with_optimizer()
                .with_dataloader()
                .build())
    
    def _call_callbacks(self, method_name: str, context: TrainContext):
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method:
                method(self, context)

    def train(self, checkpoint: Optional[Checkpoint] = None) -> Checkpoint:
        context = self._build_train_context(checkpoint)
        
        self._call_callbacks('on_train_begin', context)
        
        try:
            self.parameter.model.train()
            # 1.epoch
            for epoch in range(context.epoch, self.train_config.n_epoch):
                context.epoch = epoch
                self._call_callbacks('on_epoch_begin', context)
                
                for batch in context.dataloader:
                    if context.current_iter % self.train_config.accumulation_steps == 0:
                        # 2. step
                        self._call_callbacks('on_step_begin', context)
                        self.train_config.optimizer.step()
                        self.train_config.optimizer.zero_grad()
                        self._call_callbacks('on_step_end', context)
                    
                    # 3. batch
                    self._call_callbacks('on_batch_begin', context)
                    loss = self.train_config.strategy(batch)
                    context.loss = loss.item()
                    context.current_iter += 1
                    
                    # to make the loss normalized by accumulation steps
                    normalized_loss = loss / self.train_config.accumulation_steps
                    normalized_loss.backward()
                    
                    self._call_callbacks('on_batch_end', context)

                self._call_callbacks('on_epoch_end', context)
        
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            self._call_callbacks('on_error', context)
            raise
        finally:
            self._call_callbacks('on_train_end', context)
            return context.checkpoint