import logging
from typing import Optional, List, cast
from torch.utils.data import DataLoader

from khaosz.core import ModelParameter, Checkpoint
from khaosz.trainer.data_util import RandomSampler
from khaosz.trainer.strategy import TrainConfig, ScheduleConfig
from khaosz.trainer.trainer_callback import (
    TrainerCallback, 
    ProgressBarCallback, 
    CheckpointCallback, 
    GradientClippingCallback,
    SchedulerCallback
)

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        parameter: ModelParameter,
        train_config: TrainConfig,
        schedule_config: ScheduleConfig,
        callbacks: Optional[List[TrainerCallback]] = None
    ):
        self.parameter = parameter
        self.train_config = train_config
        self.schedule_config = schedule_config
        self.callbacks = callbacks or self._get_default_callbacks()
        
    def _get_default_callbacks(self) -> List[TrainerCallback]:
        return [
            ProgressBarCallback(),
            CheckpointCallback(self.train_config.checkpoint_interval),
            GradientClippingCallback(),
            SchedulerCallback(self.schedule_config),
        ]

    def _set_train_kwargs(self, kwargs: dict):
        seed = self.train_config.random_seed
        sampler = RandomSampler(data_source=self.train_config.dataset, seed=seed)
        optim = self.train_config.optimizer
        checkpoint = cast(Checkpoint, kwargs.get('checkpoint', None))
        
        if checkpoint is None:
            checkpoint = Checkpoint(
                model=self.parameter.model,
                tokenizer=self.parameter.tokenizer,
                config=self.parameter.config,
                sampler_state=None,
                optim_state=None,
                loss_list=[]
            )
        
        sampler_state = checkpoint.sampler_state
        optim_state = checkpoint.optim_state
        
        if sampler_state: 
            sampler.load_state_dict(sampler_state)
        
        if optim_state: 
            optim.load_state_dict(optim_state)
            
        checkpoint.optim_state = optim.state_dict()
        checkpoint.sampler_state = sampler.state_dict()

        dataloader = DataLoader(
            self.train_config.dataset, 
            batch_size=self.train_config.batch_size, 
            sampler=sampler
        )
        
        kwargs["dataloader"] = dataloader
        kwargs["optimizer"] = self.train_config.optimizer
        kwargs["epoch"] = sampler.epoch
        kwargs["current_iter"] = sampler.current_iter
        kwargs["sampler"] = sampler
        kwargs["checkpoint"] = checkpoint
        
    def _call_callbacks(self, method_name: str, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method:
                method(self, **kwargs)

    def train(
        self,
        checkpoint: Optional[Checkpoint] = None
    ) -> Checkpoint:
        
        # train        
        train_kwargs = {
            'checkpoint': checkpoint,
            'dataloader': None,
            'optimizer': None,
            'sampler': None,
            'epoch':  0,
            'current_iter': 0, 
            'loss': 0.0,
        }
        
        self._set_train_kwargs(train_kwargs)
        self._call_callbacks('on_train_begin', **train_kwargs)
        
        dataloader = train_kwargs['dataloader']
        checkpoint = train_kwargs['checkpoint']
        start_epoch = train_kwargs['epoch']
        
        try:
            self.parameter.model.train()
            for epoch in range(start_epoch, self.train_config.n_epoch):
                # epoch
                train_kwargs["epoch"] = epoch
                self._call_callbacks('on_epoch_begin', **train_kwargs)
                for batch in dataloader:
                    
                    if train_kwargs["current_iter"] % self.train_config.accumulation_steps == 0:
                        # step
                        self._call_callbacks('on_step_begin', **train_kwargs)
                        self.train_config.optimizer.step()
                        self.train_config.optimizer.zero_grad()
                        self._call_callbacks('on_step_end', **train_kwargs)
                        
                    # batch
                    self._call_callbacks('on_batch_begin', **train_kwargs)
                    loss = self.train_config.strategy(batch)
                    train_kwargs["loss"] = loss.item()
                    train_kwargs["current_iter"] += 1
                    loss.backward()
                    
                    self._call_callbacks('on_batch_end', **train_kwargs)

                self._call_callbacks('on_epoch_end', **train_kwargs)
        
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise
        finally:
            self._call_callbacks('on_train_end', **train_kwargs)
            return checkpoint