import torch
import itertools
from typing import Optional, List
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


class Trainer:
    def __init__(
        self,
        parameter: ModelParameter,
        train_config: TrainConfig,
        schedule_config: ScheduleConfig,
        callbacks: Optional[List[TrainerCallback]] = None
    ):
        self.checkpoint = Checkpoint(
            model=parameter.model,
            tokenizer=parameter.tokenizer,
            config=parameter.config,
        )
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
        
    def _create_dataloader(self, start_index: int = 0) -> DataLoader:
        seed = self.train_config.random_seed
        generator = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(
            self.train_config.dataset, 
            generator=generator,
            seed=seed
        )
        dataloader = DataLoader(
            self.train_config.dataset, 
            batch_size=self.train_config.batch_size, 
            sampler=sampler
        )
        
        if start_index > 0:
            dataloader = itertools.islice(dataloader, start_index, None)
        
        return dataloader

    def _call_callbacks(self, method_name: str, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method:
                method(self, **kwargs)
        
    def train(
        self,
        train_checkpoint: Optional[Checkpoint] = None
    ) -> Checkpoint:
                
        if train_checkpoint:
            self.checkpoint = train_checkpoint
            self.train_config.optimizer.load_state_dict(train_checkpoint.optim_state)
        else:
            self.checkpoint.optim_state = self.train_config.optimizer.state_dict()
            
        current_iter = len(self.checkpoint.loss_list)
        total_steps_per_epoch = len(self.train_config.dataset) // self.train_config.batch_size
        total_reamining_steps = total_steps_per_epoch * self.train_config.n_epoch - current_iter
        
        current_epochs = total_reamining_steps // total_steps_per_epoch 
        current_steps = total_reamining_steps % total_steps_per_epoch

        # train
        self._call_callbacks('on_train_begin', checkpoint=self.checkpoint)
        self.checkpoint.model.train()
        
        try:
            for epoch in range(current_epochs):
                # epoch
                self._call_callbacks('on_epoch_begin', epoch=epoch)
                dataloader = self._create_dataloader(start_index=current_steps)
                for batch in dataloader:
                    # batch
                    self._call_callbacks('on_batch_begin', batch=batch)
                    loss = self.train_config.strategy(batch)
                    self.checkpoint.loss_list.append(loss.item())
                    loss.backward()
                    self._call_callbacks('on_batch_end', batch=batch, loss=loss.item(), current_iter=current_iter)
                    
                    if current_iter % self.train_config.accumulation_steps == 0:
                        # step
                        self._call_callbacks('on_step_begin', current_iter=current_iter)
                        self.train_config.optimizer.step()
                        self.train_config.optimizer.zero_grad()
                        self._call_callbacks('on_step_end', current_iter=current_iter)
                    
                    current_iter += 1
                
                self._call_callbacks('on_epoch_end', epoch=epoch, loss_list=self.checkpoint.loss_list)
                
        except Exception as e:
            raise e

        finally:
            self._call_callbacks('on_train_end', checkpoint=self.checkpoint)
            return self.checkpoint