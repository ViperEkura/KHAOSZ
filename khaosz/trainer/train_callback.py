import os
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, TYPE_CHECKING
from khaosz.trainer.strategy import ScheduleConfig, SchedulerFactory

if TYPE_CHECKING:
    from khaosz.trainer.trainer import Trainer
    from khaosz.trainer.train_context import TrainContext


class TrainCallback:
    """ 
    Callback interface for trainer.
    and we use '_' to ignore unused parameters.
    """

    def on_train_begin(self, trainer: 'Trainer', context: 'TrainContext'):
        """ Called at the beginning of training. """
        _ = trainer, context
    
    def on_train_begin(self, trainer: 'Trainer', context: 'TrainContext'):
        """ Called at the end of training. """
        _ = trainer, context

    def on_train_begin(self, trainer: 'Trainer', context: 'TrainContext'):
        """ Called at the beginning of each epoch. """
        _ = trainer, context

    def on_train_begin(self, trainer: 'Trainer', context: 'TrainContext'):
        """  Called at the end of each epoch. """
        _ = trainer, context

    def on_train_begin(self, trainer: 'Trainer', context: 'TrainContext'):
        """ Called at the beginning of each batch. """
        _ = trainer, context

    def on_train_begin(self, trainer: 'Trainer', context: 'TrainContext'):
        """ Called at the end of each batch. """
        _ = trainer, context

    def on_train_begin(self, trainer: 'Trainer', context: 'TrainContext'):
        """ Called at the beginning of each step. """
        _ = trainer, context

    def on_train_begin(self, trainer: 'Trainer', context: 'TrainContext'):
        """ Called at the end of each step."""
        _ = trainer, context


class ProgressBarCallback(TrainCallback):
    """ 
    Progress bar callback for trainer.
    """
    def __init__(self):
        self.progress_bar: tqdm = None
    
    def on_epoch_begin(self, trainer: 'Trainer', context: 'TrainContext'):
        self.progress_bar = tqdm(
            context.dataloader, 
            desc=f"Epoch {context.epoch+1}/{trainer.train_config.n_epoch}", 
            dynamic_ncols=True
        )
    
    def on_batch_end(self, trainer: 'Trainer', context: 'TrainContext'):
        _ = trainer
        self.progress_bar.set_postfix({
            "loss": f"{context.loss:.4f}",
            "lr": f"{context.optimizer.param_groups[-1]['lr']:.2e}"
        })
        self.progress_bar.update(1)
    
    def on_epoch_end(self, trainer: 'Trainer', context: 'TrainContext'):
        _ = trainer, context
        if self.progress_bar:
            self.progress_bar.close()


class CheckpointCallback(TrainCallback):
    """ 
    Checkpoint callback for trainer.
    """
    def __init__(self, checkpoint_interval: int):
        self.checkpoint_interval = checkpoint_interval
        self.last_ckpt_iter = 0
    
    def _save_checkpoint(self, trainer: 'Trainer', context: 'TrainContext'):
        save_path = os.path.join(trainer.train_config.checkpoint_dir, f"iter_{context.current_iter}")
        context.checkpoint.sampler_state = context.sampler.state_dict()
        context.checkpoint.optimizer_state = context.optimizer.state_dict()
        context.checkpoint.save(save_path)
        self.last_ckpt_iter = context.current_iter
    
    def on_batch_end(self, trainer: 'Trainer', context: 'TrainContext'):
        context.checkpoint.loss_list.append(context.loss)
        
        if context.current_iter - self.last_ckpt_iter >= self.checkpoint_interval:
            self._save_checkpoint(trainer, context)
            
    def on_train_end(self, trainer: 'Trainer', context: 'TrainContext'):
        if context.current_iter != self.last_ckpt_iter:
            self._save_checkpoint(trainer, context)


class GradientClippingCallback(TrainCallback):
    """ 
    Gradient clipping callback for trainer.
    """
    def on_step_begin(self, trainer: 'Trainer', context: 'TrainContext'):
        _ = context
        clip_grad_norm_(trainer.parameter.model.parameters(), trainer.train_config.max_grad_norm)


class SchedulerCallback(TrainCallback):
    """
    Scheduler callback for trainer.
    """
    def __init__(self, schedule_config: ScheduleConfig):
        self.schedule_config = schedule_config
        self.scheduler: Optional[LambdaLR] = None
    
    def on_train_begin(self, trainer: 'Trainer', context: 'TrainContext'):

        for group in trainer.train_config.optimizer.param_groups:
            if "initial_lr" not in group:
                group["initial_lr"] = group["lr"] 

        self.schedule_config.validate()
        lambda_scheduler_fn = SchedulerFactory.load_schedule_fn(
            self.schedule_config
        )
        
        self.scheduler = LambdaLR(
            trainer.train_config.optimizer,
            lambda_scheduler_fn,
            last_epoch=context.current_iter - 1
        )
    
    def on_batch_end(self, trainer: 'Trainer', context: 'TrainContext'):
        _ = trainer, context
        if self.scheduler:
            self.scheduler.step()
