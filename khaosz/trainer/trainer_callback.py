import os
from tqdm import tqdm
from khaosz.core.parameter import Checkpoint
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, cast, TYPE_CHECKING
from khaosz.trainer.strategy import ScheduleConfig, SchedulerFactory

if TYPE_CHECKING:
    from khaosz.trainer.trainer import Trainer


class TrainerCallback:
    """ 
    Callback interface for trainer.
    and we use '_' to ignore unused parameters.
    """

    def on_train_begin(self, trainer: 'Trainer', **kwargs):
        """ 
        Called at the beginning of training.
        """
        _ = trainer, kwargs
    
    def on_train_end(self, trainer: 'Trainer', **kwargs):
        """ 
        Called at the end of training.
        """
        _ = trainer, kwargs

    def on_epoch_begin(self, trainer: 'Trainer', **kwargs):
        """ 
        Called at the beginning of each epoch.
        """
        _ = trainer, kwargs

    def on_epoch_end(self, trainer: 'Trainer', **kwargs):
        """ 
        Called at the end of each epoch.
        """
        _ = trainer, kwargs

    def on_batch_begin(self, trainer: 'Trainer', **kwargs):
        """ 
        Called at the beginning of each batch.
        """
        _ = trainer, kwargs

    def on_batch_end(self, trainer: 'Trainer', **kwargs):
        """ 
        Called at the end of each batch.
        """
        _ = trainer, kwargs

    def on_step_begin(self, trainer: 'Trainer', **kwargs):
        """ 
        Called at the beginning of each step.
        """

        _ = trainer, kwargs

    def on_step_end(self, trainer: 'Trainer', **kwargs):
        """ 
        Called at the end of each step.
        """

        _ = trainer, kwargs


class ProgressBarCallback(TrainerCallback):
    """ 
    Progress bar callback for trainer.
    """
    def __init__(self):
        self.progress_bar: tqdm = None
    
    def on_epoch_begin(self, trainer: 'Trainer', **kwargs):
        epoch = kwargs.get('epoch')
        dataloader = trainer._create_dataloader()
        self.progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}/{trainer.train_config.n_epoch}", 
            dynamic_ncols=True
        )
    
    def on_batch_end(self, trainer: 'Trainer', **kwargs):
        loss = kwargs.get('loss')
        self.progress_bar.set_postfix({
            "loss": f"{loss:.4f}",
            "lr": f"{trainer.train_config.optimizer.param_groups[0]['lr']:.2e}"
        })
        self.progress_bar.update(1)
    
    def on_epoch_end(self, trainer: 'Trainer', **kwargs):
        _ = trainer, kwargs
        if self.progress_bar:
            self.progress_bar.close()


class CheckpointCallback(TrainerCallback):
    """ 
    Checkpoint callback for trainer.
    """
    def __init__(self, checkpoint_interval: int):
        self.checkpoint_interval = checkpoint_interval
        self.last_ckpt_iter = 0
    
    @staticmethod
    def _save_checkpoint(trainer: 'Trainer'):
        current_iter = len(trainer.checkpoint.loss_list)
        save_path = os.path.join(trainer.train_config.checkpoint_dir, f"iter_{current_iter}")
        trainer.checkpoint.optim_state = trainer.train_config.optimizer.state_dict()
        trainer.checkpoint.save(save_path)
    
    def on_train_begin(self, trainer: 'Trainer', **kwargs):
        _ = trainer
        checkpoint = cast(Checkpoint, kwargs.get('checkpoint'))
        self.last_ckpt_iter = len(checkpoint.loss_list)
    
    def on_batch_end(self, trainer: 'Trainer', **kwargs):
        current_iter = kwargs.get('current_iter')
        if current_iter - self.last_ckpt_iter >= self.checkpoint_interval:
            CheckpointCallback._save_checkpoint(trainer)
            self.last_ckpt_iter = current_iter
    
    def on_train_end(self, trainer: 'Trainer', **kwargs):
        checkpoint = cast(Checkpoint, kwargs.get('checkpoint'))
        current_iter = len(checkpoint.loss_list)
        if current_iter != self.last_ckpt_iter:
            CheckpointCallback._save_checkpoint(trainer)


class GradientClippingCallback(TrainerCallback):
    """ 
    Gradient clipping callback for trainer.
    """
    def on_step_begin(self, trainer: 'Trainer', **kwargs):
        _ = kwargs
        clip_grad_norm_(
            trainer.checkpoint.model.parameters(),
            trainer.train_config.max_grad_norm
        )


class SchedulerCallback(TrainerCallback):
    """
    Scheduler callback for trainer.
    """
    def __init__(self, schedule_config: ScheduleConfig):
        self.schedule_config = schedule_config
        self.scheduler: Optional[LambdaLR] = None
        self.current_iter = 0
    
    def on_train_begin(self, trainer: 'Trainer', **kwargs):
        checkpoint = cast(Checkpoint, kwargs.get('checkpoint'))
        self.current_iter = len(checkpoint.loss_list)

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
            last_epoch=self.current_iter - 1
        )
    
    def on_step_end(self, trainer: 'Trainer', **kwargs):
        _ =  trainer, kwargs
        
        if self.scheduler:
            self.scheduler.step()
            self.current_iter += 1
