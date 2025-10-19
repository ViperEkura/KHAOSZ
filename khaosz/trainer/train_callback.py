import os
import json
import time

from pathlib import Path
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from typing import List, Optional, Protocol, TYPE_CHECKING

from khaosz.config import ScheduleConfig
from khaosz.trainer.metric_util import (
    grad_max,
    grad_min,
    grad_norm,
    grad_mean,
    grad_std,
    grad_nan_num
)

if TYPE_CHECKING:
    from khaosz.trainer.trainer import Trainer
    from khaosz.trainer.train_context import TrainContext


class TrainCallback(Protocol):
    """ 
    Callback interface for trainer.
    """

    def on_train_begin(self, trainer: 'Trainer', context: 'TrainContext'):
        """ Called at the beginning of training. """
    
    def on_train_end(self, trainer: 'Trainer', context: 'TrainContext'):
        """ Called at the end of training. """

    def on_epoch_begin(self, trainer: 'Trainer', context: 'TrainContext'):
        """ Called at the beginning of each epoch. """

    def on_epoch_end(self, trainer: 'Trainer', context: 'TrainContext'):
        """  Called at the end of each epoch. """
    
    def on_step_begin(self, trainer: 'Trainer', context: 'TrainContext'):
        """ Called at the beginning of each step. """

    def on_step_end(self, trainer: 'Trainer', context: 'TrainContext'):
        """ Called at the end of each step."""

    def on_batch_begin(self, trainer: 'Trainer', context: 'TrainContext'):
        """ Called at the beginning of each batch. """

    def on_batch_end(self, trainer: 'Trainer', context: 'TrainContext'):
        """ Called at the end of each batch. """
    
    def on_error(self, trainer: 'Trainer', context: 'TrainContext'):
        """ Called when an error occurs during training. """


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
        
        self.scheduler = context.scheduler
    
    def on_batch_end(self, trainer: 'Trainer', context: 'TrainContext'):
        _ = trainer, context
        if self.scheduler:
            self.scheduler.step()


class CheckpointCallback(TrainCallback):
    """ 
    Checkpoint callback for trainer.
    """
    def __init__(self, checkpoint_interval: int):
        self.checkpoint_interval = checkpoint_interval
        self.last_ckpt_iter = 0
    
    def _save_checkpoint(self, trainer: 'Trainer', context: 'TrainContext'):
        save_path = os.path.join(trainer.train_config.checkpoint_dir, f"iter_{context.batch_iter}")
        context.checkpoint.optimizer_state = context.optimizer.state_dict()
        context.checkpoint.scheduler_state = context.scheduler.state_dict()
        context.checkpoint.epoch = context.epoch
        context.checkpoint.batch_iter = context.batch_iter
        context.checkpoint.save(save_path)
        self.last_ckpt_iter = context.batch_iter
    
    def on_batch_end(self, trainer: 'Trainer', context: 'TrainContext'):
        context.checkpoint.loss_list.append(context.loss)
        
        if context.batch_iter - self.last_ckpt_iter >= self.checkpoint_interval:
            self._save_checkpoint(trainer, context)
            
    def on_train_end(self, trainer: 'Trainer', context: 'TrainContext'):
        if context.batch_iter != self.last_ckpt_iter:
            self._save_checkpoint(trainer, context)


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


class StepMonitorCallback(TrainCallback):
    """
    Customizable logger callback for trainer.
    
    This callback provides flexible logging capabilities for training metrics,
    supporting multiple log formats and custom log handlers.
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_interval: int = 100,
        metrics: Optional[List[str]] = None
    ):
        """
        Args:
            log_dir: Directory to save log files. If None, logs won't be saved to file.
            log_interval: Log every N steps
            metrics: List of metrics to log. Supported: ['loss', 'lr', 'grad_norm', 'grad_std',
                    grad_max', 'grad_min', 'grad_mean', 'grad_nan_num']
            custom_handlers: List of custom log handler functions
            json_log: Whether to save logs in JSON format
        """
        
        self.log_dir = Path(log_dir) if log_dir else Path(os.getcwd()) / "logs"
        self.log_interval = log_interval
        self.metrics = metrics or ['loss', 'lr']
        self.step_num = 0
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _handle_info(self, trainer: 'Trainer', context: 'TrainContext'):
        """ Logs training information to console and file. """
        
        log_data = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "epoch": context.epoch,
            "iter": context.batch_iter,
            "metrics": self.metrics,
        }
        
        for metric in self.metrics:
            if metric == 'loss':
                log_data[metric] = context.loss
            elif metric == 'lr':
                log_data[metric] = context.optimizer.param_groups[-1]['lr']
            elif metric == 'grad_norm':
                log_data[metric] = grad_norm(trainer.parameter.model)
            elif metric == 'grad_std':
                log_data[metric] = grad_std(trainer.parameter.model)
            elif metric == 'grad_max':
                log_data[metric] = grad_max(trainer.parameter.model)
            elif metric == 'grad_min':
                log_data[metric] = grad_min(trainer.parameter.model)
            elif metric == 'grad_mean':
                log_data[metric] = grad_mean(trainer.parameter.model)
            elif metric == 'grad_nan_num':
                log_data[metric] = grad_nan_num(trainer.parameter.model)
            else:
                raise ValueError(f"Invalid metric: {metric}")
        
        return log_data
    
    def _handle_log(self, trainer: 'Trainer', context: 'TrainContext'):
        """ Logs training information to console and file. """
        log_data = self._handle_info(trainer, context)
        try:
            log_file = self.log_dir / f"log_epoch_{context.epoch}_iter_{context.batch_iter}.json"
            with open(log_file, 'a') as f:
                json.dump(log_data, f, indent=4)
        except Exception:
            raise
    
    def on_step_end(self, trainer: 'Trainer', context: 'TrainContext'):
        if self.step_num % self.log_interval == 0:
            self._handle_log(trainer, context)
        
        self.step_num += 1