import os
import json
import time
import torch.nn as nn

from pathlib import Path
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LRScheduler
from typing import Callable, Optional, Protocol

from khaosz.parallel import only_on_rank
from khaosz.trainer.metric_util import (
    grad_max,
    grad_min,
    grad_norm,
    grad_mean,
    grad_std,
    grad_nan_num
)
from khaosz.data.checkpoint import Checkpoint
from khaosz.trainer.train_context import TrainContext


class TrainCallback(Protocol):
    """ 
    Callback interface for trainer.
    """

    def on_train_begin(self, context: TrainContext):
        """ Called at the beginning of training. """
    
    def on_train_end(self, context: TrainContext):
        """ Called at the end of training. """

    def on_epoch_begin(self, context: TrainContext):
        """ Called at the beginning of each epoch. """

    def on_epoch_end(self, context: TrainContext):
        """  Called at the end of each epoch. """
    
    def on_step_begin(self, context: TrainContext):
        """ Called at the beginning of each step. """

    def on_step_end(self, context: TrainContext):
        """ Called at the end of each step."""

    def on_batch_begin(self, context: TrainContext):
        """ Called at the beginning of each batch. """

    def on_batch_end(self, context: TrainContext):
        """ Called at the end of each batch. """
    
    def on_error(self, context: TrainContext):
        """ Called when an error occurs during training. """


class GradientClippingCallback(TrainCallback):
    """ 
    Gradient clipping callback for trainer.
    """
    def __init__(self, max_grad_norm: float):
        self.max_grad_norm = max_grad_norm
    
    def on_step_begin(self, context: TrainContext):
        _ = context
        clip_grad_norm_(context.model.parameters(), self.max_grad_norm)


class SchedulerCallback(TrainCallback):
    """
    Scheduler callback for trainer.
    """
    def __init__(self):
        self.scheduler: LRScheduler = None
    
    def on_train_begin(self, context: TrainContext):
        for group in context.optimizer.param_groups:
            if "initial_lr" not in group:
                group["initial_lr"] = group["lr"] 
        
        self.scheduler = context.scheduler
    
    def on_batch_end(self, context: TrainContext):
        _ = context
        if self.scheduler:
            self.scheduler.step()


class CheckpointCallback(TrainCallback):
    """ 
    Checkpoint callback for trainer.
    """
    def __init__(
        self, 
        save_dir: str,  
        interval: int,
        weight_only: bool = False,
        state_dict_fn: Optional[Callable[[nn.Module], dict]] = None
    ):
        self.save_dir = save_dir
        self.interval = interval
        self.weight_only = weight_only
        self.state_dict_fn = state_dict_fn
        self.last_ckpt_iter = 0
    
    def _save_checkpoint(self, context: TrainContext):
        save_path = os.path.join(self.save_dir, f"epoch_{context.epoch}_iter_{context.iteration}")
        state_dict = self.state_dict_fn(context.model) if self.state_dict_fn else context.model.state_dict()
        
        context.checkpoint = Checkpoint(
            state_dict=state_dict,
            epoch=context.epoch,
            iteration=context.iteration
        )

        context.checkpoint.save(save_path)
        self.last_ckpt_iter = context.iteration
    
    def on_batch_end(self, context: TrainContext):
        if context.iteration - self.last_ckpt_iter >= self.interval:
            self._save_checkpoint(context)
    
    def on_train_end(self, context: TrainContext):
        if context.iteration != self.last_ckpt_iter:
            self._save_checkpoint(context)
    
    def on_error(self, context: TrainContext):
        self._save_checkpoint(context)


class ProgressBarCallback(TrainCallback):
    """ 
    Progress bar callback for trainer.
    """
    def __init__(self, num_epoch: int):
        self.num_epoch = num_epoch
        self.progress_bar: tqdm = None
    
    @only_on_rank(0)
    def on_epoch_begin(self, context: TrainContext):
        self.progress_bar = tqdm(
            context.dataloader, 
            desc=f"Epoch {context.epoch+1}/{self.num_epoch}", 
            dynamic_ncols=True
        )
    
    @only_on_rank(0)
    def on_batch_end(self, context: TrainContext):
        self.progress_bar.set_postfix({
            "loss": f"{context.loss:.4f}",
            "lr": f"{context.optimizer.param_groups[-1]['lr']:.2e}"
        })
        self.progress_bar.update(1)
    
    @only_on_rank(0)
    def on_epoch_end(self, context: TrainContext):
        _ = context
        if self.progress_bar:
            self.progress_bar.close()


class StepMonitorCallback(TrainCallback):
    def __init__(self, log_dir=None, log_interval=100, metrics=None):
        
        self.step_num = 0
        self.log_interval = log_interval
        self.metrics = metrics or ['loss', 'lr']
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._metric_funcs = {
            'loss': lambda ctx: ctx.loss,
            'lr': lambda ctx: ctx.optimizer.param_groups[-1]['lr'],
            'grad_norm': lambda ctx: grad_norm(ctx.model),
            'grad_std': lambda ctx: grad_std(ctx.model),
            'grad_max': lambda ctx: grad_max(ctx.model),
            'grad_min': lambda ctx: grad_min(ctx.model),
            'grad_mean': lambda ctx: grad_mean(ctx.model),
            'grad_nan_num':  lambda ctx: grad_nan_num(ctx.model)
        }
    
    def _get_log_data(self, context: TrainContext):
        return {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "epoch": context.epoch,
            "iter": context.iteration,
            **{m: self._metric_funcs[m](context) for m in self.metrics}
        }
    
    @only_on_rank(0)
    def on_step_end(self, context):
        self.step_num += 1
        if self.step_num % self.log_interval != 0:
            return
        
        log_data = self._get_log_data(context)
        
        log_file = self.log_dir / f"epoch_{context.epoch}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_data) + '\n')