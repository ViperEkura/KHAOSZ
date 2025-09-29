from tqdm import tqdm
from khaosz.core.parameter import Checkpoint
from khaosz.trainer.trainer import Trainer
from torch.nn.utils import clip_grad_norm_
from typing import cast


class TrainerCallback:
    def on_train_begin(self, trainer: 'Trainer', **kwargs):
        pass
    
    def on_train_end(self, trainer: 'Trainer', **kwargs):
        pass

    def on_epoch_begin(self, trainer: 'Trainer', **kwargs):
        pass

    def on_epoch_end(self, trainer: 'Trainer', **kwargs):
        pass

    def on_batch_begin(self, trainer: 'Trainer', **kwargs):
        pass

    def on_batch_end(self, trainer: 'Trainer', **kwargs):
        pass

    def on_step_begin(self, trainer: 'Trainer', **kwargs):
        pass

    def on_step_end(self, trainer: 'Trainer', **kwargs):
        pass


class ProgressBarCallback(TrainerCallback):
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
        if self.progress_bar:
            self.progress_bar.close()


class CheckpointCallback(TrainerCallback):
    def __init__(self, checkpoint_interval: int):
        self.checkpoint_interval = checkpoint_interval
        self.last_ckpt_iter = 0
    
    def on_train_begin(self, trainer: 'Trainer', **kwargs):
        checkpoint = cast(Checkpoint, kwargs.get('checkpoint'))
        self.last_ckpt_iter = len(checkpoint.loss_list)
    
    def on_batch_end(self, trainer: 'Trainer', **kwargs):
        current_iter = kwargs.get('current_iter')
        if current_iter - self.last_ckpt_iter >= self.checkpoint_interval:
            trainer._save_checkpoint()
            self.last_ckpt_iter = current_iter
    
    def on_train_end(self, trainer: 'Trainer', **kwargs):
        checkpoint = cast(Checkpoint, kwargs.get('checkpoint'))
        current_iter = len(checkpoint.loss_list)
        if current_iter != self.last_ckpt_iter:
            trainer._save_checkpoint()


class GradientClippingCallback(TrainerCallback):
    
    def on_step_begin(self, trainer: 'Trainer', **kwargs):
        clip_grad_norm_(
            trainer.checkpoint.model.parameters(),
            trainer.train_config.max_grad_norm
        )