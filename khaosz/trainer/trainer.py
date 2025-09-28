import os
import torch

from typing import Optional
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from khaosz.core import ModelParameter, Checkpoint
from khaosz.trainer.strategy import SchedulerFactory, TrainConfig, ScheduleConfig


class Trainer:
    def __init__(
        self,
        parameter: ModelParameter,
        train_config: TrainConfig,
        schedule_config: ScheduleConfig
    ):
        self.checkpoint = Checkpoint(
            model=parameter.model,
            tokenizer=parameter.tokenizer,
            config=parameter.config,
        )
        self.train_config = train_config
        self.schedule_config = schedule_config
        
    def save_checkpoint(
        self, 
        loss_list: list,
    ):
        current_iter = len(loss_list)
        save_path = os.path.join(self.train_config.checkpoint_dir, f"iter_{current_iter}")
        self.checkpoint.loss_list = loss_list
        self.checkpoint.optim_state = self.train_config.optimizer.state_dict()
        self.checkpoint.save(save_path)

    def train(
        self,
        train_checkpoint: Optional[Checkpoint] = None
    ) -> Checkpoint:
        assert self.schedule_config.schedule_type in ["cosine", "sgdr"]
        
        if train_checkpoint:
            self.checkpoint = train_checkpoint
            self.train_config.optimizer.load_state_dict(train_checkpoint.optim_state)
        
        self.checkpoint.optim_state = self.train_config.optimizer.state_dict()
        loss_list = self.checkpoint.loss_list
        current_iter = len(self.checkpoint.loss_list)
        last_ckpt_iter = current_iter

        for group in self.train_config.optimizer.param_groups:
            if "initial_lr" not in group:
                group["initial_lr"] = group["lr"] 
            
        
        lambda_scheduler_fn  = SchedulerFactory.load_schedule_fn(
            **self.schedule_config.get_kwargs()
        )
        
        scheduler = LambdaLR(
            self.train_config.optimizer,
            lambda_scheduler_fn,
            last_epoch=current_iter - 1 if train_checkpoint else -1
        )
        
        seed = self.train_config.random_seed
        generator = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(self.train_config.dataset, generator=generator)
        remaining_epochs = self.train_config.n_epoch - current_iter // (
                        len(self.train_config.dataset) // self.train_config.batch_size)
        
        for epoch in range(remaining_epochs):
            self.checkpoint.model.train()
            dataloader = DataLoader(
                self.train_config.dataset, 
                batch_size=self.train_config.batch_size, 
                sampler=sampler
            )
            progress_bar = tqdm(
                dataloader, 
                desc=f"Epoch {epoch+1}/{self.train_config.n_epoch}", 
                dynamic_ncols=True
            )
            for batch in progress_bar:
                #forward
                loss = self.train_config.strategy(batch)
                loss_list.append(loss.item())
                #backward
                loss.backward()
                #step
                if current_iter % self.train_config.accumulation_steps == 0:
                    clip_grad_norm_(
                        self.checkpoint.model.parameters(),
                        self.train_config.max_grad_norm
                    )
                    self.train_config.optimizer.step()
                    self.train_config.optimizer.zero_grad()
                    
                current_iter += 1
                scheduler.step()
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.train_config.optimizer.param_groups[0]['lr']:.2e}"
                })
                #save checkpotint
                if current_iter - last_ckpt_iter >= self.train_config.checkpoint_interval:
                    self.save_checkpoint(loss_list)
                    last_ckpt_iter = current_iter

        if current_iter != last_ckpt_iter:
            self.save_checkpoint(loss_list)
            last_ckpt_iter = current_iter
        
        return self.checkpoint