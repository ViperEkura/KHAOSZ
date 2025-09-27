import os
import torch
import logging

from typing import Tuple
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from khaosz.core import ModelParameter, Checkpoint
from khaosz.trainer.strategy import SchedulerFactory, StrategyFactory, TrainConfig, ScheduleConfig


class Trainer:
    def __init__(
        self,
        parameter: ModelParameter,
        log_path: str="./train_log.log"
    ):
        logger = logging.getLogger()
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(log_path)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
        logger.addHandler(handler)
        logger.info("initializing trainer ...")
        
        self.logger = logger
        self.model = parameter.model
        self.tokenizer = parameter.tokenizer
        self.config = parameter.config
        
    def save_checkpoint(
        self, 
        loss_list: list, 
        ckpt_dir: str, 
        current_iter: int, 
        last_ckpt_iter: int
    ):
        save_path = os.path.join(ckpt_dir, f"iter_{current_iter}")
        Checkpoint(
            self.model, 
            self.tokenizer, 
            self.config, 
            loss_list, 
            current_iter
        ).save(save_path)
        
        diff_iter = current_iter - last_ckpt_iter
        avg_loss = sum(loss_list[last_ckpt_iter:current_iter]) / diff_iter
        self.logger.info(f"iter: {current_iter} loss: {avg_loss}")  
        
        return current_iter
    
    def load_checkpoint(self, train_checkpoint: Checkpoint) -> Tuple[list, int]:
        self.model = train_checkpoint.model
        self.tokenizer = train_checkpoint.tokenizer
        self.config = train_checkpoint.config
        loss_list = train_checkpoint.loss_list
        last_ckpt_iter = train_checkpoint.current_iter
        
        return loss_list, last_ckpt_iter

    def train(
        self,
        train_config: TrainConfig,
        schedule_config: ScheduleConfig,
        train_checkpoint: Checkpoint = None
    ):
        assert schedule_config.schedule_type in ["cosine", "sgdr"]
        assert train_config.train_type in ["seq", "sft", "dpo"]
        
        if train_checkpoint:
            loss_list, last_ckpt_iter = self.load_checkpoint(train_checkpoint)
            current_iter = train_checkpoint.current_iter + 1
            self.logger.info(f"Resuming training from checkpoint: iter {current_iter}")
        else:
            current_iter = 0
            last_ckpt_iter = 0
            loss_list = []
            
        lambda_scheduler_fn  = SchedulerFactory.load_schedule_fn(
            **schedule_config.get_kwargs()
        )
        
        strategy = StrategyFactory.load(
            self.model, 
            train_type=train_config.train_type,
            bos_token_id=self.tokenizer.bos_id,
            eos_token_id=self.tokenizer.eos_id,
            pad_token_id=self.tokenizer.pad_id,
            dpo_beta=train_config.dpo_beta
        )
        
        scheduler = LambdaLR(
            train_config.optimizer,
            lambda_scheduler_fn,
            last_epoch=current_iter - 1 if train_checkpoint else -1
        )
        
        seed = train_config.random_seed
        generator = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(train_config.dataset, generator=generator)
        remaining_epochs = train_config.n_epoch - current_iter // (len(train_config.dataset) // train_config.batch_size)
        
        self.logger.info(f"Starting {train_config.train_type.upper()} training for {train_config.n_epoch} epochs")
        self.logger.info(f"Checkpoint interval: {train_config.n_iter_ckpt} iterations")
        
        for epoch in range(remaining_epochs):
            self.model.train()
            dataloader = DataLoader(
                train_config.dataset, 
                batch_size=train_config.batch_size, 
                sampler=sampler
            )
            progress_bar = tqdm(
                dataloader, 
                desc=f"Epoch {epoch+1}/{train_config.n_epoch}", 
                dynamic_ncols=True
            )
            for batch in progress_bar:
                #forward
                loss = strategy(batch)
                loss_list.append(loss.item())
                #backward
                loss.backward()
                #step
                if current_iter % train_config.n_iter_step == 0:
                    clip_grad_norm_(
                        self.model.parameters(),
                        train_config.max_grad_norm
                    )
                    train_config.optimizer.step()
                    train_config.optimizer.zero_grad()
                    
                current_iter += 1
                scheduler.step()
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{train_config.optimizer.param_groups[0]['lr']:.2e}"
                })
                #save checkpotint
                if current_iter - last_ckpt_iter >= train_config.n_iter_ckpt:
                    last_ckpt_iter = self.save_checkpoint(
                        loss_list, 
                        train_config.ckpt_dir, 
                        current_iter, 
                        last_ckpt_iter
                    )

        if current_iter != last_ckpt_iter:
            last_ckpt_iter = self.save_checkpoint(
                loss_list, 
                train_config.ckpt_dir, 
                current_iter, 
                last_ckpt_iter
            )

        self.logger.info("Training completed")
        
        return Checkpoint(
            self.model,
            self.tokenizer,
            self.config,
            loss_list,
            current_iter,
            train_config.optimizer
        )
