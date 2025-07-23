import os
import copy
import torch
import logging
import pickle as pkl
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from khaosz.module import ModelParameter, BpeTokenizer, TransformerConfig
from .strategy import SchedulerFactory, StrategyFactory, TrainConfig, ScheduleConfig


class TrainCheckPoint(ModelParameter):
    def __init__(
            self, 
            model: nn.Module, 
            tokenizer: BpeTokenizer, 
            config: TransformerConfig, 
            loss_list: list, 
            current_iter: int
        ):
            super().__init__(model, tokenizer, config)
            self.loss_list = loss_list
            self.current_iter = current_iter

    def save_ckpt(self, save_dir: str):
        super().save(save_dir)
        paths = {
            "loss_list": os.path.join(save_dir, "loss.pkl"),
            "lossfig": os.path.join(save_dir, "loss.png")
        }
        
        pkl.dump(self.loss_list, open(paths["loss_list"], "wb"))
        plt.figure()
        plt.plot(self.loss_list)
        plt.title(f"Training Loss - iter {self.current_iter}")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.savefig(paths["lossfig"])
        plt.close()


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
        diff_iter = current_iter - last_ckpt_iter
        avg_loss = sum(loss_list[last_ckpt_iter:current_iter]) / diff_iter
        self.logger.info(f"iter: {current_iter} loss: {avg_loss}")
        save_path = os.path.join(ckpt_dir, f"iter_{current_iter}")
        TrainCheckPoint(
            self.model, 
            self.tokenizer, 
            self.config, 
            loss_list, 
            current_iter
        ).save_ckpt(save_path)
        
        return current_iter

    def train(
        self,
        train_config: TrainConfig,
        schedule_config: ScheduleConfig,
    ):
        assert schedule_config.schedule_type in ["cosine", "sgdr"]
        assert train_config.train_type in ["seq", "sft", "dpo"]
        
        current_iter = 0
        last_ckpt_iter = 0
        loss_list = []
            
        lambda_scheduler_fn  = SchedulerFactory.load_schedule_fn(
            strategy=schedule_config.schedule_type, 
            **schedule_config.get_kwargs()
        )
        
        ref_model = None
        if train_config.train_type == "dpo":
            ref_model = copy.deepcopy(self.model)
            ref_model.requires_grad_(False)
            ref_model.eval()
        
        strategy = StrategyFactory.load(
            self.model, 
            train_config.train_type, 
            self.tokenizer.pad_id, 
            train_config.dpo_beta
        )
        
        scheduler = LambdaLR(
            train_config.optimizer, 
            lambda_scheduler_fn
        )
        
        seed = train_config.random_seed
        generator = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(train_config.dataset, generator=generator)

        self.logger.info(f"Starting {train_config.train_type.upper()} training for {train_config.n_epoch} epochs")
        self.logger.info(f"Checkpoint interval: {train_config.n_iter_ckpt} iterations")

        for epoch in range(train_config.n_epoch):
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
