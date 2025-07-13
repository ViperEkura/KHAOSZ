import os
import copy
import torch
import logging

from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm
from typing import Literal, Dict,  Callable

from khaosz.module import ModelParameter
from .checkpoint import TrainCheckPoint
from .strategy import SeqStrategy, SftStrategy, DpoStrategy, BaseStrategy, SchedulerFactory


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

    def train(
        self,
        train_type: Literal["seq", "sft", "dpo"],
        dataset: Dataset,
        optimizer: Optimizer,
        ckpt_dir: str,
        n_epoch: int = 1,
        batch_size: int = 4,
        n_iter_ckpt: int = 5000,
        n_iter_step: int = 1,
        max_grad_norm: float = 1.0,
        warning_step: int = 1000,
        min_rate: float = 0.1,
        dpo_beta: float = 0.1,
        random_seed: int = 3306
    ):
        assert train_type in ["seq", "sft", "dpo"]
        
        n_iter = 0
        loss_list = []
        last_ckpt_iter = 0
        
        total_iters = len(dataset) // batch_size * n_epoch
        get_lambda_scheduler_fn = SchedulerFactory.get_sgdr_schedule
        lambda_scheduler_fn = get_lambda_scheduler_fn(warning_step, total_iters, min_rate)
        pad_token_id = self.tokenizer.pad_id
        
        ref_model = None
        if train_type == "dpo":
            ref_model = copy.deepcopy(self.model)
            ref_model.requires_grad_(False)
            ref_model.eval()
            
        train_strategy: Dict[str, Callable[[], BaseStrategy]] = {
            "seq": lambda: SeqStrategy(self.model),
            "sft": lambda: SftStrategy(self.model),
            "dpo": lambda: DpoStrategy(self.model, pad_token_id, dpo_beta)
        }
        strategy = train_strategy[train_type]()
        
        scheduler = LambdaLR(optimizer, lambda_scheduler_fn)
        sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(random_seed))

        ckpt_saver = lambda current_iter: TrainCheckPoint(
            self.model, self.tokenizer, self.config, loss_list, current_iter
        ).save_ckpt(os.path.join(ckpt_dir, f"iter_{current_iter}"))

        self.logger.info(f"Starting {train_type.upper()} training for {n_epoch} epochs")
        self.logger.info(f"Checkpoint interval: {n_iter_ckpt} iterations")

        for epoch in range(n_epoch):
            self.model.train()
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epoch}", dynamic_ncols=True)

            for batch in progress_bar:
                #forward
                loss = strategy(batch)
                loss_list.append(loss.item())
                #backward
                loss.backward()
                #step
                if n_iter % n_iter_step == 0:
                    clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                n_iter += 1
                scheduler.step()
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                #save checkpotint
                if n_iter - last_ckpt_iter >= n_iter_ckpt:
                    ckpt_saver(n_iter)
                    diff_iter = n_iter - last_ckpt_iter
                    avg_loss = sum(loss_list[last_ckpt_iter:n_iter]) / diff_iter
                    self.logger.info(f"iter: {n_iter} loss: {avg_loss}")
                    last_ckpt_iter = n_iter

        if n_iter != last_ckpt_iter:
            ckpt_saver(n_iter)
            diff_iter = n_iter - last_ckpt_iter
            avg_loss = sum(loss_list[last_ckpt_iter:n_iter]) / diff_iter
            self.logger.info(f"iter: {n_iter} loss: {avg_loss}")

        self.logger.info("Training completed")
