import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed.fsdp as fsdp

from typing import List, Optional
from functools import partial
from khaosz.config import ModelParameter, TrainConfig, CosineScheduleConfig
from khaosz.trainer import Trainer, SchedulerFactory
from khaosz.data import DatasetLoader


def parse_args() -> argparse.Namespace:
    def parse_device_ids(s: Optional[str]) -> Optional[List[int]]:
        if s is None or s.strip() == "":
            return None
        try:
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Invalid device_ids format: {s}. Expected comma-separated integers like '0,1,2'.")


    parser = argparse.ArgumentParser(description="Train the Transformer model.")
    
    parser.add_argument("--train_type",choices=["seq", "sft", "dpo"], help="Train type.")
    parser.add_argument("--data_root_path", type=str, required=True, help="Path to the root directory of the dataset.")
    parser.add_argument("--param_path", type=str, required=True, help="Path to the model parameters or resume checkpoint.")
    
    parser.add_argument("--n_epoch", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Number of iterations between each optimizer step.")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of iters between warnings.")
    parser.add_argument("--max_lr", type=float, default=3e-4, help="Max learning rate for training.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--adamw_beta1", type=float, default=0.9, help="Beta values for AdamW optimizer.")
    parser.add_argument("--adamw_beta2", type=float, default=0.95, help="Beta values for AdamW optimizer.")
    parser.add_argument("--adamw_weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--random_seed", type=int, default=3407, help="Random seed for reproducibility.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory", help="Disable pin memory")
    parser.add_argument("--window_size", type=int, default=None, help="the max length of the input sequence.")
    parser.add_argument("--stride", type=int, default=None, help="the step size of the input sequence.")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta value.")
    
    parser.add_argument("--checkpoint_interval", type=int, default=5000, help="Number of iters between checkpoints.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", help="Directory to save checkpoints.")
    parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch for training.")
    parser.add_argument("--start_batch", type=int, default=0, help="Start batch for training.")
    
    parser.add_argument("--nprocs", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--device_ids", type=parse_device_ids, default=None, help="Device IDs to use.")
    parser.add_argument("--device_type", type=str, default="cuda", help="Device type to use.")
    
    args = parser.parse_args()

    return args

def fsdp_wrap(model: nn.Module):
    
    fsdp_model = fsdp.FullyShardedDataParallel(
        model,
        sharding_strategy=fsdp.ShardingStrategy.SHARD_GRAD_OP,
        mixed_precision=fsdp.MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=fsdp.BackwardPrefetch.BACKWARD_PRE
    )
    return fsdp_model

def create_optimizer(model: nn.Module, **kwargs) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), **kwargs)

def create_scheduler(optimizer: optim.Optimizer, **kwargs) -> optim.lr_scheduler.LRScheduler:
    return SchedulerFactory.load(optimizer, **kwargs)

def train(
    train_type: str,
    param_path: str,
    data_root_path: str,
    max_lr: int,
    n_epoch: int,
    batch_size: int,
    start_epoch: int,
    start_batch: int,
    accumulation_steps: int,
    warmup_steps: int,
    checkpoint_interval: int,
    checkpoint_dir: str,
    dpo_beta: float,
    adamw_beta1: float,
    adamw_beta2: float,
    adamw_weight_decay: float,
    max_grad_norm: float,
    random_seed: int,
    num_workers: int,
    pin_memory: bool,
    window_size: int,
    stride: int,
    nprocs: int,
    device_ids: List[int],
    device_type: str,
):
    assert train_type in ["seq", "sft", "dpo"]
    assert os.path.exists(param_path)
    
    parameter = ModelParameter()
    parameter.load(param_path)

    if window_size is None:
        window_size = parameter.config.max_len

    model = parameter.model
    
    kwargs = {
        "dpo_beta": dpo_beta,
        "bos_token_id": parameter.tokenizer.bos_id,
        "eos_token_id": parameter.tokenizer.eos_id,
        "pad_token_id": parameter.tokenizer.pad_id,
    }

    dataset = DatasetLoader.load(
        train_type=train_type,
        load_path=data_root_path,
        window_size=window_size,
        stride=stride
    )
    
    schedule_config = CosineScheduleConfig(
        warmup_steps=warmup_steps,
        total_steps=len(dataset) * n_epoch // (batch_size * nprocs), 
    )
    

    optimizer_fn = partial(create_optimizer, **{"lr": max_lr, "betas": (adamw_beta1, adamw_beta2), "weight_decay": adamw_weight_decay})
    scheduler_fn = partial(create_scheduler, **{"schedule_config": schedule_config})
    optimizer, scheduler = None, None
    
    if nprocs == 1:
        optimizer = optimizer_fn(model.parameters())
        scheduler = scheduler_fn(optimizer)
    
    train_config = TrainConfig(
        model=model,
        strategy=train_type,
        dataset=dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        n_epoch=n_epoch,
        batch_size=batch_size,
        start_epoch=start_epoch,
        start_batch=start_batch,
        checkpoint_interval=checkpoint_interval,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        nprocs=nprocs,
        parallel_wrapper=fsdp_wrap,
        optimizer_factory=optimizer_fn,
        scheduler_factory=scheduler_fn,
        device_ids=device_ids,
        device_type=device_type,
        extra_kwargs=kwargs,
    )
    
    trainer = Trainer(train_config)
    trainer.train()
    

if __name__ == "__main__":
    args = parse_args()
    train(**vars(args))