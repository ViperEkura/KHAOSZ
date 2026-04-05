import argparse
import os
from functools import partial

import safetensors.torch as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from astrai.config import ModelConfig, TrainConfig
from astrai.dataset import DatasetFactory
from astrai.model import Transformer
from astrai.parallel import get_rank
from astrai.trainer import SchedulerFactory, Trainer


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Train the Transformer model.")

    parser.add_argument(
        "--train_type",
        type=str,
        required=True,
        choices=["seq", "sft", "dpo"],
        help="Train type.",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        required=True,
        help="Path to the root directory of the dataset.",
    )
    parser.add_argument(
        "--param_path",
        type=str,
        required=True,
        help="Path to the model parameters or resume checkpoint.",
    )

    parser.add_argument(
        "--n_epoch", type=int, default=1, help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training."
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Number of iterations between each optimizer step.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of iters between warnings.",
    )
    parser.add_argument(
        "--max_lr", type=float, default=3e-4, help="Max learning rate for training."
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping.",
    )
    parser.add_argument(
        "--adamw_beta1",
        type=float,
        default=0.9,
        help="Beta values for AdamW optimizer.",
    )
    parser.add_argument(
        "--adamw_beta2",
        type=float,
        default=0.95,
        help="Beta values for AdamW optimizer.",
    )
    parser.add_argument(
        "--adamw_weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=3407, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--no_pin_memory",
        action="store_false",
        dest="pin_memory",
        help="Disable pin memory",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="the max length of the input sequence.",
    )
    parser.add_argument(
        "--stride", type=int, default=None, help="the step size of the input sequence."
    )
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta value.")
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="cross_entropy function label smoothing parameter",
    )

    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=5000,
        help="Number of iters between checkpoints.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoint",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--start_epoch", type=int, default=0, help="Start epoch for training."
    )
    parser.add_argument(
        "--start_batch", type=int, default=0, help="Start batch for training."
    )

    parser.add_argument("--nprocs", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument(
        "--device_type", type=str, default="cuda", help="Device type to use."
    )

    args = parser.parse_args()

    return args


def ddp_wrap(model: nn.Module):
    local_rank = get_rank()
    model = model.to(device=f"cuda:{local_rank}", dtype=torch.bfloat16)
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    return ddp_model


def create_optimizer(model: nn.Module, **kwargs) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), **kwargs)


def create_scheduler(
    optimizer: optim.Optimizer, **kwargs
) -> optim.lr_scheduler.LRScheduler:
    return SchedulerFactory.create(optimizer, **kwargs)


def prepare_checkpoint(model: nn.Module) -> dict:
    return model.module.state_dict()


def train(
    train_type: str,
    param_path: str,
    data_root_path: str,
    max_lr: float,
    n_epoch: int,
    batch_size: int,
    start_epoch: int,
    start_batch: int,
    accumulation_steps: int,
    warmup_steps: int,
    ckpt_interval: int,
    ckpt_dir: str,
    dpo_beta: float,
    adamw_beta1: float,
    adamw_beta2: float,
    adamw_weight_decay: float,
    max_grad_norm: float,
    label_smoothing: float,
    random_seed: int,
    num_workers: int,
    pin_memory: bool,
    window_size: int,
    stride: int,
    nprocs: int,
    device_type: str,
):
    assert train_type in ["seq", "sft", "dpo"]
    assert os.path.exists(param_path)

    # Load config
    config = ModelConfig()
    config_path = os.path.join(param_path, "config.json")
    if os.path.exists(config_path):
        config.load(config_path)

    if window_size is None:
        window_size = config.max_len

    # Create bare Transformer (for training, no tokenizer needed)
    model = Transformer(config)

    # Load weights if available
    weights_path = os.path.join(param_path, "model.safetensors")
    if os.path.exists(weights_path):
        state_dict = st.load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)

    strategy_kwargs = {"dpo_beta": dpo_beta, "label_smoothing": label_smoothing}

    dataset = DatasetFactory.load(
        train_type=train_type,
        load_path=data_root_path,
        window_size=window_size,
        stride=stride,
    )

    optimizer_fn = partial(
        create_optimizer,
        **{
            "lr": max_lr,
            "betas": (adamw_beta1, adamw_beta2),
            "weight_decay": adamw_weight_decay,
        },
    )

    toltal_steps = len(dataset) * n_epoch // (batch_size * nprocs)
    scheduler_fn = partial(
        create_scheduler,
        **{
            "schedule_type": "cosine",
            "warmup_steps": warmup_steps,
            "lr_decay_steps": toltal_steps - warmup_steps,
        },
    )

    train_config = TrainConfig(
        model=model,
        strategy=train_type,
        dataset=dataset,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        ckpt_dir=ckpt_dir,
        n_epoch=n_epoch,
        batch_size=batch_size,
        start_epoch=start_epoch,
        start_batch=start_batch,
        ckpt_interval=ckpt_interval,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        nprocs=nprocs,
        parallel_wrapper=ddp_wrap,
        state_dict_fn=prepare_checkpoint,
        device_type=device_type,
        extra_kwargs=strategy_kwargs,
    )

    trainer = Trainer(train_config)
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    train(**vars(args))
