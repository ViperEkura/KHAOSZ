import argparse
import os
from functools import partial

import torch
import torch.optim as optim

from astrai.config import AutoRegressiveLMConfig, TrainConfig
from astrai.dataset import DatasetFactory
from astrai.model import AutoRegressiveLM
from astrai.trainer import SchedulerFactory, Trainer


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Train the AutoRegressiveLM model.")

    parser.add_argument(
        "--train_type",
        type=str,
        required=True,
        choices=["seq", "sft", "dpo", "grpo"],
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
        "--batch_per_device", type=int, default=1, help="Batch size per GPU."
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of iterations between each optimizer step.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Fraction of total steps used for LR warmup.",
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
        help="Beta1 for AdamW optimizer.",
    )
    parser.add_argument(
        "--adamw_beta2",
        type=float,
        default=0.95,
        help="Beta2 for AdamW optimizer.",
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
        help="Max length of the input sequence.",
    )
    parser.add_argument(
        "--stride", type=int, default=None, help="Step size of the input sequence."
    )
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta value.")
    parser.add_argument("--group_size", type=int, default=4, help="GRPO group size.")
    parser.add_argument(
        "--grpo_clip_eps", type=float, default=0.2, help="GRPO clipping epsilon."
    )
    parser.add_argument(
        "--grpo_kl_coef", type=float, default=0.01, help="GRPO KL penalty coefficient."
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.05,
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
        "--grpo_sync_interval",
        type=int,
        default=200,
        help="GRPO ref model sync interval (steps).",
    )
    parser.add_argument(
        "--start_epoch", type=int, default=0, help="Start epoch for training."
    )
    parser.add_argument(
        "--start_batch", type=int, default=0, help="Start batch for training."
    )

    parser.add_argument("--nprocs", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument(
        "--parallel_mode",
        type=str,
        default="none",
        choices=["none", "ddp", "fsdp"],
        help="Parallel training strategy (none, ddp, fsdp).",
    )
    parser.add_argument(
        "--device_type", type=str, default="cuda", help="Device type to use."
    )
    parser.add_argument(
        "--start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method.",
    )

    args = parser.parse_args()

    return args


def create_model(config):
    return AutoRegressiveLM(config).to(dtype=torch.bfloat16)


def create_optimizer(model, **kwargs) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), fused=True, **kwargs)


def create_scheduler(
    optimizer: optim.Optimizer, **kwargs
) -> optim.lr_scheduler.LRScheduler:
    return SchedulerFactory.create(optimizer, **kwargs)


def compute_total_steps(
    dataset_len: int,
    n_epoch: int,
    batch_per_device: int,
    nprocs: int,
    grad_accum_steps: int,
) -> int:

    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    samples_per_replica = ceil_div(dataset_len, nprocs)
    batches_per_replica = ceil_div(samples_per_replica, batch_per_device)
    total_steps = (batches_per_replica // grad_accum_steps) * n_epoch
    return total_steps


def train(
    train_type: str,
    param_path: str,
    data_root_path: str,
    max_lr: float,
    n_epoch: int,
    batch_per_device: int,
    start_epoch: int,
    start_batch: int,
    grad_accum_steps: int,
    warmup_ratio: float,
    ckpt_interval: int,
    ckpt_dir: str,
    dpo_beta: float,
    grpo_clip_eps: float,
    grpo_kl_coef: float,
    group_size: int,
    grpo_sync_interval: int,
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
    parallel_mode: str,
    device_type: str,
    start_method: str,
):
    assert train_type in ["seq", "sft", "dpo", "grpo"]
    assert os.path.exists(param_path)
    if nprocs > 1 and parallel_mode == "none":
        raise ValueError("--nprocs > 1 requires --parallel_mode to be 'ddp' or 'fsdp'")

    # Load config
    config_path = os.path.join(param_path, "config.json")
    config = AutoRegressiveLMConfig.from_file(config_path)

    if window_size is None:
        window_size = config.max_len

    strategy_kwargs = {
        "beta": dpo_beta,
        "label_smoothing": label_smoothing,
        "clip_eps": grpo_clip_eps,
        "kl_coef": grpo_kl_coef,
        "group_size": group_size,
        "sync_interval": grpo_sync_interval,
    }

    executor_kwargs = {
        "gradient_as_bucket_view": True,
        "broadcast_buffers": False,
    }

    model_fn = partial(create_model, config)
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

    total_steps = compute_total_steps(
        len(dataset), n_epoch, batch_per_device, nprocs, grad_accum_steps
    )
    warmup_steps = int(warmup_ratio * total_steps)

    scheduler_fn = partial(
        create_scheduler,
        **{
            "schedule_type": "cosine",
            "warmup_steps": min(warmup_steps, total_steps),
            "lr_decay_steps": total_steps - min(warmup_steps, total_steps),
        },
    )

    train_config = TrainConfig(
        model_fn=model_fn,
        strategy=train_type,
        dataset=dataset,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        ckpt_dir=ckpt_dir,
        n_epoch=n_epoch,
        batch_per_device=batch_per_device,
        start_epoch=start_epoch,
        start_batch=start_batch,
        ckpt_interval=ckpt_interval,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        nprocs=nprocs,
        parallel_mode=parallel_mode,
        device_type=device_type,
        start_method=start_method,
        executor_kwargs=executor_kwargs,
        extra_kwargs=strategy_kwargs,
    )

    trainer = Trainer(train_config)
    trainer.train(resume_dir=param_path)


if __name__ == "__main__":
    args = parse_args()
    train(**vars(args))
