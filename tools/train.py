import os
import argparse
import torch

from torch.optim import AdamW
from khaosz.config import ModelParameter, TrainConfig, CosineScheduleConfig
from khaosz.trainer import Trainer, SchedulerFactory
from khaosz.data import DatasetLoader


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--embdeding_lr_rate", type=float, default=1.0, help="The rate between the embedding layers lr rate and the max lr rate.")
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
    
    args = parser.parse_args()

    return args

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
    embdeding_lr_rate: int,
    random_seed: int,
    num_workers: int,
    pin_memory: bool,
    window_size: int,
    stride: int,
):
    assert train_type in ["seq", "sft", "dpo"]
    assert os.path.exists(param_path)
    
    parameter = ModelParameter()
    parameter.load(param_path)

    if window_size is None:
        window_size = parameter.config.m_len

    model = parameter.model
    device = torch.device("cuda")
    model = model.to(device=device, dtype=torch.bfloat16)
    
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
        stride=stride,
        **kwargs
    )
    
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "embed" in n], "lr": max_lr * embdeding_lr_rate},
        {"params": [p for n, p in model.named_parameters() if "embed" not in n], "lr": max_lr}
    ]

    optimizer = AdamW(
        param_groups,
        betas=(adamw_beta1, adamw_beta2),
        weight_decay=adamw_weight_decay
    )
    
    schedule_config = CosineScheduleConfig(
        warmup_steps=warmup_steps,
        total_steps=len(dataset) * n_epoch // batch_size, 
    )
    
    scheduler = SchedulerFactory.load(optimizer, schedule_config)
    
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
        pin_memory=pin_memory
    )
    
    trainer = Trainer(train_config)
    trainer.train()
    

if __name__ == "__main__":
    args = parse_args()
    train(**vars(args))