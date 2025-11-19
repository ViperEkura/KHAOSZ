import os
import argparse
import torch

from torch.optim import AdamW
from khaosz.config import ParameterLoader, Checkpoint, TrainConfig, CosineScheduleConfig
from khaosz.trainer import Trainer, StrategyFactory
from khaosz.data import DatasetLoader


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def get_files(root_path: str) -> list[str]:
    paths = []
    for root, _, files in os.walk(root_path):
        paths.extend([os.path.join(root, file) for file in files])

    return paths

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
    adamw_betas: tuple,
    adamw_weight_decay: float,
    max_grad_norm: float,
    embdeding_lr_rate: int,
    random_seed: int,
    window_size: int,
    stride: int,
    resume_from_checkpoint: bool
):
    assert train_type in ["seq", "sft", "dpo"]
    assert os.path.exists(param_path)
    
    parameter = ParameterLoader.load(param_path)
    checkpoint = None
    
    if isinstance(parameter, Checkpoint) and resume_from_checkpoint:
        checkpoint = parameter

    if window_size is None:
        window_size = parameter.config.m_len

    model = parameter.model
    device = torch.device("cuda")
    model = model.to(device=device, dtype=torch.bfloat16)
    cache_files = get_files(data_root_path)
    
    kwargs = {
        "dpo_beta": dpo_beta,
        "bos_token_id": parameter.tokenizer.bos_id,
        "eos_token_id": parameter.tokenizer.eos_id,
        "pad_token_id": parameter.tokenizer.pad_id,
    }
    
    strategy = StrategyFactory.load(
        model, 
        train_type,
        device,
        **kwargs
    )

    dataset = DatasetLoader.load(
        train_type=train_type,
        load_path=cache_files,
        window_size=window_size,
        stride=stride,
        **kwargs
    )
    
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "embed" in n], "lr": max_lr * embdeding_lr_rate},
        {"params": [p for n, p in model.named_parameters() if "embed" not in n], "lr": max_lr}
    ]

    optim = AdamW(
        param_groups,
        betas=adamw_betas,
        weight_decay=adamw_weight_decay
    )
    
    train_config = TrainConfig(
        strategy=strategy,
        dataset=dataset,
        optimizer=optim,
        checkpoint_dir=checkpoint_dir,
        n_epoch=n_epoch,
        batch_size=batch_size,
        start_epoch=start_epoch,
        start_batch=start_batch,
        checkpoint_interval=checkpoint_interval,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
        num_workers=4,
        pin_memory=True
    )
    
    schedule_config = CosineScheduleConfig(
        warmup_steps=warmup_steps,
        total_steps=len(dataset) * n_epoch // batch_size, 
    )
    
    trainer = Trainer(
        parameter=parameter,
        train_config=train_config,
        schedule_config=schedule_config,
    )
    trainer.train(checkpoint)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Transformer model.")
    # train args
    parser.add_argument("--train_type",choices=["seq", "sft", "dpo"], help="Train type.")
    parser.add_argument("--data_root_path", type=str, required=True, help="Path to the root directory of the dataset.")
    parser.add_argument("--param_path", type=str, required=True, help="Path to the model parameters or resume checkpoint.")
    parser.add_argument("--n_epoch", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Number of iterations between each optimizer step.")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of iters between warnings.")
    parser.add_argument("--max_lr", type=float, default=3e-4, help="Max learning rate for training.")
    parser.add_argument("--checkpoint_interval", type=int, default=5000, help="Number of iters between checkpoints.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", help="Directory to save checkpoints.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--adamw_betas", type=tuple, default=(0.9, 0.95), help="Beta values for AdamW optimizer.")
    parser.add_argument("--adamw_weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--embdeding_lr_rate", type=float, default=1.0, help="The rate between the embedding layers lr rate and the max lr rate.")
    parser.add_argument("--random_seed", type=int, default=3407, help="Random seed for reproducibility.")
    
    # other configs
    parser.add_argument("--window_size", type=int, default=None, help="the max length of the input sequence.")
    parser.add_argument("--stride", type=int, default=None, help="the step size of the input sequence.")
    parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch for training.")
    parser.add_argument("--start_batch", type=int, default=0, help="Start batch for training.")
    parser.add_argument("--resume_from_checkpoint", type=bool, default=False, help="train from checkpoint or not.")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta value.")
    
    args = parser.parse_args()

    train(**vars(args))