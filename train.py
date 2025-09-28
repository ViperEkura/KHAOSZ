import os
import argparse
import torch

from torch.optim import AdamW
from khaosz.core import ParameterLoader
from khaosz.trainer import Trainer, DatasetLoader, TrainConfig, CosineScheduleConfig
from khaosz.trainer import StrategyFactory


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
    multi_turn: bool,
):
    assert train_type in ["seq", "sft", "dpo"]
    assert os.path.exists(param_path)
    
    parameter = ParameterLoader.load(param_path)
    model = parameter.model
    
    device = torch.device("cuda")
    model = model.to(device=device, dtype=torch.bfloat16)
    
    cache_files = get_files(data_root_path)
    
    strategy_kwargs = {
        "multi_turn": multi_turn,
        "bos_token_id": parameter.tokenizer.bos_id,
        "eos_token_id": parameter.tokenizer.eos_id,
        "user_token_id":parameter.tokenizer.encode("<|user|>")[0],
        "dpo_beta": dpo_beta
    }
    
    strategy = StrategyFactory.load(
        model, 
        train_type
        **strategy_kwargs
    )

    dataset = DatasetLoader.load(
        train_type=train_type,
        load_path=cache_files,
        max_len=parameter.config.m_len,
        device=device,
        dataset_kwargs=strategy_kwargs
    )
    
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "embedding" in n], "lr": max_lr * embdeding_lr_rate},
        {"params": [p for n, p in model.named_parameters() if "embedding" not in n], "lr": max_lr}
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
        checkpoint_interval=checkpoint_interval,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
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
    trainer.train()
    

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
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta value.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--adamw_betas", type=tuple, default=(0.9, 0.95), help="Beta values for AdamW optimizer.")
    parser.add_argument("--adamw_weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--embdeding_lr_rate", type=float, default=1.0, help="The rate between the embedding layers lr rate and the max lr rate.")
    parser.add_argument("--random_seed", type=int, default=3407, help="Random seed for reproducibility.")
    
    # other configs
    parser.add_argument("--multi_turn", type=bool, default=False, help="Whether to use multi-turn convsersation training.")

    args = parser.parse_args()

    train(
        param_path=args.param_path,
        data_root_path=args.data_root_path,
        n_epoch=args.n_epoch,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_lr=args.max_lr,
        dpo_beta=args.dpo_beta,
        adamw_betas=args.adamw_betas,
        adamw_weight_decay=args.adamw_weight_decay,
        max_grad_norm=args.max_grad_norm,
        embdeding_lr_rate=args.embdeding_lr_rate,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        train_type=args.train_type,
        random_seed=args.random_seed,
        multi_turn=args.multi_turn
    )