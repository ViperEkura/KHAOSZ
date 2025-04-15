import os
import argparse
import torch
import safetensors.torch as st

from typing import Callable, Dict
from torch.optim import AdamW
from torch.utils.data import DataLoader
from module import Transformer, Config, BpeTokenizer
from trainer import Trainer, SeqDataset, SftDataset, DpoDataset

dirname = os.path.dirname(__file__)

def get_files(root_path: str) -> list[str]:
    paths = list()
    for root, _, files in os.walk(root_path):
        for file in files:
            path = os.path.join(root, file)
            paths.append(path)
    return paths

def train(
    train_type: str,
    data_root_path: str,
    n_epoch: int,
    batch_size: int,
    n_iter_step: int,
    warning_step: int,
    max_lr: int,
    n_iter_ckpt: int,
    ckpt_dir: str,
    dpo_beta: float,
    adamw_betas: tuple,
    adamw_weight_decay: float,
    max_grad_norm: float,
    resume_dir: str = None
):
    assert train_type in ["seq", "sft", "dpo"]
    if train_type in ["sft", "dpo"]:
        assert resume_dir is not None
    
    if resume_dir is None:
        config_path = os.path.join(dirname, "params", "config.json")
        tokenizer_path = os.path.join(dirname, "params", "tokenizer.json")
    else:
        config_path = os.path.join(resume_dir, "config.json")
        tokenizer_path = os.path.join(resume_dir, "tokenizer.json")

    config = Config(config_path)
    model = Transformer(config)
    tokenizer = BpeTokenizer(tokenizer_path)
    
    if resume_dir is not None:
        weight_path = os.path.join(resume_dir, "model.safetensors")
        model.load_state_dict(st.load_file(weight_path))
        
    device = torch.device("cuda")
    model = model.to(device=device, dtype=torch.bfloat16)
    
    dataset_factories: Dict[str, Callable[[int, torch.device], SeqDataset | SftDataset | DpoDataset]] = {
        "seq": lambda m_len, device: SeqDataset(m_len, device=device),
        "sft": lambda m_len, device: SftDataset(m_len, device=device),
        "dpo": lambda m_len, device: DpoDataset(m_len, device=device),
    }

    dataset_generator = dataset_factories[train_type]
    dataset = dataset_generator(config.m_len, device)
    
    cache_files = get_files(data_root_path)
    dataset.load(cache_files)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    optim = AdamW(
        model.parameters(),
        lr=max_lr,
        betas=adamw_betas,
        weight_decay=adamw_weight_decay
    )
    
    trainer = Trainer(model, tokenizer, config)
    trainer.train(
        train_type=train_type,
        dataloader=dataloader,
        optimizer=optim,
        ckpt_dir=ckpt_dir,
        n_epoch=n_epoch,
        n_iter_ckpt=n_iter_ckpt,
        n_iter_step=n_iter_step,
        warning_step=warning_step,
        dpo_beta=dpo_beta,
        max_grad_norm=max_grad_norm,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Transformer model.")
    parser.add_argument("--train_type",choices=["seq", "sft", "dpo"], help="Train type.")
    parser.add_argument("--data_root_path", type=str, required=True, help="Path to the root directory of the dataset.")
    parser.add_argument("--n_epoch", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--n_iter_step", type=int, default=1, help="Number of iterations between each optimizer step.")
    parser.add_argument("--warning_step", type=int, default=1000, help="Number of iters between warnings.")
    parser.add_argument("--max_lr", type=float, default=3e-4, help="Max learning rate for training.")
    parser.add_argument("--n_iter_ckpt", type=int, default=5000, help="Number of iters between checkpoints.")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint", help="Directory to save checkpoints.")
    parser.add_argument("--resume_dir", type=str, default=None, help="Path to the checkpoint file to resume training.")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta value.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--adamw_betas", type=tuple, default=(0.9, 0.95), help="Beta values for AdamW optimizer.")
    parser.add_argument("--adamw_weight_decay", type=float, default=0.1, help="Weight decay for AdamW optimizer.")

    args = parser.parse_args()

    train(
        data_root_path=args.data_root_path,
        n_epoch=args.n_epoch,
        batch_size=args.batch_size,
        n_iter_step=args.n_iter_step,
        warning_step=args.warning_step,
        max_lr=args.max_lr,
        dpo_beta=args.dpo_beta,
        adamw_betas=args.adamw_betas,
        adamw_weight_decay=args.adamw_weight_decay,
        max_grad_norm=args.max_grad_norm,
        n_iter_ckpt=args.n_iter_ckpt,
        ckpt_dir=args.ckpt_dir,
        resume_dir=args.resume_dir,
        train_type=args.train_type
    )