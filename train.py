import os
import argparse
import torch
import safetensors.torch as st

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
    max_lr: int,
    n_iter_ckpt: int,
    ckpt_dir: str,
    resume_dir: str = None,
):
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
    
    dataset = None
    if train_type == "seq":
        dataset = SeqDataset(config.m_len, device=device)
    elif train_type == "sft":
        dataset = SftDataset(config.m_len, device=device)
    elif train_type == "dpo":
        dataset = DpoDataset(config.m_len, device=device)
    
    cache_files = get_files(data_root_path)
    dataset.load(cache_files)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    optim = AdamW(
        model.parameters(),
        eps=5e-5,
        lr=max_lr,
        betas=(0.9, 0.99),
        weight_decay=0.05
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
        max_grad_norm=1.0
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Transformer model.")
    parser.add_argument("--train_type",choices=["seq", "sft", "dpo"], help="Train type.")
    parser.add_argument("--data_root_path", type=str, required=True, help="Path to the root directory of the dataset.")
    parser.add_argument("--n_epoch", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--n_iter_step", type=int, default=1, help="Number of iterations between each optimizer step.")
    parser.add_argument("--max_lr", type=float, default=3e-4, help="Max learning rate for training.")
    parser.add_argument("--n_iter_ckpt", type=int, default=5000, help="Number of iters between checkpoints.")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint", help="Directory to save checkpoints.")
    parser.add_argument("--resume_train", type=bool, default=False, help="Resume training from a checkpoint.")
    parser.add_argument("--resume_dir", type=str, default=None, help="Path to the checkpoint file to resume training.")

    args = parser.parse_args()
    
    resume_dir = None
    if args.resume_train and args.resume_dir is not None:
        resume_dir = args.resume_dir

    train(
        data_root_path=args.data_root_path,
        n_epoch=args.n_epoch,
        batch_size=args.batch_size,
        n_iter_step=args.n_iter_step,
        max_lr=args.max_lr,
        n_iter_ckpt=args.n_iter_ckpt,
        ckpt_dir=args.ckpt_dir,
        resume_dir=resume_dir,
        train_type=args.train_type
    )