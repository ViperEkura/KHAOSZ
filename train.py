import os
import argparse
import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torch.utils.data import DataLoader

from module.transfomer import Transfomer, Config
from module.tokenizer import BpeTokenizer
from trainer.dataset import SeqDataset
from trainer.trainer import Trainer

dirname = os.path.dirname(__file__)

def get_files(root_path: str) -> list[str]:
    paths = list()
    for root, _, files in os.walk(root_path):
        for file in files:
            path = os.path.join(root, file)
            paths.append(path)
    return paths

def train(
    data_root_path: str,
    n_epoch: int,
    batch_size: int,
    n_epoch_ckpt: int,
    ckpt_dir: str,
    resume_dir: str = None
):
    if resume_dir is None:
        config_path = os.path.join(dirname, "params", "config.json")
        tokenizer_path = os.path.join(dirname, "params", "tokenizer.json")

        config = Config(config_path)
        model = Transfomer(config)
        tokenizer = BpeTokenizer(tokenizer_path)
    else:
        config_path = os.path.join(resume_dir, "config.json")
        tokenizer_path = os.path.join(resume_dir, "tokenizer.json")
        weight_path = os.path.join(resume_dir, "model.pt")
        
        config = Config(config_path)
        model = Transfomer(config)
        model.load_state_dict(torch.load(weight_path))
        tokenizer = BpeTokenizer(tokenizer_path)
    
    dataset = SeqDataset(config.m_len)
    
    cache_files = get_files(data_root_path)
    dataset.load(cache_files)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    device = torch.device("cuda")
    model.to(device=device, dtype=torch.bfloat16)

    optim = AdamW(
        model.parameters(),
        eps=config.eps,
        weight_decay=0.02
    )
    criterion = F.cross_entropy
    
    trainer = Trainer(model, tokenizer, config)
    trainer.train(
        dataloader=dataloader,
        optimizer=optim,
        criterion=criterion,
        ckpt_dir=ckpt_dir,
        n_epoch=n_epoch, 
        n_epoch_checkpoint=n_epoch_ckpt,
        max_grad_norm=1.0
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Transformer model.")
    parser.add_argument("--data_root_path", type=str, required=True, help="Path to the root directory of the dataset.")
    parser.add_argument("--n_epoch", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--n_epoch_ckpt", type=int, default=1, help="Number of epochs between checkpoints.")
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
        n_epoch_ckpt=args.n_epoch_ckpt,
        ckpt_dir=args.ckpt_dir,
        resume_dir=resume_dir
    )