import os
import sys

parent_dir = os.path.join(os.path.dirname(__file__), '..')
abs_parent_dir = os.path.abspath(parent_dir)
sys.path.insert(0, abs_parent_dir)

import tempfile
import json
import shutil
import pytest
import torch
import pickle

from torch.utils.data import Dataset
from khaosz.module import ModelParameter, BpeTokenizer,  TransformerConfig, Transformer
from khaosz.trainer import Trainer, DatasetLoader, TrainConfig, CosineScheduleConfig, SgdrScheduleConfig

@pytest.fixture
def test_env():
    test_dir = tempfile.mkdtemp()
    config_path = os.path.join(test_dir, "config.json")
    
    config = {
        "vocab_size": 1000,
        "n_dim": 128,
        "n_head": 4,
        "n_kvhead": 2,
        "d_ffn": 256,
        "m_len": 64,
        "n_layer": 2,
        "norm_eps": 1e-5
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    transformer_config = TransformerConfig(config_path)
    model = Transformer(transformer_config)
    tokenizer = BpeTokenizer()
    
    class DummyDataset(Dataset):
        def __init__(self, length=10):
            self.length = length
            
        def __len__(self):
            return self.length
            
        def __getitem__(self, idx):
            return (
                torch.randint(0, 1000, (64,)),
                torch.randint(0, 1000, (64,))
            )
    
    dataset = DummyDataset()
    
    yield {
        "test_dir": test_dir,
        "config_path": config_path,
        "transformer_config": transformer_config,
        "model": model,
        "tokenizer": tokenizer,
        "dataset": dataset
    }
    
    shutil.rmtree(test_dir)

def test_dataset_loader(test_env):
    test_dir = test_env["test_dir"]
    pkl_path = os.path.join(test_dir, "test_data.pkl")
    
    dummy_data = {"sequence": torch.randint(0, 1000, (64,))}
    with open(pkl_path, "wb") as f:
        pickle.dump(dummy_data, f)
    
    loaded_dataset = DatasetLoader.load(train_type="seq", load_path=pkl_path, max_len=64, device="cpu")
    assert loaded_dataset is not None

def test_training_config(test_env):
    optimizer = torch.optim.AdamW(test_env["model"].parameters())
    train_config = TrainConfig(
        train_type="seq",
        dataset=test_env["dataset"],
        optimizer=optimizer,
        ckpt_dir=test_env["test_dir"],
        n_epoch=1,
        batch_size=2,
        n_iter_ckpt=5,
        n_iter_step=1,
        max_grad_norm=1.0,
        random_seed=42
    )
    assert train_config.get_kwargs()["batch_size"] == 2

def test_cosine_schedule(test_env):
    schedule_config = CosineScheduleConfig(
        warning_step=100,
        total_iters=1000
    )
    kwargs = schedule_config.get_kwargs()
    assert kwargs["warning_step"] == 100
    assert kwargs["lr_decay_iters"] == 900
    

def test_sgdr_schedule(test_env):
    schedule_config = SgdrScheduleConfig(
        warning_step=100,
        cycle_length=200,
        T_mult=2
    )
    kwargs = schedule_config.get_kwargs()
    assert kwargs["warning_step"] == 100
    assert kwargs["cycle_length"] == 200
    assert kwargs["T_mult"] == 2
    
def test_trainer_train(test_env):
    optimizer = torch.optim.AdamW(test_env["model"].parameters())
    train_config = TrainConfig(
        train_type="seq",
        dataset=test_env["dataset"],
        optimizer=optimizer,
        ckpt_dir=test_env["test_dir"],
        n_epoch=1,
        batch_size=2,
        n_iter_ckpt=5,
        n_iter_step=1,
        max_grad_norm=1.0,
        random_seed=42
    )
    schedule_config = CosineScheduleConfig(
        warning_step=100,
        total_iters=1000
    )
    model_parameter = ModelParameter(
        test_env["model"], 
        test_env["tokenizer"], 
        test_env["transformer_config"]
    )
    trainer = Trainer(model_parameter)
    trainer.train(train_config, schedule_config)