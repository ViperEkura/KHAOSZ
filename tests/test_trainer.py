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
from torch.utils.data import Dataset
from khaosz.trainer import Trainer, DatasetLoader, TrainConfig, CosineScheduleConfig
from khaosz.module import ModelParameter, ParameterLoader
from khaosz.module.transformer import TransformerConfig, Transformer

@pytest.fixture
def test_env():
    # 创建临时测试环境
    test_dir = tempfile.mkdtemp()
    config_path = os.path.join(test_dir, "config.json")
    
    # 创建测试配置
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
    
    # 初始化模型参数
    transformer_config = TransformerConfig(config_path)
    model = Transformer(transformer_config)
    tokenizer = ParameterLoader.load(test_dir).tokenizer
    
    # 创建测试数据集
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
    
    # 清理临时文件
    shutil.rmtree(test_dir)

def test_dataset_loader(test_env):
    # 测试数据集加载器
    loaded_dataset = DatasetLoader.load(
        train_type="seq",
        load_path=test_env["test_dir"],
        max_len=64,
        device="cpu"
    )
    assert loaded_dataset is not None

def test_training_config(test_env):
    # 测试训练配置
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
    assert train_config.get_kargs()["batch_size"] == 2

def test_cosine_schedule(test_env):
    # 测试余弦学习率调度器
    schedule_config = CosineScheduleConfig(
        warning_step=100,
        total_iters=1000
    )
    assert schedule_config.get_kargs()["warning_step"] == 100

def test_trainer_initialization(test_env):
    # 测试训练器初始化
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
    
    trainer = Trainer(ModelParameter(test_env["model"], test_env["model"].tokenizer, test_env["transformer_config"]))
    assert trainer is not None