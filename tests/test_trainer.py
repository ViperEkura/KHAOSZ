import os
import json
import torch
import shutil
import pytest
import pickle
import tempfile
import matplotlib

from torch.utils.data import Dataset
from khaosz.core import *
from khaosz.trainer import *

# to avoid _tkinter.TclError
matplotlib.use('Agg')


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
    
    transformer_config = TransformerConfig().load(config_path)
    model = Transformer(transformer_config)
    tokenizer = BpeTokenizer()
    
    class DummyDataset(Dataset):
        def __init__(self, length=10):
            self.length = length
            
        def __len__(self):
            return self.length
            
        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 1000, (64,)),
                "target_ids": torch.randint(0, 1000, (64,))
            }
    
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
        dataset=test_env["dataset"],
        optimizer=optimizer,
        checkpoint_dir=test_env["test_dir"],
        n_epoch=1,
        batch_size=2,
        checkpoint_interval=5,
        accumulation_steps=1,
        max_grad_norm=1.0,
        random_seed=42
    )
    assert train_config.get_kwargs()["batch_size"] == 2

def test_cosine_schedule(test_env):
    assert test_env is not None
    schedule_config = CosineScheduleConfig(
        warmup_steps=100,
        total_steps=1000
    )
    kwargs = schedule_config.get_kwargs()
    assert kwargs["warmup_steps"] == 100
    assert kwargs["lr_decay_steps"] == 900
    

def test_sgdr_schedule(test_env):
    assert test_env is not None
    schedule_config = SgdrScheduleConfig(
        warmup_steps=100,
        cycle_length=200,
        t_mult=2
    )
    kwargs = schedule_config.get_kwargs()
    assert kwargs["warmup_steps"] == 100
    assert kwargs["cycle_length"] == 200
    assert kwargs["t_mult"] == 2
    
def test_trainer_train(test_env):
    optimizer = torch.optim.AdamW(test_env["model"].parameters())
    train_config = TrainConfig(
        dataset=test_env["dataset"],
        optimizer=optimizer,
        checkpoint_dir=test_env["test_dir"],
        n_epoch=1,
        batch_size=2,
        checkpoint_interval=5,
        accumulation_steps=1,
        max_grad_norm=1.0,
        random_seed=42
    )
    schedule_config = CosineScheduleConfig(
        warmup_steps=100,
        total_steps=1000
    )
    
    train_config.strategy = StrategyFactory.load(
        test_env["model"], 
        "seq",
        pad_token_id=test_env["tokenizer"].pad_id
    )
    model_parameter = ModelParameter(
        test_env["model"], 
        test_env["tokenizer"], 
        test_env["transformer_config"]
    )
    trainer = Trainer(model_parameter, train_config, schedule_config)
    trainer.train()

def test_checkpoint(test_env):
    temp_dir = test_env["test_dir"]
    config = test_env["transformer_config"]
    model = test_env["model"]
    tokenizer = test_env["tokenizer"]
    optimizer = torch.optim.AdamW(model.parameters())
    for _ in range(3):
        optimizer.step()
    
    checkpoint = Checkpoint(
        model=model, 
        tokenizer=tokenizer, 
        config=config, 
        loss_list=[1.0, 2.0, 3.0],
        optim_state=optimizer.state_dict()
    )
    ckpt_dir = os.path.join(temp_dir, "ckpt")
    checkpoint.save(ckpt_dir)

    loaded_ckpt = Checkpoint()
    loaded_ckpt.load(ckpt_dir)
    
    assert loaded_ckpt.loss_list == [1.0, 2.0, 3.0]
    assert loaded_ckpt.optim_state == optimizer.state_dict()

    for p1, p2 in zip(model.parameters(), loaded_ckpt.model.parameters()):
        assert torch.allclose(p1, p2)


def test_checkpoint_train(test_env):
    config = test_env["transformer_config"]
    model = test_env["model"]
    tokenizer = test_env["tokenizer"]
    
    class InterruptDataset(Dataset):
        def __init__(self, length, interrupt_idx=0):
            self.length = length
            self.interrupt_idx = interrupt_idx
            
        def __len__(self):
            return self.length
            
        def __getitem__(self, idx):
            if idx == self.interrupt_idx:
                self.interrupt_idx = -1
                raise Exception("Interrupt")
            
            return {
                "input_ids": torch.randint(0, 1000, (64,)),
                "target_ids": torch.randint(0, 1000, (64,))
            }


    dataset = InterruptDataset(length=10, interrupt_idx=3)
    param = ModelParameter(model, tokenizer, config)
    optimizer = torch.optim.AdamW(test_env["model"].parameters())
    train_config = TrainConfig(
        dataset=dataset,
        optimizer=optimizer,
        checkpoint_dir=test_env["test_dir"],
        n_epoch=1,
        batch_size=2,
        checkpoint_interval=1,
        accumulation_steps=1,
        max_grad_norm=1.0,
        random_seed=42
    )
    
    train_config.strategy = StrategyFactory.load(
        test_env["model"], 
        "seq",
        pad_token_id=test_env["tokenizer"].pad_id
    )
    schedule_config = CosineScheduleConfig(
        warmup_steps=1,
        total_steps=5
    )
    trainer = Trainer(param, train_config, schedule_config)
    
    checkpoint = None
    
    try:
        checkpoint = trainer.train()
    except Exception:
        pass

    checkpoint = trainer.train(train_checkpoint=checkpoint)
    assert len(checkpoint.loss_list) == 5 - 1
    