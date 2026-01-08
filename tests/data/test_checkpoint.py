import os
import torch
import tempfile

from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from khaosz.data.checkpoint import Checkpoint
from khaosz.parallel.setup import spawn_parallel_fn

def test_single_process():
    model = torch.nn.Linear(10, 5)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(3):
        for iteration in range(10):

            x = torch.randn(32, 10)
            y = torch.randn(32, 5)
            loss = model(x).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()
    
    checkpoint = Checkpoint(
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        epoch=3,
        iteration=30,
        metrics={
            "loss": [0.5, 0.4, 0.3, 0.2, 0.1],
            "accuracy": [0.6, 0.7, 0.8, 0.85, 0.9]
        }
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint.save(tmpdir, save_metric_plot=True)
        
        loaded_checkpoint = Checkpoint.load(tmpdir)
        
        assert loaded_checkpoint.epoch == 3
        assert loaded_checkpoint.iteration == 30
        assert loaded_checkpoint.metrics["loss"] == [0.5, 0.4, 0.3, 0.2, 0.1]
        
        assert 'param_groups' in loaded_checkpoint.optimizer_state_dict
        assert 'state' in loaded_checkpoint.optimizer_state_dict
        
        png_files = list(Path(tmpdir).glob("*.png"))
        assert png_files

def simple_training():
    rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # 简单的训练逻辑
    model = torch.nn.Linear(10, 5)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    
    # 训练步骤
    for epoch in range(2):
        for iteration in range(5):
            x = torch.randn(16, 10)
            y = torch.randn(16, 5)
            loss = model(x).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

    checkpoint = Checkpoint(
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        epoch=2,
        iteration=10,
        metrics={"loss": [0.3, 0.2, 0.1]}
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint.save(tmpdir)
        loaded = Checkpoint.load(tmpdir)
        assert loaded.epoch == 2
        print(f"Rank {rank}: Checkpoint test passed")

def test_multi_process():
    spawn_parallel_fn(
        simple_training,
        world_size=2,
        backend="gloo"
    )