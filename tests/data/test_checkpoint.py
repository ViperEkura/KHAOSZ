import torch
import tempfile
import torch.distributed as dist

from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from khaosz.data.checkpoint import Checkpoint
from khaosz.parallel.setup import get_rank, spawn_parallel_fn

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
        state_dict=model.state_dict(),
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
        
        png_files = list(Path(tmpdir).glob("*.png"))
        assert png_files

def simple_training():
    model = torch.nn.Linear(10, 5)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    
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
        state_dict=model.state_dict(),
        epoch=2,
        iteration=10,
        metrics={"loss": [0.3, 0.2, 0.1]}
    )
    
    rank = get_rank()
    
    if rank == 0:
        shared_dir = tempfile.mkdtemp()
        checkpoint.save(shared_dir)
    else:
        shared_dir = None

    
    if dist.is_initialized():
        dir_list = [shared_dir]
        dist.broadcast_object_list(dir_list, src=0)
        shared_dir = dir_list[0]
    
 
    loaded = Checkpoint.load(shared_dir)
    assert loaded.epoch == 2

def test_multi_process():
    spawn_parallel_fn(
        simple_training,
        world_size=2,
        backend="gloo"
    )