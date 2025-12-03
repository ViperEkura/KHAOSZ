import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import contextmanager


def get_device_count() -> int:
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.xpu.device_count()
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        return 1
    else:
        return 1

def get_current_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device(f"xpu:{torch.xpu.current_device()}")
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_available_backend():
    if torch.cuda.is_available():
        return "nccl"
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return "ccl"  # Intel XPU use ccl
    else:
        return "gloo"


@contextmanager
def setup_parallel(
    rank: int = 0, 
    world_size: int = 1,
    master_addr: str = "localhost",
    master_port: str = "29500"
):

    if dist.is_available() and dist.is_initialized():
        yield dist.group.WORLD
        return 

    if world_size <= 1:
        yield None
        return
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    
    backend = get_available_backend()
    
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    
    try:
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(rank)
        elif backend == "ccl" and hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.set_device(rank)
        
        yield dist.group.WORLD
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def wrapper_spawn_func(rank, world_size, func, kwargs_dict):
    with setup_parallel(rank, world_size):
        func(**kwargs_dict)

def spawn_parallel_fn(func, world_size=None, **kwargs):
 
    if world_size is None:
        world_size = get_device_count()
    
    if world_size < 1:
        raise ValueError("world_size must be greater than 0")
    
    device_count = get_device_count()
    if world_size > device_count:
        raise ValueError(f"world_size ({world_size}) exceeds available devices ({device_count})")
    
    if world_size == 1:
        func(**kwargs)
        return
    
    mp.spawn(
        wrapper_spawn_func,
        nprocs=world_size,
        args=(world_size, func, kwargs),  
        join=True
    )