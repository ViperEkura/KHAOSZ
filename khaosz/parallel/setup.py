import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from typing import Callable
from functools import wraps
from contextlib import contextmanager


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1

def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0

def get_current_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device(f"xpu:{torch.xpu.current_device()}")
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

@contextmanager
def setup_parallel(
    rank: int, 
    world_size: int,
    backend: str = "nccl",
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
    
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{master_addr}:{master_port}",
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

def only_on_rank(rank, sync=False):
    """
    decorator to run a function only on a specific rank.
    """
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if get_rank() == rank:
                return func(*args, **kwargs)
            if sync:
                dist.barrier()
        
        return wrapper
    
    return decorator

def wrapper_spawn_func(rank, world_size, backend, func, kwargs):
    with setup_parallel(rank, world_size, backend):
        func(**kwargs)

def spawn_parallel_fn(func: Callable, world_size: int, backend: str, **kwargs):
    
    if world_size == 1:
        func(**kwargs)
        return

    # clear environment variables
    for key in ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK']:
        if key in os.environ:
            del os.environ[key]
    
    mp.spawn(
        wrapper_spawn_func,
        nprocs=world_size,
        args=(world_size, backend, func, kwargs),  
        join=True
    )