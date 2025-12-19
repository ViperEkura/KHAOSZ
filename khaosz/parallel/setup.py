import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from functools import wraps
from contextlib import contextmanager
from typing import Callable, List, Optional
from khaosz.parallel.device import device_registry


def get_current_device():
    return device_registry.get_current_device()

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

@contextmanager
def setup_parallel(
    rank: int, 
    world_size: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: str = "29500",
    avail_ids: Optional[List[int]] = None
):

    if dist.is_available() and dist.is_initialized():
        yield dist.group.WORLD
        return 

    if world_size <= 1:
        yield None
        return
    
    if avail_ids is None:
        avail_ids = [i for i in range(world_size)]
    
    rank = avail_ids[rank % len(avail_ids)]
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
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

def wrapper_spawn_func(
    rank: int, 
    world_size: int, 
    backend: str, 
    master_addr: str, 
    master_port: str, 
    avail_ids: List[int], 
    func: Callable, 
    kwargs: dict
):
    try:
        with setup_parallel(
            rank=rank, 
            world_size=world_size, 
            backend=backend, 
            master_addr=master_addr, 
            master_port=master_port,
            avail_ids=avail_ids
        ):
            func(**kwargs)
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise

def spawn_parallel_fn(
    func: Callable, 
    world_size: int, 
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: str = "29500",
    avail_ids: Optional[List[int]] = None,
    **kwargs
):
    
    if world_size == 1:
        func(**kwargs)
        return

    # clear environment variables
    for key in ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK']:
        if key in os.environ:
            del os.environ[key]

    wrapper_spawn_func_args = (world_size, backend, 
                        master_addr, master_port, avail_ids, func, kwargs)
    
    mp.spawn(
        wrapper_spawn_func, 
        nprocs=world_size, 
        args=wrapper_spawn_func_args, 
        join=True
    )