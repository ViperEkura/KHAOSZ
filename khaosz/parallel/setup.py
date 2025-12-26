import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from functools import wraps
from contextlib import contextmanager
from typing import Callable, List, Optional


def get_current_device():
    return os.environ["LOCAL_DEVICE"]

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
    device_type: str = "cuda",
    device_ids: Optional[List[int]] = None
):

    if dist.is_available() and dist.is_initialized():
        yield dist.group.WORLD
        return 

    if world_size <= 1:
        yield None
        return
    
    if device_ids is None:
        device_ids = [i for i in range(world_size)]
    
    rank = device_ids[rank % len(device_ids)]
    device_id = torch.device(device_type, device_ids[rank])
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ["LOCAL_DEVICE"] = str(device_id)
    
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        device_id=device_id
    )
    
    try:
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(device_id)
        elif backend == "ccl" and hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.set_device(device_id)
        
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
    device_type: str,
    device_ids: List[int], 
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
            device_type=device_type,
            device_ids=device_ids
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
    device_type: str = "cuda",
    device_ids: Optional[List[int]] = None,
    **kwargs
):
    # clear environment variables
    for key in ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_DEVICE']:
        if key in os.environ:
            del os.environ[key]
    
    if world_size == 1:
        device_ids = device_ids or [0]
        deice_id = torch.device(device_type, device_ids[0])
        os.environ["LOCAL_DEVICE"] = str(deice_id)
        
        func(**kwargs)
        return

    wrapper_spawn_func_args = (world_size, backend, master_addr, master_port, 
                               device_type, device_ids, func, kwargs)

    mp.spawn(
        wrapper_spawn_func, 
        nprocs=world_size, 
        args=wrapper_spawn_func_args, 
        join=True
    )