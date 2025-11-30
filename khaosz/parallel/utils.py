import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from contextlib import contextmanager


@contextmanager
def setup_parallel(
    rank: int = 0, 
    world_size: int=1,
    master_addr: str="localhost",
    master_port: str="29500"
):
    if dist.is_available() and dist.is_initialized():
        yield dist.group.WORLD

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo", 
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)
    
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


def wrapper_func(rank, world_size, func, config_pack):
    with setup_parallel(rank, world_size):
        func(**config_pack)


def spawn_parallel_fn(func, world_size, kwargs_dict):
    mp.spawn(
        wrapper_func,
        nprocs=world_size,
        args=(world_size, func, kwargs_dict,),
        join=True
    )


def get_current_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device(f"xpu:{torch.xpu.current_device()}")
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")