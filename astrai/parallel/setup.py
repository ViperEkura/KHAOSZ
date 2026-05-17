import os
from contextlib import contextmanager
from functools import wraps
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


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
):

    if dist.is_available() and dist.is_initialized():
        yield dist.group.WORLD
        return

    if world_size <= 1:
        yield None
        return

    device_id = torch.device(device_type, rank)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_DEVICE"] = str(device_id)

    dist.init_process_group(
        rank=rank, world_size=world_size, backend=backend, device_id=device_id
    )

    try:
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(device_id)
        elif backend == "ccl" and hasattr(torch, "xpu") and torch.xpu.is_available():
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
            ret_args = None
            if get_rank() == rank:
                ret_args = func(*args, **kwargs)

            if sync and dist.is_available() and dist.is_initialized():
                dist.barrier()

            return ret_args

        return wrapper

    return decorator


def wrapper_spawn_func(
    rank: int,
    world_size: int,
    backend: str,
    master_addr: str,
    master_port: str,
    device_type: str,
    func: Callable,
    kwargs: dict,
):
    try:
        with setup_parallel(
            rank=rank,
            world_size=world_size,
            backend=backend,
            master_addr=master_addr,
            master_port=master_port,
            device_type=device_type,
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
    start_method: str = "spawn",
    **kwargs,
):
    # clear environment variables
    for key in [
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_DEVICE",
    ]:
        if key in os.environ:
            del os.environ[key]

    if world_size == 1:
        device_id = torch.device(device_type, 0)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_DEVICE"] = str(device_id)

        func(**kwargs)
        return

    wrapper_spawn_func_args = (
        world_size,
        backend,
        master_addr,
        master_port,
        device_type,
        func,
        kwargs,
    )

    mp.start_processes(
        wrapper_spawn_func,
        args=wrapper_spawn_func_args,
        nprocs=world_size,
        start_method=start_method,
        join=True,
        daemon=True,
    )
