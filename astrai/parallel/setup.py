import ctypes
import logging
import os
import signal
import socket
import sys
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import timedelta
from functools import wraps
from typing import Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from astrai.signal_handler import install_early_signal_handlers

logger = logging.getLogger(__name__)

# Rank processes get this long to honour a termination signal before the
# launcher force-kills them; the same budget bounds the launcher's own cleanup.
SHUTDOWN_GRACE_SECONDS = 30.0


def find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])


def get_current_device():
    return os.environ["LOCAL_DEVICE"]


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


@contextmanager
def setup_parallel(
    rank: int,
    world_size: int,
    local_rank: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: str = "29500",
    device_type: str = "cuda",
    process_group_timeout: Optional[float] = None,
):

    if dist.is_available() and dist.is_initialized():
        yield dist.group.WORLD
        return

    if world_size <= 1:
        device_id = torch.device(device_type, local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_DEVICE"] = str(device_id)
        yield None
        return

    device_id = torch.device(device_type, local_rank)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_DEVICE"] = str(device_id)

    pg_kwargs = dict(rank=rank, world_size=world_size, backend=backend)
    if backend in ("nccl", "ccl"):
        pg_kwargs["device_id"] = device_id
    if process_group_timeout:
        # Bounds how long a stalled collective (dead or wedged peer) can block,
        # which is also the worst-case latency of a graceful stop.
        pg_kwargs["timeout"] = timedelta(seconds=process_group_timeout)

    dist.init_process_group(**pg_kwargs)

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


def die_with_parent() -> None:
    """SIGTERM this rank when the launching process disappears."""
    if not sys.platform.startswith("linux"):
        return
    try:
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except OSError:
        return
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGTERM)


def follow_parent(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: tie a worker's life to its training rank."""
    die_with_parent()


def _reap(processes, grace: float) -> None:
    """Join rank processes, force-killing any that outlive the grace period."""
    deadline = time.monotonic() + grace
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            logger.warning(
                "Rank %s did not exit within %.0fs, killing it", process.pid, grace
            )
            process.kill()
            process.join()


def agree_stop(requested: bool, device) -> bool:
    """Agree on a stop with every rank, so they all leave on the same step."""
    if not (dist.is_available() and dist.is_initialized()):
        return requested
    if dist.get_world_size() == 1:
        return requested
    flag = torch.tensor([1.0 if requested else 0.0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _run_single_rank(
    rank: int,
    world_size: int,
    backend: str,
    master_addr: str,
    master_port: str,
    device_type: str,
    func: Callable,
    kwargs: Dict,
    process_group_timeout: Optional[float] = None,
    child: bool = False,
):
    install_early_signal_handlers()
    if child:
        os.environ["ASTRAI_RANK_CHILD"] = "1"
        die_with_parent()
    with setup_parallel(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        backend=backend,
        master_addr=master_addr,
        master_port=master_port,
        device_type=device_type,
        process_group_timeout=process_group_timeout,
    ):
        func(**kwargs)


class LaunchStrategy(ABC):
    """Strategy for launching a function in a distributed context."""

    def __init__(
        self,
        world_size: int,
        backend: str,
        master_addr: str,
        master_port: str,
        device_type: str,
        start_method: str,
        process_group_timeout: Optional[float] = None,
    ):
        self.world_size = world_size
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port
        self.device_type = device_type
        self.start_method = start_method
        self.process_group_timeout = process_group_timeout

    @abstractmethod
    def launch(self, func: Callable, **kwargs):
        raise NotImplementedError


class TorchrunStrategy(LaunchStrategy):
    """External orchestrator (torchrun, SLURM, K8s) — env vars pre-set."""

    def launch(self, func: Callable, **kwargs):
        install_early_signal_handlers()
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        with setup_parallel(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            backend=self.backend,
            master_addr=os.environ.get("MASTER_ADDR", self.master_addr),
            master_port=os.environ.get("MASTER_PORT", self.master_port),
            device_type=self.device_type,
            process_group_timeout=self.process_group_timeout,
        ):
            func(**kwargs)


class LocalStrategy(LaunchStrategy):
    """Local launcher — single-process or mp.start_processes."""

    def launch(self, func: Callable, **kwargs):
        args = (
            self.world_size,
            self.backend,
            self.master_addr,
            self.master_port,
            self.device_type,
            func,
            kwargs,
            self.process_group_timeout,
        )

        if self.world_size == 1:
            _run_single_rank(0, *args, False)
            return

        install_early_signal_handlers()
        ctx = mp.start_processes(
            _run_single_rank,
            args=(*args, True),
            nprocs=self.world_size,
            start_method=self.start_method,
            join=False,
        )

        parent_stop = threading.Event()
        received: list = []
        original_handlers = {}

        def _parent_handler(signum, frame):
            sig = signal.Signals(signum)
            logger.warning(
                "Parent (pid=%d) received %s, forwarding to children...",
                os.getpid(),
                sig.name,
            )
            received.append(signum)
            parent_stop.set()
            for p in ctx.processes:
                if p.is_alive():
                    p.terminate()

        for sig in (signal.SIGTERM, signal.SIGINT):
            prev = signal.signal(sig, _parent_handler)
            if prev not in (signal.SIG_DFL, signal.SIG_IGN, None, _parent_handler):
                original_handlers[sig] = prev

        try:
            # Keep joining after a stop signal: a rank may need more than the
            # failure-cleanup grace period to finish its step and checkpoint.
            # A failed rank still makes join raise and enters bounded cleanup.
            while True:
                if ctx.join(timeout=1.0, grace_period=SHUTDOWN_GRACE_SECONDS):
                    break
        except BaseException:
            logger.warning(
                "Parent received unexpected exception, terminating children..."
            )
            for p in ctx.processes:
                if p.is_alive():
                    p.terminate()
            raise
        finally:
            for sig, handler in original_handlers.items():
                signal.signal(sig, handler)
            _reap(ctx.processes, SHUTDOWN_GRACE_SECONDS)

        if parent_stop.is_set():
            raise SystemExit(128 + (received[0] if received else signal.SIGTERM))


def _is_external_launcher() -> bool:
    """Whether an external launcher (torchrun/elastic/manual env) started us."""
    if dist.is_torchelastic_launched():
        return True
    if "LOCAL_WORLD_SIZE" in os.environ:
        return True
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return True
    return False


def spawn_parallel_fn(
    func: Callable,
    world_size: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: Optional[str] = None,
    device_type: str = "cuda",
    start_method: str = "spawn",
    process_group_timeout: Optional[float] = None,
    **kwargs,
):
    if master_port is None:
        master_port = find_free_port()
    if _is_external_launcher():
        strategy = TorchrunStrategy(
            world_size,
            backend,
            master_addr,
            master_port,
            device_type,
            start_method,
            process_group_timeout,
        )
    else:
        strategy = LocalStrategy(
            world_size,
            backend,
            master_addr,
            master_port,
            device_type,
            start_method,
            process_group_timeout,
        )
    strategy.launch(func, **kwargs)
