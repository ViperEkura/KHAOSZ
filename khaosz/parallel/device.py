import os
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class DeviceStrategy:
    """
    A class representing a device strategy.
    
    Attributes:
        name: Name of the device backend (e.g., 'cuda', 'xpu').
        priority: Higher number means higher priority.
        is_available: A callable that returns True if the device is available.
        make_device: A callable that takes a rank (int) and returns a torch.device.
    """
    name: str
    priority: int
    is_available: Callable[[], bool]
    make_device: Callable[[int], torch.device]


class DeviceStrategyRegistry:
    """
    A registry for device strategies that automatically selects the best available device.
    And allows overriding the device backend via environment variable.
    """
    
    _instance: Optional["DeviceStrategyRegistry"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
            
        self._strategies: List[DeviceStrategy] = []
        
        self.register(DeviceStrategy(
            name="cuda",
            priority=100,
            is_available=torch.cuda.is_available,
            make_device=lambda rank: torch.device(f"cuda:{rank}")
        ))
        
        self.register(DeviceStrategy(
            name="xpu",
            priority=90,
            is_available=torch.xpu.is_available,
            make_device=lambda rank: torch.device(f"xpu:{rank}")
        ))
        
        self.register(DeviceStrategy(
            name="mps",
            priority=80,
            is_available=torch.mps.is_available,
            make_device=lambda _: torch.device("mps")  # MPS ignores rank
        ))
        
        self.register(DeviceStrategy(
            name="cpu",
            priority=0,
            is_available=lambda: True,
            make_device=lambda _: torch.device("cpu")
        ))
        
        self._initialized = True

    def register(self, strategy: DeviceStrategy):
        self._strategies.append(strategy)

    def get_current_device(self) -> torch.device:
        """Return the best available device for the current process."""
        override = os.getenv("TORCH_DEVICE_OVERRIDE")
        sorted_strategies = sorted(self._strategies, key=lambda s: -s.priority)
        
        rank = 0

        if dist.is_available() and dist.is_initialized():
            rank = os.environ["LOCAL_RANK"]
    
        if override:
            return torch.device(override, rank)
        
        
        for strategy in sorted_strategies:
            if strategy.is_available():
               
                return strategy.make_device(rank)

        raise RuntimeError("No device backend is available, including CPU.")


device_registry = DeviceStrategyRegistry()