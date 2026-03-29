from typing import Any, Dict, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ScheduleConfig(ABC):
    """Base configuration class for learning rate schedulers.
    
    Provides common validation and interface for all schedule types.
    """
    
    schedule_type: str = field(
        default="cosine",
        metadata={
            "help": "Type of learning rate schedule.", 
            "choices": ["cosine", "sgdr"]
        }
    )
    warmup_steps: int = field(
        default=1000,
        metadata={"help": "Number of warmup steps."}
    )
    min_rate: float = field(
        default=0.05,
        metadata={"help": "Minimum learning rate multiplier."}
    )
    
    @abstractmethod
    def get_kwargs(self) -> Dict[str, Any]:
        """Get configuration kwargs for scheduler creation."""
        raise NotImplementedError
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if not 0 <= self.min_rate <= 1:
            raise ValueError(f"min_rate must be between 0 and 1, got {self.min_rate}")


@dataclass
class CosineScheduleConfig(ScheduleConfig):
    """Cosine annealing learning rate schedule configuration."""
    
    total_steps: int = field(
        default=None,
        metadata={"help": "Total training steps for cosine schedule."}
    )
    
    def __post_init__(self) -> None:
        self.schedule_type = "cosine"
        self.validate()
    
    def get_kwargs(self) -> Dict[str, Any]:
        if self.total_steps is None:
            raise ValueError("total_steps must be specified for cosine schedule")
            
        return {
            "schedule_type": self.schedule_type,
            "warmup_steps": self.warmup_steps,
            "lr_decay_steps": self.total_steps - self.warmup_steps,
            "min_rate": self.min_rate
        }
    
    def validate(self) -> None:
        super().validate()
        if self.total_steps is not None and self.total_steps <= self.warmup_steps:
            raise ValueError(f"total_steps ({self.total_steps}) must be greater than warmup_steps ({self.warmup_steps})")


@dataclass
class SGDRScheduleConfig(ScheduleConfig):
    """Stochastic Gradient Descent with Warm Restarts schedule configuration."""
    
    cycle_length: int = field(
        default=1000,
        metadata={"help": "Length of the first cycle in steps."}
    )
    t_mult: int = field( 
        default=2,
        metadata={"help": "Multiplier for cycle length growth."}
    )

    def __post_init__(self) -> None:
        self.schedule_type = "sgdr"
        self.validate()

    def get_kwargs(self) -> Dict[str, Any]:
        return {
            "schedule_type": self.schedule_type,
            "warmup_steps": self.warmup_steps,
            "cycle_length": self.cycle_length,
            "min_rate": self.min_rate,
            "t_mult": self.t_mult
        }
    
    def validate(self) -> None:
        super().validate()
        if self.cycle_length <= 0:
            raise ValueError(f"cycle_length must be positive, got {self.cycle_length}")
        if self.t_mult < 1:
            raise ValueError(f"t_mult must be >= 1, got {self.t_mult}")


class ScheduleConfigFactory:
    """Factory class for creating ScheduleConfig instances.
    
    Supports both direct instantiation and factory creation methods.
    
    Example usage:
        # Direct creation
        config = CosineScheduleConfig(total_steps=10000)
        
        # Factory method
        config = ScheduleConfigFactory.create("cosine", total_steps=10000)
    """
    
    CONFIG_MAP: Dict[str, Type[ScheduleConfig]] = {
        "cosine": CosineScheduleConfig,
        "sgdr": SGDRScheduleConfig,
    }
    
    @classmethod
    def create(cls, schedule_type: str, **kwargs) -> ScheduleConfig:
        """Create a schedule config instance.
        
        Args:
            schedule_type: Type of schedule ("cosine", "sgdr")
            **kwargs: Arguments passed to the config constructor
            
        Returns:
            ScheduleConfig instance
            
        Raises:
            ValueError: If schedule_type is not supported
        """
        if schedule_type not in cls.CONFIG_MAP:
            raise ValueError(
                f"Unknown schedule type: '{schedule_type}'. "
                f"Supported types: {sorted(cls.CONFIG_MAP.keys())}"
            )
        
        config_cls = cls.CONFIG_MAP[schedule_type]
        return config_cls(**kwargs)
    
    @classmethod
    def available_types(cls) -> list:
        """Return list of available schedule type names."""
        return list(cls.CONFIG_MAP.keys())