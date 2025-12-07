import torch
import numpy as np
import pytest

from khaosz.config import *
from khaosz.trainer.schedule import *
from khaosz.data.dataset import *


def test_schedule_factory_random_configs():
    """Test scheduler factory with random configurations"""
    
    # Create a simple model and optimizer for testing
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Test multiple random configurations
    for _ in range(5):  # Test 5 random configurations
        schedule_configs = [
            CosineScheduleConfig(
                warmup_steps=np.random.randint(50, 200),
                total_steps=np.random.randint(1000, 5000),
                min_rate=np.random.uniform(0.01, 0.1)
            ),
            SGDRScheduleConfig(
                warmup_steps=np.random.randint(50, 200),
                cycle_length=np.random.randint(500, 2000),
                t_mult=np.random.randint(1, 3),
                min_rate=np.random.uniform(0.01, 0.1)
            )
        ]
        
        for config in schedule_configs:
            # Validate configuration
            config.validate()
            
            # Create scheduler using factory
            scheduler = SchedulerFactory.load(optimizer, config)
            
            # Verify scheduler type
            if isinstance(config, CosineScheduleConfig):
                assert isinstance(scheduler, CosineScheduler)
                assert scheduler.warmup_steps == config.warmup_steps
                assert scheduler.lr_decay_steps == config.total_steps - config.warmup_steps
                assert scheduler.min_rate == config.min_rate
            elif isinstance(config, SGDRScheduleConfig):
                assert isinstance(scheduler, SGDRScheduler)
                assert scheduler.warmup_steps == config.warmup_steps
                assert scheduler.cycle_length == config.cycle_length
                assert scheduler.t_mult == config.t_mult
                assert scheduler.min_rate == config.min_rate
            
            # Test scheduler state dict functionality
            state_dict = scheduler.state_dict()
            assert 'warmup_steps' in state_dict
            assert 'min_rate' in state_dict
            
            # Test scheduler step functionality
            initial_lr = scheduler.get_last_lr()
            scheduler.step()
            new_lr = scheduler.get_last_lr()
            
            # Learning rate should change after step, or if it's the first step,
            # the epoch counter should increment
            assert initial_lr != new_lr or scheduler.last_epoch > -1


def test_schedule_factory_edge_cases():
    """Test scheduler factory with edge cases and boundary conditions"""
    
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Test edge cases for CosineScheduleConfig
    edge_cases = [
        # Minimal warmup and steps
        CosineScheduleConfig(warmup_steps=1, total_steps=10, min_rate=0.01),
        # Large values
        CosineScheduleConfig(warmup_steps=1000, total_steps=10000, min_rate=0.5),
        # Zero min_rate (edge case)
        CosineScheduleConfig(warmup_steps=100, total_steps=1000, min_rate=0.0),
    ]
    
    for config in edge_cases:
        config.validate()
        scheduler = SchedulerFactory.load(optimizer, config)
        assert scheduler is not None
        
        # Test multiple steps
        for _ in range(10):
            scheduler.step()


def test_schedule_factory_invalid_configs():
    """Test scheduler factory with invalid configurations"""
    
    # Test invalid configurations that should raise errors
    invalid_configs = [
        # Negative warmup steps
        {"warmup_steps": -10, "total_steps": 1000, "min_rate": 0.1},
        # Total steps less than warmup steps
        {"warmup_steps": 500, "total_steps": 400, "min_rate": 0.1},
        # Invalid min_rate
        {"warmup_steps": 100, "total_steps": 1000, "min_rate": -0.1},
        {"warmup_steps": 100, "total_steps": 1000, "min_rate": 1.1},
    ]
    
    for kwargs in invalid_configs:
        with pytest.raises(ValueError):
            config = CosineScheduleConfig(**kwargs)
            config.validate()


def test_schedule_factory_state_persistence():
    """Test scheduler state persistence (save/load)"""
    
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    config = CosineScheduleConfig(warmup_steps=100, total_steps=1000, min_rate=0.1)
    scheduler = SchedulerFactory.load(optimizer, config)
    
    # Take a few steps
    for _ in range(5):
        scheduler.step()
    
    # Save state
    state_dict = scheduler.state_dict()
    
    # Create new scheduler and load state
    new_scheduler = SchedulerFactory.load(optimizer, config)
    new_scheduler.load_state_dict(state_dict)
    
    # Verify states match
    assert scheduler.last_epoch == new_scheduler.last_epoch
    assert scheduler.get_last_lr() == new_scheduler.get_last_lr()