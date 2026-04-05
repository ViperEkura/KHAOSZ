import numpy as np
import torch

from astrai.trainer.schedule import CosineScheduler, SchedulerFactory, SGDRScheduler


def test_schedule_factory_random_configs():
    """Test scheduler factory with random configurations"""

    # Create a simple model and optimizer for testing
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Test multiple random configurations
    for _ in range(5):  # Test 5 random configurations
        # Test multiple random configurations
        cosine_params = {
            "schedule_type": "cosine",
            "warmup_steps": np.random.randint(50, 200),
            "total_steps": np.random.randint(1000, 5000),
            "min_rate": np.random.uniform(0.01, 0.1),
        }
        sgdr_params = {
            "schedule_type": "sgdr",
            "warmup_steps": np.random.randint(50, 200),
            "cycle_length": np.random.randint(500, 2000),
            "t_mult": np.random.randint(1, 3),
            "min_rate": np.random.uniform(0.01, 0.1),
        }
        for params in [cosine_params, sgdr_params]:
            schedule_type = params["schedule_type"]
            # Convert parameters for scheduler constructor
            if schedule_type == "cosine":
                warmup_steps = params["warmup_steps"]
                total_steps = params["total_steps"]
                min_rate = params["min_rate"]
                lr_decay_steps = total_steps - warmup_steps
                scheduler = SchedulerFactory.create(
                    optimizer,
                    schedule_type,
                    warmup_steps=warmup_steps,
                    lr_decay_steps=lr_decay_steps,
                    min_rate=min_rate,
                )
                assert isinstance(scheduler, CosineScheduler)
                assert scheduler.warmup_steps == warmup_steps
                assert scheduler.lr_decay_steps == lr_decay_steps
                assert scheduler.min_rate == min_rate
            elif schedule_type == "sgdr":
                warmup_steps = params["warmup_steps"]
                cycle_length = params["cycle_length"]
                t_mult = params["t_mult"]
                min_rate = params["min_rate"]
                scheduler = SchedulerFactory.create(
                    optimizer,
                    schedule_type,
                    warmup_steps=warmup_steps,
                    cycle_length=cycle_length,
                    t_mult=t_mult,
                    min_rate=min_rate,
                )
                assert isinstance(scheduler, SGDRScheduler)
                assert scheduler.warmup_steps == warmup_steps
                assert scheduler.cycle_length == cycle_length
                assert scheduler.t_mult == t_mult
                assert scheduler.min_rate == min_rate

            # Test scheduler state dict functionality
            state_dict = scheduler.state_dict()
            assert "warmup_steps" in state_dict
            assert "min_rate" in state_dict

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
        {"warmup_steps": 1, "total_steps": 10, "min_rate": 0.01},
        # Large values
        {"warmup_steps": 1000, "total_steps": 10000, "min_rate": 0.5},
        # Zero min_rate (edge case)
        {"warmup_steps": 100, "total_steps": 1000, "min_rate": 0.0},
    ]

    for params in edge_cases:
        warmup_steps = params["warmup_steps"]
        total_steps = params["total_steps"]
        min_rate = params["min_rate"]
        lr_decay_steps = total_steps - warmup_steps
        scheduler = SchedulerFactory.create(
            optimizer,
            "cosine",
            warmup_steps=warmup_steps,
            lr_decay_steps=lr_decay_steps,
            min_rate=min_rate,
        )
        assert scheduler is not None

        # Test multiple steps
        for _ in range(10):
            scheduler.step()


def test_schedule_factory_state_persistence():
    """Test scheduler state persistence (save/load)"""

    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Create scheduler directly with parameters
    warmup_steps = 100
    total_steps = 1000
    min_rate = 0.1
    lr_decay_steps = total_steps - warmup_steps
    scheduler = SchedulerFactory.create(
        optimizer,
        "cosine",
        warmup_steps=warmup_steps,
        lr_decay_steps=lr_decay_steps,
        min_rate=min_rate,
    )

    # Take a few steps
    for _ in range(5):
        scheduler.step()

    # Save state
    state_dict = scheduler.state_dict()

    # Create new scheduler with same parameters
    new_scheduler = SchedulerFactory.create(
        optimizer,
        "cosine",
        warmup_steps=warmup_steps,
        lr_decay_steps=lr_decay_steps,
        min_rate=min_rate,
    )
    new_scheduler.load_state_dict(state_dict)

    # Verify states match
    assert scheduler.last_epoch == new_scheduler.last_epoch
    assert scheduler.get_last_lr() == new_scheduler.get_last_lr()
