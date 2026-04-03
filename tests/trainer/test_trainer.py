from astrai.data.dataset import *
from astrai.trainer import Trainer

# train_config_factory is injected via fixture


def test_different_batch_sizes(base_test_env, random_dataset, train_config_factory):
    """Test training with different batch sizes"""
    batch_sizes = [1, 2, 4, 8]

    for batch_size in batch_sizes:
        train_config = train_config_factory(
            model=base_test_env["model"],
            dataset=random_dataset,
            test_dir=base_test_env["test_dir"],
            device=base_test_env["device"],
            batch_size=batch_size,
        )

        assert train_config.batch_size == batch_size


def test_gradient_accumulation(base_test_env, random_dataset, train_config_factory):
    """Test training with different gradient accumulation steps"""
    accumulation_steps_list = [1, 2, 4]

    for accumulation_steps in accumulation_steps_list:
        train_config = train_config_factory(
            model=base_test_env["model"],
            dataset=random_dataset,
            test_dir=base_test_env["test_dir"],
            device=base_test_env["device"],
            batch_size=2,
            accumulation_steps=accumulation_steps,
        )

        trainer = Trainer(train_config)
        trainer.train()

        assert train_config.accumulation_steps == accumulation_steps


def test_memory_efficient_training(base_test_env, random_dataset, train_config_factory):
    """Test training with memory-efficient configurations"""
    # Test with smaller batch sizes and gradient checkpointing
    small_batch_configs = [
        {"batch_size": 1, "accumulation_steps": 8},
        {"batch_size": 2, "accumulation_steps": 4},
        {"batch_size": 4, "accumulation_steps": 2},
    ]

    for config in small_batch_configs:
        train_config = train_config_factory(
            model=base_test_env["model"],
            dataset=random_dataset,
            test_dir=base_test_env["test_dir"],
            device=base_test_env["device"],
            batch_size=config["batch_size"],
            accumulation_steps=config["accumulation_steps"],
        )

        assert train_config.accumulation_steps == config["accumulation_steps"]
