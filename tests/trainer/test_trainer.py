from astrai.trainer import Trainer

# train_config_factory is injected via fixture


def test_different_batch_sizes(base_test_env, random_dataset, train_config_factory):
    """Test training with different batch sizes"""
    batch_sizes = [1, 2, 4, 8]

    for batch_per_device in batch_sizes:
        train_config = train_config_factory(
            model_fn=lambda: base_test_env["model"],
            dataset=random_dataset,
            test_dir=base_test_env["test_dir"],
            device=base_test_env["device"],
            batch_per_device=batch_per_device,
        )

        assert train_config.batch_per_device == batch_per_device


def test_gradient_accumulation(base_test_env, random_dataset, train_config_factory):
    """Test training with different gradient accumulation steps"""
    grad_accum_steps_list = [1, 2, 4]

    for grad_accum_steps in grad_accum_steps_list:
        train_config = train_config_factory(
            model_fn=lambda: base_test_env["model"],
            dataset=random_dataset,
            test_dir=base_test_env["test_dir"],
            device=base_test_env["device"],
            batch_per_device=2,
            grad_accum_steps=grad_accum_steps,
        )

        trainer = Trainer(train_config)
        trainer.train()

        assert train_config.grad_accum_steps == grad_accum_steps


def test_memory_efficient_training(base_test_env, random_dataset, train_config_factory):
    """Test training with memory-efficient configurations"""
    # Test with smaller batch sizes and gradient checkpointing
    small_batch_configs = [
        {"batch_per_device": 1, "grad_accum_steps": 8},
        {"batch_per_device": 2, "grad_accum_steps": 4},
        {"batch_per_device": 4, "grad_accum_steps": 2},
    ]

    for config in small_batch_configs:
        train_config = train_config_factory(
            model_fn=lambda: base_test_env["model"],
            dataset=random_dataset,
            test_dir=base_test_env["test_dir"],
            device=base_test_env["device"],
            batch_per_device=config["batch_per_device"],
            grad_accum_steps=config["grad_accum_steps"],
        )

        assert train_config.grad_accum_steps == config["grad_accum_steps"]
        assert train_config.batch_per_device == config["batch_per_device"]
