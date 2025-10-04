import torch

from torch.utils.data import Dataset
from khaosz.core import *
from khaosz.trainer import *
from khaosz.trainer.data_util import *

def test_early_stopping_simulation(base_test_env):
    """Simulate early stopping behavior"""
    class EarlyStoppingDataset(Dataset):
        def __init__(self, length=10, stop_after=5):
            self.length = length
            self.stop_after = stop_after
            self.count = 0
            
        def __len__(self):
            return self.length
            
        def __getitem__(self, idx):
            self.count += 1
            if self.count == self.stop_after:
                raise RuntimeError("Simulated early stopping")
            
            return {
                "input_ids": torch.randint(0, 1000, (64,)),
                "target_ids": torch.randint(0, 1000, (64,))
            }
    
    dataset = EarlyStoppingDataset()
    
    optimizer = torch.optim.AdamW(base_test_env["model"].parameters())
    train_config = TrainConfig(
        dataset=dataset,
        optimizer=optimizer,
        checkpoint_dir=base_test_env["test_dir"],
        n_epoch=2,
        batch_size=2,
        checkpoint_interval=1,
        accumulation_steps=1,
        max_grad_norm=1.0,
        random_seed=42
    )
    
    train_config.strategy = StrategyFactory.load(base_test_env["model"], "seq", base_test_env["device"])
    model_parameter = ModelParameter(
        base_test_env["model"], 
        base_test_env["tokenizer"], 
        base_test_env["transformer_config"]
    )
    schedule_config = CosineScheduleConfig(warmup_steps=10, total_steps=20)
    trainer = Trainer(model_parameter, train_config, schedule_config)
    
    # Should handle early stopping gracefully
    checkpoint = None
    try:
        checkpoint = trainer.train()
        assert len(checkpoint.loss_list) == 2
    except Exception:
        # Handle any exceptions
        pass
    
    checkpoint = trainer.train(checkpoint)
    assert len(checkpoint.loss_list) == 10