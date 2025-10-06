import torch

from khaosz.core import *
from khaosz.trainer import *
from khaosz.trainer.data_util import *

def test_callback_integration(base_test_env, random_dataset):
    """Test that all callbacks are properly integrated"""
    optimizer = torch.optim.AdamW(base_test_env["model"].parameters())
    train_config = TrainConfig(
        dataset=random_dataset,
        optimizer=optimizer,
        checkpoint_dir=base_test_env["test_dir"],
        n_epoch=1,
        batch_size=2,
        checkpoint_interval=3,
        accumulation_steps=1,
        max_grad_norm=1.0,
        random_seed=42
    )
    
    schedule_config = CosineScheduleConfig(
        warmup_steps=10,
        total_steps=20
    )
    
    # Create custom callbacks to track calls
    callback_calls = []
    
    class TrackingCallback(TrainCallback):
        def on_train_begin(self, trainer, context):
            callback_calls.append('on_train_begin')
        
        def on_batch_end(self, trainer, context):
            callback_calls.append('on_batch_end')
        
        def on_epoch_end(self, trainer, context):
            callback_calls.append('on_epoch_end')
    
    train_config.strategy = StrategyFactory.load(base_test_env["model"], "seq", base_test_env["device"])
    model_parameter = ModelParameter(
        base_test_env["model"], 
        base_test_env["tokenizer"], 
        base_test_env["transformer_config"]
    )
    
    trainer = Trainer(
        model_parameter, 
        train_config, 
        schedule_config,
        callbacks=[TrackingCallback(), ProgressBarCallback()]
    )
    
    trainer.train()
    
    # Verify callbacks were called
    assert 'on_train_begin' in callback_calls
    assert 'on_batch_end' in callback_calls
    assert 'on_epoch_end' in callback_calls