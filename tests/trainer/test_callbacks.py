import torch

from khaosz.config import *
from khaosz.trainer import *

def test_callback_integration(base_test_env, random_dataset):
    """Test that all callbacks are properly integrated"""
    schedule_config = CosineScheduleConfig(
        warmup_steps=10,
        total_steps=20
    )
    
    optimizer = torch.optim.AdamW(base_test_env["model"].parameters())
    scheduler = SchedulerFactory.load(optimizer, schedule_config)
    
    train_config = TrainConfig(
        model=base_test_env["model"],
        strategy='seq',
        dataset=random_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=base_test_env["test_dir"],
        n_epoch=1,
        batch_size=2,
        checkpoint_interval=3,
        accumulation_steps=1,
        max_grad_norm=1.0,
        random_seed=42
    )
    

    
    # Create custom callbacks to track calls
    callback_calls = []
    
    class TrackingCallback(TrainCallback):
        def on_train_begin(self, context):
            callback_calls.append('on_train_begin')
        
        def on_batch_end(self, context):
            callback_calls.append('on_batch_end')
        
        def on_epoch_end(self, context):
            callback_calls.append('on_epoch_end')
    

    
    trainer = Trainer(
        train_config, 
        callbacks=[TrackingCallback()]
    )
    
    trainer.train()
    
    # Verify callbacks were called
    assert 'on_train_begin' in callback_calls
    assert 'on_batch_end' in callback_calls
    assert 'on_epoch_end' in callback_calls