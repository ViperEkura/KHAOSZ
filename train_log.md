## 训练记录

### 训练时间：25/1/20

参数设置：

```json
{
    "vocab_size": 60000,
    "n_dim": 1024,
    "n_head": 16,
    "d_ffn": 4096,
    "m_len": 1024,
    "n_layer": 12,
    "norm_eps": 1e-5,
    "flash_attn": true
}
```

训练脚本设置：

```bash
python KHAOSZ/train.py \
--data_root_path=/root/autodl-fs/pre_train \
--n_epoch=4 \
--batch_size=16 \
--max_lr=6e-4 \
--n_iter_ckpt=10000 \
--ckpt_dir checkpoints 
```

存在问题， 在训练的时候，模型收敛似乎十分困难

训练日志：
```
2025-01-20 15:24:58,150 -- initializing trainer ...
2025-01-20 15:24:58,151 -- start training ...
2025-01-20 16:20:40,815 -- Epoch 1/4 Loss: 7.642834375
2025-01-20 16:20:46,550 -- Saved checkpoint to checkpoints/epoch_01_iter_10000
2025-01-20 17:16:28,386 -- Epoch 2/4 Loss: 6.37259375
2025-01-20 17:16:34,884 -- Saved checkpoint to checkpoints/epoch_02_iter_20000
2025-01-20 18:12:16,817 -- Epoch 2/4 Loss: 6.08589375
2025-01-20 18:12:22,207 -- Saved checkpoint to checkpoints/epoch_02_iter_30000

```
