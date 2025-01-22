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

其他：
使用wiki_zh 作为训练数据进行预训练， 但是发现loss 停滞在5左右
最终发现是位置编码精度问题， 在代码中使用成torch.long 的精度


### 训练时间：25/1/21

参数设置：
同上

训练脚本设置：

```bash
python KHAOSZ/train.py \
--data_root_path=/root/autodl-fs/pre_train \
--n_epoch=3 \
--batch_size=16 \
--max_lr=6e-4 \
--n_iter_ckpt=10000 \
--ckpt_dir checkpoints 

python KHAOSZ/train.py \
--data_root_path=/root/autodl-fs/kol \
--n_epoch=4 \
--batch_size=16 \
--max_lr=6e-4 \
--n_iter_ckpt=10000 \
--ckpt_dir checkpoints \
--resume_train=true \
--resume_dir=checkpoints/epoch_03_iter_40000

python KHAOSZ/train.py \
--data_root_path=/root/autodl-fs/belle_0.5M \
--n_epoch=4 \
--batch_size=16 \
--max_lr=6e-4 \
--n_iter_ckpt=10000 \
--ckpt_dir checkpoints \
--resume_train=true \
--resume_dir=checkpoints/epoch_03_iter_40000
```

训练日志：
```
2025-01-20 23:01:40,397 -- initializing trainer ...
2025-01-20 23:01:40,397 -- start training ...
2025-01-20 23:56:39,860 -- Epoch 1/3 Loss: 5.0279171875
2025-01-20 23:56:45,289 -- Saved checkpoint to checkpoints/epoch_01_iter_10000
2025-01-21 00:51:47,797 -- Epoch 2/3 Loss: 4.023115625
2025-01-21 00:51:53,505 -- Saved checkpoint to checkpoints/epoch_02_iter_20000
2025-01-21 01:46:57,027 -- Epoch 2/3 Loss: 3.770625
2025-01-21 01:47:02,859 -- Saved checkpoint to checkpoints/epoch_02_iter_30000
2025-01-21 02:42:04,113 -- Epoch 3/3 Loss: 3.58589375
2025-01-21 02:42:09,868 -- Saved checkpoint to checkpoints/epoch_03_iter_40000

2025-01-21 12:15:04,580 -- initializing trainer ...
2025-01-21 12:15:04,580 -- start training ...
2025-01-21 13:09:50,554 -- Epoch 1/3 Loss: 5.110409375
2025-01-21 13:09:56,266 -- Saved checkpoint to checkpoints/epoch_01_iter_10000
2025-01-21 14:04:45,571 -- Epoch 2/3 Loss: 4.6440875
2025-01-21 14:04:51,345 -- Saved checkpoint to checkpoints/epoch_02_iter_20000
2025-01-21 14:59:40,389 -- Epoch 2/3 Loss: 4.448953125
2025-01-21 14:59:46,012 -- Saved checkpoint to checkpoints/epoch_02_iter_30000
2025-01-21 15:54:34,669 -- Epoch 3/3 Loss: 4.280875
2025-01-21 15:54:40,486 -- Saved checkpoint to checkpoints/epoch_03_iter_40000


2025-01-21 21:33:29,629 -- initializing trainer ...
2025-01-21 21:33:29,629 -- start training ...
2025-01-21 22:28:25,492 -- Epoch 3/3 Loss: 2.33816796875
2025-01-21 22:28:31,664 -- Saved checkpoint to checkpoints/epoch_03_iter_10000
```

其他：
从前天的模型继续训练， 也是从前天的检查点开始继续处理的

