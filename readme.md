## KHAOSZ

这是一个支持中文和英文双语言的Transfomer模型，包含模型设置和训练流程， 通过加载`params/config.json` 中的设定的参数完成训练， 使用`argparse`库解析命令行参数，包括数据集根目录、训练轮数、批处理大小、保存检查点的间隔轮数以及检查点保存目录。

- **设备选择**：当前代码默认使用CUDA进行训练
- **性能优化**：代码中设置了`dtype=torch.bfloat16`来启用混合精度训练，这有助于提高训练速度和降低显存消耗，但需确保硬件支持此特性。
- **多语言支持**：该模型支持中文和英文双语，通过先前训练好的tokenzier 完成分词

1. 如何训练

要训练这个支持中文和英文双语言的Transformer模型，您可以按照以下步骤进行操作：

(1). 准备数据集：

确保您的数据集位于一个指定的根目录下。数据集应包含用于训练的文本文件，这些文件可以是中文、英文或两者混合。
数据文件的格式应与模型的输入要求一致，最好是经过tokenizer处理过后的token_id

(2).安装依赖：

确保您已经安装了所有必要的Python库。根据代码中的导入语句，您需要安装以下库：

```bash
pip install -r requirements.txt1
```

(3).运行训练脚本：

使用以下命令运行训练脚本，并根据需要调整参数：

```bash
python train.py \
--data_root_path=/path/to/dataset \
--n_epoch=5 \
--batch_size=8 \
--n_epoch_ckpt=1 \
--ckpt_dir checkpoints 
```

--data_root_path：指定数据集的根目录路径。

--n_epoch：指定训练的总轮数。

--batch_size：指定每个批次的样本数量。

--n_epoch_ckpt：指定每多少轮保存一次检查点。

--ckpt_dir：指定保存检查点的目录。

--resume_train: 是否从检查点恢复训练

--resume_ckpt_path: 恢复训练的checkpoint路径

训练过程中，您可以在终端中查看训练日志，了解训练进度、损失值等信息。
检查点文件会保存在指定的检查点目录中，您可以使用这些检查点文件来恢复训练或进行评估。

