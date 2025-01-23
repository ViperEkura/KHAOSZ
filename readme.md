## KHAOSZ

这是一个支持中文和英文双语言的Transfomer模型，包含模型设置和训练流程， 通过加载`params/config.json` 中的设定的参数完成训练， 使用`train.py`解析命令行参数，包括数据集根目录、训练轮数、批处理大小、保存检查点的间隔轮数以及检查点保存目录。

- **设备选择**：当前代码默认使用CUDA进行训练
- **性能优化**：代码中设置了`dtype=torch.bfloat16`来启用混合精度训练，这有助于提高训练速度和降低显存消耗，但需确保硬件支持此特性。
- **语言支持**：该模型目前仅仅在中文数据集上训练， 因此通过英文对话可能出现问题， 但是训练tokenzier 的时候加入了英文文段， 也可以解码英文token

### 1. 如何训练

要训练这个Transformer模型，您可以按照以下步骤进行操作：

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
--max_lr=2e-4 \
--n_iter_ckpt=10000 \
--ckpt_dir checkpoints 
```

--data_root_path：指定数据集的根目录路径。

--n_epoch：指定训练的总轮数。

--batch_size：指定每个批次的样本数量。

--max_lr: 指定过程中最大的学习率（学习率采用的是预热 + 余弦衰减）

--n_iter_ckpt：指定每多少迭代次数保存一次检查点。

--ckpt_dir：指定保存检查点的目录。

--resume_train: 是否从检查点恢复训练

--resume_dir: 恢复训练的checkpoint路径

训练过程中，您可以在终端中查看训练日志，了解训练进度、损失值等信息。
检查点文件会保存在指定的检查点目录中，您可以使用这些检查点文件来恢复训练或进行评估。


### 2. 如何使用
如果您想使用这个模型进行对话聊天, 请打开 chat.py 文件，并运行它。
或者， 您可以使用流式输出接口/对话生成接口完成对话

```python
from module import Khaosz

model = Khaosz("params")
model = model.to(device='cuda', dtype=torch.bfloat16)
histroy = []

while True:
    query = input(">> ")
    if query == "!exit":
        break
    
    response_size = 0
    for response, histroy in model.stream_generate(
        query=query, 
        history=histroy,
        temperature=1.0,
        top_p=0.5
    ):
        print(response[response_size:], end="")
        response_size = len(response)       
    print()

```

或者您可以使用非流式输出的方式完成对话

```python
from module import Khaosz

model = Khaosz("params")
model = model.to(device='cuda', dtype=torch.bfloat16)
histroy = []

while True:
    query = input(">> ")
    if query == "!exit":
        break
    
    response_size = 0
    response =  model.generate(
        query=query, 
        history=histroy,
        temperature=1.0,
        top_p=0.5
    )
    print(response)
```

### 其他问题
本模型基于12层的transformer，参数大致设置如`config.json`，参数大小为2.6亿（0.26b）

模型采用权重绑定， embedding层的权重和最后线性层的权重是共享的（比较小的模型都采用这种方式节省参数大小， 因为不采用权重绑定， embedding层假设有14000单词， 将会占用 14000 * 1024 = 143,200,000 参数 ， 也就是 0.14b 参数， 因为词表会占用太多的参数， 所以采用权重绑定是小模型的通用方法）

另外， 模型参数比较小， 生成速度快， 但是由于训练数据只使用了7gb 的中文数据集， 所以存在生成文段比较混乱的情况， 作为个聊天机器比较适合， 但是对于没有训练过的知识点，会存在胡言乱语的问题
