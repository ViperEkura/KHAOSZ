![image-20250306182014120](/assets/images/project_logo_clipped.png)

<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; font-size: 16px; font-weight: bold; margin-top: 50px;">
  
  <div>
    <a href="#english" style="text-decoration: none; margin: 0 10px; color: blue;">English</a> | 
    <a href="#chinese" style="text-decoration: none; margin: 0 10px; color: blue;">中文</a>
  </div>

  <h1 style="margin: 20px 0 0 0; font-size: 2.5em; font-weight: bold;">KHAOSZ </h1>
</div>

<h2 id="english">English Version</h2>

A training and inference framework for autoregressive Transformer language models.

**Model Download Options (choose one):**

1. Visit [HuggingFace](https://huggingface.co/ViperEk/KHAOSZ) and check **Files and versions**
2. Run `scripts/download.py` to download model parameters

**Demo Video:** [bilibili](https://www.bilibili.com/video/BV1z5RPYHEkd)

For training data sources, please refer to the **Model Card** section on the HuggingFace download page.

**License:** The code follows the GPL-3.0 license. Please provide attribution when using it.

- **📊 Device Selection:** Uses CUDA for training by default
- **🌐 Performance Optimization:** Enable `dtype=torch.bfloat16` to accelerate training and reduce memory usage. Ensure your hardware supports this feature
- **🤖 Language Support:** The model supports training in Chinese and English. Since the BBPE tokenizer hasn't been trained on multilingual text, OOV (Out-of-Vocabulary) issues are minimal for Chinese and English, but may exist for other languages


### 📌 Training Guide

To train this Transformer model, follow these steps:

**(1). Prepare the Dataset:**

Place the dataset in the specified root directory. This system uses the BBPE tokenizer for tokenization and requires training with pre-tokenized segments (stored as *.h5 format files).

**(2). Install Dependencies:**

```bash
pip install -e .
```

**(3). Run the Training Script:**

```bash
python train.py \
--train_type=train_type[seq, sft, dpo] \
--data_root_path=/path/to/dataset \
--param_path=/path/to/param_path \
--n_epoch=5 \
--batch_size=8 \
--max_lr=2e-4 \
--checkpoint_interval=10000 \
--checkpoint_dir=checkpoints 
```

**Parameter Explanation:**
- `--train_type`: Training type (seq, sft, dpo)
- `--data_root_path`: Dataset root directory
- `--param_path`: Path to model training parameters
- `--n_epoch`: Total number of training epochs
- `--batch_size`: Batch size
- `--accumulation_steps`: Number of batches per training step
- `--warmup_steps`: Warmup steps
- `--max_lr`: Maximum learning rate (using warmup + cosine decay)
- `--checkpoint_interval`: Checkpoint saving interval
- `--checkpoint_dir`: Checkpoint saving directory
- `--resume_dir`: Resume training from specified path



### 👉 Usage Guide

**(1). Chat with the Model:**

Open `chat.py` or use the streaming/non-streaming interfaces:

**Streaming Output:**
```python
import torch
from khaosz import Khaosz

model_dir = "your_model_parameter_dir"
model = Khaosz(model_dir).to(device='cuda', dtype=torch.bfloat16)
history = []

while True:
    query = input(">> ")
    if query == "!exit":
        break
    
    response_size = 0
    for response, history in model.stream_generate(
        query=query, 
        history=history,
        temperature=0.85,
        top_p=0.95,
        top_k=50
    ):
        print(response[response_size:], end="")
        response_size = len(response)       
```

**Non-streaming Output:**
```python
import torch
from khaosz import Khaosz

model_dir = "your_model_parameter_dir"
model = Khaosz(model_dir).to(device='cuda', dtype=torch.bfloat16)
history = []

while True:
    query = input(">> ")
    if query == "!exit":
        break
    
    response = model.generate(
        query=query, 
        history=history,
        temperature=0.85,
        top_p=0.95,
        top_k=50
    )
    print(response)
```

**(2). Retrieval-Augmented Generation (RAG):**

```python
import torch
from khaosz import Khaosz

model_dir = "your_model_parameter_dir"
model = Khaosz(model_dir).to(device='cuda', dtype=torch.bfloat16)

retrieved_content = model.retrieve_generate(
    query=query,
    retrieve_top_k=5,
    temperature=0.6,
    top_k=30,
    top_p=0.95
)
print(retrieved_content)
```

<h2 id="chinese">中文版本</h2>
这是一个支持基于自回归模式的 Transfomer 语言模型训练以及推理框架

**模型下载选项（任选其一）：**

1. 访问 [HuggingFace](https://huggingface.co/ViperEk/KHAOSZ) 查看 **Files and versions**
2. 运行 `scripts/download.py` 下载模型参数

**演示视频：** [bilibili](https://www.bilibili.com/video/BV1z5RPYHEkd)

训练数据来源请参见 HuggingFace 下载页面中的 **Model Card** 部分。

**许可证：** 代码遵循 GPL-3.0 协议，使用时请注明出处。

- **📊 设备选择：** 默认使用 CUDA 进行训练
- **🌐 性能优化：** 启用 `dtype=torch.bfloat16` 以加速训练并减少内存占用，请确保硬件支持该特性
- **🤖 语言支持：** 模型支持中文和英文训练。由于 BBPE 分词器未使用多语言文本训练，因此中英文的 OOV（未登录词）问题较少，其他语言可能存在 OOV 问题


### 📌 训练指南

要训练该 Transformer 模型，请按照以下步骤操作：

**(1). 准备数据集：**

将数据集放置在指定的根目录下， 本系统采用 BBPE 分词器进行分词，并且要求使用已经经过分词的 token 分段训练（分段存储为 *.h5 格式）

**(2). 安装依赖：**

```bash
pip install -e .
```

**(3). 运行训练脚本：**

```bash
python train.py \
--train_type=train_type[seq, sft, dpo] \
--data_root_path=/path/to/dataset \
--param_path=/path/to/param_path \
--n_epoch=5 \
--batch_size=8 \
--max_lr=2e-4 \
--checkpoint_interval=10000 \
--checkpoint_dir=checkpoints 
```

**参数说明：**
- `--train_type`: 训练类型（seq, sft, dpo）
- `--data_root_path`: 数据集根目录
- `--param_path`: 模型训练参数路径
- `--n_epoch`: 总训练轮数
- `--batch_size`: 批量大小
- `--accumulation_steps`: 每个训练步骤的 batch 数量
- `--warmup_steps`: 预热步数（warmup steps）
- `--max_lr`: 最大学习率（使用预热 + 余弦衰减）
- `--checkpoint_interval`: 检查点保存间隔
- `--checkpoint_dir`: 检查点保存目录
- `--resume_dir`: 从指定路径恢复训练



### 👉 使用指南

**(1). 与模型对话：**

打开 `chat.py` 或使用流式/非流式接口：

**流式输出：**
```python
import torch
from khaosz import Khaosz

model_dir = "your_model_parameter_dir"
model = Khaosz(model_dir).to(device='cuda', dtype=torch.bfloat16)
history = []

while True:
    query = input(">> ")
    if query == "!exit":
        break
    
    response_size = 0
    for response, history in model.stream_generate(
        query=query, 
        history=history,
        temperature=0.85,
        top_p=0.95,
        top_k=50
    ):
        print(response[response_size:], end="")
        response_size = len(response)       
```

**非流式输出：**
```python
import torch
from khaosz import Khaosz

model_dir = "your_model_parameter_dir"
model = Khaosz(model_dir).to(device='cuda', dtype=torch.bfloat16)
history = []

while True:
    query = input(">> ")
    if query == "!exit":
        break
    
    response = model.generate(
        query=query, 
        history=history,
        temperature=0.85,
        top_p=0.95,
        top_k=50
    )
    print(response)
```

**(2). 基于检索的生成（RAG）：**

```python
import torch
from khaosz import Khaosz

model_dir = "your_model_parameter_dir"
model = Khaosz(model_dir).to(device='cuda', dtype=torch.bfloat16)

retrieved_content = model.retrieve_generate(
    query=query,
    retrieve_top_k=5,
    temperature=0.6,
    top_k=30,
    top_p=0.95
)
print(retrieved_content)
```