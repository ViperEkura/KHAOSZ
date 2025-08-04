![image-20250306182014120](/resources/images/project_logo_clipped.png)

<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; font-size: 16px; font-weight: bold; margin-top: 50px;">
  
  <div>
    <a href="#english" style="text-decoration: none; margin: 0 10px; color: blue;">English</a> | 
    <a href="#chinese" style="text-decoration: none; margin: 0 10px; color: blue;">中文</a>
  </div>

  <h1 style="margin: 20px 0 0 0; font-size: 2.5em; font-weight: bold;">KHAOSZ </h1>
</div>

<h2 id="english">English Version</h2>

This is a Chinese-English bilingual Transformer model supporting both languages. It contains model configurations and training workflows, completing training by loading parameters defined in `params/config.json`. The training script `train.py` parses command-line arguments, including dataset root directory, number of training epochs, batch size, checkpoint interval, and checkpoint directory.

**Model Download Options (Choose One):**

1. Visit [HuggingFace](https://huggingface.co/ViperEk/KHAOSZ) to access **Files and versions**
2. Run `params/download.py` to download parameters

**Demo Video:** [bilibili](https://www.bilibili.com/video/BV1z5RPYHEkd)

Training dataset sources are listed in the **Model Card** section of the HuggingFace download link.

**License:** Code follows Apache-2.0 protocol. Please credit the source code when used.

- **📊 Device Selection:** Code defaults to CUDA training
- **🌐 Performance Optimization:** `dtype=torch.bfloat16` is enabled to accelerate training and reduce memory usage. Ensure hardware supports this feature.
- **🤖 Language Support:** Model supports Chinese and English training. The BBPE tokenizer was trained without multilingual text, so OOV (out-of-vocabulary) issues are minimized for these languages but may exist for others.

### 📌 Training Guide

To train this Transformer model, follow these steps:

**(1). Prepare Dataset:**

Place datasets in the designated root directory. Files should be text documents in Chinese, English, or mixed. Format should align with model input requirements - preferably pre-tokenized token_ids stored as `torch.Tensor` (using `torch.Tensor` saves memory compared to Python lists, which default to 64-bit precision).

**(2). Install Dependencies:**

```bash
pip install -r requirements.txt
pip install .
```

**(3). Run Training Script:**

```bash
python train.py \
--train_type=train_type[seq, sft, dpo] \
--data_root_path=/path/to/dataset \
--n_epoch=5 \
--batch_size=8 \
--max_lr=2e-4 \
--n_iter_ckpt=10000 \
--ckpt_dir checkpoints 
```

**Parameters Explanation:**
- `--train_type`: Training type (seq, sft, dpo)
- `--data_root_path`: Dataset root directory
- `--n_epoch`: Total training epochs
- `--batch_size`: Batch size
- `--n_iter_step`: Number of batches per training step
- `--warning_step`: Warmup steps
- `--max_lr`: Maximum learning rate (uses warmup + cosine decay)
- `--n_iter_ckpt`: Checkpoint saving interval
- `--ckpt_dir`: Checkpoint directory
- `--resume_dir`: Path to resume training from checkpoint

Training logs are saved in `train_log.txt`. Checkpoints will be stored in the specified directory for resuming training or evaluation.

### 👉 Usage Guide

**(1). Chatting with the Model:**

Open `chat.py` or use streaming/non-streaming interfaces:

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

**(2) Retrieval-Augmented Generation (RAG):**

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

### 📌 Model Specifications

This model is based on a 24-layer Transformer with parameters defined in `config.json`, totaling approximately 1.0 billion (1.0B) parameters.

**Key Design Choices:**
- Weight tying between embedding and final linear layers (standard for small models to save parameters)
- Embedding layer optimization: Without weight tying, a 10,000-word vocabulary would consume ~102M parameters (0.1B)

**Limitations:**
- May struggle with complex language phenomena due to smaller parameter size
- Prone to overfitting on specialized datasets
- Limited multilingual capabilities

**Advantages:**
- Runs efficiently on lower-spec hardware
- Shorter training time compared to larger models

**Training Pipeline:** 
The model has completed pre-training + SFT (Supervised Fine-Tuning) + DPO (Direct Preference Optimization) workflows. All corresponding training code is included in the repository.


<h2 id="chinese">中文版本</h2>
这是一个支持中英文双语的 Transformer 模型，能够处理两种语言。模型包含配置文件和训练流程，通过加载 `params/config.json` 中定义的参数完成训练。训练脚本 `train.py` 支持命令行参数解析，包括数据集根目录、训练轮数（epochs）、批量大小（batch size）、检查点保存间隔、检查点目录等。

**模型下载选项（任选其一）：**

1. 访问 [HuggingFace](https://huggingface.co/ViperEk/KHAOSZ) 查看 **Files and versions**
2. 运行 `params/download.py` 下载模型参数

**演示视频：** [bilibili](https://www.bilibili.com/video/BV1z5RPYHEkd)

训练数据来源请参见 HuggingFace 下载页面中的 **Model Card** 部分。

**许可证：** 代码遵循 Apache-2.0 协议，使用时请注明出处。

- **📊 设备选择：** 默认使用 CUDA 进行训练
- **🌐 性能优化：** 启用 `dtype=torch.bfloat16` 以加速训练并减少内存占用，请确保硬件支持该特性
- **🤖 语言支持：** 模型支持中文和英文训练。由于 BBPE 分词器未使用多语言文本训练，因此中英文的 OOV（未登录词）问题较少，其他语言可能存在 OOV 问题



### 📌 训练指南

要训练该 Transformer 模型，请按照以下步骤操作：

#### **(1). 准备数据集：**

将数据集放置在指定的根目录下。文件应为包含中文、英文或混合文本的文本文档。格式应符合模型输入要求——建议使用预分词后的 `token_ids` 并以 `torch.Tensor` 格式保存（使用 `torch.Tensor` 相比 Python 列表更节省内存，列表默认为 64 位精度）。

#### **(2). 安装依赖：**

```bash
pip install -r requirements.txt
pip install .
```

#### **(3). 运行训练脚本：**

```bash
python train.py \
--train_type=train_type[seq, sft, dpo] \
--data_root_path=/path/to/dataset \
--n_epoch=5 \
--batch_size=8 \
--max_lr=2e-4 \
--n_iter_ckpt=10000 \
--ckpt_dir checkpoints 
```

**参数说明：**
- `--train_type`: 训练类型（seq, sft, dpo）
- `--data_root_path`: 数据集根目录
- `--n_epoch`: 总训练轮数
- `--batch_size`: 批量大小
- `--n_iter_step`: 每个训练步骤的 batch 数量
- `--warning_step`: 预热步数（warmup steps）
- `--max_lr`: 最大学习率（使用预热 + 余弦衰减）
- `--n_iter_ckpt`: 检查点保存间隔
- `--ckpt_dir`: 检查点保存目录
- `--resume_dir`: 从指定路径恢复训练

训练日志将保存在 `train_log.txt` 中。检查点将保存在指定目录，用于恢复训练或评估。



### 👉 使用指南

#### **(1). 与模型对话：**

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

#### **(2). 基于检索的生成（RAG）：**

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



### 📌 模型规格说明（重复部分）

该模型基于一个 24 层的 Transformer 架构，参数配置定义在 `config.json` 中，总参数量约为 10 亿（1.0B）。

**关键设计选择：**
- 在嵌入层（embedding）与最终线性层之间进行权重绑定（weight tying），这是小型模型中常见的节省参数量的做法
- 嵌入层优化：若不进行权重绑定，一个包含 10,000 个词的词汇表将消耗约 1.02 亿（0.1B）参数

**局限性：**
- 由于参数规模较小，可能在处理复杂语言现象时表现受限
- 在特定领域的数据集上容易出现过拟合
- 多语言能力有限

**优势：**
- 可在低配置硬件上高效运行
- 相较于大型模型，训练时间更短

**训练流程：**  
该模型已完成预训练（pre-training）+ 监督微调（SFT, Supervised Fine-Tuning）+ 直接偏好优化（DPO, Direct Preference Optimization）的全流程。所有相关的训练代码均已包含在代码库中。