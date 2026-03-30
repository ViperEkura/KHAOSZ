<div align="center">
  <img src="assets/images/project_logo.png" width="auto" alt="Logo">
  
  <h1>KHAOSZ</h1>
  
  <div>
    <a href="#english">English</a> • 
    <a href="#chinese">中文</a>
  </div>
  
  <p>
    <strong>A lightweight Transformer training & inference framework</strong>
  </p>
</div>

## 📖 Table of Contents | 目录

<details open>
<summary><b>English</b></summary>

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)

</details>

<details>
<summary><b>中文</b></summary>

- [安装](#安装)
- [快速开始](#快速开始)
- [文档](#文档)

</details>

---

<a id="english"></a>
## English

### Features

- 🚀 **High Performance**: Optimized for both training and inference
- 🔧 **Flexible**: Support for seq/sft/dpo training
- 💡 **Easy to Use**: Simple API with comprehensive examples
- 📦 **Lightweight**: Minimal dependencies

### Installation

```bash
git clone https://github.com/username/khaosz.git
cd khaosz
pip install -e .
```

### Quick Start

```bash
# Train
python tools/train.py \
  --train_type=seq \
  --data_root_path=/path/to/dataset \
  --param_path=/path/to/param_path

# Generate
python tools/generate.py --param_path=/path/to/param_path
```

### Demo

```bash
# run download before using
python demo/download.py

# run demo
python demo/stream_chat.py
python demo/generate_batch.py
python demo/generate_ar.py
```

- [bilibili](https://www.bilibili.com/video/BV1z5RPYHEkd)

---

<a id="chinese"></a>
## 中文

### 特性

- 🚀 **高性能**: 训练与推理双向优化
- 🔧 **灵活**: 支持 seq/sft/dpo 多种训练方式
- 💡 **易用**: 简洁的 API 与丰富的示例
- 📦 **轻量**: 依赖少，部署简单

### 安装

```bash
git clone https://github.com/username/khaosz.git
cd khaosz
pip install -e .
```

### 快速开始

```bash
# 训练
python tools/train.py \
  --train_type=seq \
  --data_root_path=/path/to/dataset \
  --param_path=/path/to/param_path

# 生成
python tools/generate.py --param_path=/path/to/param_path
```

### 演示

```bash
# 使用前先下载模型
python demo/download.py

# 运行示例
python demo/stream_chat.py
python demo/generate_batch.py
python demo/generate_ar.py
```

- [bilibili](https://www.bilibili.com/video/BV1z5RPYHEkd)


---

<a id="documentation"></a>

### Document | 文档

| Document | Description |
|----------|-------------|
| *Parameter Guide* <br> [参数说明](./assets/docs/params.md) | *Training & inference parameters* <br> 训练与推理参数配置 |
| *Design Document* <br> [设计文档](./assets/docs/design.md) | *Framework architecture & module design* <br> 系统架构与模块设计 |
| *Data Flow* <br> [数据流程](./assets/docs/dataflow.md) | *Data processing pipeline details* <br> 数据处理管道详解 |
| *Model Introduction* <br> [模型介绍](./assets/docs/introduction.md) | *Model architecture & technical details* <br> 模型架构与技术细节 |

### Download | 下载

- [HuggingFace](https://huggingface.co/ViperEk/KHAOSZ)
- `python demo/download.py`

### Lincence | 许可证

- [GPL-3.0](LICENSE)