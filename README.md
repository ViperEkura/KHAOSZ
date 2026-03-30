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

<div align="center">

| English | 中文 |
|---------|------|
| [Installation](#installation) | [安装](#安装) |
| [Quick Start](#quick-start) | [快速开始](#快速开始) |
| [Documentation](#documentation) | [文档](#文档) |
| [License](#license) | [许可证](#许可证) |

</div>

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

<a id="license"></a>
### License

GPL-3.0

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

### 许可证

GPL-3.0

---

<a id="documentation"></a>
## 📚 Documentation | 文档

| Document | 说明 |
|----------|------|
| [参数说明](assets/docs/params.md) | Training & inference parameters |
| [设计文档](assets/docs/design.md) | Framework design |
| [数据流程](assets/docs/dataflow.md) | Data processing pipeline |
| [模型介绍](assets/docs/introduction.md) | Model architecture |

### Download | 下载

- [HuggingFace](https://huggingface.co/ViperEk/KHAOSZ)
- `python demo/download.py`