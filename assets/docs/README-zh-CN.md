<div align="center">
  
  <img src="../images/logo.png" width="auto" alt="Logo">
  
  <div>
    <a href="../../README.md">English</a> • 
    <a href="#chinese">中文</a>
  </div>
  
  <p>
    <strong>轻量级 Transformer 训练与推理框架</strong>
  </p>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="python">
  <img src="https://img.shields.io/badge/license-GPL--3.0-blue.svg" alt="license">
  <img src="https://img.shields.io/github/v/release/ViperEkura/AstrAI?color=76bad9" alt="release">
  <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.github.com%2Frepos%2FViperEkura%2FAstrAI&query=%24.stargazers_count&label=stars&suffix=%20stars&color=76bad9" alt="stars">
  <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.github.com%2Frepos%2FViperEkura%2FAstrAI&query=%24.forks_count&label=forks&suffix=%20forks&color=76bad9" alt="forks">
</div>

<br>

<div align="center">
  <a href="../../README.md">English</a> •
  <a href="#chinese">中文</a> •
  <a href="https://github.com/ViperEkura/AstrAI/issues">问题追踪</a> •
  <a href="https://github.com/ViperEkura/AstrAI/discussions">讨论区</a> •
  <a href="https://huggingface.co/ViperEk">HuggingFace</a>
</div>
<br>

## 📖 目录

- [特性](#特性)
- [快速开始](#快速开始)
- [文档](#文档)
- [贡献](#贡献)
- [社区](#社区)
- [许可证](#许可证)

---

<a id="chinese"></a>
## 中文

### 特性

- 🚀 **高性能**: 训练与推理双向优化，高效并行。
- 🔧 **灵活**: 支持 seq/sft/dpo 多种训练方式，可定制模型架构。
- 💡 **易用**: 简洁的 API 与丰富的示例、演示。
- 📦 **轻量**: 依赖少，部署简单。
- 🔬 **研究友好**: 模块化设计，便于实验新想法。
- 🤗 **HuggingFace 集成**: 兼容 HuggingFace 模型与数据集。

### 快速开始

#### 安装

```bash
git clone https://github.com/ViperEkura/AstrAI.git
cd AstrAI
pip install -e .
```

安装开发依赖：

```bash
pip install -e ".[dev]"
```

#### 训练模型

```bash
python scripts/tools/train.py \
  --train_type=seq \
  --data_root_path=/path/to/dataset \
  --param_path=/path/to/param_path
```

#### 文本生成

```bash
python scripts/tools/generate.py --param_path=/path/to/param_path
```

#### 演示

查看 `scripts/demo/` 文件夹中的演示：

```bash
# 下载预处理数据（运行演示前必需）
python scripts/demo/download.py

# 交互式流式聊天
python scripts/demo/stream_chat.py

# 批量生成
python scripts/demo/generate_batch.py

# 自回归生成
python scripts/demo/generate_ar.py
```

观看 [bilibili](https://www.bilibili.com/video/BV1z5RPYHEkd) 上的视频演示。

### 文档

| 文档 | 说明 |
|------|------|
| [参数说明](./params.md) | 训练与推理参数配置 |
| [设计文档](./design.md) | 系统架构与模块设计 |
| [数据流程](./dataflow.md) | 数据处理管道详解 |
| [模型介绍](./introduction.md) | 模型架构与技术细节 |

### 贡献

我们欢迎贡献！请参阅[贡献指南](../../CONTRIBUTING.md)了解详情。

1. Fork 本仓库。
2. 创建功能分支。
3. 提交更改。
4. 发起 Pull Request。

重大更改请先开 issue 讨论。

### 社区

- **GitHub Issues**: [问题追踪](https://github.com/ViperEkura/AstrAI/issues)
- **Discussions**: [GitHub 讨论区](https://github.com/ViperEkura/AstrAI/discussions)
- **HuggingFace**: [模型中心](https://huggingface.co/ViperEk)

### 许可证

本项目采用 [GPL-3.0 许可证](../../LICENSE)。

---

<div align="center">
  <em>专为高性能与易用性设计的轻量级 Transformer 框架。</em>
</div>