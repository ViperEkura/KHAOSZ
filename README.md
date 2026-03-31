<div align="center">
  <!-- <img src="assets/images/project_logo.png" width="auto" alt="Logo"> -->
  
  <h1>AstrAI</h1>
  <p>
    <strong>A lightweight Transformer training & inference framework</strong>
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
  <a href="#english">English</a> •
  <a href="assets/docs/README-zh-CN.md">中文</a> •
  <a href="https://github.com/ViperEkura/AstrAI/issues">Issue Tracker</a> •
  <a href="https://github.com/ViperEkura/AstrAI/discussions">Discussions</a> •
  <a href="https://huggingface.co/ViperEk/AstrAI">HuggingFace</a>
</div>

<br>

## 📖 Table of Contents

<details open>
<summary><b>English</b></summary>

- [Features](#features)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Community](#community)
- [License](#license)

</details>

---

<a id="english"></a>
## English

### Features

- 🚀 **High Performance**: Optimized for both training and inference with efficient parallelization.
- 🔧 **Flexible**: Support for seq/sft/dpo training, customizable model architectures.
- 💡 **Easy to Use**: Simple API with comprehensive examples and demos.
- 📦 **Lightweight**: Minimal dependencies, easy to deploy.
- 🔬 **Research‑Friendly**: Modular design, easy to experiment with new ideas.
- 🤗 **HuggingFace Integration**: Compatible with HuggingFace models and datasets.

### Quick Start

#### Installation

```bash
git clone https://github.com/ViperEkura/AstrAI.git
cd AstrAI
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

#### Train a Model

```bash
python scripts/tools/train.py \
  --train_type=seq \
  --data_root_path=/path/to/dataset \
  --param_path=/path/to/param_path
```

#### Generate Text

```bash
python scripts/tools/generate.py --param_path=/path/to/param_path
```

#### Demo

Check out the demos in the `scripts/demo/` folder:

```bash
# Download pre‑processed data (required before running demos)
python scripts/demo/download.py

# Interactive streaming chat
python scripts/demo/stream_chat.py

# Batch generation
python scripts/demo/generate_batch.py

# Auto‑regressive generation
python scripts/demo/generate_ar.py
```

Watch a video walkthrough on [bilibili](https://www.bilibili.com/video/BV1z5RPYHEkd).

### Documentation

| Document | Description |
|----------|-------------|
| [Parameter Guide](./assets/docs/params.md) | Training & inference parameters |
| [Design Document](./assets/docs/design.md) | Framework architecture & module design |
| [Data Flow](./assets/docs/dataflow.md) | Data processing pipeline details |
| [Model Introduction](./assets/docs/introduction.md) | Model architecture & technical details |

### Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a Pull Request.

For major changes, please open an issue first to discuss what you would like to change.

### Community

- **GitHub Issues**: [Issue Tracker](https://github.com/ViperEkura/AstrAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ViperEkura/AstrAI/discussions)
- **HuggingFace**: [Model Hub](https://huggingface.co/ViperEk/AstrAI)

### License

This project is licensed under the [GPL-3.0 License](LICENSE).

---

<div align="center">
  <em>A lightweight Transformer framework designed for both high performance and ease of use.</em>
</div>