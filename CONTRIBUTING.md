# Contributing to AstrAI

Thank you for your interest in contributing to AstrAI! This document provides guidelines and steps for contributing.

## How to Contribute

### Reporting Issues
If you encounter a bug or have a feature request, please open an issue on GitHub. Include as much detail as possible:
- A clear description of the problem or request.
- Steps to reproduce (for bugs).
- Your environment (Python version, OS, etc.).

### Submitting Changes
1. **Fork** the repository.
2. **Clone** your fork:
   ```bash
   git clone https://github.com/your-username/AstrAI.git
   cd AstrAI
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes**. Follow the code style guidelines below.
5. **Commit your changes** with a descriptive commit message:
   ```bash
   git commit -m "Add: brief description of the change"
   ```
6. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request** (PR) against the `main` branch of the upstream repository.

## Code Style

AstrAI uses [Ruff](https://docs.astral.sh/ruff/) for code formatting and linting. Please ensure your code is formatted before submitting.

- Run Ruff to format and lint:
  ```bash
  ruff format .
  ruff check --fix .
  ```
- The project uses **double quotes** for strings and **4‑space indentation** (as configured in `pyproject.toml`).

## Testing

If you add or modify functionality, please include appropriate tests.

- Run the test suite with:
  ```bash
  pytest
  ```
- Ensure all tests pass before submitting your PR.

## Code Review

All submissions will be reviewed. We may request changes or discuss alternatives. Please be responsive to feedback.

## License

By contributing, you agree that your contributions will be licensed under the same [GPL-3.0 License](LICENSE) that covers the project.

---

If you have any questions, feel free to ask in the [GitHub Discussions](https://github.com/ViperEkura/AstrAI/discussions) or open an issue.

Happy contributing!