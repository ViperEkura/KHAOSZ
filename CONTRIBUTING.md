# Contributing to AstrAI

Thank you for your interest in contributing! This document provides step-by-step guidelines.

## Quick Start

```bash
git clone https://github.com/ViperEkura/AstrAI.git
cd AstrAI
pip install -e ".[dev]"     # install with dev dependencies (pytest, ruff)
```

## Before You Commit

Run the following checks **in order** — CI will reject if any fail.

### 1. Format

```bash
ruff format .
```

### 2. Import sorting

```bash
ruff check . --select I
```

If this fails, **manually fix** import ordering (ruff does not auto-fix in this project's CI):

```bash
ruff check . --select I --fix .
ruff format .    # re-format after fix
```

### 3. Run tests

```bash
python -u -m pytest tests/ -v
```

See [docs/developer/testing.md](docs/developer/testing.md) for the tier
layout and skip rules.

> Failed tests may leave orphan tempdirs under the system temp directory
> (`$TMPDIR` on Linux/macOS, `%TEMP%` on Windows). Clean them manually if needed.

### 4. (Optional) Full pre-commit check script

If you have `bash` available (Git Bash on Windows works too):

```bash
bash scripts/pre_commit.sh
```

The script installs development dependencies by default, then runs the format
check, import sort check, and tests. If dependencies are already installed, use:

```bash
bash scripts/pre_commit.sh --skip-deps
```

## Commit Style

Use this layout. Keep the subject, each body bullet, the benchmark description,
and each benchmark result on a single physical line.

```text
type: short description

- first change
- second change
```

- The subject starts with one allowed type and a colon: fix, feat, chore, docs,
  refactor, perf, test, style, ci, build, or revert.
- Keep the subject near 50 characters and at most 72 characters. Do not add a
  scope in parentheses or a final period.
- Put exactly one blank line between the subject and the first body line.
- Write body items as dash bullets, one item per line. Do not wrap a bullet onto
  another line and do not leave blank lines between bullets.
- Keep each body bullet at or below 72 characters. Shorten or split longer
  items into additional bullets.
- For performance-affecting changes (perf, or refactor/feat that change
  measured performance), include a Benchmark section after the change bullets.
  Separate it from the preceding bullets with one blank line.
- Write Benchmark: and the environment/measurement method on one line. Follow
  it immediately with one dash bullet per data point; keep those bullets
  consecutive and use old -> new unit (ratio, +-%) format.

### Example: regular commit

```text
fix: keep async rollouts version-consistent

- serialize shared-model optimizer updates with generation
- reject future or over-lagged results after async scoring
- close cache publication races
- persist policy versions in online checkpoints
```

### Example: performance commit

```text
refactor: standardize packed 3d inference

- keep training attention on dense 4d tensors
- use packed 3d tensors with KV cache for inference
- extend CUDA rotary embedding to packed 3d inputs
- adapt torch, CUDA and FlashAttention backend dispatch

Benchmark: NVIDIA L20, BF16 1B, paged KV and CUDA Graph, prompt 512/gen 128
- batch 1: 234.5 -> 242.6 tok/s (1.034x, +3.4%)
- batch 8: 1243.1 -> 1286.6 tok/s (1.035x, +3.5%)
```

Both examples are real commits from this repository; inspect them with git show.

## Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `ruff check --select I` fails | Wrong import order | `ruff check . --select I --fix .` then `ruff format .` |
| `ruff format` changed many files | Not formatted before commit | Review diff carefully before staging |
| Pre-commit check script fails | Dependency install, tests, or lint failed | Fix the failing step; use `--skip-deps` only when dependencies are already installed |
| Tests fail with tempdir left | Test crash | Clean `%TEMP%` manually |

## Branching

- **Trunk-based**: `main` is the only long-lived branch and must stay releasable at all times. There is no `develop` or long-lived release branch — releases are cut by pushing a `v*` tag, which triggers `release.yml` to build wheels.
- Branch from the latest `main` and keep branches short-lived; delete them after merge.
- Name branches after the commit type: `feat/<slug>`, `fix/<slug>`, `perf/<slug>`, `docs/<slug>` — e.g. `feat/paged-cache-settings`, `fix/moe-routing-consistency`.
- Rebase onto `main` before opening a PR and again whenever conflicts appear. Force-pushes are acceptable on your own feature branches only — never on `main`.
- PRs are **squash-merged**: the PR title becomes the commit subject, so write PR titles in the exact commit style (`type: subject`). The individual branch commits are discarded.
- External contributors work from forks; every PR must pass CI (`lint`, `test (3.12)`) and receive at least one review before merge.

## Submitting Changes

1. Fork the repo.
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make changes following the steps above.
4. Commit with the commit style above.
5. Push: `git push origin feat/my-feature`
6. Open a Pull Request against `main`.

## Code Review

- All PRs are reviewed. We may request changes.
- CI runs `ruff format --check .` then `ruff check . --select I` (no `--fix` in CI).
- Ensure all tests pass.

## License

By contributing, you agree that your contributions will be licensed under the [Apache-2.0 License](LICENSE).

---

Questions? Ask in [GitHub Discussions](https://github.com/ViperEkura/AstrAI/discussions) or open an issue.
