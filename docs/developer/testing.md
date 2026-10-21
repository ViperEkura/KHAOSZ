# Test suites

Tests are organized by resources and boundaries:

- unit/: pure logic and isolated fakes; no CUDA required.
- integration/: CPU interactions among components, files, HTTP, threads, or processes.
- gpu/: CUDA tensors, native kernels, CUDA Graphs, or NCCL. Run manually on suitable hardware.
- support/: reusable builders and capability marks. conftest.py files contain fixtures only.

Run the full suite (GPU cases skip when requirements are unavailable):

    python -m pytest tests/ -v

Run CPU suites only when iterating locally:

    python -m pytest tests/unit tests/integration -q

Run GPU tests on a machine with the required devices and extensions:

    python -m pytest tests/gpu -q

GPU tests skip when CUDA, the required extension, or enough visible devices are unavailable. A successful command with skips does not mean the skipped paths were exercised. The move into tier directories changes pytest node ID paths; test function names remain the same where possible.

## Skip rules

Marks live in `tests/support/capabilities.py`:

| Mark | Skip when |
|------|-----------|
| `skip_no_cuda` | `torch.cuda.is_available()` is False |
| `skip_lt2_cuda` | fewer than two visible CUDA devices |
| `skip_no_kernel` | any built extension is missing (`astrai.extension.KERNEL_NAMES`) |
| `skip_no_fp8` | no FP8-capable extension or `device_capability < (8, 9)` |
| `slow` / `integration` / `unit` | marker-only; no runtime skip |

A green run on a tier does **not** prove the skipped paths were exercised.
Check the skip summary before assuming coverage.

## Scale

- ~1181 `def test_*` across `tests/unit`, `tests/integration`, `tests/gpu`.
- `tests/support/` holds reusable builders and capability helpers; it does
  not contain test modules of its own.
- `conftest.py` files register fixtures only; assertions live in the test
  modules.

## CI entry points

- `scripts/pre_commit.sh` runs format, import sort, then tests.
- `scripts/pre_commit.sh --skip-deps` skips dependency installation.
- CI: `.github/workflows/lint.yml` then `.github/workflows/tests.yml`.
