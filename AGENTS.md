# Repository Guidelines

## Project Structure & Module Organization
- `python/needle/` contains the core Needle library: `autograd.py`, `ops/`, `nn/`, `data/`, `init/`, and `optim.py`.
- `apps/` holds runnable training examples such as `mlp_resnet.py` and `simple_ml.py`.
- `src/` contains the HW0 reference code and the C++ extension source `simple_ml_ext.cpp`.
- `tests/` and `tests/hw2/` contain the `pytest` suites for HW0-HW2.
- `data/` stores MNIST gzip files; `figures/` stores report images; notebooks and homework notes live at the repo root.

## Build, Test, and Development Commands
- `make` — builds `src/simple_ml_ext.so` with `pybind11`.
- `pytest -q tests/test_simple_ml.py` — checks the HW0 NumPy and C++ helpers.
- `pytest -q tests/test_autograd_hw.py tests/hw2` — runs autograd, data, NN, and optimizer coverage.
- `cd apps && python3 mlp_resnet.py` — runs the MNIST MLP-ResNet example.

Run tests from the repository root: several test files append `./python`, `./src`, and `./apps` to `sys.path`. Local test collection also expects packages such as `pytest`, `numpy`, `pybind11`, `numdifftools`, and the course `mugrade` module.

## Coding Style & Naming Conventions
Use 4-space indentation and follow the existing Python style: `snake_case` for functions, variables, and files; `CapWords` for classes. Keep modules focused and match the current pattern of small tensor ops and layers. Preserve scaffold markers like `### BEGIN YOUR SOLUTION` / `### END YOUR SOLUTION` when editing homework code.

## Testing Guidelines
Add tests under `tests/` or `tests/hw2/` with names matching `test_*.py`. Prefer deterministic tests: seed NumPy explicitly and check both shapes and numeric values. For autograd or optimizer work, include forward-pass and backward-pass assertions.

## Commit & Pull Request Guidelines
Recent history mixes milestone messages (`my work on hw2 is done`) with conventional prefixes (`feat: ...`). Prefer short, imperative subjects, optionally scoped by homework, e.g. `hw2: implement BatchNorm1d` or `feat: add logsumexp backward`. PRs should summarize the task, list touched paths, and include the exact test command(s) run. Attach screenshots only when updating notebooks or `figures/`.

## Configuration & Artifact Tips
Do not commit generated artifacts or local clutter: `.gitignore` already excludes `*.so`, `__pycache__/`, virtual environments, and notebook checkpoints. Keep dataset paths relative to the repo (`data/...`) rather than hard-coding machine-specific paths.
