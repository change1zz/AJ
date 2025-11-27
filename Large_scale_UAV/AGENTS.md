# Repository Guidelines

## Project Structure & Module Organization
- `MARL_QMIX/`: core multi-agent training and simulation code (`env.py`, `runner.py`, `common/arguments.py`).
- `MARL_QMIX/scripts/`: utilities for training loops, topology checks, monitoring, and report exports.
- `MARL_QMIX/model`, `network`, `policy`, `controllers`: model weights, neural modules, mixers, and HRL controller logic.
- `datasets/`: sample mobility traces and communication topologies; keep new generated datasets here.
- `3.0/result/` and `MARL_QMIX/runs/`: experiment outputs and TensorBoard logs; avoid manual edits.

## Build, Test, and Development Commands
- `python main.py` - default training plus periodic evaluation using defaults in `common/arguments.py`.
- `python train_100k.py` - fixed-step training (example: 100k steps) with one evaluation pass.
- `python scripts/short_train.py --n_steps 50000` - quick sanity run to validate code changes.
- `python scripts/memory_smoke_test.py` and `python scripts/test_uniform_topology.py` - spot-check memory usage and topology integrity.
- `python scripts/monitor_training.py` - tail training metrics during long runs; pair with `tmp_train_log.txt` for debugging.

## Coding Style & Naming Conventions
- Python 3 with 4-space indents; keep lines readable (<100 chars) and favor explicit arguments over globals.
- Modules/functions/variables use `snake_case`; classes use `CamelCase`; constants use `UPPER_SNAKE`.
- Centralize configuration in `common/arguments.py`; add defaults and brief comments instead of hard-coding values in scripts.
- Reuse helpers in `common/utils.py` and `common/replay_buffer.py` before introducing new utilities.

## Testing Guidelines
- Before opening a PR, run at least one fast training script (`short_train.py` or `train_100k.py`) and a topology check (`test_uniform_topology.py`).
- For stability-sensitive changes (masking, sampling, loss), run `python scripts/test_nan_fix.py` to watch for NaNs.
- Capture key metrics (reward/BLER/transmit rate) from logs; note the dataset or topology path used (e.g., `datasets/topologies/16_nodes/...`).
- Use deterministic seeds from `common/arguments.py` when comparing experiments to keep regressions reproducible.

## Commit & Pull Request Guidelines
- Commit messages: imperative mood with scope, e.g., `fix: guard env reset when nodes drop`.
- PRs should describe what changed, why it is needed, commands run, and datasets/topologies touched; attach log snippets or screenshots when behavior changes.
- Avoid committing derived artifacts (`runs/`, `tmp_train_log.txt`, `*.pth`, large CSVs); keep them local or ignored.
- Update relevant docs (`MARL_QMIX/README.md`, `DEVLOG.md`) when altering default parameters or workflows.

## Security & Configuration Tips
- Do not check in proprietary datasets; store local copies under `datasets/` and reference paths in notes instead.
- When adding new config knobs, set conservative defaults, document them in `common/arguments.py`, and keep CLI flags consistent with existing naming.
- Prefer `scripts/start_train_with_tb.py` for long experiments to centralize logging instead of ad-hoc print statements.
