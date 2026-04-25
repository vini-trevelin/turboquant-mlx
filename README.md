# turboquant-mlx

Local MLX research repo for **TurboQuant-style KV-cache compression** on **Llama-family models**.

This project implements a repo-local Llama adapter, a custom packed TurboQuant cache, compressed-domain attention, offline outlier-mask calibration, and a small evaluation harness for long-context experiments on Apple Silicon.

## Scope

- Focus: **KV-cache quantization for inference**
- Backend: **MLX / mlx-lm**
- Model family: **Llama-compatible models**
- Design goal: stay close to the TurboQuant paper while keeping the code modular and inspectable

This repo does **not** aim to be a generic vector-search implementation or a drop-in patch to your `.venv`.

## What Is Here

- `turboquant_mlx/llama_adapter.py`
  Local Llama attention/cache adapter used instead of monkey-patching `mlx-lm`.
- `turboquant_mlx/quantizer.py`
  Shared TurboQuant setup, packed-code quantization, MLX-native decode, and compressed-domain attention math.
- `turboquant_mlx/cache.py`
  Packed KV-cache storage with chunked streaming updates.
- `turboquant_mlx/calibration.py`
  Offline calibration for the fixed outlier mask used by mixed-bit presets.
- `turboquant_mlx/needle.py`
  Needle-in-a-Haystack style diagnostic benchmark.
- `turboquant_mlx/longbench.py`
  Long-context QA evaluation utilities.
- `turboquant_mlx/teacher_forcing.py`
  Teacher-forced parity checks between standard KV and TurboQuant modes.
- `turboquant_mlx/reporting.py`
  End-to-end report runner that aggregates raw rows, summaries, and plots.
- `tests/`
  Unit, numeric, and smoke coverage for the core primitives and evaluation flow.

## Modes

- `standard`
  Regular `mlx-lm` KV cache.
- `core`
  Data-oblivious integer-bit TurboQuant.
- `preset`
  Mixed-bit TurboQuant presets with a fixed offline-calibrated outlier mask.

The current implementation also supports an optional QJL-style residual correction on the key-score path.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Main Commands

```bash
turboquant-calibrate
turboquant-needle
turboquant-longbench
turboquant-parity
turboquant-report
pytest -q
```

`turboquant-report` writes to `./results/` and reads LongBench inputs from
`./assets/longbench/data/` by default. Override with the CLI flags
`--output-root` / `--longbench-source-dir`, or with the environment variables
`TURBOQUANT_OUTPUT_ROOT` / `TURBOQUANT_LONGBENCH_DIR`.

If you prefer module execution, the same entrypoints are available through `python -m turboquant_mlx.<module>`.

## Read This First

- [docs/study-notes.md](docs/study-notes.md)
  Working notes on KV cache compression, TurboQuant, PolarQuant, and the motivation for the repo.
- [docs/implementation-plan.md](docs/implementation-plan.md)
  The implementation plan and design constraints that shaped the codebase.

External context:

- [Google Research blog: TurboQuant - Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [TurboQuant paper](https://openreview.net/forum?id=tO3ASKZlok)

## Project Notes

- The code is written to be easy to inspect and modify first, not to hide the method behind a highly fused abstraction.
- Packed cache storage is treated as the persistent source of truth; temporary dense tensors are only used as working state during attention.
- The evaluation harness is intended for local validation and engineering iteration, not as an official paper reproduction claim.
