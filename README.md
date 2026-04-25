# turboquant-mlx

Local MLX research repo for **TurboQuant-style KV-cache compression** on **Llama-family models**.

This project implements a repo-local Llama adapter, a custom packed TurboQuant cache, compressed-domain attention, offline outlier-mask calibration, and a small evaluation harness for long-context experiments on Apple Silicon.

## Read This First

- [docs/study-notes.md](docs/study-notes.md)
  My working notes on KV cache compression, TurboQuant, PolarQuant, and the motivation for the repo.
- [docs/implementation-plan.md](docs/implementation-plan.md)
  The implementation plan and design constraints that shaped the codebase.

External context:

- [Google Research blog: TurboQuant - Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [TurboQuant paper](https://openreview.net/forum?id=tO3ASKZlok)


## Scope

- Focus: **KV-cache quantization for inference**
- Backend: **MLX / mlx-lm**
- Model family: **Llama-compatible models**
- Design goal: stay close to the TurboQuant paper while keeping the code modular and inspectable
- The code is written to be easy to inspect and modify first, not to hide the method behind a highly fused abstraction, mainly used into validating the method and understanding its behavior on Apple Silicon.
- The evaluation harness is intended for local validation and engineering iteration, not as an official paper reproduction claim. It's a sandbox for me to understand the method and its tradeoffs, not a polished artifact for public consumption.


This repo does **not** aim to be a generic vector-search implementation or a drop-in patch to your `.venv`.

## Modes

- `standard`
  Regular `mlx-lm` KV cache.
- `core`
  Data-oblivious integer-bit TurboQuant.
- `preset`
  Mixed-bit TurboQuant presets with a fixed offline-calibrated outlier mask.

The current implementation also supports an optional QJL-style residual correction on the key-score path.

`turboquant-report` writes to `./results/` and reads LongBench inputs from
`./assets/longbench/data/` by default. Override with the CLI flags
`--output-root` / `--longbench-source-dir`, or with the environment variables
`TURBOQUANT_OUTPUT_ROOT` / `TURBOQUANT_LONGBENCH_DIR`.

## Before You Run a Benchmark

- Wipe `./results/` (or any earlier `--output-root`) before believing any
  `preset_*_qjl` numbers produced before the QJL projection-basis fix landed.
  The QJL correction was being applied in the wrong basis, which inflated
  attention error rather than reducing it; every prior `preset_*_qjl` row is
  invalidated.
- Re-run the calibration step (or rely on the new on-disk cache, see below).

## Knobs Worth Knowing

- `--seed N` — master seed threaded through Python `random`, NumPy, and MLX
  via `turboquant_mlx._seed.set_global_seed`. Decoding is greedy, so the
  visible effect is mostly on the needle prompt construction.
- `--quiet` — silences per-(mode, tier, dataset) progress prints. Default is
  verbose so a long run is obviously alive.
- `--resume <result_dir>` — continue a previous run that was killed
  mid-execution. The streamed `raw/*.jsonl` files act as a resume log; any
  `(mode, tier, dataset, index)` (or per-position-seed for needle) already
  present is skipped.
- Calibration is now cached on disk under
  `<output_root>/.calibration_cache/<model>/`. The cache key covers
  `(model, head_dim, outlier_count, quantile, calibration_text_hashes)`, so
  any change forces a recompute; reusing the same model + texts is
  instantaneous.

## What's in `summary.json`

Every report stamps a top-level `provenance` block with `git_sha`, the
parsed CLI args, and the active `mlx` / `mlx-lm` / `numpy` / Python versions,
so two timestamped runs are never ambiguous about which code revision
produced them.

The acceptance gate compares the 95% lower bound of each metric delta
(headline minus standard) against a documented threshold; thresholds are
named module-level constants in `turboquant_mlx/reporting.py` and printed
in `annotations/SUMMARY.md`.

![image.png](docs/image.png)

