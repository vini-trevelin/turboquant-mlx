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


![image.png](docs/image.png)

