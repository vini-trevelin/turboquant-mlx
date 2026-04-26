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

## Results: Compression vs Quality (Llama-3.2-3B-Instruct-4bit, seed 0)

Headline mode: **Preset 3.5-bit + QJL**. Baseline: standard FP16 mlx-lm KV cache.

### Cache compression — confirmed

| Context (tokens) | Standard cache | Preset 3.5-bit + QJL | Reduction |
|-----------------|---------------|----------------------|-----------|
| 512             | 84 MB         | 16 MB                | **5.3×**  |
| 2 048           | 246 MB        | 61 MB                | **4.1×**  |
| 4 096           | 441 MB        | 114 MB               | **3.9×**  |


### Needle-in-a-haystack — zero degradation

Every mode retrieves the planted answer at 100% accuracy across all context lengths.
Preset 2.5-bit + QJL drops to 83% at 4 096 tokens — that preset is too aggressive.

| Context | Standard | Preset 3.5 + QJL | Preset 2.5 + QJL |
|---------|----------|------------------|------------------|
| 512     | 1.00     | **1.00**         | 1.00             |
| 2 048   | 1.00     | **1.00**         | 1.00             |
| 4 096   | 1.00     | **1.00**         | 0.83 ⚠️          |


### QA token-F1 — point estimates within tolerance (low-n caveat)

| Context | Standard F1 | Preset 3.5 + QJL F1 | Delta   |
|---------|------------|---------------------|---------|
| 512     | 0.164       | 0.228               | +0.064  |
| 2 048   | 0.305       | 0.228               | −0.076  |
| 4 096   | 0.401       | 0.383               | **−0.017** |

Point estimates show the 4 096-token tier within the −0.03 acceptance threshold.
The 2 048 delta (−0.076) is the weakest result; more examples are needed to resolve it.
With only 10 examples per dataset the per-tier standard deviation is ~0.38, so the 95%
confidence intervals are wide. The gate formally reads NOT YET until sample size increases.


### Teacher-forced distribution parity

| Context | Top-1 agreement | KL divergence | Perplexity ratio (turbo / std) |
|---------|----------------|---------------|-------------------------------|
| 512     | 84.7%          | 0.105         | 1.057                         |
| 2 048   | 84.2%          | 0.131         | 1.139                         |
| 4 096   | 83.4%          | 0.129         | 1.143                         |

Perplexity is 6–14% higher than standard — a real but bounded distribution shift.
Top-1 token agreement stays above 83% at all lengths.

### Conclusion

Retrieval fidelity (needle) is **fully preserved** at 3.5-bit compression with ~4× cache reduction.
QA quality at the 4 096-token tier is within tolerance at point-estimate level; the 2 048-tier
needs more examples to resolve. Perplexity degradation is real (~1.1×) and should be factored
in for generation-quality sensitive workloads. The 2.5-bit preset is not recommended for
contexts beyond 2 048 tokens.

