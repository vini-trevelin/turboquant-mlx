from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

import mlx.core as mx

from ._charts import bar_chart_svg as _bar_chart_svg
from ._charts import format_bytes as _format_bytes
from ._charts import line_chart_svg as _line_chart_svg
from ._charts import scatter_chart_svg as _scatter_chart_svg
from ._seed import set_global_seed
from .calibration import calibrate_outlier_mask_cached
from .config import CalibrationArtifact, CalibrationConfig, EvaluationConfig, TurboQuantConfig
from .longbench import evaluate_longbench_loaded, prepare_longbench_examples
from .load import load_turboquant
from .needle import build_needle_case, run_loaded_needle_case
from .qa_eval import read_jsonl, write_jsonl
from .generate import append_jsonl
from .teacher_forcing import evaluate_teacher_forced_loaded


DEFAULT_QUALITY_DATASETS = ["triviaqa_e", "hotpotqa_e", "2wikimqa_e"]
DEFAULT_CALIBRATION_DATASETS = ["multifieldqa_en", "multi_news"]
DEFAULT_CONTEXT_TIERS = [512, 2048, 4096]

# Acceptance thresholds for the headline (Preset 3.5 + QJL) vs Standard KV claim.
# Calibrated against the Llama-3.2-3B preset 3.5 + QJL profile; revisit if the
# headline mode or model class changes.
CACHE_REDUCTION_THRESHOLD = 3.0   # standard cache bytes / headline cache bytes
QA_F1_DELTA_THRESHOLD = -0.03     # headline F1 - standard F1, must be >= this
NEEDLE_DELTA_THRESHOLD = -0.02    # headline accuracy - standard accuracy


@dataclass(frozen=True)
class ModeSpec:
    slug: str
    label: str
    config: Optional[TurboQuantConfig]
    headline: bool = False
    diagnostic: bool = False


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _serializable_prepared_example(example: Mapping) -> dict:
    return {
        "dataset": example["_dataset_name"],
        "index": example["_index"],
        "prompt": example["_prompt"],
        "prompt_token_target": example["_prompt_token_target"],
        "prompt_token_count": example["_prompt_token_count"],
        "context_token_count": example["_context_token_count"],
        "answers": example.get("answers") or [example.get("answer", "")],
    }


def _mode_specs(head_dim: int, calibration_artifact_path: Path, *, headline_only: bool = False) -> List[ModeSpec]:
    calibration = CalibrationConfig(artifact_path=str(calibration_artifact_path))
    modes = [
        ModeSpec(slug="standard", label="Standard KV", config=None),
        ModeSpec(
            slug="preset_3p5_qjl",
            label="Preset 3.5-bit + QJL",
            config=TurboQuantConfig(
                mode="preset",
                head_dim=head_dim,
                preset_name="3.5",
                calibration=calibration,
                qjl_enabled=True,
                qjl_dim=64,
            ),
            headline=True,
        ),
    ]
    if headline_only:
        return modes
    modes.extend(
        [
            ModeSpec(
                slug="core_4bit",
                label="Core 4-bit MSE",
                config=TurboQuantConfig(mode="core", head_dim=head_dim, core_bits=4),
                diagnostic=True,
            ),
            ModeSpec(
                slug="preset_2p5_qjl",
                label="Preset 2.5-bit + QJL",
                config=TurboQuantConfig(
                    mode="preset",
                    head_dim=head_dim,
                    preset_name="2.5",
                    calibration=calibration,
                    qjl_enabled=True,
                    qjl_dim=64,
                ),
                diagnostic=True,
            ),
        ]
    )
    return modes


def _prepare_quality_slices(
    source_dir: Path,
    tokenizer,
    output_dir: Path,
    *,
    dataset_names: List[str],
    context_tiers: List[int],
    examples_per_dataset: int,
) -> dict[int, dict[str, List[dict]]]:
    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    prepared_by_tier: dict[int, dict[str, List[dict]]] = {tier: {} for tier in context_tiers}

    for tier in context_tiers:
        for dataset_name in dataset_names:
            path = source_dir / f"{dataset_name}.jsonl"
            prepared = prepare_longbench_examples(
                path,
                tokenizer,
                prompt_token_limit=tier,
                max_examples=examples_per_dataset,
                dataset_name=dataset_name,
            )
            prepared_by_tier[tier][dataset_name] = prepared
            serializable_rows = [_serializable_prepared_example(example) for example in prepared]
            write_jsonl(datasets_dir / f"{dataset_name}_ctx{tier}.jsonl", serializable_rows)
    return prepared_by_tier


def _prepare_calibration_texts(
    source_dir: Path,
    tokenizer,
    output_dir: Path,
    *,
    dataset_names: List[str],
    examples_per_dataset: int,
    prompt_token_limit: int,
) -> List[str]:
    rows: List[dict] = []
    texts: List[str] = []
    for dataset_name in dataset_names:
        prepared = prepare_longbench_examples(
            source_dir / f"{dataset_name}.jsonl",
            tokenizer,
            prompt_token_limit=prompt_token_limit,
            max_examples=examples_per_dataset,
            dataset_name=dataset_name,
        )
        for prepared_example in prepared:
            text = prepared_example["_prompt"]
            texts.append(text)
            rows.append({"dataset": dataset_name, "text": text})
    write_jsonl(output_dir / "datasets" / "calibration_prompts.jsonl", rows)
    return texts


def _summarize_quality_rows(rows: List[dict]) -> List[dict]:
    grouped: dict[tuple[str, int], List[dict]] = {}
    for row in rows:
        grouped.setdefault((row["mode_slug"], row["context_tier"]), []).append(row)
    summary = []
    for (mode_slug, context_tier), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        summary.append(
            {
                "mode_slug": mode_slug,
                "mode_label": group[0]["mode_label"],
                "context_tier": context_tier,
                "examples": len(group),
                "mean_em": _mean(row["em"] for row in group),
                "mean_f1": _mean(row["f1"] for row in group),
                "mean_headline_score": _mean(row["headline_score"] for row in group),
                "cache_nbytes_mean": _mean(row["cache_nbytes"] for row in group),
                "peak_memory_delta_bytes_mean": _mean(row["peak_memory_delta_bytes"] for row in group),
                "prefill_seconds_mean": _mean(row["prefill_seconds"] for row in group),
                "decode_seconds_mean": _mean(row["decode_seconds"] for row in group),
            }
        )
    return summary


def _summarize_needle_rows(rows: List[dict]) -> List[dict]:
    grouped: dict[tuple[str, int], List[dict]] = {}
    for row in rows:
        grouped.setdefault((row["mode_slug"], row["context_tier"]), []).append(row)
    summary = []
    for (mode_slug, context_tier), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        summary.append(
            {
                "mode_slug": mode_slug,
                "mode_label": group[0]["mode_label"],
                "context_tier": context_tier,
                "examples": len(group),
                "accuracy_mean": _mean(row["correct"] for row in group),
                "cache_nbytes_mean": _mean(row["cache_nbytes"] for row in group),
                "peak_memory_delta_bytes_mean": _mean(row["peak_memory_delta_bytes"] for row in group),
            }
        )
    return summary


def _summarize_parity_rows(rows: List[dict]) -> List[dict]:
    grouped: dict[int, List[dict]] = {}
    for row in rows:
        grouped.setdefault(row["context_tier"], []).append(row)
    summary = []
    for context_tier, group in sorted(grouped.items()):
        std_ppl = _mean(row["standard_perplexity"] for row in group if row.get("standard_perplexity"))
        turbo_ppl = _mean(row["turbo_perplexity"] for row in group if row.get("turbo_perplexity"))
        summary.append(
            {
                "context_tier": context_tier,
                "examples": len(group),
                "top1_agreement_mean": _mean(row["top1_agreement"] for row in group),
                "top5_overlap_mean": _mean(row["top5_overlap"] for row in group),
                "kl_divergence_mean": _mean(row["kl_divergence_mean"] for row in group),
                "standard_perplexity_mean": std_ppl,
                "turbo_perplexity_mean": turbo_ppl,
                "perplexity_ratio_mean": (turbo_ppl / std_ppl) if std_ppl else 0.0,
            }
        )
    return summary


def _index_summary(summary: List[dict]) -> dict[tuple[str, int], dict]:
    return {(row["mode_slug"], row["context_tier"]): row for row in summary}


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        return out.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - python < 3.8
        return "unknown"
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _provenance(args: Optional[argparse.Namespace] = None) -> dict:
    return {
        "git_sha": _git_sha(),
        "python_version": sys.version.split()[0],
        "mlx_version": _package_version("mlx"),
        "mlx_lm_version": _package_version("mlx-lm"),
        "numpy_version": _package_version("numpy"),
        "args": vars(args) if args is not None else {},
    }


def _swap_latest_symlink(latest: Path, *, target_name: str) -> None:
    """Point <latest> at <target_name>, refusing to clobber a real directory."""
    if latest.is_symlink():
        latest.unlink()
    elif latest.exists():
        raise RuntimeError(
            f"refusing to replace non-symlink {latest!r} with a symlink; remove it manually if intended"
        )
    latest.symlink_to(target_name)


def _series_from_index(
    index: dict[tuple[str, int], dict],
    *,
    mode_slug: str,
    tiers: List[int],
    field: str,
) -> List[float]:
    return [index[(mode_slug, tier)][field] for tier in tiers if (mode_slug, tier) in index]


def _build_acceptance_summary(
    quality_summary: List[dict],
    needle_summary: List[dict],
    *,
    headline_mode_slug: str = "preset_3p5_qjl",
    standard_mode_slug: str = "standard",
) -> dict:
    context_tiers = sorted({row["context_tier"] for row in quality_summary})
    quality_index = _index_summary(quality_summary)
    needle_index = _index_summary(needle_summary)
    per_tier = []
    overall_pass = True
    for tier in context_tiers:
        quality_standard = quality_index.get((standard_mode_slug, tier))
        quality_headline = quality_index.get((headline_mode_slug, tier))
        needle_standard = needle_index.get((standard_mode_slug, tier))
        needle_headline = needle_index.get((headline_mode_slug, tier))
        if not all([quality_standard, quality_headline, needle_standard, needle_headline]):
            continue

        cache_reduction = (
            quality_standard["cache_nbytes_mean"] / quality_headline["cache_nbytes_mean"]
            if quality_headline["cache_nbytes_mean"]
            else 0.0
        )
        qa_f1_delta = quality_headline["mean_f1"] - quality_standard["mean_f1"]
        needle_delta = needle_headline["accuracy_mean"] - needle_standard["accuracy_mean"]
        tier_pass = (
            cache_reduction >= CACHE_REDUCTION_THRESHOLD
            and qa_f1_delta >= QA_F1_DELTA_THRESHOLD
            and needle_delta >= NEEDLE_DELTA_THRESHOLD
        )
        per_tier.append(
            {
                "context_tier": tier,
                "cache_reduction": cache_reduction,
                "qa_f1_delta": qa_f1_delta,
                "needle_accuracy_delta": needle_delta,
                "pass": tier_pass,
            }
        )
        overall_pass = overall_pass and tier_pass
    return {
        "headline_mode_slug": headline_mode_slug,
        "overall_pass": overall_pass,
        "cache_reduction_threshold": CACHE_REDUCTION_THRESHOLD,
        "qa_f1_delta_threshold": QA_F1_DELTA_THRESHOLD,
        "needle_accuracy_delta_threshold": NEEDLE_DELTA_THRESHOLD,
        "per_tier": per_tier,
    }


def _log(message: str, *, quiet: bool) -> None:
    if not quiet:
        print(message, flush=True)


def _load_streamed_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def _quality_key(row: Mapping) -> tuple:
    return (row["mode_slug"], row["context_tier"], row.get("dataset", row.get("dataset_name")), row["index"])


def _needle_key(row: Mapping) -> tuple:
    return (row["mode_slug"], row["context_tier"], row["needle_position_label"], row["seed"])


def _parity_key(row: Mapping) -> tuple:
    return (row["context_tier"], row["dataset"], row["index"])


def _run_quality_suite_loaded(
    model,
    tokenizer,
    mode: ModeSpec,
    prepared_by_tier: dict[int, dict[str, List[dict]]],
    *,
    max_tokens: int,
    quiet: bool = False,
    output_path: Optional[Path] = None,
    skip_keys: Optional[set] = None,
) -> List[dict]:
    skip_keys = skip_keys or set()
    rows: List[dict] = []
    for tier, datasets in prepared_by_tier.items():
        for dataset_name, prepared_examples in datasets.items():
            pending = [ex for ex in prepared_examples if (mode.slug, tier, dataset_name, ex["_index"]) not in skip_keys]
            if not pending:
                _log(f"[quality][{mode.slug}] tier={tier} dataset={dataset_name} skipped (resumed)", quiet=quiet)
                continue
            _log(
                f"[quality][{mode.slug}] tier={tier} dataset={dataset_name} examples={len(pending)}/{len(prepared_examples)}",
                quiet=quiet,
            )
            result = evaluate_longbench_loaded(
                model,
                tokenizer,
                pending,
                max_tokens=max_tokens,
                turboquant_config=mode.config,
            )
            for row in result["rows"]:
                row["mode_slug"] = mode.slug
                row["mode_label"] = mode.label
                row["context_tier"] = tier
                row["dataset_name"] = dataset_name
                rows.append(row)
                if output_path is not None:
                    append_jsonl(output_path, row)
    return rows


def _run_needle_suite_loaded(
    model,
    tokenizer,
    mode: ModeSpec,
    *,
    context_tiers: List[int],
    seeds_per_position: int,
    max_tokens: int,
    quiet: bool = False,
    output_path: Optional[Path] = None,
    skip_keys: Optional[set] = None,
) -> List[dict]:
    skip_keys = skip_keys or set()
    rows: List[dict] = []
    for tier in context_tiers:
        _log(f"[needle][{mode.slug}] tier={tier} positions=3 seeds={seeds_per_position}", quiet=quiet)
        for position_label in ("front", "middle", "back"):
            for seed in range(seeds_per_position):
                key = (mode.slug, tier, position_label, seed)
                if key in skip_keys:
                    continue
                case = build_needle_case(
                    tokenizer,
                    context_tokens=tier,
                    needle="The launch code is 314159.",
                    question="What is the launch code?",
                    needle_position=position_label,
                    seed=seed,
                )
                row = run_loaded_needle_case(
                    model=model,
                    tokenizer=tokenizer,
                    case=case,
                    max_tokens=max_tokens,
                    model_label="loaded-model",
                    turboquant_config=mode.config,
                )
                row["mode_slug"] = mode.slug
                row["mode_label"] = mode.label
                row["context_tier"] = tier
                rows.append(row)
                if output_path is not None:
                    append_jsonl(output_path, row)
    return rows


def _run_parity_suite(
    standard_model,
    headline_model,
    prepared_by_tier: dict[int, dict[str, List[dict]]],
    *,
    context_tiers: List[int],
    examples_per_dataset: int,
    quiet: bool = False,
    output_path: Optional[Path] = None,
    skip_keys: Optional[set] = None,
) -> List[dict]:
    skip_keys = skip_keys or set()
    rows: List[dict] = []
    for tier in context_tiers:
        _log(f"[parity] tier={tier}", quiet=quiet)
        for dataset_name, examples in prepared_by_tier[tier].items():
            for example in examples[:examples_per_dataset]:
                key = (tier, dataset_name, example["_index"])
                if key in skip_keys:
                    continue
                metrics = evaluate_teacher_forced_loaded(
                    standard_model,
                    headline_model,
                    example["_prompt_tokens"],
                )
                row = {
                    "context_tier": tier,
                    "dataset": dataset_name,
                    "index": example["_index"],
                    **metrics.to_dict(),
                }
                rows.append(row)
                if output_path is not None:
                    append_jsonl(output_path, row)
    return rows


def _generate_plots(
    plots_dir: Path,
    *,
    quality_summary: List[dict],
    needle_summary: List[dict],
    parity_summary: List[dict],
    acceptance: dict,
) -> None:
    tiers = sorted({row["context_tier"] for row in quality_summary})
    tier_labels = [str(tier) for tier in tiers]
    mode_order = []
    for row in quality_summary:
        if row["mode_slug"] not in mode_order:
            mode_order.append(row["mode_slug"])

    quality_index = _index_summary(quality_summary)
    needle_index = _index_summary(needle_summary)
    mode_labels = {row["mode_slug"]: row["mode_label"] for row in quality_summary}

    quality_series = {}
    cache_series = {}
    needle_series = {}
    for mode_slug in mode_order:
        label = mode_labels[mode_slug]
        quality_series[label] = _series_from_index(quality_index, mode_slug=mode_slug, tiers=tiers, field="mean_f1")
        cache_series[label] = _series_from_index(quality_index, mode_slug=mode_slug, tiers=tiers, field="cache_nbytes_mean")
        needle_series[label] = _series_from_index(needle_index, mode_slug=mode_slug, tiers=tiers, field="accuracy_mean")

    tradeoff_points = []
    for row in quality_summary:
        tradeoff_points.append(
            {
                "mode_slug": row["mode_slug"],
                "x": row["cache_nbytes_mean"],
                "y": row["mean_f1"],
                "label": f"{row['mode_label']}@{row['context_tier']}",
            }
        )

    (plots_dir / "quality_vs_context.svg").write_text(
        _line_chart_svg(
            "Quality vs Context Length",
            quality_series,
            tier_labels,
            subtitle="Mean token-F1 on curated long-context QA tasks",
            y_label="Mean F1",
        )
    )
    (plots_dir / "cache_bytes_vs_context.svg").write_text(
        _line_chart_svg(
            "Observed Cache Bytes vs Context Length",
            cache_series,
            tier_labels,
            subtitle="FP16 mlx-lm KV cache vs TurboQuant packed cache (lower is better)",
            y_label="Cache size",
            formatter=_format_bytes,
            zero_baseline=True,
        )
    )
    (plots_dir / "quality_memory_tradeoff.svg").write_text(
        _scatter_chart_svg(
            "Quality vs Memory Tradeoff",
            tradeoff_points,
            subtitle="Each point is a mode at a context tier — top-left is best",
        )
    )
    (plots_dir / "needle_accuracy_vs_context.svg").write_text(
        _line_chart_svg(
            "Needle Accuracy vs Context Length",
            needle_series,
            tier_labels,
            subtitle="Structured short-answer recall under long context",
            y_label="Accuracy",
            zero_baseline=True,
        )
    )

    if parity_summary:
        (plots_dir / "parity_kl_vs_context.svg").write_text(
            _bar_chart_svg(
                "Teacher-Forced KL vs Context Length",
                tier_labels,
                [row["kl_divergence_mean"] for row in parity_summary],
                subtitle="Standard KV vs Preset 3.5 + QJL (lower is closer)",
                x_label="Context length (tokens)",
                y_label="Mean KL divergence",
                color="#0F766E",
            )
        )
        ratios = [row.get("perplexity_ratio_mean", 0.0) for row in parity_summary]
        if any(ratios):
            (plots_dir / "parity_perplexity_ratio_vs_context.svg").write_text(
                _bar_chart_svg(
                    "Teacher-Forced Perplexity Ratio vs Context Length",
                    tier_labels,
                    ratios,
                    subtitle="TurboQuant perplexity / Standard perplexity on the prompt suffix (1.0 = identical)",
                    x_label="Context length (tokens)",
                    y_label="Perplexity ratio (turbo / standard)",
                    color="#7C3AED",
                    reference_line=1.0,
                    reference_label="standard = 1.0",
                )
            )


def _write_annotations(
    annotations_dir: Path,
    *,
    model: str,
    quality_summary: List[dict],
    needle_summary: List[dict],
    parity_summary: List[dict],
    acceptance: dict,
    context_tiers: List[int],
) -> None:
    headline_mode = "preset_3p5_qjl"
    status = "PASS" if acceptance["overall_pass"] else "NOT YET"
    headline_rows = [row for row in quality_summary if row["mode_slug"] == headline_mode]
    standard_rows = [row for row in quality_summary if row["mode_slug"] == "standard"]
    smallest_cache = min(headline_rows, key=lambda row: row["cache_nbytes_mean"]) if headline_rows else None
    best_standard = max(standard_rows, key=lambda row: row["mean_f1"]) if standard_rows else None
    summary_lines = [
        "# Validation Summary",
        "",
        f"Model: `{model}`",
        "",
        f"Headline claim status: `{status}`",
        "",
        "## Claim",
        "",
        "`Preset 3.5 + QJL` should use materially less KV-cache memory than `Standard KV` while keeping quality within a small drop.",
        "",
        "## Acceptance Thresholds",
        "",
        f"- Cache reduction (FP16 baseline / TurboQuant cache) must be at least `{acceptance['cache_reduction_threshold']:.2f}x`.",
        f"- QA token-F1 delta (headline - standard) must be at least `{acceptance['qa_f1_delta_threshold']:.3f}` (i.e. drop of at most `{-acceptance['qa_f1_delta_threshold']:.3f}`).",
        f"- Needle accuracy delta (headline - standard) must be at least `{acceptance['needle_accuracy_delta_threshold']:.3f}` (i.e. drop of at most `{-acceptance['needle_accuracy_delta_threshold']:.3f}`).",
        "",
        "All three conditions must hold per tier; overall PASS requires every tier to pass.",
        "",
        "## Acceptance by Context Tier",
        "",
    ]
    for row in acceptance["per_tier"]:
        summary_lines.append(
            f"- `{row['context_tier']}` tokens: cache reduction `{row['cache_reduction']:.2f}x`, QA F1 delta `{row['qa_f1_delta']:.3f}`, Needle delta `{row['needle_accuracy_delta']:.3f}`, pass=`{row['pass']}`"
        )
    summary_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Quality uses curated exact-answer-friendly LongBench datasets with normalized EM and token-F1.",
            "- Needle uses canonical short-answer matching, including numeric recovery such as `314159` for the launch-code prompt.",
            "- Memory proof should be read primarily from `cache_nbytes` (the on-device packed cache size) compared to the FP16 mlx-lm KV cache baseline. `peak_memory_delta_bytes` is a noisier secondary signal.",
        ]
    )
    if smallest_cache is not None:
        summary_lines.append(
            f"- Smallest headline cache footprint: `{int(smallest_cache['cache_nbytes_mean']):,}` bytes at `{smallest_cache['context_tier']}` tokens."
        )
    if best_standard is not None:
        summary_lines.append(
            f"- Best baseline QA F1: `{best_standard['mean_f1']:.3f}` at `{best_standard['context_tier']}` tokens."
        )
    if parity_summary:
        worst_parity = max(parity_summary, key=lambda row: row["kl_divergence_mean"])
        summary_lines.append(
            f"- Worst teacher-forced KL in the headline comparison: `{worst_parity['kl_divergence_mean']:.4f}` at `{worst_parity['context_tier']}` tokens."
        )
        ratios = [row.get("perplexity_ratio_mean", 0.0) for row in parity_summary if row.get("perplexity_ratio_mean")]
        if ratios:
            worst_ppl = max(parity_summary, key=lambda row: row.get("perplexity_ratio_mean", 0.0))
            summary_lines.append(
                f"- Worst teacher-forced perplexity ratio (turbo / standard): "
                f"`{worst_ppl['perplexity_ratio_mean']:.3f}` at `{worst_ppl['context_tier']}` tokens "
                f"(standard PPL `{worst_ppl['standard_perplexity_mean']:.2f}`, "
                f"turbo PPL `{worst_ppl['turbo_perplexity_mean']:.2f}`)."
            )
    (annotations_dir / "SUMMARY.md").write_text("\n".join(summary_lines))

    limitations_lines = [
        "# Limitations",
        "",
        "- This report is a local 3B validation suite, not a paper-parity claim on the 8B setup.",
        "- `Preset 3.5 + QJL` is the only mode used for the headline pass/fail decision; other modes remain diagnostic.",
        "- Runtime is tracked but not gated. A pass on this report does not imply the current Python-level implementation is production-fast.",
        "- Teacher-forced parity is a support signal about distribution drift, not a substitute for downstream task quality.",
    ]
    (annotations_dir / "LIMITATIONS.md").write_text("\n".join(limitations_lines))


def run_report(
    *,
    model: str,
    longbench_source_dir: Path,
    output_root: Path,
    context_tiers: List[int] | None = None,
    quality_datasets: List[str] | None = None,
    calibration_datasets: List[str] | None = None,
    quality_examples_per_dataset: int = 10,
    parity_examples_per_dataset: int = 1,
    calibration_examples_per_dataset: int = 2,
    quality_max_tokens: int = 24,
    needle_max_tokens: int = 12,
    needle_seeds_per_position: int = 4,
    headline_only: bool = False,
    quiet: bool = False,
    seed: int = 0,
    resume_from: Optional[Path] = None,
    provenance: Optional[dict] = None,
) -> Path:
    set_global_seed(seed)
    context_tiers = context_tiers or list(DEFAULT_CONTEXT_TIERS)
    quality_datasets = quality_datasets or list(DEFAULT_QUALITY_DATASETS)
    calibration_datasets = calibration_datasets or list(DEFAULT_CALIBRATION_DATASETS)

    if resume_from is not None:
        result_dir = Path(resume_from)
        if not result_dir.exists():
            raise FileNotFoundError(f"--resume target does not exist: {result_dir}")
        timestamp = result_dir.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_dir = output_root / timestamp
    raw_dir = result_dir / "raw"
    plots_dir = result_dir / "plots"
    annotations_dir = result_dir / "annotations"
    raw_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    quality_jsonl = raw_dir / "quality_rows.jsonl"
    needle_jsonl = raw_dir / "needle_rows.jsonl"
    parity_jsonl = raw_dir / "parity_rows.jsonl"
    quality_rows = _load_streamed_rows(quality_jsonl) if resume_from else []
    needle_rows = _load_streamed_rows(needle_jsonl) if resume_from else []
    parity_rows = _load_streamed_rows(parity_jsonl) if resume_from else []
    quality_skip = {_quality_key(r) for r in quality_rows}
    needle_skip = {_needle_key(r) for r in needle_rows}
    parity_skip = {_parity_key(r) for r in parity_rows}
    if resume_from:
        _log(
            f"[resume] loaded quality={len(quality_rows)} needle={len(needle_rows)} parity={len(parity_rows)} from {result_dir}",
            quiet=quiet,
        )

    standard_model, tokenizer = load_turboquant(model, turboquant_config=None)
    prepared_by_tier = _prepare_quality_slices(
        longbench_source_dir,
        tokenizer,
        result_dir,
        dataset_names=quality_datasets,
        context_tiers=context_tiers,
        examples_per_dataset=quality_examples_per_dataset,
    )
    calibration_texts = _prepare_calibration_texts(
        longbench_source_dir,
        tokenizer,
        result_dir,
        dataset_names=calibration_datasets,
        examples_per_dataset=calibration_examples_per_dataset,
        prompt_token_limit=max(context_tiers),
    )
    calibration_cache_dir = output_root / ".calibration_cache" / model.replace("/", "_")
    artifact = calibrate_outlier_mask_cached(
        model,
        calibration_texts,
        outlier_count=32,
        quantile=99.9,
        max_examples=len(calibration_texts),
        cache_dir=calibration_cache_dir,
    )
    calibration_artifact_path = raw_dir / "calibration_artifact.json"
    artifact.save(calibration_artifact_path)

    modes = _mode_specs(artifact.head_dim, calibration_artifact_path, headline_only=headline_only)
    quality_rows.extend(
        _run_quality_suite_loaded(
            standard_model,
            tokenizer,
            modes[0],
            prepared_by_tier,
            max_tokens=quality_max_tokens,
            quiet=quiet,
            output_path=quality_jsonl,
            skip_keys=quality_skip,
        )
    )
    needle_rows.extend(
        _run_needle_suite_loaded(
            standard_model,
            tokenizer,
            modes[0],
            context_tiers=context_tiers,
            seeds_per_position=needle_seeds_per_position,
            max_tokens=needle_max_tokens,
            quiet=quiet,
            output_path=needle_jsonl,
            skip_keys=needle_skip,
        )
    )

    # Free the standard model before loading the headline model so we never hold
    # both fp16 weight sets in memory at once. Standard is reloaded later just
    # for the parity pass, which is short.
    del standard_model
    del tokenizer
    mx.clear_cache()

    headline_mode = next(mode for mode in modes if mode.headline)
    headline_model, headline_tokenizer = load_turboquant(model, turboquant_config=headline_mode.config)
    quality_rows.extend(
        _run_quality_suite_loaded(
            headline_model,
            headline_tokenizer,
            headline_mode,
            prepared_by_tier,
            max_tokens=quality_max_tokens,
            quiet=quiet,
            output_path=quality_jsonl,
            skip_keys=quality_skip,
        )
    )
    needle_rows.extend(
        _run_needle_suite_loaded(
            headline_model,
            headline_tokenizer,
            headline_mode,
            context_tiers=context_tiers,
            seeds_per_position=needle_seeds_per_position,
            max_tokens=needle_max_tokens,
            quiet=quiet,
            output_path=needle_jsonl,
            skip_keys=needle_skip,
        )
    )

    standard_model_for_parity, _ = load_turboquant(model, turboquant_config=None)
    parity_rows.extend(
        _run_parity_suite(
            standard_model_for_parity,
            headline_model,
            prepared_by_tier,
            context_tiers=context_tiers,
            examples_per_dataset=parity_examples_per_dataset,
            quiet=quiet,
            output_path=parity_jsonl,
            skip_keys=parity_skip,
        )
    )
    del standard_model_for_parity
    mx.clear_cache()

    for mode in modes:
        if mode.slug in {modes[0].slug, headline_mode.slug}:
            continue
        loaded_model, loaded_tokenizer = load_turboquant(model, turboquant_config=mode.config)
        try:
            quality_rows.extend(
                _run_quality_suite_loaded(
                    loaded_model,
                    loaded_tokenizer,
                    mode,
                    prepared_by_tier,
                    max_tokens=quality_max_tokens,
                    quiet=quiet,
                    output_path=quality_jsonl,
                    skip_keys=quality_skip,
                )
            )
            needle_rows.extend(
                _run_needle_suite_loaded(
                    loaded_model,
                    loaded_tokenizer,
                    mode,
                    context_tiers=context_tiers,
                    seeds_per_position=needle_seeds_per_position,
                    max_tokens=needle_max_tokens,
                    quiet=quiet,
                    output_path=needle_jsonl,
                    skip_keys=needle_skip,
                )
            )
        finally:
            del loaded_model
            del loaded_tokenizer
            mx.clear_cache()

    quality_summary = _summarize_quality_rows(quality_rows)
    needle_summary = _summarize_needle_rows(needle_rows)
    parity_summary = _summarize_parity_rows(parity_rows)
    acceptance = _build_acceptance_summary(quality_summary, needle_summary)
    summary_payload = {
        "model": model,
        "timestamp": timestamp,
        "provenance": provenance if provenance is not None else _provenance(),
        "context_tiers": context_tiers,
        "quality_datasets": quality_datasets,
        "calibration_datasets": calibration_datasets,
        "evaluation_defaults": asdict(
            EvaluationConfig(
                model=model,
                mode="preset",
                suite="quality",
                context_tier=context_tiers[0],
                calibration_artifact_path=str(calibration_artifact_path),
            )
        ),
        "calibration_artifact": asdict(CalibrationArtifact.load(calibration_artifact_path)),
        "quality_summary": quality_summary,
        "needle_summary": needle_summary,
        "parity_summary": parity_summary,
        "acceptance": acceptance,
    }
    _write_json(raw_dir / "summary.json", summary_payload)

    _generate_plots(
        plots_dir,
        quality_summary=quality_summary,
        needle_summary=needle_summary,
        parity_summary=parity_summary,
        acceptance=acceptance,
    )
    _write_annotations(
        annotations_dir,
        model=model,
        quality_summary=quality_summary,
        needle_summary=needle_summary,
        parity_summary=parity_summary,
        acceptance=acceptance,
        context_tiers=context_tiers,
    )

    _swap_latest_symlink(output_root / "latest", target_name=result_dir.name)

    del headline_model
    del headline_tokenizer
    mx.clear_cache()
    return result_dir


def _default_longbench_dir() -> str:
    return os.environ.get("TURBOQUANT_LONGBENCH_DIR", "assets/longbench/data")


def _default_output_root() -> str:
    return os.environ.get("TURBOQUANT_OUTPUT_ROOT", "results")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TurboQuant validation report.")
    parser.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--longbench-source-dir", default=_default_longbench_dir())
    parser.add_argument("--output-root", default=_default_output_root())
    parser.add_argument("--context-tiers", default="512,2048,4096")
    parser.add_argument("--quality-datasets", default="triviaqa_e,hotpotqa_e,2wikimqa_e")
    parser.add_argument("--calibration-datasets", default="multifieldqa_en,multi_news")
    parser.add_argument("--quality-examples-per-dataset", type=int, default=10)
    parser.add_argument("--parity-examples-per-dataset", type=int, default=1)
    parser.add_argument("--calibration-examples-per-dataset", type=int, default=2)
    parser.add_argument("--quality-max-tokens", type=int, default=24)
    parser.add_argument("--needle-max-tokens", type=int, default=12)
    parser.add_argument("--needle-seeds-per-position", type=int, default=4)
    parser.add_argument("--headline-only", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-tier/per-dataset progress lines")
    parser.add_argument("--seed", type=int, default=0, help="Master seed; threaded through every RNG.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to an existing result directory to continue. Re-uses its raw_rows JSONL, "
        "skips already-completed examples, and writes additional outputs into the same dir.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    provenance = _provenance(args=args)
    result_dir = run_report(
        model=args.model,
        longbench_source_dir=Path(args.longbench_source_dir),
        output_root=Path(args.output_root),
        context_tiers=_parse_int_list(args.context_tiers),
        quality_datasets=_parse_csv_list(args.quality_datasets),
        calibration_datasets=_parse_csv_list(args.calibration_datasets),
        quality_examples_per_dataset=args.quality_examples_per_dataset,
        parity_examples_per_dataset=args.parity_examples_per_dataset,
        calibration_examples_per_dataset=args.calibration_examples_per_dataset,
        quality_max_tokens=args.quality_max_tokens,
        needle_max_tokens=args.needle_max_tokens,
        needle_seeds_per_position=args.needle_seeds_per_position,
        headline_only=args.headline_only,
        quiet=args.quiet,
        seed=args.seed,
        resume_from=Path(args.resume) if args.resume else None,
        provenance=provenance,
    )
    print(result_dir)


if __name__ == "__main__":
    main()
