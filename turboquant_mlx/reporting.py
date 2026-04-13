from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import mlx.core as mx

from .calibration import calibrate_outlier_mask
from .config import CalibrationConfig, TurboQuantConfig
from .generate import append_jsonl
from .longbench import evaluate_longbench
from .load import load_turboquant
from .needle import build_needle_case, run_loaded_needle_case


DEFAULT_LONG_DATASETS = [
    "lcc_e",
    "trec_e",
]
DEFAULT_CALIBRATION_DATASETS = [
    "multifieldqa_en",
    "multi_news",
]


@dataclass
class ModeSpec:
    slug: str
    label: str
    config: Optional[TurboQuantConfig]


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _build_prompt_text(example: dict) -> str:
    return "\n\n".join(
        part
        for part in [
            f"Context:\n{example.get('context', '')}".strip(),
            f"Question:\n{example.get('input', '')}".strip(),
        ]
        if part
    )


def _truncate_words(text: str, limit: Optional[int]) -> str:
    if limit is None:
        return text
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit])


def build_dataset_slices(
    source_dir: Path,
    output_dir: Path,
    *,
    eval_examples_per_dataset: int = 1,
    calibration_examples_per_dataset: int = 2,
    context_word_limit: Optional[int] = 120,
) -> tuple[Path, Path]:
    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    eval_rows: List[dict] = []
    for dataset_name in DEFAULT_LONG_DATASETS:
        rows = _read_jsonl(source_dir / f"{dataset_name}.jsonl")
        for row in rows[:eval_examples_per_dataset]:
            eval_rows.append(
                {
                    **row,
                    "context": _truncate_words(row.get("context", ""), context_word_limit),
                }
            )

    calibration_rows: List[dict] = []
    for dataset_name in DEFAULT_CALIBRATION_DATASETS:
        rows = _read_jsonl(source_dir / f"{dataset_name}.jsonl")
        for row in rows[:calibration_examples_per_dataset]:
            calibration_rows.append(
                {
                    "text": _build_prompt_text(
                        {
                            **row,
                            "context": _truncate_words(row.get("context", ""), context_word_limit),
                        }
                    )
                }
            )

    eval_path = datasets_dir / "longbench_eval_slice.jsonl"
    calibration_path = datasets_dir / "calibration_prompts.jsonl"
    _write_jsonl(eval_path, eval_rows)
    _write_jsonl(calibration_path, calibration_rows)
    return eval_path, calibration_path


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _bar_chart_svg(
    title: str,
    labels: List[str],
    values: List[float],
    *,
    subtitle: str = "",
    width: int = 900,
    height: int = 520,
    color: str = "#2463EB",
    formatter=_format_float,
) -> str:
    if not values:
        raise ValueError("bar chart requires at least one value")
    max_value = max(values) or 1.0
    left, right, top, bottom = 90, 40, 110, 110
    chart_width = width - left - right
    chart_height = height - top - bottom
    gap = 24
    bar_width = max(24, (chart_width - gap * (len(values) - 1)) / len(values))
    origin_y = top + chart_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        f'<text x="{left}" y="44" font-size="28" font-family="Menlo, Monaco, monospace" fill="#0F172A">{title}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="{left}" y="72" font-size="14" font-family="Menlo, Monaco, monospace" fill="#475569">{subtitle}</text>'
        )
    parts.append(f'<line x1="{left}" y1="{origin_y}" x2="{width-right}" y2="{origin_y}" stroke="#CBD5E1" stroke-width="2"/>')

    for i in range(5):
        tick_value = max_value * i / 4
        y = origin_y - chart_height * i / 4
        parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#E2E8F0" stroke-width="1"/>')
        parts.append(
            f'<text x="{left-14}" y="{y+5}" text-anchor="end" font-size="12" font-family="Menlo, Monaco, monospace" fill="#64748B">{formatter(tick_value)}</text>'
        )

    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * (bar_width + gap)
        bar_height = 0 if max_value == 0 else chart_height * (value / max_value)
        y = origin_y - bar_height
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" rx="6" fill="{color}"/>')
        parts.append(
            f'<text x="{x + bar_width/2}" y="{y-10}" text-anchor="middle" font-size="12" font-family="Menlo, Monaco, monospace" fill="#0F172A">{formatter(value)}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width/2}" y="{origin_y+26}" text-anchor="middle" font-size="13" font-family="Menlo, Monaco, monospace" fill="#334155">{label}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _line_chart_svg(
    title: str,
    series: dict[str, List[float]],
    x_labels: List[str],
    *,
    subtitle: str = "",
    width: int = 960,
    height: int = 520,
) -> str:
    if not series:
        raise ValueError("line chart requires series")
    max_value = max((max(values) for values in series.values() if values), default=1.0) or 1.0
    left, right, top, bottom = 90, 40, 110, 100
    chart_width = width - left - right
    chart_height = height - top - bottom
    origin_y = top + chart_height
    colors = ["#2463EB", "#DC2626", "#059669", "#7C3AED", "#EA580C"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        f'<text x="{left}" y="44" font-size="28" font-family="Menlo, Monaco, monospace" fill="#0F172A">{title}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="{left}" y="72" font-size="14" font-family="Menlo, Monaco, monospace" fill="#475569">{subtitle}</text>'
        )
    for i in range(5):
        tick_value = max_value * i / 4
        y = origin_y - chart_height * i / 4
        parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#E2E8F0" stroke-width="1"/>')
        parts.append(
            f'<text x="{left-14}" y="{y+5}" text-anchor="end" font-size="12" font-family="Menlo, Monaco, monospace" fill="#64748B">{tick_value:.2f}</text>'
        )

    if len(x_labels) == 1:
        xs = [left + chart_width / 2]
    else:
        xs = [left + i * (chart_width / (len(x_labels) - 1)) for i in range(len(x_labels))]
    for x, label in zip(xs, x_labels):
        parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{origin_y}" stroke="#E2E8F0" stroke-width="1"/>')
        parts.append(
            f'<text x="{x}" y="{origin_y+26}" text-anchor="middle" font-size="13" font-family="Menlo, Monaco, monospace" fill="#334155">{label}</text>'
        )

    legend_x = left
    legend_y = height - 28
    for color, (name, values) in zip(colors, series.items()):
        points = []
        for x, value in zip(xs, values):
            y = origin_y - (0 if max_value == 0 else chart_height * (value / max_value))
            points.append((x, y, value))
        polyline = " ".join(f"{x},{y}" for x, y, _ in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{polyline}"/>')
        for x, y, value in points:
            parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>')
            parts.append(
                f'<text x="{x}" y="{y-10}" text-anchor="middle" font-size="12" font-family="Menlo, Monaco, monospace" fill="{color}">{value:.2f}</text>'
            )
        parts.append(f'<rect x="{legend_x}" y="{legend_y-10}" width="14" height="14" rx="3" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x+22}" y="{legend_y+2}" font-size="13" font-family="Menlo, Monaco, monospace" fill="#334155">{name}</text>'
        )
        legend_x += 180
    parts.append("</svg>")
    return "\n".join(parts)


def _safe_slug(value: str) -> str:
    return value.replace(".", "p").replace("-", "_")


def _mode_specs(head_dim: int, calibration_artifact_path: Path) -> List[ModeSpec]:
    calibration = CalibrationConfig(artifact_path=str(calibration_artifact_path))
    return [
        ModeSpec(slug="standard", label="Standard KV", config=None),
        ModeSpec(
            slug="core_4bit",
            label="Core 4-bit MSE",
            config=TurboQuantConfig(mode="core", head_dim=head_dim, core_bits=4),
        ),
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
        ),
    ]


def run_report(
    *,
    model: str,
    longbench_source_dir: Path,
    output_root: Path,
    eval_examples_per_dataset: int = 1,
    calibration_examples_per_dataset: int = 2,
    longbench_max_tokens: int = 24,
    needle_context_words: int = 4096,
    needle_max_tokens: int = 32,
    context_word_limit: Optional[int] = 120,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = output_root / timestamp
    raw_dir = result_dir / "raw"
    plots_dir = result_dir / "plots"
    annotations_dir = result_dir / "annotations"
    raw_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    eval_path, calibration_path = build_dataset_slices(
        longbench_source_dir,
        result_dir,
        eval_examples_per_dataset=eval_examples_per_dataset,
        calibration_examples_per_dataset=calibration_examples_per_dataset,
        context_word_limit=context_word_limit,
    )
    calibration_texts = [row["text"] for row in _read_jsonl(calibration_path)]
    artifact = calibrate_outlier_mask(
        model,
        calibration_texts,
        outlier_count=32,
        quantile=99.9,
        max_examples=len(calibration_texts),
    )
    calibration_artifact_path = raw_dir / "calibration_artifact.json"
    artifact.save(calibration_artifact_path)

    head_dim = artifact.head_dim
    modes = _mode_specs(head_dim, calibration_artifact_path)
    longbench_summary = []
    for mode in modes:
        result = evaluate_longbench(
            model,
            str(eval_path),
            turboquant_config=mode.config,
            max_tokens=longbench_max_tokens,
        )
        mode_rows_path = raw_dir / f"longbench_{mode.slug}.jsonl"
        _write_jsonl(mode_rows_path, result["rows"])
        prompt_tps_mean = sum(row["prompt_tps"] for row in result["rows"]) / len(result["rows"])
        generation_tps_mean = sum(row["generation_tps"] for row in result["rows"]) / len(result["rows"])
        peak_memory_mean = sum(row["peak_memory_gb"] for row in result["rows"]) / len(result["rows"])
        cache_nbytes_mean = sum(row["cache_nbytes"] for row in result["rows"]) / len(result["rows"])
        longbench_summary.append(
            {
                "mode_slug": mode.slug,
                "mode_label": mode.label,
                "mean_score": result["mean_score"],
                "examples": result["examples"],
                "prompt_tps_mean": prompt_tps_mean,
                "generation_tps_mean": generation_tps_mean,
                "peak_memory_gb_mean": peak_memory_mean,
                "cache_nbytes_mean": cache_nbytes_mean,
            }
        )

    needle_positions = [128, needle_context_words // 2, max(needle_context_words - 128, 128)]
    needle_modes = modes
    needle_rows = []
    for mode in needle_modes:
        loaded_model, tokenizer = load_turboquant(model, turboquant_config=mode.config)
        try:
            for position in needle_positions:
                case = build_needle_case(
                    context_words=needle_context_words,
                    needle="The launch code is 314159.",
                    question="What is the launch code?",
                    needle_position=position,
                    seed=42 + position,
                )
                row = run_loaded_needle_case(
                    model=loaded_model,
                    tokenizer=tokenizer,
                    case=case,
                    max_tokens=needle_max_tokens,
                    model_label=model,
                    turboquant_config=mode.config,
                )
                row["mode_slug"] = mode.slug
                row["mode_label"] = mode.label
                append_jsonl(raw_dir / "needle_results.jsonl", row)
                needle_rows.append(row)
        finally:
            del loaded_model
            del tokenizer
            mx.clear_cache()

    summary_payload = {
        "model": model,
        "timestamp": timestamp,
        "context_word_limit": context_word_limit,
        "calibration_artifact": asdict(artifact),
        "longbench_summary": longbench_summary,
        "needle_rows": needle_rows,
        "eval_dataset_path": str(eval_path),
        "calibration_dataset_path": str(calibration_path),
    }
    (raw_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))

    labels = [row["mode_label"] for row in longbench_summary]
    (plots_dir / "longbench_mean_score.svg").write_text(
        _bar_chart_svg(
            "LongBench-E Mean Score",
            labels,
            [row["mean_score"] for row in longbench_summary],
            subtitle="Compact evaluation slice, exact-match style scoring",
            color="#0F766E",
        )
    )
    (plots_dir / "generation_tps.svg").write_text(
        _bar_chart_svg(
            "Generation Tokens / Second",
            labels,
            [row["generation_tps_mean"] for row in longbench_summary],
            subtitle="Mean over the LongBench slice",
            color="#2563EB",
        )
    )
    (plots_dir / "peak_memory_gb.svg").write_text(
        _bar_chart_svg(
            "Peak Memory (GB)",
            labels,
            [row["peak_memory_gb_mean"] for row in longbench_summary],
            subtitle="Mean peak memory reported during generation",
            color="#DC2626",
        )
    )
    (plots_dir / "cache_nbytes.svg").write_text(
        _bar_chart_svg(
            "Observed Cache Size (bytes)",
            labels,
            [row["cache_nbytes_mean"] for row in longbench_summary],
            subtitle="Mean observed prompt cache size",
            color="#7C3AED",
            formatter=lambda v: f"{int(v):,}",
        )
    )

    needle_series = {}
    for mode in needle_modes:
        mode_rows = [row for row in needle_rows if row["mode_slug"] == mode.slug]
        needle_series[mode.label] = [float(row["correct"]) for row in sorted(mode_rows, key=lambda r: r["needle_position"])]
    (plots_dir / "needle_accuracy_by_position.svg").write_text(
        _line_chart_svg(
            "Needle Accuracy by Needle Position",
            needle_series,
            [str(position) for position in needle_positions],
            subtitle="1.0 means exact recovery of the inserted answer string",
        )
    )

    best_score = max(longbench_summary, key=lambda row: row["mean_score"])
    fastest = max(longbench_summary, key=lambda row: row["generation_tps_mean"])
    smallest_cache = min(longbench_summary, key=lambda row: row["cache_nbytes_mean"])
    summary_md = f"""# TurboQuant Run Summary

Model: `{model}`

Result directory: `{result_dir}`

## Highlights

- Best LongBench mean score: `{best_score['mode_label']}` at `{best_score['mean_score']:.3f}`
- Fastest generation: `{fastest['mode_label']}` at `{fastest['generation_tps_mean']:.3f}` tokens/s
- Smallest observed cache: `{smallest_cache['mode_label']}` at `{int(smallest_cache['cache_nbytes_mean']):,}` bytes

## Annotations

- The run uses a compact held-out LongBench slice built from `{', '.join(DEFAULT_LONG_DATASETS)}` and a separate calibration slice from `{', '.join(DEFAULT_CALIBRATION_DATASETS)}`.
- Each LongBench example in this first results pack is truncated to `{context_word_limit}` context words so the Python-level TurboQuant path can finish end-to-end on local hardware.
- `Preset 3.5-bit + QJL` uses a fixed outlier mask calibrated at the `99.9` percentile on held-out prompts.
- `Core 4-bit MSE` is the calibration-free reference path for the local adapter.
- The `needle_accuracy_by_position.svg` chart helps separate retrieval breakage from general generation drift.

## Files

- Raw summaries: [`raw/summary.json`](raw/summary.json)
- LongBench plots:
  - [`plots/longbench_mean_score.svg`](plots/longbench_mean_score.svg)
  - [`plots/generation_tps.svg`](plots/generation_tps.svg)
  - [`plots/peak_memory_gb.svg`](plots/peak_memory_gb.svg)
  - [`plots/cache_nbytes.svg`](plots/cache_nbytes.svg)
- Needle plot: [`plots/needle_accuracy_by_position.svg`](plots/needle_accuracy_by_position.svg)
"""
    (annotations_dir / "SUMMARY.md").write_text(summary_md)

    latest = output_root / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_dir() and not latest.is_symlink():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    latest.symlink_to(result_dir.name)
    return result_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a compact TurboQuant results pack with plots.")
    parser.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument(
        "--longbench-source-dir",
        default="/Users/vinitl/prog/turboquant-mlx/assets/longbench/data",
    )
    parser.add_argument("--output-root", default="/Users/vinitl/prog/turboquant-mlx/results")
    parser.add_argument("--eval-examples-per-dataset", type=int, default=1)
    parser.add_argument("--calibration-examples-per-dataset", type=int, default=2)
    parser.add_argument("--longbench-max-tokens", type=int, default=24)
    parser.add_argument("--needle-context-words", type=int, default=4096)
    parser.add_argument("--needle-max-tokens", type=int, default=32)
    parser.add_argument("--context-word-limit", type=int, default=120)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result_dir = run_report(
        model=args.model,
        longbench_source_dir=Path(args.longbench_source_dir),
        output_root=Path(args.output_root),
        eval_examples_per_dataset=args.eval_examples_per_dataset,
        calibration_examples_per_dataset=args.calibration_examples_per_dataset,
        longbench_max_tokens=args.longbench_max_tokens,
        needle_context_words=args.needle_context_words,
        needle_max_tokens=args.needle_max_tokens,
        context_word_limit=args.context_word_limit,
    )
    print(result_dir)


if __name__ == "__main__":
    main()
