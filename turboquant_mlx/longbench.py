from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional

from .config import CalibrationConfig, TurboQuantConfig
from .generate import append_jsonl, metrics_to_dict, run_generation
from .load import load_turboquant


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_longbench_examples(path: str | Path) -> List[dict]:
    examples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def build_prompt(example: dict) -> str:
    sections = []
    for key in ("instruction", "context", "input", "question"):
        value = example.get(key)
        if value:
            sections.append(f"{key.capitalize()}:\n{value}")
    sections.append("Answer:")
    return "\n\n".join(sections)


def score_example(prediction: str, answers: Iterable[str]) -> float:
    normalized_prediction = normalize_text(prediction)
    return float(any(normalize_text(answer) in normalized_prediction for answer in answers))


def evaluate_longbench(
    model_path: str,
    dataset_path: str,
    *,
    turboquant_config: Optional[TurboQuantConfig],
    max_examples: Optional[int] = None,
    max_tokens: int = 64,
) -> dict:
    model, tokenizer = load_turboquant(model_path, turboquant_config=turboquant_config)
    examples = load_longbench_examples(dataset_path)
    if max_examples is not None:
        examples = examples[:max_examples]

    total_score = 0.0
    rows = []
    for idx, example in enumerate(examples):
        prompt = build_prompt(example)
        prediction, metrics, _ = run_generation(model, tokenizer, prompt, max_tokens=max_tokens)
        answers = example.get("answers") or [example.get("answer", "")]
        score = score_example(prediction, answers)
        total_score += score
        rows.append(
            {
                "index": idx,
                "prompt": prompt,
                "prediction": prediction,
                "answers": answers,
                "score": score,
                **metrics_to_dict(metrics),
            }
        )

    mean_score = total_score / len(rows) if rows else 0.0
    return {
        "model": model_path,
        "mode": None if turboquant_config is None else turboquant_config.mode,
        "preset_name": None if turboquant_config is None else turboquant_config.preset_name,
        "examples": len(rows),
        "mean_score": mean_score,
        "rows": rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Llama-3.1-8B-Instruct on LongBench-E.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--output")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--mode", choices=["standard", "core", "preset"], default="standard")
    parser.add_argument("--core-bits", type=int, default=4)
    parser.add_argument("--preset-name", choices=["2.5", "3.5"])
    parser.add_argument("--calibration-artifact")
    parser.add_argument("--qjl", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = None
    if args.mode != "standard":
        config = TurboQuantConfig(
            mode="core" if args.mode == "core" else "preset",
            core_bits=args.core_bits,
            preset_name=args.preset_name,
            qjl_enabled=args.qjl,
            calibration=CalibrationConfig(artifact_path=args.calibration_artifact),
        )
    result = evaluate_longbench(
        args.model,
        args.dataset_jsonl,
        turboquant_config=config,
        max_examples=args.max_examples,
        max_tokens=args.max_tokens,
    )
    if args.output:
        for row in result["rows"]:
            append_jsonl(args.output, row)
    print(
        {
            "model": result["model"],
            "mode": result["mode"],
            "preset_name": result["preset_name"],
            "examples": result["examples"],
            "mean_score": result["mean_score"],
        }
    )


if __name__ == "__main__":
    main()
