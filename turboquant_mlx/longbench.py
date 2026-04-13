from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

from .config import CalibrationConfig, TurboQuantConfig
from .generate import append_jsonl, metrics_to_dict, run_generation
from .load import load_turboquant
from .qa_eval import build_prompt, read_jsonl, score_qa_prediction, truncate_example_to_prompt_tokens


def load_longbench_examples(path: str | Path) -> List[dict]:
    return read_jsonl(path)


def prepare_longbench_examples(
    dataset_path: str | Path,
    tokenizer,
    *,
    prompt_token_limit: Optional[int] = None,
    max_examples: Optional[int] = None,
    dataset_name: Optional[str] = None,
) -> List[dict]:
    examples = load_longbench_examples(dataset_path)
    if max_examples is not None:
        examples = examples[:max_examples]
    dataset_name = dataset_name or Path(dataset_path).stem
    prepared = []
    for idx, example in enumerate(examples):
        truncated = (
            truncate_example_to_prompt_tokens(example, tokenizer, prompt_token_limit)
            if prompt_token_limit is not None
            else {
                **example,
                "_prompt": build_prompt(example),
                "_prompt_tokens": tokenizer.encode(build_prompt(example), add_special_tokens=False),
                "_prompt_token_count": len(tokenizer.encode(build_prompt(example), add_special_tokens=False)),
                "_context_token_count": len(tokenizer.encode(example.get("context", ""), add_special_tokens=False))
                if example.get("context")
                else 0,
            }
        )
        prepared.append(
            {
                **truncated,
                "_dataset_name": dataset_name,
                "_index": idx,
            }
        )
    return prepared


def evaluate_longbench_loaded(
    model,
    tokenizer,
    examples: Iterable[dict],
    *,
    max_tokens: int = 64,
    turboquant_config: Optional[TurboQuantConfig],
) -> dict:
    total_em = 0.0
    total_f1 = 0.0
    total_headline = 0.0
    rows = []
    examples = list(examples)

    for example in examples:
        prompt = example["_prompt"]
        prompt_tokens = example["_prompt_tokens"]
        prediction, metrics, _ = run_generation(model, tokenizer, prompt_tokens, max_tokens=max_tokens)
        answers = example.get("answers") or [example.get("answer", "")]
        qa_score = score_qa_prediction(
            prediction,
            answers,
            dataset_name=example["_dataset_name"],
        )
        total_em += qa_score.em
        total_f1 += qa_score.f1
        total_headline += qa_score.headline_score
        rows.append(
            {
                "index": example["_index"],
                "dataset": example["_dataset_name"],
                "prompt": prompt,
                "prediction": prediction,
                "answers": answers,
                "prompt_token_target": example.get("_prompt_token_target", example["_prompt_token_count"]),
                "prompt_token_count": example["_prompt_token_count"],
                "context_token_count": example["_context_token_count"],
                **qa_score.to_dict(),
                **metrics_to_dict(metrics),
            }
        )

    count = len(rows) or 1
    return {
        "model": getattr(model, "model_type", "loaded-model"),
        "mode": None if turboquant_config is None else turboquant_config.mode,
        "preset_name": None if turboquant_config is None else turboquant_config.preset_name,
        "examples": len(rows),
        "mean_em": total_em / count,
        "mean_f1": total_f1 / count,
        "mean_headline_score": total_headline / count,
        "rows": rows,
    }


def evaluate_longbench(
    model_path: str,
    dataset_path: str,
    *,
    turboquant_config: Optional[TurboQuantConfig],
    max_examples: Optional[int] = None,
    max_tokens: int = 64,
    prompt_token_limit: Optional[int] = None,
    dataset_name: Optional[str] = None,
) -> dict:
    model, tokenizer = load_turboquant(model_path, turboquant_config=turboquant_config)
    prepared = prepare_longbench_examples(
        dataset_path,
        tokenizer,
        prompt_token_limit=prompt_token_limit,
        max_examples=max_examples,
        dataset_name=dataset_name,
    )
    result = evaluate_longbench_loaded(
        model,
        tokenizer,
        prepared,
        max_tokens=max_tokens,
        turboquant_config=turboquant_config,
    )
    result["model"] = model_path
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate exact-answer-friendly LongBench datasets.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument("--output")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-token-limit", type=int)
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
        prompt_token_limit=args.prompt_token_limit,
        dataset_name=args.dataset_name,
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
            "mean_em": result["mean_em"],
            "mean_f1": result["mean_f1"],
            "mean_headline_score": result["mean_headline_score"],
        }
    )


if __name__ == "__main__":
    main()
