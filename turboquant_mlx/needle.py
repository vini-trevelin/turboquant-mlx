from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .config import CalibrationConfig, TurboQuantConfig
from .generate import append_jsonl, metrics_to_dict, run_generation
from .load import load_turboquant


@dataclass
class NeedleCase:
    prompt: str
    needle: str
    question: str
    context_words: int
    needle_position: int


def build_needle_case(
    *,
    context_words: int,
    needle: str,
    question: str,
    needle_position: int,
    seed: int = 0,
) -> NeedleCase:
    rng = random.Random(seed)
    vocab = [
        "atlas",
        "harbor",
        "copper",
        "ember",
        "linen",
        "signal",
        "forest",
        "delta",
        "marble",
        "engine",
    ]
    filler = [rng.choice(vocab) for _ in range(context_words)]
    insert_at = max(0, min(needle_position, len(filler)))
    filler.insert(insert_at, needle)
    prompt = (
        "Read the following context and answer the question exactly.\n\n"
        f"Context:\n{' '.join(filler)}\n\nQuestion: {question}\nAnswer:"
    )
    return NeedleCase(
        prompt=prompt,
        needle=needle,
        question=question,
        context_words=context_words,
        needle_position=insert_at,
    )


def run_needle_case(
    *,
    model_path: str,
    turboquant_config: Optional[TurboQuantConfig],
    case: NeedleCase,
    max_tokens: int = 32,
) -> dict:
    model, tokenizer = load_turboquant(model_path, turboquant_config=turboquant_config)
    return run_loaded_needle_case(
        model=model,
        tokenizer=tokenizer,
        case=case,
        max_tokens=max_tokens,
        model_label=model_path,
        turboquant_config=turboquant_config,
    )


def run_loaded_needle_case(
    *,
    model,
    tokenizer,
    case: NeedleCase,
    max_tokens: int = 32,
    model_label: str = "loaded-model",
    turboquant_config: Optional[TurboQuantConfig] = None,
) -> dict:
    text, metrics, _ = run_generation(model, tokenizer, case.prompt, max_tokens=max_tokens)
    correct = case.needle.lower() in text.lower()
    result = {
        "model": model_label,
        "mode": None if turboquant_config is None else turboquant_config.mode,
        "preset_name": None if turboquant_config is None else turboquant_config.preset_name,
        "context_words": case.context_words,
        "needle_position": case.needle_position,
        "needle": case.needle,
        "question": case.question,
        "response": text,
        "correct": correct,
        **metrics_to_dict(metrics),
    }
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Needle in a Haystack sanity benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output")
    parser.add_argument("--context-words", type=int, default=4096)
    parser.add_argument("--needle", default="The launch code is 314159.")
    parser.add_argument("--question", default="What is the launch code?")
    parser.add_argument("--needle-position", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=32)
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
    case = build_needle_case(
        context_words=args.context_words,
        needle=args.needle,
        question=args.question,
        needle_position=args.needle_position,
        seed=args.seed,
    )
    result = run_needle_case(
        model_path=args.model,
        turboquant_config=config,
        case=case,
        max_tokens=args.max_tokens,
    )
    if args.output:
        append_jsonl(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
