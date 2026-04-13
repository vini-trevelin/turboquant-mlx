from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Optional, Sequence

from .config import CalibrationConfig, TurboQuantConfig
from .generate import append_jsonl, metrics_to_dict, run_generation
from .load import load_turboquant
from .qa_eval import score_short_answer


@dataclass
class NeedleCase:
    prompt: str
    prompt_tokens: list[int]
    needle: str
    question: str
    context_tokens: int
    needle_position: int
    needle_position_label: str
    seed: int


def _position_ratio(needle_position: str | int, target_tokens: int) -> tuple[str, float]:
    if isinstance(needle_position, str):
        label = needle_position.lower()
        if label == "front":
            return label, 0.15
        if label == "back":
            return label, 0.85
        return "middle", 0.5
    if target_tokens <= 0:
        return "custom", 0.5
    return "custom", max(0.0, min(float(needle_position) / float(target_tokens), 1.0))


def build_needle_case(
    tokenizer,
    *,
    context_tokens: int,
    needle: str,
    question: str,
    needle_position: str | int = "middle",
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
        "saturn",
        "willow",
    ]
    prefix = "Read the following context and answer the question exactly.\n\nContext:\n"
    suffix = f"\n\nQuestion: {question}\nAnswer:"
    position_label, ratio = _position_ratio(needle_position, context_tokens)
    pre_words: list[str] = []
    post_words: list[str] = []

    def render(pre: Sequence[str], post: Sequence[str]) -> str:
        pieces = []
        if pre:
            pieces.append(" ".join(pre))
        pieces.append(needle)
        if post:
            pieces.append(" ".join(post))
        context = " ".join(piece for piece in pieces if piece).strip()
        return f"{prefix}{context}{suffix}"

    prompt = render(pre_words, post_words)
    prompt_tokens = list(tokenizer.encode(prompt, add_special_tokens=False))
    if len(prompt_tokens) > context_tokens:
        raise ValueError(
            f"context_tokens={context_tokens} is smaller than the minimum prompt size {len(prompt_tokens)}"
        )
    while len(prompt_tokens) < context_tokens:
        candidate_word = rng.choice(vocab)
        target_pre = (len(pre_words) + len(post_words) + 1) * ratio
        add_to_pre = len(pre_words) < target_pre
        next_pre = pre_words + [candidate_word] if add_to_pre else pre_words
        next_post = post_words if add_to_pre else post_words + [candidate_word]
        next_prompt = render(next_pre, next_post)
        next_tokens = list(tokenizer.encode(next_prompt, add_special_tokens=False))
        if len(next_tokens) > context_tokens:
            break
        pre_words, post_words = next_pre, next_post
        prompt, prompt_tokens = next_prompt, next_tokens

    return NeedleCase(
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        needle=needle,
        question=question,
        context_tokens=len(prompt_tokens),
        needle_position=len(pre_words),
        needle_position_label=position_label,
        seed=seed,
    )


def run_needle_case(
    *,
    model_path: str,
    turboquant_config: Optional[TurboQuantConfig],
    context_tokens: int,
    needle: str = "The launch code is 314159.",
    question: str = "What is the launch code?",
    needle_position: str | int = "middle",
    seed: int = 0,
    max_tokens: int = 32,
) -> dict:
    model, tokenizer = load_turboquant(model_path, turboquant_config=turboquant_config)
    case = build_needle_case(
        tokenizer,
        context_tokens=context_tokens,
        needle=needle,
        question=question,
        needle_position=needle_position,
        seed=seed,
    )
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
    text, metrics, _ = run_generation(model, tokenizer, case.prompt_tokens, max_tokens=max_tokens)
    score = score_short_answer(text, [case.needle])
    result = {
        "model": model_label,
        "mode": None if turboquant_config is None else turboquant_config.mode,
        "preset_name": None if turboquant_config is None else turboquant_config.preset_name,
        "context_tokens": case.context_tokens,
        "needle_position": case.needle_position,
        "needle_position_label": case.needle_position_label,
        "needle": case.needle,
        "question": case.question,
        "response": text,
        "normalized_prediction": score.normalized_prediction,
        "normalized_gold": score.normalized_gold,
        "match_type": score.match_type,
        "correct": float(score.correct),
        **metrics_to_dict(metrics),
    }
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Needle in a Haystack sanity benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output")
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--needle", default="The launch code is 314159.")
    parser.add_argument("--question", default="What is the launch code?")
    parser.add_argument("--needle-position", default="middle")
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
    needle_position: str | int
    try:
        needle_position = int(args.needle_position)
    except ValueError:
        needle_position = args.needle_position

    result = run_needle_case(
        model_path=args.model,
        turboquant_config=config,
        context_tokens=args.context_tokens,
        needle=args.needle,
        question=args.question,
        needle_position=needle_position,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )
    if args.output:
        append_jsonl(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
