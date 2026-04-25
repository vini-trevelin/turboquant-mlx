from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import mlx.core as mx
import numpy as np

from .config import CalibrationConfig, TurboQuantConfig
from .load import load_turboquant


@dataclass(frozen=True)
class TeacherForcedMetrics:
    context_tokens: int
    steps: int
    top1_agreement: float
    top5_overlap: float
    kl_divergence_mean: float
    standard_nll_mean: float
    turbo_nll_mean: float
    standard_perplexity: float
    turbo_perplexity: float

    def to_dict(self) -> dict:
        return asdict(self)


def _encode_prompt(tokenizer, prompt: str | Sequence[int]) -> list[int]:
    if isinstance(prompt, str):
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
        return list(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))
    return list(prompt)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - logsumexp


def _compare_logits(
    standard_logits: np.ndarray,
    turbo_logits: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, float, float, float, float]:
    steps = standard_logits.shape[0]
    top1_matches = 0.0
    top5_overlap = 0.0
    kl_total = 0.0

    standard_log_probs = _log_softmax(standard_logits)
    turbo_log_probs = _log_softmax(turbo_logits)
    standard_probs = np.exp(standard_log_probs)

    standard_nll = -float(np.sum(standard_log_probs[np.arange(steps), targets]))
    turbo_nll = -float(np.sum(turbo_log_probs[np.arange(steps), targets]))

    for idx in range(steps):
        std_row = standard_logits[idx]
        turbo_row = turbo_logits[idx]
        if int(np.argmax(std_row)) == int(np.argmax(turbo_row)):
            top1_matches += 1.0

        std_top5 = np.argpartition(std_row, -5)[-5:]
        turbo_top5 = np.argpartition(turbo_row, -5)[-5:]
        top5_overlap += len(set(std_top5.tolist()) & set(turbo_top5.tolist())) / 5.0

        kl_total += float(
            np.sum(
                standard_probs[idx]
                * (standard_log_probs[idx] - turbo_log_probs[idx])
            )
        )

    return top1_matches, top5_overlap, kl_total, standard_nll, turbo_nll


def evaluate_teacher_forced_loaded(
    standard_model,
    turbo_model,
    prompt_tokens: Sequence[int],
    *,
    chunk_size: int = 64,
) -> TeacherForcedMetrics:
    prompt_tokens = list(prompt_tokens)
    if len(prompt_tokens) < 2:
        return TeacherForcedMetrics(
            context_tokens=len(prompt_tokens),
            steps=0,
            top1_agreement=0.0,
            top5_overlap=0.0,
            kl_divergence_mean=0.0,
            standard_nll_mean=0.0,
            turbo_nll_mean=0.0,
            standard_perplexity=0.0,
            turbo_perplexity=0.0,
        )

    standard_cache = standard_model.make_cache()
    turbo_cache = turbo_model.make_cache()
    teacher_inputs = prompt_tokens[:-1]
    teacher_targets = prompt_tokens[1:]

    top1_matches = 0.0
    top5_overlap = 0.0
    kl_total = 0.0
    standard_nll_total = 0.0
    turbo_nll_total = 0.0
    steps = 0

    for start in range(0, len(teacher_inputs), chunk_size):
        chunk = teacher_inputs[start : start + chunk_size]
        targets = np.asarray(teacher_targets[start : start + chunk_size], dtype=np.int64)
        inputs = mx.array([chunk], dtype=mx.uint32)
        standard_logits = np.asarray(standard_model(inputs, cache=standard_cache)[0])
        turbo_logits = np.asarray(turbo_model(inputs, cache=turbo_cache)[0])
        chunk_top1, chunk_top5, chunk_kl, chunk_std_nll, chunk_turbo_nll = _compare_logits(
            standard_logits, turbo_logits, targets
        )
        top1_matches += chunk_top1
        top5_overlap += chunk_top5
        kl_total += chunk_kl
        standard_nll_total += chunk_std_nll
        turbo_nll_total += chunk_turbo_nll
        steps += len(chunk)

    standard_nll_mean = standard_nll_total / steps if steps else 0.0
    turbo_nll_mean = turbo_nll_total / steps if steps else 0.0
    return TeacherForcedMetrics(
        context_tokens=len(prompt_tokens),
        steps=steps,
        top1_agreement=top1_matches / steps if steps else 0.0,
        top5_overlap=top5_overlap / steps if steps else 0.0,
        kl_divergence_mean=kl_total / steps if steps else 0.0,
        standard_nll_mean=standard_nll_mean,
        turbo_nll_mean=turbo_nll_mean,
        standard_perplexity=float(np.exp(standard_nll_mean)) if steps else 0.0,
        turbo_perplexity=float(np.exp(turbo_nll_mean)) if steps else 0.0,
    )


def evaluate_teacher_forced(
    model_path: str,
    prompt: str | Sequence[int],
    *,
    turboquant_config: TurboQuantConfig,
    chunk_size: int = 64,
) -> dict:
    standard_model, tokenizer = load_turboquant(model_path, turboquant_config=None)
    turbo_model, _ = load_turboquant(model_path, turboquant_config=turboquant_config)
    prompt_tokens = _encode_prompt(tokenizer, prompt)
    metrics = evaluate_teacher_forced_loaded(
        standard_model,
        turbo_model,
        prompt_tokens,
        chunk_size=chunk_size,
    )
    return {
        "model": model_path,
        "mode": turboquant_config.mode,
        "preset_name": turboquant_config.preset_name,
        **metrics.to_dict(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare standard KV vs TurboQuant under teacher forcing.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--mode", choices=["core", "preset"], default="preset")
    parser.add_argument("--core-bits", type=int, default=4)
    parser.add_argument("--preset-name", choices=["2.5", "3.5"], default="3.5")
    parser.add_argument("--calibration-artifact")
    parser.add_argument("--qjl", action="store_true")
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = TurboQuantConfig(
        mode=args.mode,
        core_bits=args.core_bits,
        preset_name=args.preset_name if args.mode == "preset" else None,
        qjl_enabled=args.qjl,
        calibration=CalibrationConfig(artifact_path=args.calibration_artifact),
    )
    result = evaluate_teacher_forced(
        args.model,
        args.prompt,
        turboquant_config=config,
        chunk_size=args.chunk_size,
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
