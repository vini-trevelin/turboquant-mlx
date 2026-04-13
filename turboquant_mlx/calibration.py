from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

import mlx.core as mx
import numpy as np

from .config import CalibrationArtifact
from .load import load_turboquant


@dataclass
class ActivationRecorder:
    samples: List[np.ndarray] = field(default_factory=list)

    def observe(self, keys: mx.array, values: mx.array) -> None:
        key_abs = np.abs(np.asarray(keys)).reshape(-1, keys.shape[-1])
        value_abs = np.abs(np.asarray(values)).reshape(-1, values.shape[-1])
        self.samples.append(np.concatenate([key_abs, value_abs], axis=0))

    def finalize(self, quantile: float) -> np.ndarray:
        if not self.samples:
            raise ValueError("No activation samples were recorded during calibration.")
        activations = np.concatenate(self.samples, axis=0)
        return np.quantile(activations, quantile / 100.0, axis=0)


def load_texts(path: str | Path) -> List[str]:
    lines = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "text" in payload:
                lines.append(payload["text"])
            elif "prompt" in payload:
                lines.append(payload["prompt"])
            else:
                lines.append(" ".join(str(v) for v in payload.values()))
    return lines


def calibrate_outlier_mask(
    model_path: str,
    texts: Iterable[str],
    *,
    outlier_count: int = 32,
    quantile: float = 99.9,
    max_examples: int = 32,
) -> CalibrationArtifact:
    texts = list(texts)
    model, tokenizer, config = load_turboquant(model_path, return_config=True)
    recorder = ActivationRecorder()
    model.attach_collector(recorder)

    for idx, text in enumerate(texts):
        if idx >= max_examples:
            break
        tokens = tokenizer.encode(text)
        cache = model.make_cache()
        token_array = mx.array(tokens, dtype=mx.uint32)
        model(token_array[None], cache=cache)

    scores = recorder.finalize(quantile)
    outlier_indices = tuple(np.argsort(scores)[-outlier_count:].astype(int).tolist())
    return CalibrationArtifact(
        head_dim=int(config.get("head_dim", model.args.head_dim or model.args.hidden_size // model.args.num_attention_heads)),
        outlier_indices=outlier_indices,
        quantile=quantile,
        metadata={
            "model": model_path,
            "examples": str(min(len(texts), max_examples)),
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate a shared TurboQuant outlier mask.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--texts-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--outlier-count", type=int, default=32)
    parser.add_argument("--quantile", type=float, default=99.9)
    parser.add_argument("--max-examples", type=int, default=32)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    texts = load_texts(args.texts_jsonl)
    artifact = calibrate_outlier_mask(
        args.model,
        texts,
        outlier_count=args.outlier_count,
        quantile=args.quantile,
        max_examples=args.max_examples,
    )
    artifact.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
