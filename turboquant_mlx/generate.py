from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Sequence, Union

import mlx.core as mx

from mlx_lm.generate import generate_step, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tokenizer_utils import TokenizerWrapper


def stream_generate_turboquant(model, tokenizer: TokenizerWrapper, prompt: Union[str, Sequence[int]], **kwargs):
    return stream_generate(model, tokenizer, prompt, **kwargs)


def generate_tokens(
    model,
    prompt: Sequence[int],
    *,
    max_tokens: int = 32,
    sampler=None,
    **kwargs,
) -> List[int]:
    prompt_array = mx.array(prompt, dtype=mx.uint32)
    outputs: List[int] = []
    sampler = sampler or (lambda logprobs: mx.argmax(logprobs, axis=-1))
    for token, _ in generate_step(
        prompt_array,
        model,
        max_tokens=max_tokens,
        sampler=sampler,
        **kwargs,
    ):
        outputs.append(int(token))
        if len(outputs) >= max_tokens:
            break
    return outputs


@dataclass
class RunMetrics:
    prompt_tokens: int
    generated_tokens: int
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float
    cache_nbytes: int


def _estimate_cache_nbytes(cache: List[Any]) -> int:
    total = 0
    for entry in cache:
        if hasattr(entry, "observed_nbytes"):
            total += int(entry.observed_nbytes())
            continue
        for attr in ("keys", "values"):
            value = getattr(entry, attr, None)
            if value is not None and hasattr(value, "nbytes"):
                total += int(value.nbytes)
    return total


def _ensure_tokenizer(tokenizer) -> TokenizerWrapper:
    return tokenizer if isinstance(tokenizer, TokenizerWrapper) else TokenizerWrapper(tokenizer)


def run_generation(
    model,
    tokenizer,
    prompt: Union[str, Sequence[int]],
    *,
    max_tokens: int = 64,
    sampler=None,
    prompt_cache=None,
    **kwargs,
):
    tokenizer = _ensure_tokenizer(tokenizer)
    if isinstance(prompt, str):
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    else:
        prompt_tokens = list(prompt)
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)
    detokenizer = tokenizer.detokenizer
    sampler = sampler or (lambda logprobs: mx.argmax(logprobs, axis=-1))
    prompt_cache = prompt_cache or make_prompt_cache(model)

    generated = []
    prompt_start = time.perf_counter()
    first_token_time = None
    for step_idx, (token, logprobs) in enumerate(
        generate_step(
            prompt_array,
            model,
            max_tokens=max_tokens,
            sampler=sampler,
            prompt_cache=prompt_cache,
            **kwargs,
        )
    ):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        token_value = int(token)
        if token_value in tokenizer.eos_token_ids:
            break
        generated.append(token_value)
        detokenizer.add_token(token_value)
        if len(generated) >= max_tokens:
            break
    detokenizer.finalize()
    prompt_elapsed = max((first_token_time or time.perf_counter()) - prompt_start, 1e-9)
    generation_elapsed = max(time.perf_counter() - (first_token_time or prompt_start), 1e-9)
    metrics = RunMetrics(
        prompt_tokens=len(prompt_tokens),
        generated_tokens=len(generated),
        prompt_tps=len(prompt_tokens) / prompt_elapsed,
        generation_tps=len(generated) / generation_elapsed if generated else 0.0,
        peak_memory_gb=mx.get_peak_memory() / 1e9,
        cache_nbytes=_estimate_cache_nbytes(prompt_cache),
    )
    return detokenizer.text, metrics, prompt_cache


def append_jsonl(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def metrics_to_dict(metrics: RunMetrics) -> dict:
    return asdict(metrics)
