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
    context_tokens: int
    generated_tokens: int
    prefill_seconds: float
    decode_seconds: float
    prompt_tps: float
    generation_tps: float
    cache_nbytes: int
    active_memory_before_bytes: int
    active_memory_after_bytes: int
    peak_memory_delta_bytes: int
    metal_cache_memory_before_bytes: int
    metal_cache_memory_after_bytes: int


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


def _memory_api():
    required = ("get_active_memory", "get_cache_memory", "get_peak_memory", "reset_peak_memory")
    if all(hasattr(mx, name) for name in required):
        return mx
    return getattr(mx, "metal", mx)


def _reset_peak_memory() -> None:
    api = _memory_api()
    if hasattr(api, "reset_peak_memory"):
        api.reset_peak_memory()


def _get_active_memory() -> int:
    api = _memory_api()
    if hasattr(api, "get_active_memory"):
        return int(api.get_active_memory())
    return 0


def _get_cache_memory() -> int:
    api = _memory_api()
    if hasattr(api, "get_cache_memory"):
        return int(api.get_cache_memory())
    return 0


def _get_peak_memory() -> int:
    api = _memory_api()
    if hasattr(api, "get_peak_memory"):
        return int(api.get_peak_memory())
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return 0


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
    _reset_peak_memory()
    active_memory_before = _get_active_memory()
    cache_memory_before = _get_cache_memory()
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
    prefill_seconds = max((first_token_time or time.perf_counter()) - prompt_start, 1e-9)
    decode_seconds = max(time.perf_counter() - (first_token_time or prompt_start), 1e-9)
    active_memory_after = _get_active_memory()
    cache_memory_after = _get_cache_memory()
    peak_memory_delta_bytes = _get_peak_memory()
    metrics = RunMetrics(
        prompt_tokens=len(prompt_tokens),
        context_tokens=len(prompt_tokens),
        generated_tokens=len(generated),
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        prompt_tps=len(prompt_tokens) / prefill_seconds,
        generation_tps=len(generated) / decode_seconds if generated else 0.0,
        cache_nbytes=_estimate_cache_nbytes(prompt_cache),
        active_memory_before_bytes=active_memory_before,
        active_memory_after_bytes=active_memory_after,
        peak_memory_delta_bytes=peak_memory_delta_bytes,
        metal_cache_memory_before_bytes=cache_memory_before,
        metal_cache_memory_after_bytes=cache_memory_after,
    )
    return detokenizer.text, metrics, prompt_cache


def append_jsonl(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def metrics_to_dict(metrics: RunMetrics) -> dict:
    payload = asdict(metrics)
    payload["peak_memory_delta_gb"] = metrics.peak_memory_delta_bytes / 1e9
    payload["peak_memory_gb"] = payload["peak_memory_delta_gb"]
    payload["active_memory_before_gb"] = metrics.active_memory_before_bytes / 1e9
    payload["active_memory_after_gb"] = metrics.active_memory_after_bytes / 1e9
    payload["metal_cache_memory_before_gb"] = metrics.metal_cache_memory_before_bytes / 1e9
    payload["metal_cache_memory_after_gb"] = metrics.metal_cache_memory_after_bytes / 1e9
    return payload
