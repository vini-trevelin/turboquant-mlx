from __future__ import annotations

from typing import Optional

import mlx.core as mx

from .cache import TurboQuantKVCache
from .quantizer import apply_attention_to_values, score_queries_against_keys


def turboquant_scaled_dot_product_attention(
    queries: mx.array,
    cache: TurboQuantKVCache,
    *,
    scale: float,
    mask: Optional[mx.array],
) -> mx.array:
    scores = score_queries_against_keys(
        queries,
        cache.keys,
        cache.setup,
        scale=scale,
        apply_qjl=cache.config.qjl_enabled,
    )

    if mask is not None:
        if isinstance(mask, str):
            q_len, k_len = scores.shape[-2:]
            q_indices = mx.arange(k_len - q_len, k_len)
            k_indices = mx.arange(k_len)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores = scores + mask

    weights = mx.softmax(scores, axis=-1, precise=True)
    return apply_attention_to_values(weights, cache.values, cache.setup)

