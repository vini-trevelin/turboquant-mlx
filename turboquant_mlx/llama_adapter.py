from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope

from .attention import turboquant_scaled_dot_product_attention
from .cache import TurboQuantKVCache
from .config import TurboQuantConfig


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    layer_types: Optional[List[str]] = None
    sliding_window: Optional[int] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or args.hidden_size // self.n_heads
        self.scale = self.head_dim**-0.5
        attention_bias = getattr(args, "attention_bias", False)

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=attention_bias)
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )
        self.collector = None

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        batch, length, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(batch, length, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch, length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(batch, length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            if self.collector is not None:
                self.collector.observe(keys, values)
            if isinstance(cache, TurboQuantKVCache):
                cache.update_and_fetch(keys, values)
                output = turboquant_scaled_dot_product_attention(
                    queries,
                    cache,
                    scale=self.scale,
                    mask=mask,
                )
            else:
                keys, values = cache.update_and_fetch(keys, values)
                output = scaled_dot_product_attention(
                    queries, keys, values, cache=cache, scale=self.scale, mask=mask
                )
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
            if self.collector is not None:
                self.collector.observe(keys, values)
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )

        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        mlp_bias = getattr(args, "mlp_bias", False)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, use_sliding: bool = False):
        super().__init__()
        self.use_sliding = use_sliding
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, use_sliding=layer_type == "sliding_attention")
            for layer_type in self.layer_types
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fa_idx = self.layer_types.index("full_attention")
        self.swa_idx = None
        for idx, layer in enumerate(self.layers):
            if layer.use_sliding:
                self.swa_idx = idx
                break

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(h, cache[self.swa_idx], window_size=self.sliding_window)

        for layer, layer_cache in zip(self.layers, cache):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, mask, cache=layer_cache)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LlamaModel(args)
        self.turboquant_config: Optional[TurboQuantConfig] = None
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        weights = {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers

    def enable_turboquant(self, config: TurboQuantConfig):
        self.turboquant_config = config

    def attach_collector(self, collector):
        for layer in self.layers:
            layer.self_attn.collector = collector

    def make_cache(self):
        if self.turboquant_config is None:
            return [
                (
                    RotatingKVCache(max_size=self.model.sliding_window)
                    if layer.use_sliding
                    else KVCache()
                )
                for layer in self.layers
            ]
        return [TurboQuantKVCache(self.turboquant_config) for _ in self.layers]

