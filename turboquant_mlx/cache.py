from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import mlx.core as mx

from mlx_lm.models.cache import create_attention_mask

from .config import TurboQuantConfig
from .quantizer import (
    ChunkedCompressedTensor,
    CompressedTensor,
    CompressedTensorLike,
    SharedTurboQuantSetup,
    quantize_tensor,
)


def _concat_optional(existing: Optional[mx.array], new: Optional[mx.array], axis: int) -> Optional[mx.array]:
    if new is None:
        return existing
    if existing is None:
        return new
    return mx.concatenate([existing, new], axis=axis)


def _merge_compressed_chunks(existing: CompressedTensor, new: CompressedTensor) -> CompressedTensor:
    states = []
    for old_state, new_state in zip(existing.subset_states, new.subset_states):
        states.append(
            old_state.__class__(
                packed_codes=mx.concatenate([old_state.packed_codes, new_state.packed_codes], axis=2),
                norms=mx.concatenate([old_state.norms, new_state.norms], axis=2),
                residual_signs=_concat_optional(old_state.residual_signs, new_state.residual_signs, axis=2),
                residual_norms=_concat_optional(old_state.residual_norms, new_state.residual_norms, axis=2),
            )
        )
    base_shape = existing.original_shape or new.original_shape
    merged_shape = (base_shape[0], base_shape[1], existing.sequence_length + new.sequence_length, base_shape[3])
    return CompressedTensor(subset_states=tuple(states), original_shape=merged_shape)


def _as_chunk_tuple(tensor: Optional[CompressedTensorLike]) -> tuple[CompressedTensor, ...]:
    if tensor is None:
        return ()
    if isinstance(tensor, ChunkedCompressedTensor):
        return tensor.chunks
    return (tensor,)


def _wrap_chunks(chunks: tuple[CompressedTensor, ...]) -> Optional[CompressedTensorLike]:
    if not chunks:
        return None
    if len(chunks) == 1:
        return chunks[0]
    return ChunkedCompressedTensor(chunks=chunks)


def _tail_capacity(tensor: Optional[CompressedTensorLike], chunk_size: int) -> int:
    chunks = _as_chunk_tuple(tensor)
    if not chunks:
        return 0
    return max(chunk_size - chunks[-1].sequence_length, 0)


def _append_quantized_chunk(
    existing: Optional[CompressedTensorLike],
    new: CompressedTensor,
    *,
    chunk_size: int,
) -> CompressedTensorLike:
    chunks = list(_as_chunk_tuple(existing))
    if not chunks:
        return new

    tail_capacity = max(chunk_size - chunks[-1].sequence_length, 0)
    if tail_capacity >= new.sequence_length and tail_capacity > 0:
        chunks[-1] = _merge_compressed_chunks(chunks[-1], new)
    else:
        chunks.append(new)
    return _wrap_chunks(tuple(chunks))


def _slice_single_chunk(tensor: CompressedTensor, length: int) -> Optional[CompressedTensor]:
    if length <= 0:
        return None
    if length >= tensor.sequence_length:
        return tensor
    states = []
    for state in tensor.subset_states:
        states.append(
            state.__class__(
                packed_codes=state.packed_codes[:, :, :length, :],
                norms=state.norms[:, :, :length],
                residual_signs=None
                if state.residual_signs is None
                else state.residual_signs[:, :, :length, :],
                residual_norms=None
                if state.residual_norms is None
                else state.residual_norms[:, :, :length],
            )
        )
    base_shape = tensor.original_shape
    sliced_shape = (base_shape[0], base_shape[1], length, base_shape[3])
    return CompressedTensor(subset_states=tuple(states), original_shape=sliced_shape)


def _slice_compressed(tensor: Optional[CompressedTensorLike], length: int) -> Optional[CompressedTensorLike]:
    if tensor is None:
        return None
    if length <= 0:
        return None

    kept = []
    remaining = length
    for chunk in _as_chunk_tuple(tensor):
        if remaining <= 0:
            break
        if remaining >= chunk.sequence_length:
            kept.append(chunk)
            remaining -= chunk.sequence_length
            continue
        sliced = _slice_single_chunk(chunk, remaining)
        if sliced is not None:
            kept.append(sliced)
        remaining = 0
        break
    return _wrap_chunks(tuple(kept))


@dataclass
class TurboQuantCacheState:
    keys: Optional[CompressedTensorLike]
    values: Optional[CompressedTensorLike]


class TurboQuantKVCache:
    step = 256

    def __init__(self, config: TurboQuantConfig, setup: Optional[SharedTurboQuantSetup] = None):
        self.config = config
        self.setup = setup or SharedTurboQuantSetup.from_config(config)
        self.keys: Optional[CompressedTensorLike] = None
        self.values: Optional[CompressedTensorLike] = None
        self.offset = 0
        self.perf_stats: dict[str, float] = {}

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        update_start = perf_counter()
        total_tokens = int(keys.shape[2])
        cursor = 0

        while cursor < total_tokens:
            tail_capacity = _tail_capacity(self.keys, self.step)
            chunk_tokens = tail_capacity if tail_capacity > 0 else self.step
            chunk_tokens = min(chunk_tokens, total_tokens - cursor)
            key_slice = keys[:, :, cursor : cursor + chunk_tokens, :]
            value_slice = values[:, :, cursor : cursor + chunk_tokens, :]
            quantized_keys = quantize_tensor(key_slice, self.setup, with_qjl=self.config.qjl_enabled)
            quantized_values = quantize_tensor(value_slice, self.setup, with_qjl=False)
            self.keys = _append_quantized_chunk(self.keys, quantized_keys, chunk_size=self.step)
            self.values = _append_quantized_chunk(self.values, quantized_values, chunk_size=self.step)
            cursor += chunk_tokens

        self.offset += total_tokens
        self.perf_stats["cache_update_seconds"] = self.perf_stats.get("cache_update_seconds", 0.0) + (
            perf_counter() - update_start
        )
        return self.keys, self.values

    @property
    def state(self) -> TurboQuantCacheState:
        return TurboQuantCacheState(keys=self.keys, values=self.values)

    @property
    def meta_state(self):
        return (
            str(self.offset),
            self.config.mode,
            str(self.config.head_dim),
            str(self.config.seed),
            str(self.config.core_bits),
            self.config.preset_name or "",
            str(int(self.config.qjl_enabled)),
            str(self.config.qjl_dim),
        )

    @meta_state.setter
    def meta_state(self, value):
        self.offset = int(value[0])

    def __len__(self) -> int:
        return self.offset

    def is_trimmable(self):
        return True

    def trim(self, num_tokens: int):
        num_tokens = min(num_tokens, self.offset)
        self.offset -= num_tokens
        self.keys = _slice_compressed(self.keys, self.offset)
        self.values = _slice_compressed(self.values, self.offset)
        return num_tokens

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def observed_nbytes(self) -> int:
        total = 0
        if self.keys is not None:
            total += self.keys.nbytes
        if self.values is not None:
            total += self.values.nbytes
        return total
