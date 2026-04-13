from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from mlx_lm.models.cache import create_attention_mask

from .config import TurboQuantConfig
from .quantizer import CompressedTensor, SharedTurboQuantSetup, quantize_tensor


def _concat_optional(existing: Optional[mx.array], new: Optional[mx.array], axis: int) -> Optional[mx.array]:
    if new is None:
        return existing
    if existing is None:
        return new
    return mx.concatenate([existing, new], axis=axis)


def _append_compressed(existing: Optional[CompressedTensor], new: CompressedTensor) -> CompressedTensor:
    if existing is None:
        return new

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
    return CompressedTensor(subset_states=tuple(states), original_shape=new.original_shape)


def _slice_compressed(tensor: Optional[CompressedTensor], length: int) -> Optional[CompressedTensor]:
    if tensor is None:
        return None
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
    return CompressedTensor(subset_states=tuple(states), original_shape=tensor.original_shape)


@dataclass
class TurboQuantCacheState:
    keys: Optional[CompressedTensor]
    values: Optional[CompressedTensor]


class TurboQuantKVCache:
    step = 256

    def __init__(self, config: TurboQuantConfig, setup: Optional[SharedTurboQuantSetup] = None):
        self.config = config
        self.setup = setup or SharedTurboQuantSetup.from_config(config)
        self.keys: Optional[CompressedTensor] = None
        self.values: Optional[CompressedTensor] = None
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        quantized_keys = quantize_tensor(keys, self.setup, with_qjl=self.config.qjl_enabled)
        quantized_values = quantize_tensor(values, self.setup, with_qjl=False)
        self.keys = _append_compressed(self.keys, quantized_keys)
        self.values = _append_compressed(self.values, quantized_values)
        self.offset += keys.shape[2]
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

