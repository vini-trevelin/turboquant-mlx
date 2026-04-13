from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import sqrt
from typing import Iterable, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from .codebooks import beta_lloyd_max_codebook, codebook_thresholds
from .config import SubsetSpec, TurboQuantConfig
from .packing import pack_codes, pack_signs, unpack_codes, unpack_signs
from .rotation import orthogonal_matrix

EPS = 1e-8
QJL_SCALE = sqrt(np.pi / 2.0)


@dataclass(frozen=True)
class SubsetSetup:
    spec: SubsetSpec
    rotation: np.ndarray
    inverse_rotation: np.ndarray
    codebook: np.ndarray
    thresholds: np.ndarray
    projection: Optional[np.ndarray] = None

    @cached_property
    def indices_mx(self) -> mx.array:
        return mx.array(self.spec.indices, dtype=mx.int32)

    @cached_property
    def rotation_mx(self) -> mx.array:
        return mx.array(self.rotation, dtype=mx.float32)

    @cached_property
    def inverse_rotation_mx(self) -> mx.array:
        return mx.array(self.inverse_rotation, dtype=mx.float32)

    @cached_property
    def codebook_mx(self) -> mx.array:
        return mx.array(self.codebook, dtype=mx.float32)

    @cached_property
    def projection_mx(self) -> Optional[mx.array]:
        if self.projection is None:
            return None
        return mx.array(self.projection, dtype=mx.float32)


@dataclass(frozen=True)
class PackedSubsetState:
    packed_codes: mx.array
    norms: mx.array
    residual_signs: Optional[mx.array] = None
    residual_norms: Optional[mx.array] = None

    @property
    def sequence_length(self) -> int:
        return int(self.norms.shape[2])

    @property
    def nbytes(self) -> int:
        total = self.packed_codes.nbytes + self.norms.nbytes
        if self.residual_signs is not None:
            total += self.residual_signs.nbytes
        if self.residual_norms is not None:
            total += self.residual_norms.nbytes
        return total


@dataclass(frozen=True)
class CompressedTensor:
    subset_states: Tuple[PackedSubsetState, ...]
    original_shape: Tuple[int, ...]

    @property
    def sequence_length(self) -> int:
        if not self.subset_states:
            return 0
        return self.subset_states[0].sequence_length

    @property
    def nbytes(self) -> int:
        return sum(state.nbytes for state in self.subset_states)


@dataclass(frozen=True)
class SharedTurboQuantSetup:
    config: TurboQuantConfig
    subsets: Tuple[SubsetSetup, ...]
    ordered_indices: Tuple[int, ...]
    inverse_permutation: Tuple[int, ...]

    @classmethod
    def from_config(cls, config: TurboQuantConfig) -> "SharedTurboQuantSetup":
        subsets = []
        ordered_indices = []
        for subset_idx, spec in enumerate(config.subset_specs()):
            rotation = orthogonal_matrix(spec.dim, seed=config.seed + subset_idx)
            codebook = beta_lloyd_max_codebook(spec.dim, spec.bits)
            projection = None
            if spec.qjl_dim > 0:
                rng = np.random.default_rng(config.seed + 1000 + subset_idx)
                projection = rng.standard_normal((spec.dim, spec.qjl_dim)).astype(np.float32)
            subsets.append(
                SubsetSetup(
                    spec=spec,
                    rotation=rotation,
                    inverse_rotation=rotation.T.astype(np.float32),
                    codebook=codebook,
                    thresholds=codebook_thresholds(codebook),
                    projection=projection,
                )
            )
            ordered_indices.extend(spec.indices)

        inverse_permutation = tuple(np.argsort(np.asarray(ordered_indices, dtype=np.int32)).tolist())
        return cls(
            config=config,
            subsets=tuple(subsets),
            ordered_indices=tuple(ordered_indices),
            inverse_permutation=inverse_permutation,
        )

    @cached_property
    def inverse_permutation_mx(self) -> mx.array:
        return mx.array(self.inverse_permutation, dtype=mx.int32)


def _to_numpy(array: mx.array) -> np.ndarray:
    return np.asarray(array)


def _quantize_subset(
    vectors: np.ndarray,
    subset: SubsetSetup,
    with_qjl: bool,
) -> PackedSubsetState:
    values = np.take(vectors, subset.spec.indices, axis=-1).astype(np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    safe_norms = np.maximum(norms, EPS)
    unit = np.divide(values, safe_norms, out=np.zeros_like(values), where=norms > EPS)
    rotated = np.einsum("...d,dm->...m", unit, subset.rotation, optimize=True)

    codes = np.digitize(rotated, subset.thresholds, right=False).astype(np.uint8)
    packed_codes = pack_codes(codes, bits=subset.spec.bits)

    residual_signs = None
    residual_norms = None
    if with_qjl and subset.projection is not None:
        approx_rotated = subset.codebook[codes]
        approx_unit = np.einsum(
            "...m,md->...d",
            approx_rotated,
            subset.inverse_rotation,
            optimize=True,
        )
        residual = unit - approx_unit
        residual_norms_np = np.linalg.norm(residual, axis=-1, keepdims=True)
        safe_residual_norms = np.maximum(residual_norms_np, EPS)
        residual_unit = np.divide(
            residual,
            safe_residual_norms,
            out=np.zeros_like(residual),
            where=residual_norms_np > EPS,
        )
        residual_unit = np.nan_to_num(residual_unit, nan=0.0, posinf=0.0, neginf=0.0)
        projected = np.einsum("...d,dm->...m", residual_unit, subset.projection, optimize=True)
        signs = (projected >= 0).astype(np.uint8)
        residual_signs = mx.array(pack_signs(signs), dtype=mx.uint8)
        residual_norms = mx.array(residual_norms_np.squeeze(-1).astype(np.float16))

    return PackedSubsetState(
        packed_codes=mx.array(packed_codes, dtype=mx.uint8),
        norms=mx.array(norms.squeeze(-1).astype(np.float16)),
        residual_signs=residual_signs,
        residual_norms=residual_norms,
    )


def quantize_tensor(
    tensor: mx.array,
    setup: SharedTurboQuantSetup,
    with_qjl: bool = False,
) -> CompressedTensor:
    tensor_np = _to_numpy(tensor)
    subset_states = tuple(
        _quantize_subset(tensor_np, subset, with_qjl=with_qjl) for subset in setup.subsets
    )
    return CompressedTensor(subset_states=subset_states, original_shape=tuple(tensor.shape))


def _decompress_rotated_subset(state: PackedSubsetState, subset: SubsetSetup) -> mx.array:
    codes = unpack_codes(
        _to_numpy(state.packed_codes),
        bits=subset.spec.bits,
        num_codes=subset.spec.dim,
    )
    rotated = subset.codebook[codes]
    return mx.array(rotated.astype(np.float32))


def dequantize_tensor(
    compressed: CompressedTensor,
    setup: SharedTurboQuantSetup,
) -> mx.array:
    subset_outputs = []
    for subset, state in zip(setup.subsets, compressed.subset_states):
        rotated = _decompress_rotated_subset(state, subset)
        dense = mx.matmul(rotated, subset.inverse_rotation_mx)
        scaled = mx.expand_dims(state.norms.astype(mx.float32), axis=-1) * dense
        subset_outputs.append(scaled)

    if len(subset_outputs) == 1:
        return subset_outputs[0]

    combined = mx.concatenate(subset_outputs, axis=-1)
    return mx.take(combined, setup.inverse_permutation_mx, axis=-1)


def _group_queries(queries: mx.array, n_kv_heads: int) -> tuple[mx.array, int]:
    batch, n_q_heads, q_len, dim = queries.shape
    repeats = n_q_heads // n_kv_heads
    grouped = queries.reshape(batch, n_kv_heads, repeats, q_len, dim)
    return grouped, repeats


def score_queries_against_keys(
    queries: mx.array,
    compressed_keys: CompressedTensor,
    setup: SharedTurboQuantSetup,
    *,
    scale: float,
    apply_qjl: bool,
) -> mx.array:
    n_kv_heads = int(compressed_keys.subset_states[0].norms.shape[1])
    grouped_queries, repeats = _group_queries(queries, n_kv_heads)
    scores = None

    for subset, state in zip(setup.subsets, compressed_keys.subset_states):
        query_subset = mx.take(grouped_queries, subset.indices_mx, axis=-1)
        rotated_query = mx.matmul(query_subset.astype(mx.float32), subset.rotation_mx)

        rotated_keys = _decompress_rotated_subset(state, subset)
        scaled_rotated_keys = mx.expand_dims(state.norms.astype(mx.float32), axis=-1) * rotated_keys

        contribution = mx.matmul(
            rotated_query,
            mx.expand_dims(scaled_rotated_keys.transpose(0, 1, 3, 2), axis=2),
        )

        if apply_qjl and subset.projection_mx is not None and state.residual_signs is not None:
            packed_signs = unpack_signs(
                _to_numpy(state.residual_signs),
                num_codes=subset.spec.qjl_dim,
            ).astype(np.float32)
            signs = mx.array(packed_signs * 2.0 - 1.0, dtype=mx.float32)
            projected_query = mx.matmul(rotated_query, subset.projection_mx)
            correction = mx.matmul(
                projected_query,
                mx.expand_dims(signs.transpose(0, 1, 3, 2), axis=2),
            )
            residual_scale = (
                mx.expand_dims(state.norms.astype(mx.float32), axis=-1)
                * mx.expand_dims(state.residual_norms.astype(mx.float32), axis=-1)
            )
            contribution = contribution + (
                QJL_SCALE / subset.spec.qjl_dim
            ) * correction * mx.expand_dims(residual_scale.transpose(0, 1, 3, 2), axis=2)

        scores = contribution if scores is None else scores + contribution

    scores = scores * scale
    if repeats == 1:
        return scores[:, :, 0, :, :]
    batch, n_kv, _, q_len, kv_len = scores.shape
    return scores.reshape(batch, n_kv * repeats, q_len, kv_len)


def apply_attention_to_values(
    attention: mx.array,
    compressed_values: CompressedTensor,
    setup: SharedTurboQuantSetup,
) -> mx.array:
    n_kv_heads = int(compressed_values.subset_states[0].norms.shape[1])
    grouped_attention, repeats = _group_queries(attention, n_kv_heads)
    subset_outputs = []

    for subset, state in zip(setup.subsets, compressed_values.subset_states):
        rotated_values = _decompress_rotated_subset(state, subset)
        scaled_rotated_values = mx.expand_dims(state.norms.astype(mx.float32), axis=-1) * rotated_values
        weighted_rotated = mx.matmul(grouped_attention, mx.expand_dims(scaled_rotated_values, axis=2))
        subset_outputs.append(mx.matmul(weighted_rotated, subset.inverse_rotation_mx))

    combined = subset_outputs[0] if len(subset_outputs) == 1 else mx.concatenate(subset_outputs, axis=-1)
    if len(subset_outputs) > 1:
        combined = mx.take(combined, setup.inverse_permutation_mx, axis=-1)

    batch, n_kv_heads, repeats, q_len, dim = combined.shape
    return combined.reshape(batch, n_kv_heads * repeats, q_len, dim)


def observed_bytes_per_token(setup: SharedTurboQuantSetup, include_qjl: bool) -> int:
    total = 0
    for subset in setup.subsets:
        total += (subset.spec.bits * subset.spec.dim + 7) // 8
        total += 2
        if include_qjl and subset.spec.qjl_dim > 0:
            total += (subset.spec.qjl_dim + 7) // 8
            total += 2
    return total
