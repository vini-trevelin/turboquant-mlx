from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import sqrt
from time import perf_counter
from typing import Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from .codebooks import beta_lloyd_max_codebook, codebook_thresholds
from .config import TurboQuantConfig
from .packing import pack_codes, pack_signs, packed_length
from .rotation import orthogonal_matrix

EPS = 1e-8
QJL_SCALE = sqrt(np.pi / 2.0)


def _record_perf(perf_stats: Optional[dict], key: str, duration: float) -> None:
    if perf_stats is None:
        return
    perf_stats[key] = perf_stats.get(key, 0.0) + duration


@dataclass(frozen=True)
class PackedDecodePlan:
    bits: int
    num_codes: int
    byte_indices: np.ndarray
    next_byte_indices: np.ndarray
    shifts: np.ndarray
    next_shifts: np.ndarray
    has_spill: np.ndarray
    mask: int

    @classmethod
    def build(cls, num_codes: int, bits: int) -> "PackedDecodePlan":
        bit_positions = np.arange(num_codes, dtype=np.int32) * bits
        byte_indices = bit_positions // 8
        shifts = (bit_positions % 8).astype(np.int32)
        next_byte_indices = np.minimum(byte_indices + 1, packed_length(num_codes, bits) - 1)
        next_shifts = (8 - shifts).astype(np.int32)
        has_spill = (shifts + bits > 8)
        return cls(
            bits=bits,
            num_codes=num_codes,
            byte_indices=byte_indices,
            next_byte_indices=next_byte_indices.astype(np.int32),
            shifts=shifts,
            next_shifts=next_shifts,
            has_spill=has_spill,
            mask=(1 << bits) - 1,
        )

    @property
    def needs_spill(self) -> bool:
        return bool(np.any(self.has_spill))

    @cached_property
    def byte_indices_mx(self) -> mx.array:
        return mx.array(self.byte_indices, dtype=mx.int32)

    @cached_property
    def next_byte_indices_mx(self) -> mx.array:
        return mx.array(self.next_byte_indices, dtype=mx.int32)

    @cached_property
    def shifts_mx(self) -> mx.array:
        return mx.array(self.shifts, dtype=mx.uint16)

    @cached_property
    def next_shifts_mx(self) -> mx.array:
        return mx.array(self.next_shifts, dtype=mx.uint16)

    @cached_property
    def has_spill_mx(self) -> mx.array:
        return mx.array(self.has_spill, dtype=mx.bool_)

    @cached_property
    def mask_mx(self) -> mx.array:
        return mx.array(self.mask, dtype=mx.uint16)


@dataclass(frozen=True)
class SubsetSetup:
    spec: object
    rotation: np.ndarray
    inverse_rotation: np.ndarray
    codebook: np.ndarray
    thresholds: np.ndarray
    code_decode: PackedDecodePlan
    projection: Optional[np.ndarray] = None
    sign_decode: Optional[PackedDecodePlan] = None

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

    @property
    def chunk_count(self) -> int:
        return 1


@dataclass(frozen=True)
class ChunkedCompressedTensor:
    chunks: Tuple[CompressedTensor, ...]

    @property
    def sequence_length(self) -> int:
        return sum(chunk.sequence_length for chunk in self.chunks)

    @property
    def nbytes(self) -> int:
        return sum(chunk.nbytes for chunk in self.chunks)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def original_shape(self) -> Tuple[int, ...]:
        if not self.chunks:
            return ()
        base = self.chunks[0].original_shape
        return (base[0], base[1], self.sequence_length, base[3])


CompressedTensorLike = Union[CompressedTensor, ChunkedCompressedTensor]


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
            sign_decode = None
            if spec.qjl_dim > 0:
                rng = np.random.default_rng(config.seed + 1000 + subset_idx)
                projection = rng.standard_normal((spec.dim, spec.qjl_dim)).astype(np.float32)
                sign_decode = PackedDecodePlan.build(spec.qjl_dim, bits=1)
            subsets.append(
                SubsetSetup(
                    spec=spec,
                    rotation=rotation,
                    inverse_rotation=rotation.T.astype(np.float32),
                    codebook=codebook,
                    thresholds=codebook_thresholds(codebook),
                    code_decode=PackedDecodePlan.build(spec.dim, spec.bits),
                    projection=projection,
                    sign_decode=sign_decode,
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


def _iter_chunks(compressed: CompressedTensorLike) -> Tuple[CompressedTensor, ...]:
    if isinstance(compressed, ChunkedCompressedTensor):
        return compressed.chunks
    return (compressed,)


def _first_chunk(compressed: CompressedTensorLike) -> CompressedTensor:
    return _iter_chunks(compressed)[0]


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


def _decode_packed_with_plan(packed: mx.array, plan: PackedDecodePlan) -> mx.array:
    packed_u16 = packed.astype(mx.uint16)
    values = mx.right_shift(mx.take(packed_u16, plan.byte_indices_mx, axis=-1), plan.shifts_mx)
    if plan.needs_spill:
        spill = mx.left_shift(mx.take(packed_u16, plan.next_byte_indices_mx, axis=-1), plan.next_shifts_mx)
        values = mx.where(plan.has_spill_mx, mx.bitwise_or(values, spill), values)
    return mx.bitwise_and(values, plan.mask_mx).astype(mx.int32)


def decode_packed_codes_mx(packed: mx.array, *, bits: int, num_codes: int) -> mx.array:
    return _decode_packed_with_plan(packed, PackedDecodePlan.build(num_codes, bits))


def decode_packed_signs_mx(packed: mx.array, *, num_codes: int) -> mx.array:
    return _decode_packed_with_plan(packed, PackedDecodePlan.build(num_codes, 1)).astype(mx.uint8)


def _decode_rotated_subset(
    state: PackedSubsetState,
    subset: SubsetSetup,
    *,
    perf_stats: Optional[dict] = None,
) -> mx.array:
    start = perf_counter()
    codes = _decode_packed_with_plan(state.packed_codes, subset.code_decode)
    rotated = mx.take(subset.codebook_mx, codes, axis=0)
    _record_perf(perf_stats, "packed_decode_seconds", perf_counter() - start)
    return rotated


def _decode_signs_subset(
    state: PackedSubsetState,
    subset: SubsetSetup,
    *,
    perf_stats: Optional[dict] = None,
) -> mx.array:
    if state.residual_signs is None or subset.sign_decode is None:
        raise ValueError("QJL sign decoding requested without residual sign state")
    start = perf_counter()
    decoded = _decode_packed_with_plan(state.residual_signs, subset.sign_decode).astype(mx.float32)
    signs = decoded * 2.0 - 1.0
    _record_perf(perf_stats, "packed_decode_seconds", perf_counter() - start)
    return signs


def _dequantize_single_chunk(
    compressed: CompressedTensor,
    setup: SharedTurboQuantSetup,
) -> mx.array:
    subset_outputs = []
    for subset, state in zip(setup.subsets, compressed.subset_states):
        rotated = _decode_rotated_subset(state, subset)
        dense = mx.matmul(rotated, subset.inverse_rotation_mx)
        scaled = mx.expand_dims(state.norms.astype(mx.float32), axis=-1) * dense
        subset_outputs.append(scaled)

    if len(subset_outputs) == 1:
        return subset_outputs[0]

    combined = mx.concatenate(subset_outputs, axis=-1)
    return mx.take(combined, setup.inverse_permutation_mx, axis=-1)


def dequantize_tensor(
    compressed: CompressedTensorLike,
    setup: SharedTurboQuantSetup,
) -> mx.array:
    chunks = [_dequantize_single_chunk(chunk, setup) for chunk in _iter_chunks(compressed)]
    if len(chunks) == 1:
        return chunks[0]
    return mx.concatenate(chunks, axis=2)


def _group_queries(queries: mx.array, n_kv_heads: int) -> tuple[mx.array, int]:
    batch, n_q_heads, q_len, dim = queries.shape
    repeats = n_q_heads // n_kv_heads
    grouped = queries.reshape(batch, n_kv_heads, repeats, q_len, dim)
    return grouped, repeats


def score_queries_against_keys(
    queries: mx.array,
    compressed_keys: CompressedTensorLike,
    setup: SharedTurboQuantSetup,
    *,
    scale: float,
    apply_qjl: bool,
    perf_stats: Optional[dict] = None,
) -> mx.array:
    first_chunk = _first_chunk(compressed_keys)
    n_kv_heads = int(first_chunk.subset_states[0].norms.shape[1])
    grouped_queries, repeats = _group_queries(queries, n_kv_heads)

    prepared_queries = []
    for subset in setup.subsets:
        query_subset = mx.take(grouped_queries, subset.indices_mx, axis=-1)
        rotated_query = mx.matmul(query_subset.astype(mx.float32), subset.rotation_mx)
        projected_query = None
        if apply_qjl and subset.projection_mx is not None:
            projected_query = mx.matmul(rotated_query, subset.projection_mx)
        prepared_queries.append((rotated_query, projected_query))

    chunk_scores = []
    for chunk in _iter_chunks(compressed_keys):
        score_chunk = None
        for subset, state, prepared in zip(setup.subsets, chunk.subset_states, prepared_queries):
            rotated_query, projected_query = prepared
            rotated_keys = _decode_rotated_subset(state, subset, perf_stats=perf_stats)
            scaled_rotated_keys = mx.expand_dims(state.norms.astype(mx.float32), axis=-1) * rotated_keys

            score_start = perf_counter()
            contribution = mx.matmul(
                rotated_query,
                mx.expand_dims(scaled_rotated_keys.transpose(0, 1, 3, 2), axis=2),
            )

            if apply_qjl and subset.projection_mx is not None and state.residual_signs is not None:
                signs = _decode_signs_subset(state, subset, perf_stats=perf_stats)
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

            _record_perf(perf_stats, "key_score_seconds", perf_counter() - score_start)
            score_chunk = contribution if score_chunk is None else score_chunk + contribution
        chunk_scores.append(score_chunk)

    scores = chunk_scores[0] if len(chunk_scores) == 1 else mx.concatenate(chunk_scores, axis=-1)
    scores = scores * scale
    if repeats == 1:
        return scores[:, :, 0, :, :]
    batch, n_kv, _, q_len, kv_len = scores.shape
    return scores.reshape(batch, n_kv * repeats, q_len, kv_len)


def apply_attention_to_values(
    attention: mx.array,
    compressed_values: CompressedTensorLike,
    setup: SharedTurboQuantSetup,
    *,
    perf_stats: Optional[dict] = None,
) -> mx.array:
    first_chunk = _first_chunk(compressed_values)
    n_kv_heads = int(first_chunk.subset_states[0].norms.shape[1])
    grouped_attention, repeats = _group_queries(attention, n_kv_heads)
    subset_weighted_outputs = [None] * len(setup.subsets)

    start_idx = 0
    for chunk in _iter_chunks(compressed_values):
        end_idx = start_idx + chunk.sequence_length
        attention_chunk = grouped_attention[..., start_idx:end_idx]
        start_idx = end_idx
        for subset_idx, (subset, state) in enumerate(zip(setup.subsets, chunk.subset_states)):
            rotated_values = _decode_rotated_subset(state, subset, perf_stats=perf_stats)
            scaled_rotated_values = mx.expand_dims(state.norms.astype(mx.float32), axis=-1) * rotated_values
            apply_start = perf_counter()
            weighted_rotated = mx.matmul(attention_chunk, mx.expand_dims(scaled_rotated_values, axis=2))
            subset_weighted_outputs[subset_idx] = (
                weighted_rotated
                if subset_weighted_outputs[subset_idx] is None
                else subset_weighted_outputs[subset_idx] + weighted_rotated
            )
            _record_perf(perf_stats, "value_apply_seconds", perf_counter() - apply_start)

    subset_outputs = []
    for subset, weighted_rotated in zip(setup.subsets, subset_weighted_outputs):
        rotate_start = perf_counter()
        subset_outputs.append(mx.matmul(weighted_rotated, subset.inverse_rotation_mx))
        _record_perf(perf_stats, "value_apply_seconds", perf_counter() - rotate_start)

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
