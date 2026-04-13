import numpy as np
import mlx.core as mx

from turboquant_mlx.codebooks import beta_lloyd_max_codebook
from turboquant_mlx.packing import pack_codes, pack_signs, unpack_codes, unpack_signs
from turboquant_mlx.quantizer import decode_packed_codes_mx, decode_packed_signs_mx
from turboquant_mlx.rotation import orthogonal_matrix


def test_rotation_is_deterministic_and_orthogonal():
    matrix_a = orthogonal_matrix(8, seed=7)
    matrix_b = orthogonal_matrix(8, seed=7)
    assert np.allclose(matrix_a, matrix_b)
    assert np.allclose(matrix_a.T @ matrix_a, np.eye(8), atol=1e-4)


def test_beta_lloyd_max_codebook_is_monotonic():
    codebook = beta_lloyd_max_codebook(8, 3)
    assert codebook.shape == (8,)
    assert np.all(np.diff(codebook) > 0)
    assert np.isclose(codebook[0], -codebook[-1], atol=1e-2)


def test_pack_unpack_codes_roundtrip():
    codes = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uint8)
    packed = pack_codes(codes, bits=3)
    unpacked = unpack_codes(packed, bits=3, num_codes=codes.shape[-1])
    assert np.array_equal(codes, unpacked)


def test_pack_unpack_signs_roundtrip():
    signs = np.array([[0, 1, 1, 0, 1, 0, 0, 1, 1]], dtype=np.uint8)
    packed = pack_signs(signs)
    unpacked = unpack_signs(packed, num_codes=signs.shape[-1])
    assert np.array_equal(signs, unpacked)


def test_mlx_decode_matches_numpy_unpack_for_used_bit_widths():
    rng = np.random.default_rng(0)
    for bits in (1, 2, 3, 4, 5):
        codes = rng.integers(0, 1 << bits, size=(2, 3, 7), dtype=np.uint8)
        packed = pack_codes(codes, bits=bits)
        decoded = decode_packed_codes_mx(
            mx.array(packed, dtype=mx.uint8),
            bits=bits,
            num_codes=codes.shape[-1],
        )
        expected = unpack_codes(packed, bits=bits, num_codes=codes.shape[-1])
        assert np.array_equal(np.asarray(decoded), expected.astype(np.int32))


def test_mlx_sign_decode_matches_numpy_unpack():
    rng = np.random.default_rng(1)
    signs = rng.integers(0, 2, size=(2, 4, 9), dtype=np.uint8)
    packed = pack_signs(signs)
    decoded = decode_packed_signs_mx(mx.array(packed, dtype=mx.uint8), num_codes=signs.shape[-1])
    expected = unpack_signs(packed, num_codes=signs.shape[-1])
    assert np.array_equal(np.asarray(decoded), expected)
