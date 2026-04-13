import numpy as np

from turboquant_mlx.codebooks import beta_lloyd_max_codebook
from turboquant_mlx.packing import pack_codes, pack_signs, unpack_codes, unpack_signs
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

