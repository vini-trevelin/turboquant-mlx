from __future__ import annotations

import numpy as np


def packed_length(num_codes: int, bits: int) -> int:
    return (num_codes * bits + 7) // 8


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    if bits < 1 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    values = np.asarray(codes, dtype=np.uint16)
    rows = values.reshape(-1, values.shape[-1])
    out = np.zeros((rows.shape[0], packed_length(rows.shape[1], bits)), dtype=np.uint8)
    mask = (1 << bits) - 1

    for row_idx, row in enumerate(rows):
        bit_cursor = 0
        for code in row:
            value = int(code) & mask
            byte_idx = bit_cursor // 8
            shift = bit_cursor % 8
            out[row_idx, byte_idx] |= (value << shift) & 0xFF
            spill = shift + bits - 8
            if spill > 0:
                out[row_idx, byte_idx + 1] |= (value >> (bits - spill)) & 0xFF
            bit_cursor += bits

    return out.reshape(*values.shape[:-1], out.shape[-1])


def unpack_codes(packed: np.ndarray, bits: int, num_codes: int) -> np.ndarray:
    if bits < 1 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    packed = np.asarray(packed, dtype=np.uint8)
    rows = packed.reshape(-1, packed.shape[-1])
    out = np.zeros((rows.shape[0], num_codes), dtype=np.uint8)
    mask = (1 << bits) - 1

    for row_idx, row in enumerate(rows):
        bit_cursor = 0
        for code_idx in range(num_codes):
            byte_idx = bit_cursor // 8
            shift = bit_cursor % 8
            value = int(row[byte_idx]) >> shift
            spill = shift + bits - 8
            if spill > 0 and byte_idx + 1 < row.shape[0]:
                value |= int(row[byte_idx + 1]) << (8 - shift)
            out[row_idx, code_idx] = value & mask
            bit_cursor += bits

    return out.reshape(*packed.shape[:-1], num_codes)


def pack_signs(signs: np.ndarray) -> np.ndarray:
    signs = np.asarray(signs, dtype=np.uint8)
    return pack_codes(signs, bits=1)


def unpack_signs(packed: np.ndarray, num_codes: int) -> np.ndarray:
    return unpack_codes(packed, bits=1, num_codes=num_codes)

