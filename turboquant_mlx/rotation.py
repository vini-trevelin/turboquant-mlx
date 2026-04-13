from __future__ import annotations

import numpy as np


def orthogonal_matrix(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((dim, dim), dtype=np.float32)
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q * signs
    return q.astype(np.float32)

