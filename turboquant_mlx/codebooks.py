from __future__ import annotations

from functools import lru_cache

import numpy as np

TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _coordinate_pdf(grid: np.ndarray, dim: int) -> np.ndarray:
    if dim < 2:
        raise ValueError("dim must be >= 2")
    exponent = (dim - 3.0) / 2.0
    safe = np.clip(1.0 - np.square(grid), 1e-12, None)
    values = np.power(safe, exponent, dtype=np.float64)
    area = TRAPEZOID(values, grid)
    return values / area


@lru_cache(maxsize=128)
def beta_lloyd_max_codebook(
    dim: int,
    bits: int,
    grid_size: int = 32769,
    max_iters: int = 80,
    tol: float = 1e-8,
) -> np.ndarray:
    levels = 1 << bits
    grid = np.linspace(-1.0 + 1e-6, 1.0 - 1e-6, grid_size, dtype=np.float64)
    pdf = _coordinate_pdf(grid, dim)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]

    targets = (np.arange(levels, dtype=np.float64) + 0.5) / levels
    centroids = np.interp(targets, cdf, grid)

    for _ in range(max_iters):
        boundaries = np.empty(levels + 1, dtype=np.float64)
        boundaries[0], boundaries[-1] = -1.0, 1.0
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])

        new_centroids = centroids.copy()
        for level in range(levels):
            left = boundaries[level]
            right = boundaries[level + 1]
            mask = (grid >= left) & (grid <= right)
            cell_grid = grid[mask]
            cell_pdf = pdf[mask]
            mass = TRAPEZOID(cell_pdf, cell_grid)
            if mass <= 0:
                continue
            new_centroids[level] = TRAPEZOID(cell_grid * cell_pdf, cell_grid) / mass

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids.astype(np.float32)


def codebook_thresholds(codebook: np.ndarray) -> np.ndarray:
    codebook = np.asarray(codebook, dtype=np.float32)
    return ((codebook[:-1] + codebook[1:]) * 0.5).astype(np.float32)
