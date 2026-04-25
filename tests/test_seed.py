from __future__ import annotations

import random

import mlx.core as mx
import numpy as np

from turboquant_mlx._seed import set_global_seed


def test_set_global_seed_makes_python_random_deterministic():
    set_global_seed(123)
    a = [random.random() for _ in range(5)]
    set_global_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_global_seed_makes_numpy_deterministic():
    set_global_seed(7)
    a = np.random.standard_normal(8).tolist()
    set_global_seed(7)
    b = np.random.standard_normal(8).tolist()
    assert a == b


def test_set_global_seed_makes_mlx_deterministic():
    set_global_seed(42)
    a = mx.random.normal(shape=(8,)).tolist()
    set_global_seed(42)
    b = mx.random.normal(shape=(8,)).tolist()
    assert a == b
