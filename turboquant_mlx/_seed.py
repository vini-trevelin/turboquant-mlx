"""Single entry point for seeding every RNG the eval harness touches."""

from __future__ import annotations

import random

import mlx.core as mx
import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)
