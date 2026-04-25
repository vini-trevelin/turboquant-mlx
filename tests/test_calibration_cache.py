from __future__ import annotations

import json

from turboquant_mlx.calibration import _calibration_cache_key, calibrate_outlier_mask_cached
from turboquant_mlx.config import CalibrationArtifact


def test_cache_key_is_stable_for_identical_inputs():
    a = _calibration_cache_key(
        model_path="m",
        outlier_count=32,
        quantile=99.9,
        texts=["alpha", "beta"],
        max_examples=2,
    )
    b = _calibration_cache_key(
        model_path="m",
        outlier_count=32,
        quantile=99.9,
        texts=["alpha", "beta"],
        max_examples=2,
    )
    assert a == b


def test_cache_key_changes_when_any_input_changes():
    base = dict(model_path="m", outlier_count=32, quantile=99.9, texts=["alpha"], max_examples=1)
    base_key = _calibration_cache_key(**base)
    assert _calibration_cache_key(**{**base, "model_path": "n"}) != base_key
    assert _calibration_cache_key(**{**base, "outlier_count": 16}) != base_key
    assert _calibration_cache_key(**{**base, "quantile": 99.5}) != base_key
    assert _calibration_cache_key(**{**base, "texts": ["alpha2"]}) != base_key
    assert _calibration_cache_key(**{**base, "max_examples": 2}) != base_key


def test_cached_calibration_loads_from_disk_without_calling_calibrate(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    key = _calibration_cache_key(
        model_path="fake",
        outlier_count=4,
        quantile=99.0,
        texts=["x"],
        max_examples=1,
    )
    artifact = CalibrationArtifact(head_dim=8, outlier_indices=(0, 1, 2, 3), quantile=99.0)
    artifact.save(cache_dir / f"{key}.json")

    def boom(*args, **kwargs):
        raise AssertionError("calibrate_outlier_mask must not be called on a cache hit")

    monkeypatch.setattr("turboquant_mlx.calibration.calibrate_outlier_mask", boom)

    loaded = calibrate_outlier_mask_cached(
        "fake",
        ["x"],
        outlier_count=4,
        quantile=99.0,
        max_examples=1,
        cache_dir=cache_dir,
    )
    assert loaded.outlier_indices == (0, 1, 2, 3)
    assert loaded.head_dim == 8
