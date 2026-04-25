from __future__ import annotations

import pytest

from turboquant_mlx.reporting import _swap_latest_symlink


def test_swap_latest_symlink_refuses_to_clobber_real_directory(tmp_path):
    real_latest = tmp_path / "latest"
    real_latest.mkdir()
    (real_latest / "do_not_delete.txt").write_text("precious")
    (tmp_path / "20990101-000000").mkdir()

    with pytest.raises(RuntimeError, match="refusing to replace non-symlink"):
        _swap_latest_symlink(tmp_path / "latest", target_name="20990101-000000")

    assert real_latest.exists()
    assert (real_latest / "do_not_delete.txt").read_text() == "precious"


def test_swap_latest_symlink_replaces_existing_symlink(tmp_path):
    target_a = tmp_path / "run-a"
    target_a.mkdir()
    target_b = tmp_path / "run-b"
    target_b.mkdir()
    latest = tmp_path / "latest"
    latest.symlink_to(target_a.name)

    _swap_latest_symlink(latest, target_name=target_b.name)

    assert latest.is_symlink()
    assert (tmp_path / latest.readlink()).resolve() == target_b.resolve()


def test_swap_latest_symlink_creates_when_absent(tmp_path):
    target = tmp_path / "run-c"
    target.mkdir()
    latest = tmp_path / "latest"

    _swap_latest_symlink(latest, target_name=target.name)

    assert latest.is_symlink()
    assert (tmp_path / latest.readlink()).resolve() == target.resolve()
