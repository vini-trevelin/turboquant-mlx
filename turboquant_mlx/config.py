from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Tuple


Mode = Literal["core", "preset"]


@dataclass(frozen=True)
class CalibrationConfig:
    quantile: float = 99.9
    artifact_path: Optional[str] = None


@dataclass(frozen=True)
class CalibrationArtifact:
    head_dim: int
    outlier_indices: Tuple[int, ...]
    quantile: float = 99.9
    source_path: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["outlier_indices"] = list(self.outlier_indices)
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationArtifact":
        payload = json.loads(Path(path).read_text())
        payload["outlier_indices"] = tuple(int(v) for v in payload["outlier_indices"])
        return cls(**payload)


@dataclass(frozen=True)
class SubsetSpec:
    name: str
    indices: Tuple[int, ...]
    bits: int
    qjl_dim: int = 0

    @property
    def dim(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class TurboQuantConfig:
    mode: Mode = "core"
    head_dim: int = 128
    seed: int = 0
    core_bits: int = 4
    preset_name: Optional[str] = None
    qjl_enabled: bool = False
    qjl_dim: int = 64
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    outlier_indices: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.mode == "preset" and self.preset_name not in {"2.5", "3.5"}:
            raise ValueError("preset mode requires preset_name to be '2.5' or '3.5'")
        if self.mode == "core" and self.core_bits < 1:
            raise ValueError("core_bits must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")

    def with_outlier_indices(self, outlier_indices: Sequence[int]) -> "TurboQuantConfig":
        return TurboQuantConfig(
            mode=self.mode,
            head_dim=self.head_dim,
            seed=self.seed,
            core_bits=self.core_bits,
            preset_name=self.preset_name,
            qjl_enabled=self.qjl_enabled,
            qjl_dim=self.qjl_dim,
            calibration=self.calibration,
            outlier_indices=tuple(int(v) for v in outlier_indices),
        )

    def resolved_outlier_indices(self) -> Tuple[int, ...]:
        if self.mode != "preset":
            return tuple(range(self.head_dim))
        if self.outlier_indices is not None:
            return tuple(sorted(set(self.outlier_indices)))
        if self.calibration.artifact_path:
            artifact = CalibrationArtifact.load(self.calibration.artifact_path)
            if artifact.head_dim != self.head_dim:
                raise ValueError(
                    f"Calibration artifact head_dim {artifact.head_dim} "
                    f"does not match config head_dim {self.head_dim}"
                )
            return tuple(sorted(set(artifact.outlier_indices)))
        raise ValueError("preset mode requires outlier indices or a calibration artifact")

    def subset_specs(self) -> Tuple[SubsetSpec, ...]:
        if self.mode == "core":
            return (
                SubsetSpec(
                    name="all",
                    indices=tuple(range(self.head_dim)),
                    bits=self.core_bits,
                    qjl_dim=self.qjl_dim if self.qjl_enabled else 0,
                ),
            )

        outlier = self.resolved_outlier_indices()
        regular = tuple(i for i in range(self.head_dim) if i not in set(outlier))
        if len(outlier) == 0 or len(regular) == 0:
            raise ValueError("preset mode requires a non-empty outlier and regular split")
        if self.preset_name == "2.5":
            outlier_bits, regular_bits = 3, 2
        else:
            outlier_bits, regular_bits = 5, 3
        return (
            SubsetSpec(
                name="outlier",
                indices=outlier,
                bits=outlier_bits,
                qjl_dim=self.qjl_dim if self.qjl_enabled else 0,
            ),
            SubsetSpec(
                name="regular",
                indices=regular,
                bits=regular_bits,
                qjl_dim=self.qjl_dim if self.qjl_enabled else 0,
            ),
        )

