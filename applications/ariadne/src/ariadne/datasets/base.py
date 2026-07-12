"""Common dataset evaluation result and adapter dispatch."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetEvaluation:
    dataset: str
    status: str
    agents: tuple[str, ...]
    modalities: tuple[str, ...]
    metrics: dict[str, int | float | str]
    warnings: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def evaluate_dataset(name: str, path: Path | None = None, *, seed: int = 7) -> DatasetEvaluation:
    if name == "s3e":
        if path is None:
            raise ValueError("s3e evaluation requires --path")
        from ariadne.datasets.s3e import evaluate_s3e

        return evaluate_s3e(path)
    if name == "miluv":
        if path is None:
            raise ValueError("miluv evaluation requires --path")
        from ariadne.datasets.zip_datasets import evaluate_miluv

        return evaluate_miluv(path)
    if name == "qdrone":
        if path is None:
            raise ValueError("qdrone evaluation requires --path")
        from ariadne.datasets.zip_datasets import evaluate_qdrone

        return evaluate_qdrone(path)
    if name == "d2slam":
        if path is None:
            raise ValueError("d2slam evaluation requires --path")
        from ariadne.datasets.d2slam import evaluate_d2slam

        return evaluate_d2slam(path)
    if name == "simulation":
        from ariadne.datasets.simulation import evaluate_simulation

        return evaluate_simulation(seed)
    raise ValueError(f"unsupported dataset: {name}")
