"""Dataset/model catalog validation and experiment provenance records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    license: str
    upstream: str
    local_path: str
    role: str


@dataclass(frozen=True)
class ModelEntry:
    name: str
    version: str
    license: str
    backend: str
    task: str


@dataclass(frozen=True)
class ExperimentEntry:
    experiment_id: str
    benchmark: str
    seed: int
    config_digest: str
    artifact_path: str
    status: str

    def __post_init__(self) -> None:
        if not self.experiment_id or not self.benchmark or not self.config_digest:
            raise ValueError("experiment identifiers must not be empty")
        if self.status not in {"passed", "failed"}:
            raise ValueError("experiment status must be passed or failed")


class RegistryCatalog:
    def __init__(
        self,
        datasets: tuple[DatasetEntry, ...],
        models: tuple[ModelEntry, ...],
    ) -> None:
        if len({item.name for item in datasets}) != len(datasets):
            raise ValueError("dataset registry contains duplicate names")
        if len({(item.name, item.version) for item in models}) != len(models):
            raise ValueError("model registry contains duplicate name/version pairs")
        self.datasets = datasets
        self.models = models

    @classmethod
    def load(cls, dataset_path: Path, model_path: Path) -> RegistryCatalog:
        dataset_raw = OmegaConf.to_container(OmegaConf.load(dataset_path), resolve=True)
        model_raw = OmegaConf.to_container(OmegaConf.load(model_path), resolve=True)
        if not isinstance(dataset_raw, dict) or not isinstance(model_raw, dict):
            raise ValueError("registry roots must be mappings")
        datasets = tuple(
            DatasetEntry(
                name,
                str(payload["license"]),
                str(payload["upstream"]),
                str(payload["local_path"]),
                str(payload["role"]),
            )
            for name, payload in sorted(
                cast(dict[str, dict[str, Any]], dataset_raw.get("datasets", {})).items()
            )
        )
        models = tuple(
            ModelEntry(
                name,
                str(payload["version"]),
                str(payload["license"]),
                str(payload["backend"]),
                str(payload["task"]),
            )
            for name, payload in sorted(
                cast(dict[str, dict[str, Any]], model_raw.get("models", {})).items()
            )
        )
        if not datasets or not models:
            raise ValueError("dataset and model registries must not be empty")
        return cls(datasets, models)

    @staticmethod
    def write_experiment(entry: ExperimentEntry, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(entry), indent=2, sort_keys=True), encoding="utf-8")
