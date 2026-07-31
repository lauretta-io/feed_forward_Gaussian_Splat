"""Bounded, versioned global Gaussian scene map with provenance."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from ariadne.common import Timestamp
from ariadne.splatting.adapter import GaussianPrimitive, ReconstructionResult

LOGGER = logging.getLogger(__name__)
SNAPSHOT_SCHEMA = "ariadne.global-scene-snapshot.v1"


def _atomic_write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


@dataclass(frozen=True)
class SceneSnapshot:
    revision: int
    timestamp: Timestamp
    primitives: tuple[GaussianPrimitive, ...]
    operation: str = "update"
    source_revision: int | None = None

    def __post_init__(self) -> None:
        supported_operations = {"bootstrap", "update", "rollback", "restore"}
        if self.revision < 0 or self.operation not in supported_operations:
            raise ValueError("scene snapshot revision or operation is invalid")
        identifiers = [primitive.primitive_id for primitive in self.primitives]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("scene snapshot primitive identifiers must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SNAPSHOT_SCHEMA,
            "revision": self.revision,
            "timestamp": self.timestamp.to_dict(),
            "operation": self.operation,
            "source_revision": self.source_revision,
            "primitives": [
                {
                    "primitive_id": primitive.primitive_id,
                    "object_id": primitive.object_id,
                    "mean_m": primitive.mean_m.tolist(),
                    "scale_m": primitive.scale_m.tolist(),
                    "color_rgb": primitive.color_rgb.tolist(),
                    "opacity": primitive.opacity,
                    "confidence": primitive.confidence,
                    "source_observation_ids": list(primitive.source_observation_ids),
                }
                for primitive in self.primitives
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SceneSnapshot:
        schema = payload.get("schema", SNAPSHOT_SCHEMA)
        if schema != SNAPSHOT_SCHEMA:
            raise ValueError("unsupported global scene snapshot schema")
        return cls(
            int(payload["revision"]),
            Timestamp.from_dict(payload["timestamp"]),
            tuple(
                GaussianPrimitive(
                    str(item["primitive_id"]),
                    str(item["object_id"]),
                    np.asarray(item["mean_m"], dtype=np.float64),
                    np.asarray(item["scale_m"], dtype=np.float64),
                    np.asarray(item["color_rgb"], dtype=np.float64),
                    float(item["opacity"]),
                    float(item["confidence"]),
                    tuple(str(value) for value in item["source_observation_ids"]),
                )
                for item in payload["primitives"]
            ),
            str(payload.get("operation", "restore")),
            (
                int(payload["source_revision"])
                if payload.get("source_revision") is not None
                else None
            ),
        )

    def write_json(self, path: str | Path) -> Path:
        return _atomic_write_json(Path(path), self.to_dict())

    @classmethod
    def read_json(cls, path: str | Path) -> SceneSnapshot:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != SNAPSHOT_SCHEMA
        ):
            raise ValueError("scene snapshot root must be an object")
        return cls.from_dict(payload)


class SceneSnapshotStore:
    def __init__(self, directory: str | Path, *, max_snapshots: int = 16) -> None:
        if max_snapshots <= 0:
            raise ValueError("max_snapshots must be positive")
        self.directory = Path(directory)
        self.max_snapshots = max_snapshots

    @property
    def latest_path(self) -> Path:
        return self.directory / "latest.json"

    @property
    def snapshot_paths(self) -> tuple[Path, ...]:
        return tuple(sorted(self.directory.glob("scene-*.json")))

    def persist(self, snapshot: SceneSnapshot) -> Path:
        try:
            current = self.load_latest()
        except FileNotFoundError:
            current = None
        if current is not None:
            if snapshot.revision < current.revision:
                raise ValueError("scene snapshot revisions must be monotonic")
            if (
                snapshot.revision == current.revision
                and snapshot.to_dict() != current.to_dict()
            ):
                raise ValueError("scene snapshot revision conflicts with persisted state")
        archive = self.directory / f"scene-{snapshot.revision:020d}.json"
        snapshot.write_json(archive)
        snapshot.write_json(self.latest_path)
        paths = self.snapshot_paths
        for expired in paths[: max(0, len(paths) - self.max_snapshots)]:
            expired.unlink()
        return archive

    def load_latest(self) -> SceneSnapshot:
        candidates = (self.latest_path,) + tuple(reversed(self.snapshot_paths))
        failures: list[Exception] = []
        snapshots: list[SceneSnapshot] = []
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                snapshots.append(SceneSnapshot.read_json(candidate))
            except (OSError, TypeError, ValueError) as error:
                failures.append(error)
        if snapshots:
            latest_revision = max(snapshot.revision for snapshot in snapshots)
            latest = [
                snapshot for snapshot in snapshots if snapshot.revision == latest_revision
            ]
            if any(snapshot.to_dict() != latest[0].to_dict() for snapshot in latest[1:]):
                raise ValueError("persisted global scene snapshots conflict")
            return latest[0]
        if failures:
            raise ValueError("no valid global scene snapshot is available") from failures[-1]
        raise FileNotFoundError(f"no global scene snapshots exist in {self.directory}")


class GlobalGaussianMap:
    def __init__(self, *, max_primitives: int = 10_000, max_history: int = 16) -> None:
        if max_primitives <= 0 or max_history <= 0:
            raise ValueError("scene map bounds must be positive")
        self.max_primitives = max_primitives
        self.max_history = max_history
        self._primitives: dict[str, GaussianPrimitive] = {}
        self._revision = 0
        self._timestamp = Timestamp(0)
        self._history: dict[int, SceneSnapshot] = {
            0: SceneSnapshot(0, self._timestamp, (), "bootstrap")
        }
        self._applied_updates: set[tuple[object, ...]] = set()
        self.metrics = {
            "updates": 0,
            "duplicate_updates": 0,
            "inserted": 0,
            "merged": 0,
            "evicted": 0,
            "rollbacks": 0,
            "restores": 0,
        }

    def apply(self, result: ReconstructionResult) -> SceneSnapshot:
        if not result.primitives:
            raise ValueError("scene updates must contain at least one primitive")
        identifiers = [primitive.primitive_id for primitive in result.primitives]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("scene update primitive identifiers must be unique")
        update_key = self._update_key(result)
        if update_key in self._applied_updates:
            self.metrics["duplicate_updates"] += 1
            return self.snapshot()
        if result.timestamp < self._timestamp:
            raise ValueError("scene updates must be time ordered")
        updated = dict(self._primitives)
        inserted = 0
        merged = 0
        evicted_count = 0
        for primitive in result.primitives:
            previous = updated.get(primitive.primitive_id)
            if previous is None:
                updated[primitive.primitive_id] = primitive
                inserted += 1
            else:
                if previous.object_id != primitive.object_id:
                    raise ValueError("scene update conflicts with the existing primitive object")
                total = previous.confidence + primitive.confidence
                if total <= np.finfo(np.float64).eps:
                    left = right = 0.5
                else:
                    left = previous.confidence / total
                    right = primitive.confidence / total
                updated[primitive.primitive_id] = GaussianPrimitive(
                    primitive.primitive_id,
                    primitive.object_id,
                    previous.mean_m * left + primitive.mean_m * right,
                    previous.scale_m * left + primitive.scale_m * right,
                    previous.color_rgb * left + primitive.color_rgb * right,
                    max(previous.opacity, primitive.opacity),
                    min(1.0, total / 2),
                    tuple(
                        dict.fromkeys(
                            previous.source_observation_ids + primitive.source_observation_ids
                        )
                    ),
                )
                merged += 1
        while len(updated) > self.max_primitives:
            evicted = min(
                updated.values(), key=lambda item: (item.confidence, item.primitive_id)
            )
            del updated[evicted.primitive_id]
            evicted_count += 1
        self._primitives = updated
        self._revision += 1
        self._timestamp = result.timestamp
        self.metrics["updates"] += 1
        self.metrics["inserted"] += inserted
        self.metrics["merged"] += merged
        self.metrics["evicted"] += evicted_count
        self._applied_updates.add(update_key)
        snapshot = self.snapshot()
        self._remember(snapshot)
        LOGGER.info(
            "scene_map_updated revision=%d primitives=%d", self._revision, len(self._primitives)
        )
        return snapshot

    @property
    def history(self) -> tuple[SceneSnapshot, ...]:
        return tuple(self._history[revision] for revision in sorted(self._history))

    def rollback(self, revision: int, *, timestamp: Timestamp) -> SceneSnapshot:
        target = self._history.get(revision)
        if target is None:
            raise ValueError("scene rollback revision is unavailable")
        if revision == self._revision:
            raise ValueError("scene rollback target must differ from the current revision")
        if timestamp < self._timestamp:
            raise ValueError("scene rollback timestamp must be time ordered")
        previous_revision = self._revision
        self._primitives = {
            primitive.primitive_id: primitive for primitive in target.primitives
        }
        self._revision += 1
        self._timestamp = timestamp
        snapshot = SceneSnapshot(
            self._revision,
            timestamp,
            tuple(self._primitives[key] for key in sorted(self._primitives)),
            "rollback",
            revision,
        )
        self.metrics["rollbacks"] += 1
        self._remember(snapshot)
        LOGGER.warning(
            "scene_map_rolled_back from_revision=%d source_revision=%d new_revision=%d",
            previous_revision,
            revision,
            self._revision,
        )
        return snapshot

    @classmethod
    def restore(
        cls,
        snapshot: SceneSnapshot,
        *,
        max_primitives: int = 10_000,
        max_history: int = 16,
    ) -> GlobalGaussianMap:
        if len(snapshot.primitives) > max_primitives:
            raise ValueError("scene snapshot exceeds the primitive bound")
        scene = cls(max_primitives=max_primitives, max_history=max_history)
        scene._primitives = {
            primitive.primitive_id: primitive for primitive in snapshot.primitives
        }
        scene._revision = snapshot.revision
        scene._timestamp = snapshot.timestamp
        restored = SceneSnapshot(
            snapshot.revision,
            snapshot.timestamp,
            snapshot.primitives,
            "restore",
            snapshot.revision,
        )
        scene._history = {snapshot.revision: restored}
        scene._applied_updates = set()
        scene.metrics["restores"] = 1
        return scene

    def snapshot(self) -> SceneSnapshot:
        return SceneSnapshot(
            self._revision,
            self._timestamp,
            tuple(self._primitives[key] for key in sorted(self._primitives)),
        )

    def _remember(self, snapshot: SceneSnapshot) -> None:
        self._history[snapshot.revision] = snapshot
        while len(self._history) > self.max_history:
            del self._history[min(self._history)]

    @staticmethod
    def _update_key(result: ReconstructionResult) -> tuple[object, ...]:
        return (
            result.timestamp.monotonic_ns,
            result.model_version.name,
            result.model_version.version,
            result.model_version.checksum_sha256,
            tuple(
                (
                    primitive.primitive_id,
                    primitive.object_id,
                    tuple(primitive.mean_m),
                    tuple(primitive.scale_m),
                    tuple(primitive.color_rgb),
                    primitive.opacity,
                    primitive.confidence,
                    primitive.source_observation_ids,
                )
                for primitive in result.primitives
            ),
        )
