"""Bounded local static-object and keyframe store."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ariadne.common import ModelVersion, Timestamp
from ariadne.tracking import StaticTrackState, TrackState

LOGGER = logging.getLogger(__name__)
SNAPSHOT_SCHEMA = "ariadne.local-object-store.v1"


def _vector(value: npt.ArrayLike, length: int, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {length}")
    result = array.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class KeyframeRecord:
    frame_index: int
    timestamp: Timestamp
    quality: float
    region_id: str

    def __post_init__(self) -> None:
        if self.frame_index < 0 or not self.region_id or not 0 <= self.quality <= 1:
            raise ValueError("keyframe fields are invalid")


@dataclass(frozen=True)
class LocalObjectRecord:
    local_id: str
    agent_id: str
    timestamp: Timestamp
    position_m: npt.NDArray[np.float64]
    embedding: npt.NDArray[np.float64]
    covariance_diagonal: npt.NDArray[np.float64]
    confidence: float
    model_version: ModelVersion
    observation_count: int
    keyframes: tuple[KeyframeRecord, ...]

    def __post_init__(self) -> None:
        if not self.local_id or not self.agent_id or self.observation_count <= 0:
            raise ValueError("local object identifiers and observation count are invalid")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        position = _vector(self.position_m, 3, "position_m")
        embedding = np.asarray(self.embedding, dtype=np.float64)
        if embedding.ndim != 1 or not embedding.size or not np.all(np.isfinite(embedding)):
            raise ValueError("embedding must be a non-empty finite vector")
        epsilon = float(np.finfo(np.float64).eps)
        embedding = embedding / max(float(np.linalg.norm(embedding)), epsilon)
        embedding.setflags(write=False)
        covariance = _vector(self.covariance_diagonal, 3, "covariance_diagonal")
        if np.any(covariance < 0):
            raise ValueError("covariance_diagonal must be non-negative")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "embedding", embedding)
        object.__setattr__(self, "covariance_diagonal", covariance)


class LocalObjectStore:
    def __init__(self, *, max_objects: int = 256, max_keyframes_per_object: int = 4) -> None:
        if max_objects <= 0 or max_keyframes_per_object <= 0:
            raise ValueError("store bounds must be positive")
        self.max_objects = max_objects
        self.max_keyframes_per_object = max_keyframes_per_object
        self._objects: dict[str, LocalObjectRecord] = {}
        self.metrics = {
            "upserts": 0,
            "evictions": 0,
            "rejected": 0,
            "snapshots": 0,
            "restores": 0,
        }

    @property
    def objects(self) -> tuple[LocalObjectRecord, ...]:
        return tuple(sorted(self._objects.values(), key=lambda record: record.local_id))

    def upsert(
        self,
        track: TrackState,
        *,
        model_version: ModelVersion,
        keyframe: KeyframeRecord,
    ) -> LocalObjectRecord | None:
        if track.state is not StaticTrackState.STATIC_CONFIRMED:
            self.metrics["rejected"] += 1
            return None
        observation = track.observation
        local_id = f"{observation.agent_id}:{observation.track_id}"
        previous = self._objects.get(local_id)
        if (
            previous is not None
            and observation.timestamp.monotonic_ns < previous.timestamp.monotonic_ns
        ):
            self.metrics["rejected"] += 1
            LOGGER.warning("local_object_out_of_order local_id=%s", local_id)
            return None
        count = 0 if previous is None else previous.observation_count
        keyframes = (() if previous is None else previous.keyframes) + (keyframe,)
        keyframes = tuple(
            sorted(
                keyframes,
                key=lambda item: (item.quality, item.timestamp.monotonic_ns),
                reverse=True,
            )[: self.max_keyframes_per_object]
        )
        if previous is None:
            position = observation.position_m
            embedding = observation.embedding
        else:
            position = (previous.position_m * count + observation.position_m) / (count + 1)
            embedding = (previous.embedding * count + observation.embedding) / (count + 1)
        record = LocalObjectRecord(
            local_id,
            observation.agent_id,
            observation.timestamp,
            position,
            embedding,
            np.full(3, max(1.0 - track.static_probability, 1e-4)),
            track.static_probability,
            model_version,
            count + 1,
            keyframes,
        )
        self._objects[local_id] = record
        self.metrics["upserts"] += 1
        if len(self._objects) > self.max_objects:
            evicted = min(
                self._objects.values(),
                key=lambda item: (item.confidence, item.timestamp.monotonic_ns, item.local_id),
            )
            del self._objects[evicted.local_id]
            self.metrics["evictions"] += 1
            LOGGER.info("local_object_evicted local_id=%s", evicted.local_id)
        return record

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SNAPSHOT_SCHEMA,
            "max_objects": self.max_objects,
            "max_keyframes_per_object": self.max_keyframes_per_object,
            "objects": [
                {
                    "local_id": record.local_id,
                    "agent_id": record.agent_id,
                    "timestamp_ns": record.timestamp.monotonic_ns,
                    "position_m": record.position_m.tolist(),
                    "embedding": record.embedding.tolist(),
                    "covariance_diagonal": record.covariance_diagonal.tolist(),
                    "confidence": record.confidence,
                    "model_version": record.model_version.to_dict(),
                    "observation_count": record.observation_count,
                    "keyframes": [
                        {
                            "frame_index": keyframe.frame_index,
                            "timestamp_ns": keyframe.timestamp.monotonic_ns,
                            "quality": keyframe.quality,
                            "region_id": keyframe.region_id,
                        }
                        for keyframe in record.keyframes
                    ],
                }
                for record in self.objects
            ],
        }

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(self.to_dict(), stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.metrics["snapshots"] += 1
        return destination

    @classmethod
    def read_json(cls, path: str | Path) -> LocalObjectStore:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != SNAPSHOT_SCHEMA:
            raise ValueError("unsupported local object store snapshot schema")
        max_objects = int(payload["max_objects"])
        max_keyframes = int(payload["max_keyframes_per_object"])
        store = cls(
            max_objects=max_objects,
            max_keyframes_per_object=max_keyframes,
        )
        raw_objects = payload.get("objects")
        if not isinstance(raw_objects, list) or len(raw_objects) > max_objects:
            raise ValueError("local object store snapshot exceeds configured bounds")
        for raw_value in raw_objects:
            raw = cast(dict[str, Any], raw_value)
            raw_keyframes = cast(list[dict[str, Any]], raw["keyframes"])
            if len(raw_keyframes) > max_keyframes:
                raise ValueError("local object snapshot exceeds keyframe bounds")
            record = LocalObjectRecord(
                local_id=str(raw["local_id"]),
                agent_id=str(raw["agent_id"]),
                timestamp=Timestamp(int(raw["timestamp_ns"])),
                position_m=np.asarray(raw["position_m"], dtype=np.float64),
                embedding=np.asarray(raw["embedding"], dtype=np.float64),
                covariance_diagonal=np.asarray(
                    raw["covariance_diagonal"], dtype=np.float64
                ),
                confidence=float(raw["confidence"]),
                model_version=ModelVersion.from_dict(
                    cast(dict[str, Any], raw["model_version"])
                ),
                observation_count=int(raw["observation_count"]),
                keyframes=tuple(
                    KeyframeRecord(
                        frame_index=int(keyframe["frame_index"]),
                        timestamp=Timestamp(int(keyframe["timestamp_ns"])),
                        quality=float(keyframe["quality"]),
                        region_id=str(keyframe["region_id"]),
                    )
                    for keyframe in raw_keyframes
                ),
            )
            if record.local_id in store._objects:
                raise ValueError("local object store snapshot contains duplicate identifiers")
            store._objects[record.local_id] = record
        store.metrics["restores"] += 1
        return store
