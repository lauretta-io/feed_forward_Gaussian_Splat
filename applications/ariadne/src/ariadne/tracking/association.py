"""Confidence-gated cross-agent object association."""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ariadne.common import Timestamp
from ariadne.models.features import cosine_similarity
from ariadne.tracking.static_filter import StaticTrackState, TrackState

SNAPSHOT_SCHEMA = "ariadne.cross-agent-association.v1"


@dataclass
class GlobalObject:
    global_id: str
    position_m: npt.NDArray[np.float64]
    embedding: npt.NDArray[np.float64]
    agents: set[str] = field(default_factory=set)
    observation_count: int = 0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64)
        embedding = np.asarray(self.embedding, dtype=np.float64)
        if (
            not self.global_id
            or position.shape != (3,)
            or embedding.ndim != 1
            or not embedding.size
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(embedding))
            or self.observation_count < 0
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("global object state is invalid")
        self.position_m = position.copy()
        norm = float(np.linalg.norm(embedding))
        self.embedding = embedding.copy() / norm if norm > 0 else embedding.copy()
        self.agents = set(self.agents)


@dataclass(frozen=True)
class AssociationEvidence:
    agent_id: str
    track_id: str
    global_id: str
    timestamp: Timestamp
    decision: str
    distance_m: float | None
    cosine_similarity: float | None
    score: float | None

    def __post_init__(self) -> None:
        if (
            not self.agent_id
            or not self.track_id
            or not self.global_id
            or self.decision not in {"created", "matched", "existing"}
        ):
            raise ValueError("association evidence identifiers or decision are invalid")
        values = (self.distance_m, self.cosine_similarity, self.score)
        if any(value is not None and not np.isfinite(value) for value in values):
            raise ValueError("association evidence scores must be finite")


class CrossAgentAssociator:
    def __init__(
        self,
        *,
        max_distance_m: float = 1.5,
        min_cosine_similarity: float = 0.8,
        max_objects: int = 4096,
        max_evidence: int = 8192,
    ) -> None:
        if (
            max_distance_m <= 0
            or not 0 <= min_cosine_similarity <= 1
            or max_objects <= 0
            or max_evidence <= 0
        ):
            raise ValueError("association gates are invalid")
        self.max_distance_m = max_distance_m
        self.min_cosine_similarity = min_cosine_similarity
        self.max_objects = max_objects
        self.max_evidence = max_evidence
        self._objects: dict[str, GlobalObject] = {}
        self._local_to_global: dict[tuple[str, str], str] = {}
        self._evidence: deque[AssociationEvidence] = deque()
        self._next_id = 1
        self.metrics = {
            "created": 0,
            "matched": 0,
            "existing": 0,
            "rejected": 0,
            "capacity_rejected": 0,
            "restores": 0,
        }

    @property
    def objects(self) -> tuple[GlobalObject, ...]:
        return tuple(self._objects[key] for key in sorted(self._objects))

    @property
    def evidence(self) -> tuple[AssociationEvidence, ...]:
        return tuple(self._evidence)

    def associate(self, track: TrackState) -> GlobalObject | None:
        if track.state is not StaticTrackState.STATIC_CONFIRMED:
            self.metrics["rejected"] += 1
            return None
        observation = track.observation
        local_key = (observation.agent_id, observation.track_id)
        existing_id = self._local_to_global.get(local_key)
        if existing_id is not None:
            target = self._objects[existing_id]
            distance = float(np.linalg.norm(observation.position_m - target.position_m))
            similarity = cosine_similarity(observation.embedding, target.embedding)
            self._update(target, track)
            self._record(track, target, "existing", distance, similarity, 1.0)
            self.metrics["existing"] += 1
            return target

        candidates: list[tuple[float, float, float, GlobalObject]] = []
        for candidate in self._objects.values():
            if observation.agent_id in candidate.agents:
                continue
            distance = float(np.linalg.norm(observation.position_m - candidate.position_m))
            similarity = cosine_similarity(observation.embedding, candidate.embedding)
            if distance <= self.max_distance_m and similarity >= self.min_cosine_similarity:
                score = similarity - 0.25 * distance / self.max_distance_m
                candidates.append((score, distance, similarity, candidate))
        if candidates:
            selected_score, selected_distance, selected_similarity, target = max(
                candidates, key=lambda item: (item[0], item[3].global_id)
            )
            decision = "matched"
            self.metrics["matched"] += 1
        else:
            if len(self._objects) >= self.max_objects:
                self.metrics["capacity_rejected"] += 1
                return None
            global_id = f"global_{self._next_id:04d}"
            self._next_id += 1
            target = GlobalObject(
                global_id,
                observation.position_m.copy(),
                observation.embedding.copy(),
            )
            self._objects[global_id] = target
            selected_score = None
            selected_distance = None
            selected_similarity = None
            decision = "created"
            self.metrics["created"] += 1
        self._local_to_global[local_key] = target.global_id
        self._update(target, track)
        self._record(
            track,
            target,
            decision,
            selected_distance,
            selected_similarity,
            selected_score,
        )
        return target

    def _update(self, target: GlobalObject, track: TrackState) -> None:
        observation = track.observation
        count = target.observation_count
        target.position_m = (target.position_m * count + observation.position_m) / (count + 1)
        embedding = (target.embedding * count + observation.embedding) / (count + 1)
        norm = float(np.linalg.norm(embedding))
        target.embedding = embedding / norm if norm > 0 else embedding
        target.observation_count += 1
        target.agents.add(observation.agent_id)
        target.confidence = min(target.confidence + track.static_probability / 3.0, 1.0)

    def _record(
        self,
        track: TrackState,
        target: GlobalObject,
        decision: str,
        distance_m: float | None,
        similarity: float | None,
        score: float | None,
    ) -> None:
        observation = track.observation
        self._evidence.append(
            AssociationEvidence(
                observation.agent_id,
                observation.track_id,
                target.global_id,
                observation.timestamp,
                decision,
                distance_m,
                similarity,
                score,
            )
        )
        while len(self._evidence) > self.max_evidence:
            self._evidence.popleft()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SNAPSHOT_SCHEMA,
            "max_distance_m": self.max_distance_m,
            "min_cosine_similarity": self.min_cosine_similarity,
            "max_objects": self.max_objects,
            "max_evidence": self.max_evidence,
            "next_id": self._next_id,
            "objects": [
                {
                    "global_id": item.global_id,
                    "position_m": item.position_m.tolist(),
                    "embedding": item.embedding.tolist(),
                    "agents": sorted(item.agents),
                    "observation_count": item.observation_count,
                    "confidence": item.confidence,
                }
                for item in self.objects
            ],
            "local_to_global": [
                {
                    "agent_id": agent_id,
                    "track_id": track_id,
                    "global_id": global_id,
                }
                for (agent_id, track_id), global_id in sorted(self._local_to_global.items())
            ],
            "evidence": [
                {
                    "agent_id": item.agent_id,
                    "track_id": item.track_id,
                    "global_id": item.global_id,
                    "timestamp_ns": item.timestamp.monotonic_ns,
                    "decision": item.decision,
                    "distance_m": item.distance_m,
                    "cosine_similarity": item.cosine_similarity,
                    "score": item.score,
                }
                for item in self.evidence
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
        return destination

    @classmethod
    def read_json(cls, path: str | Path) -> CrossAgentAssociator:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != SNAPSHOT_SCHEMA:
            raise ValueError("unsupported cross-agent association snapshot schema")
        associator = cls(
            max_distance_m=float(payload["max_distance_m"]),
            min_cosine_similarity=float(payload["min_cosine_similarity"]),
            max_objects=int(payload["max_objects"]),
            max_evidence=int(payload["max_evidence"]),
        )
        raw_objects = cast(list[dict[str, Any]], payload["objects"])
        raw_evidence = cast(list[dict[str, Any]], payload["evidence"])
        if (
            len(raw_objects) > associator.max_objects
            or len(raw_evidence) > associator.max_evidence
        ):
            raise ValueError("association snapshot exceeds configured bounds")
        for raw in raw_objects:
            item = GlobalObject(
                str(raw["global_id"]),
                np.asarray(raw["position_m"], dtype=np.float64),
                np.asarray(raw["embedding"], dtype=np.float64),
                set(str(value) for value in raw["agents"]),
                int(raw["observation_count"]),
                float(raw["confidence"]),
            )
            if item.global_id in associator._objects:
                raise ValueError("association snapshot contains duplicate global identifiers")
            associator._objects[item.global_id] = item
        for raw in cast(list[dict[str, Any]], payload["local_to_global"]):
            key = (str(raw["agent_id"]), str(raw["track_id"]))
            global_id = str(raw["global_id"])
            if key in associator._local_to_global or global_id not in associator._objects:
                raise ValueError("association snapshot contains an invalid local mapping")
            associator._local_to_global[key] = global_id
        for raw in raw_evidence:
            evidence = AssociationEvidence(
                str(raw["agent_id"]),
                str(raw["track_id"]),
                str(raw["global_id"]),
                Timestamp(int(raw["timestamp_ns"])),
                str(raw["decision"]),
                float(raw["distance_m"]) if raw["distance_m"] is not None else None,
                (
                    float(raw["cosine_similarity"])
                    if raw["cosine_similarity"] is not None
                    else None
                ),
                float(raw["score"]) if raw["score"] is not None else None,
            )
            if evidence.global_id not in associator._objects:
                raise ValueError("association evidence references an unknown global object")
            associator._evidence.append(evidence)
        associator._next_id = int(payload["next_id"])
        if associator._next_id <= 0:
            raise ValueError("association snapshot next identifier is invalid")
        associator.metrics["restores"] = 1
        return associator
