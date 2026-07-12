"""Confidence-gated cross-agent object association."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ariadne.models.features import cosine_similarity
from ariadne.tracking.static_filter import StaticTrackState, TrackState


@dataclass
class GlobalObject:
    global_id: str
    position_m: npt.NDArray[np.float64]
    embedding: npt.NDArray[np.float64]
    agents: set[str] = field(default_factory=set)
    observation_count: int = 0
    confidence: float = 0.0


class CrossAgentAssociator:
    def __init__(self, *, max_distance_m: float = 1.5, min_cosine_similarity: float = 0.8) -> None:
        if max_distance_m <= 0 or not 0 <= min_cosine_similarity <= 1:
            raise ValueError("association gates are invalid")
        self.max_distance_m = max_distance_m
        self.min_cosine_similarity = min_cosine_similarity
        self._objects: dict[str, GlobalObject] = {}
        self._local_to_global: dict[tuple[str, str], str] = {}
        self._next_id = 1

    @property
    def objects(self) -> tuple[GlobalObject, ...]:
        return tuple(self._objects[key] for key in sorted(self._objects))

    def associate(self, track: TrackState) -> GlobalObject | None:
        if track.state is not StaticTrackState.STATIC_CONFIRMED:
            return None
        observation = track.observation
        local_key = (observation.agent_id, observation.track_id)
        existing_id = self._local_to_global.get(local_key)
        if existing_id is not None:
            target = self._objects[existing_id]
            self._update(target, track)
            return target

        candidates: list[tuple[float, GlobalObject]] = []
        for candidate in self._objects.values():
            if observation.agent_id in candidate.agents:
                continue
            distance = float(np.linalg.norm(observation.position_m - candidate.position_m))
            similarity = cosine_similarity(observation.embedding, candidate.embedding)
            if distance <= self.max_distance_m and similarity >= self.min_cosine_similarity:
                score = similarity - 0.25 * distance / self.max_distance_m
                candidates.append((score, candidate))
        if candidates:
            target = max(candidates, key=lambda item: item[0])[1]
        else:
            global_id = f"global_{self._next_id:04d}"
            self._next_id += 1
            target = GlobalObject(
                global_id,
                observation.position_m.copy(),
                observation.embedding.copy(),
            )
            self._objects[global_id] = target
        self._local_to_global[local_key] = target.global_id
        self._update(target, track)
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
