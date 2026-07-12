"""Hysteretic temporal static/dynamic classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from ariadne.common import Timestamp


class StaticTrackState(StrEnum):
    UNKNOWN = "unknown"
    STATIC_CANDIDATE = "static_candidate"
    STATIC_CONFIRMED = "static_confirmed"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class TrackObservation:
    timestamp: Timestamp
    agent_id: str
    track_id: str
    position_m: npt.NDArray[np.float64]
    embedding: npt.NDArray[np.float64]
    motion_residual_mps: float
    depth_consistency: float
    embedding_similarity: float

    def __post_init__(self) -> None:
        if not self.agent_id or not self.track_id:
            raise ValueError("agent_id and track_id must not be empty")
        position = np.array(self.position_m, dtype=np.float64, copy=True)
        embedding = np.array(self.embedding, dtype=np.float64, copy=True)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_m must be a finite three-vector")
        if embedding.ndim != 1 or not embedding.size or not np.all(np.isfinite(embedding)):
            raise ValueError("embedding must be a non-empty finite vector")
        scalars = (self.motion_residual_mps, self.depth_consistency, self.embedding_similarity)
        if not np.all(np.isfinite(scalars)):
            raise ValueError("observation scores must be finite")
        if self.motion_residual_mps < 0:
            raise ValueError("motion_residual_mps must be non-negative")
        if not 0 <= self.depth_consistency <= 1 or not 0 <= self.embedding_similarity <= 1:
            raise ValueError("consistency and similarity must be between zero and one")
        position.setflags(write=False)
        embedding.setflags(write=False)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "embedding", embedding)


@dataclass(frozen=True)
class TrackState:
    observation: TrackObservation
    state: StaticTrackState
    static_probability: float
    observation_count: int


@dataclass
class _FilterState:
    probability: float
    count: int
    state: StaticTrackState


class TemporalStaticFilter:
    def __init__(
        self,
        *,
        smoothing: float = 0.65,
        confirmation_threshold: float = 0.75,
        dynamic_threshold: float = 0.35,
        min_confirmations: int = 3,
    ) -> None:
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        if not 0 <= dynamic_threshold < confirmation_threshold <= 1:
            raise ValueError("classification thresholds are inconsistent")
        if min_confirmations < 2:
            raise ValueError("min_confirmations must be at least two")
        self.smoothing = smoothing
        self.confirmation_threshold = confirmation_threshold
        self.dynamic_threshold = dynamic_threshold
        self.min_confirmations = min_confirmations
        self._tracks: dict[tuple[str, str], _FilterState] = {}

    def update(self, observation: TrackObservation) -> TrackState:
        motion_score = float(np.exp(-observation.motion_residual_mps / 0.2))
        evidence = (
            0.45 * motion_score
            + 0.30 * observation.depth_consistency
            + 0.25 * observation.embedding_similarity
        )
        key = (observation.agent_id, observation.track_id)
        previous = self._tracks.get(key)
        probability = (
            evidence
            if previous is None
            else self.smoothing * previous.probability + (1.0 - self.smoothing) * evidence
        )
        count = 1 if previous is None else previous.count + 1
        if probability <= self.dynamic_threshold:
            state = StaticTrackState.DYNAMIC
        elif count >= self.min_confirmations and probability >= self.confirmation_threshold:
            state = StaticTrackState.STATIC_CONFIRMED
        elif count > 1:
            state = StaticTrackState.STATIC_CANDIDATE
        else:
            state = StaticTrackState.UNKNOWN
        self._tracks[key] = _FilterState(probability, count, state)
        return TrackState(observation, state, probability, count)

    def reset(self) -> None:
        self._tracks.clear()
