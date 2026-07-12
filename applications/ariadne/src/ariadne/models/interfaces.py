"""Stable contracts shared by reference and production model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ariadne.common import ModelVersion, Timestamp
from ariadne.replay import ImageFrame, SynchronizedPacket


@dataclass(frozen=True)
class VioEstimate:
    timestamp: Timestamp
    agent_id: str
    position_m: npt.NDArray[np.float64]
    velocity_mps: npt.NDArray[np.float64]
    covariance: npt.NDArray[np.float64]
    tracking_quality: float
    status: str


class VioBackend(Protocol):
    version: ModelVersion

    def reset(self) -> None: ...

    def process(self, packet: SynchronizedPacket) -> VioEstimate: ...


@dataclass(frozen=True)
class FeatureSet:
    keypoints_xy: npt.NDArray[np.float64]
    descriptors: npt.NDArray[np.float64]
    scores: npt.NDArray[np.float64]
    version: ModelVersion


class GeometricFeatureExtractor(Protocol):
    version: ModelVersion

    def extract(self, frame: ImageFrame) -> FeatureSet: ...


@dataclass(frozen=True)
class SemanticEmbedding:
    vector: npt.NDArray[np.float64]
    version: ModelVersion


class SemanticEmbedder(Protocol):
    version: ModelVersion

    def embed(self, frame: ImageFrame) -> SemanticEmbedding: ...
