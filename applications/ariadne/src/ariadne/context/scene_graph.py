"""Freshness-aware unified scene graph for downstream planning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from ariadne.common import Timestamp

LOGGER = logging.getLogger(__name__)


class SceneNodeKind(StrEnum):
    WINGMAN = "wingman"
    OBJECT = "object"
    REGION = "region"
    FRONTIER = "frontier"


@dataclass(frozen=True)
class SceneNode:
    node_id: str
    kind: SceneNodeKind
    position_m: npt.NDArray[np.float64]
    confidence: float
    updated_at: Timestamp

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64)
        if not self.node_id or position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("scene node identifier or position is invalid")
        if not 0 <= self.confidence <= 1:
            raise ValueError("scene node confidence must be between zero and one")
        position = position.copy()
        position.setflags(write=False)
        object.__setattr__(self, "position_m", position)


@dataclass(frozen=True)
class SceneEdge:
    source: str
    destination: str
    relation: str
    confidence: float

    def __post_init__(self) -> None:
        if (
            not self.source
            or not self.destination
            or self.source == self.destination
            or not self.relation
        ):
            raise ValueError("scene edge is invalid")
        if not 0 <= self.confidence <= 1:
            raise ValueError("scene edge confidence must be between zero and one")


@dataclass(frozen=True)
class ContextSnapshot:
    revision: int
    timestamp: Timestamp
    nodes: tuple[SceneNode, ...]
    edges: tuple[SceneEdge, ...]
    degraded: bool
    stale_node_ids: tuple[str, ...]


class UnifiedContext:
    def __init__(self) -> None:
        self._nodes: dict[str, SceneNode] = {}
        self._edges: dict[tuple[str, str, str], SceneEdge] = {}
        self._revision = 0
        self.metrics = {"node_updates": 0, "edge_updates": 0, "snapshots": 0}

    def upsert_node(self, node: SceneNode) -> None:
        previous = self._nodes.get(node.node_id)
        if previous is not None and node.updated_at < previous.updated_at:
            raise ValueError("scene node updates must be time ordered")
        self._nodes[node.node_id] = node
        self._revision += 1
        self.metrics["node_updates"] += 1

    def upsert_edge(self, edge: SceneEdge) -> None:
        if edge.source not in self._nodes or edge.destination not in self._nodes:
            raise ValueError("scene edge endpoints must exist")
        self._edges[(edge.source, edge.destination, edge.relation)] = edge
        self._revision += 1
        self.metrics["edge_updates"] += 1

    def snapshot(self, now: Timestamp, *, max_age_ns: int = 5_000_000_000) -> ContextSnapshot:
        if max_age_ns <= 0:
            raise ValueError("max_age_ns must be positive")
        stale = tuple(
            sorted(
                node.node_id
                for node in self._nodes.values()
                if node.updated_at.monotonic_ns + max_age_ns < now.monotonic_ns
            )
        )
        self.metrics["snapshots"] += 1
        degraded = not self._nodes or bool(stale)
        LOGGER.debug("context_snapshot revision=%d degraded=%s", self._revision, degraded)
        return ContextSnapshot(
            self._revision,
            now,
            tuple(self._nodes[key] for key in sorted(self._nodes)),
            tuple(self._edges[key] for key in sorted(self._edges)),
            degraded,
            stale,
        )
