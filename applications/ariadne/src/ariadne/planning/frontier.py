"""Confidence-aware deterministic frontier auction baseline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

LOGGER = logging.getLogger(__name__)


def _position(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    position = np.asarray(value, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("planning position must be a finite three-vector")
    position = position.copy()
    position.setflags(write=False)
    return position


@dataclass(frozen=True)
class Frontier:
    frontier_id: str
    position_m: npt.NDArray[np.float64]
    information_gain: float
    confidence: float

    def __post_init__(self) -> None:
        if not self.frontier_id or self.information_gain <= 0 or not 0 <= self.confidence <= 1:
            raise ValueError("frontier fields are invalid")
        object.__setattr__(self, "position_m", _position(self.position_m))


@dataclass(frozen=True)
class WingmanPlanningState:
    agent_id: str
    position_m: npt.NDArray[np.float64]
    battery_fraction: float
    available: bool = True

    def __post_init__(self) -> None:
        if not self.agent_id or not 0 <= self.battery_fraction <= 1:
            raise ValueError("Wingman planning state is invalid")
        object.__setattr__(self, "position_m", _position(self.position_m))


@dataclass(frozen=True)
class TaskAssignment:
    agent_id: str
    frontier_id: str
    utility: float
    travel_distance_m: float


class FrontierAuctionPlanner:
    def __init__(self, *, minimum_battery_fraction: float = 0.2) -> None:
        if not 0 <= minimum_battery_fraction < 1:
            raise ValueError("minimum_battery_fraction must be in [0, 1)")
        self.minimum_battery_fraction = minimum_battery_fraction
        self.metrics = {"plans": 0, "assignments": 0, "unassigned": 0}

    def plan(
        self,
        wingmen: tuple[WingmanPlanningState, ...],
        frontiers: tuple[Frontier, ...],
    ) -> tuple[TaskAssignment, ...]:
        candidates: list[tuple[float, str, str, float]] = []
        for wingman in wingmen:
            if not wingman.available or wingman.battery_fraction < self.minimum_battery_fraction:
                continue
            for frontier in frontiers:
                distance = float(np.linalg.norm(frontier.position_m - wingman.position_m))
                utility = (
                    frontier.information_gain
                    * frontier.confidence
                    * wingman.battery_fraction
                    / (1.0 + distance)
                )
                candidates.append((utility, wingman.agent_id, frontier.frontier_id, distance))
        assignments: list[TaskAssignment] = []
        used_agents: set[str] = set()
        used_frontiers: set[str] = set()
        for utility, agent_id, frontier_id, distance in sorted(
            candidates, key=lambda item: (-item[0], item[1], item[2])
        ):
            if agent_id in used_agents or frontier_id in used_frontiers:
                continue
            assignments.append(TaskAssignment(agent_id, frontier_id, utility, distance))
            used_agents.add(agent_id)
            used_frontiers.add(frontier_id)
        assignments.sort(key=lambda item: item.agent_id)
        self.metrics["plans"] += 1
        self.metrics["assignments"] += len(assignments)
        self.metrics["unassigned"] += len(frontiers) - len(assignments)
        LOGGER.info("frontier_plan assignments=%d frontiers=%d", len(assignments), len(frontiers))
        return tuple(assignments)
