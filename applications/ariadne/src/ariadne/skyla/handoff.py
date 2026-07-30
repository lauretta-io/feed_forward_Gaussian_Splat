"""Versioned ARIADNE context hand-off contract for SKYLA planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from ariadne.common import FrameId, Timestamp
from ariadne.context import ContextSnapshot
from ariadne.planning import Frontier, WingmanPlanningState

SKYLA_HANDOFF_SCHEMA = "ariadne.skyla.handoff.v1"


def _position(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    position = np.asarray(value, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(f"{name} must be a finite three-vector")
    position = position.copy()
    position.setflags(write=False)
    return position


class VehicleHealth(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass(frozen=True)
class VehicleState:
    agent_id: str
    position_m: npt.NDArray[np.float64]
    battery_fraction: float
    link_quality: float
    health: VehicleHealth = VehicleHealth.HEALTHY
    available: bool = True

    def __post_init__(self) -> None:
        if not self.agent_id or not 0 <= self.battery_fraction <= 1:
            raise ValueError("vehicle identity or battery fraction is invalid")
        if not 0 <= self.link_quality <= 1:
            raise ValueError("vehicle link quality must be between zero and one")
        object.__setattr__(self, "position_m", _position(self.position_m, "vehicle position"))

    def planning_state(self) -> WingmanPlanningState:
        return WingmanPlanningState(
            self.agent_id,
            self.position_m,
            self.battery_fraction,
            self.available and self.health is not VehicleHealth.FAILED and self.link_quality > 0,
        )


@dataclass(frozen=True)
class MissionGoal:
    goal_id: str
    priority: int
    frontier_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.goal_id or not 1 <= self.priority <= 100:
            raise ValueError("mission goal identity or priority is invalid")
        if not self.frontier_ids or any(not value for value in self.frontier_ids):
            raise ValueError("mission goal must target at least one frontier")
        if len(set(self.frontier_ids)) != len(self.frontier_ids):
            raise ValueError("mission goal frontier identifiers must be unique")


@dataclass(frozen=True)
class MissionIntent:
    mission_id: str
    revision: int
    goals: tuple[MissionGoal, ...]

    def __post_init__(self) -> None:
        if not self.mission_id or self.revision <= 0 or not self.goals:
            raise ValueError("mission identity, revision, and goals are required")
        goal_ids = [goal.goal_id for goal in self.goals]
        if len(set(goal_ids)) != len(goal_ids):
            raise ValueError("mission goal identifiers must be unique")


@dataclass(frozen=True)
class NoFlyZone:
    zone_id: str
    center_m: npt.NDArray[np.float64]
    radius_m: float

    def __post_init__(self) -> None:
        if not self.zone_id or not np.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("no-fly zone identity or radius is invalid")
        object.__setattr__(self, "center_m", _position(self.center_m, "no-fly zone center"))

    def blocks_segment(
        self,
        start_m: npt.NDArray[np.float64],
        end_m: npt.NDArray[np.float64],
    ) -> bool:
        segment = end_m - start_m
        denominator = float(np.dot(segment, segment))
        if denominator <= np.finfo(np.float64).eps:
            closest = start_m
        else:
            fraction = float(np.dot(self.center_m - start_m, segment) / denominator)
            closest = start_m + np.clip(fraction, 0.0, 1.0) * segment
        return float(np.linalg.norm(closest - self.center_m)) <= self.radius_m


@dataclass(frozen=True)
class SkylaHandoff:
    handoff_id: str
    schema: str
    global_frame: FrameId
    issued_at: Timestamp
    expires_at_ns: int
    context: ContextSnapshot
    mission: MissionIntent
    vehicles: tuple[VehicleState, ...]
    frontiers: tuple[Frontier, ...]
    no_fly_zones: tuple[NoFlyZone, ...]

    def __post_init__(self) -> None:
        if not self.handoff_id or self.schema != SKYLA_HANDOFF_SCHEMA:
            raise ValueError("SKYLA hand-off identity or schema is invalid")
        if self.global_frame != FrameId("global"):
            raise ValueError("SKYLA hand-off must use the canonical global frame")
        if self.expires_at_ns <= self.issued_at.monotonic_ns:
            raise ValueError("SKYLA hand-off expiry must follow issue time")
        if self.context.timestamp > self.issued_at:
            raise ValueError("SKYLA hand-off cannot contain future context")
        if not self.vehicles or not self.frontiers:
            raise ValueError("SKYLA hand-off requires vehicle and frontier state")
        for values, name in (
            ([item.agent_id for item in self.vehicles], "vehicle"),
            ([item.frontier_id for item in self.frontiers], "frontier"),
            ([item.zone_id for item in self.no_fly_zones], "no-fly zone"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"SKYLA hand-off {name} identifiers must be unique")
        frontier_ids = {frontier.frontier_id for frontier in self.frontiers}
        targeted_ids = {
            frontier_id for goal in self.mission.goals for frontier_id in goal.frontier_ids
        }
        unknown = targeted_ids - frontier_ids
        if unknown:
            raise ValueError(f"mission targets unknown frontiers: {sorted(unknown)}")

    @classmethod
    def create(
        cls,
        *,
        context: ContextSnapshot,
        mission: MissionIntent,
        vehicles: tuple[VehicleState, ...],
        frontiers: tuple[Frontier, ...],
        no_fly_zones: tuple[NoFlyZone, ...] = (),
        issued_at: Timestamp,
        ttl_ns: int = 60_000_000_000,
    ) -> SkylaHandoff:
        if ttl_ns <= 0:
            raise ValueError("SKYLA hand-off TTL must be positive")
        identifier_payload = {
            "schema": SKYLA_HANDOFF_SCHEMA,
            "context_revision": context.revision,
            "context_timestamp_ns": context.timestamp.monotonic_ns,
            "context_degraded": context.degraded,
            "context_stale_node_ids": list(context.stale_node_ids),
            "mission_id": mission.mission_id,
            "mission_revision": mission.revision,
            "mission_goals": [
                {
                    "goal_id": goal.goal_id,
                    "priority": goal.priority,
                    "frontier_ids": list(goal.frontier_ids),
                }
                for goal in mission.goals
            ],
            "issued_at_ns": issued_at.monotonic_ns,
            "ttl_ns": ttl_ns,
            "vehicles": [
                {
                    "agent_id": vehicle.agent_id,
                    "position_m": vehicle.position_m.tolist(),
                    "battery_fraction": vehicle.battery_fraction,
                    "link_quality": vehicle.link_quality,
                    "health": vehicle.health.value,
                    "available": vehicle.available,
                }
                for vehicle in vehicles
            ],
            "frontiers": [
                {
                    "frontier_id": frontier.frontier_id,
                    "position_m": frontier.position_m.tolist(),
                    "information_gain": frontier.information_gain,
                    "confidence": frontier.confidence,
                }
                for frontier in frontiers
            ],
            "no_fly_zones": [
                {
                    "zone_id": zone.zone_id,
                    "center_m": zone.center_m.tolist(),
                    "radius_m": zone.radius_m,
                }
                for zone in no_fly_zones
            ],
        }
        canonical = json.dumps(identifier_payload, sort_keys=True, separators=(",", ":")).encode()
        handoff_id = f"handoff-{hashlib.sha256(canonical).hexdigest()[:20]}"
        return cls(
            handoff_id,
            SKYLA_HANDOFF_SCHEMA,
            FrameId("global"),
            issued_at,
            issued_at.monotonic_ns + ttl_ns,
            context,
            mission,
            vehicles,
            frontiers,
            no_fly_zones,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "handoff_id": self.handoff_id,
            "schema": self.schema,
            "global_frame": str(self.global_frame),
            "issued_at": self.issued_at.to_dict(),
            "expires_at_ns": self.expires_at_ns,
            "context": {
                "revision": self.context.revision,
                "timestamp": self.context.timestamp.to_dict(),
                "degraded": self.context.degraded,
                "stale_node_ids": list(self.context.stale_node_ids),
                "nodes": [
                    {
                        "id": node.node_id,
                        "kind": node.kind.value,
                        "position_m": node.position_m.tolist(),
                        "confidence": node.confidence,
                        "updated_at": node.updated_at.to_dict(),
                    }
                    for node in self.context.nodes
                ],
                "edges": [
                    {
                        "source": edge.source,
                        "destination": edge.destination,
                        "relation": edge.relation,
                        "confidence": edge.confidence,
                    }
                    for edge in self.context.edges
                ],
            },
            "mission": {
                "mission_id": self.mission.mission_id,
                "revision": self.mission.revision,
                "goals": [
                    {
                        "goal_id": goal.goal_id,
                        "priority": goal.priority,
                        "frontier_ids": list(goal.frontier_ids),
                    }
                    for goal in self.mission.goals
                ],
            },
            "vehicles": [
                {
                    "agent_id": vehicle.agent_id,
                    "position_m": vehicle.position_m.tolist(),
                    "battery_fraction": vehicle.battery_fraction,
                    "link_quality": vehicle.link_quality,
                    "health": vehicle.health.value,
                    "available": vehicle.available,
                }
                for vehicle in self.vehicles
            ],
            "frontiers": [
                {
                    "frontier_id": frontier.frontier_id,
                    "position_m": frontier.position_m.tolist(),
                    "information_gain": frontier.information_gain,
                    "confidence": frontier.confidence,
                }
                for frontier in self.frontiers
            ],
            "no_fly_zones": [
                {
                    "zone_id": zone.zone_id,
                    "center_m": zone.center_m.tolist(),
                    "radius_m": zone.radius_m,
                }
                for zone in self.no_fly_zones
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
