"""Deterministic SKYLA mission allocation and route-request reference."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from ariadne.common import Timestamp
from ariadne.planning import TaskAssignment
from ariadne.skyla.handoff import SkylaHandoff, VehicleHealth

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteRequest:
    task_id: str
    idempotency_key: str
    handoff_id: str
    context_revision: int
    mission_revision: int
    agent_id: str
    frontier_id: str
    priority: int
    waypoints_m: tuple[npt.NDArray[np.float64], ...]
    expires_at_ns: int
    requires_local_safety_validation: bool = True

    def __post_init__(self) -> None:
        if (
            not self.task_id
            or not self.idempotency_key
            or not self.agent_id
            or not self.frontier_id
            or self.context_revision < 0
            or self.mission_revision <= 0
            or not 1 <= self.priority <= 100
            or len(self.waypoints_m) < 2
        ):
            raise ValueError("route request fields are invalid")
        immutable_waypoints = []
        for waypoint in self.waypoints_m:
            value = np.asarray(waypoint, dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError("route waypoints must be finite three-vectors")
            value = value.copy()
            value.setflags(write=False)
            immutable_waypoints.append(value)
        object.__setattr__(self, "waypoints_m", tuple(immutable_waypoints))


@dataclass(frozen=True)
class SkylaPlanningResult:
    handoff_id: str
    status: Literal["passed", "degraded", "failed"]
    assignments: tuple[TaskAssignment, ...]
    route_requests: tuple[RouteRequest, ...]
    blocked_frontier_ids: tuple[str, ...]
    excluded_vehicle_ids: tuple[str, ...]
    reasons: tuple[str, ...]


class SkylaMissionPlanner:
    def __init__(
        self,
        *,
        minimum_battery_fraction: float = 0.2,
        minimum_link_quality: float = 0.15,
        fail_closed_on_degraded_context: bool = True,
    ) -> None:
        if not 0 <= minimum_battery_fraction < 1:
            raise ValueError("minimum battery fraction must be in [0, 1)")
        if not 0 <= minimum_link_quality <= 1:
            raise ValueError("minimum link quality must be in [0, 1]")
        self.minimum_battery_fraction = minimum_battery_fraction
        self.minimum_link_quality = minimum_link_quality
        self.fail_closed_on_degraded_context = fail_closed_on_degraded_context
        self._results: dict[str, SkylaPlanningResult] = {}
        self._latest_context_revision = -1
        self.metrics = {
            "plans": 0,
            "idempotent_replays": 0,
            "route_requests": 0,
            "blocked_frontiers": 0,
            "excluded_vehicles": 0,
            "degraded_plans": 0,
        }

    def plan(self, handoff: SkylaHandoff, *, now: Timestamp) -> SkylaPlanningResult:
        if now.monotonic_ns >= handoff.expires_at_ns:
            raise ValueError("SKYLA hand-off has expired")
        if handoff.context.revision < self._latest_context_revision:
            raise ValueError("SKYLA hand-off context revision is stale")
        cached = self._results.get(handoff.handoff_id)
        if cached is not None:
            self.metrics["idempotent_replays"] += 1
            return cached
        if handoff.context.degraded and self.fail_closed_on_degraded_context:
            result = SkylaPlanningResult(
                handoff.handoff_id,
                "degraded",
                (),
                (),
                (),
                tuple(vehicle.agent_id for vehicle in handoff.vehicles),
                ("context_degraded",),
            )
            self._record(handoff, result)
            self.metrics["degraded_plans"] += 1
            return result

        goal_priorities = {
            frontier_id: max(
                goal.priority
                for goal in handoff.mission.goals
                if frontier_id in goal.frontier_ids
            )
            for frontier_id in {
                value for goal in handoff.mission.goals for value in goal.frontier_ids
            }
        }
        excluded = tuple(
            sorted(
                vehicle.agent_id
                for vehicle in handoff.vehicles
                if (
                    not vehicle.available
                    or vehicle.health is VehicleHealth.FAILED
                    or vehicle.battery_fraction < self.minimum_battery_fraction
                    or vehicle.link_quality < self.minimum_link_quality
                )
            )
        )
        candidates: list[tuple[float, str, str, float, int]] = []
        constrained_frontiers: set[str] = set()
        feasible_frontiers: set[str] = set()
        for vehicle in handoff.vehicles:
            if vehicle.agent_id in excluded:
                continue
            for frontier in handoff.frontiers:
                priority = goal_priorities.get(frontier.frontier_id)
                if priority is None:
                    continue
                if any(
                    zone.blocks_segment(vehicle.position_m, frontier.position_m)
                    for zone in handoff.no_fly_zones
                ):
                    constrained_frontiers.add(frontier.frontier_id)
                    continue
                feasible_frontiers.add(frontier.frontier_id)
                distance = float(np.linalg.norm(frontier.position_m - vehicle.position_m))
                utility = (
                    frontier.information_gain
                    * frontier.confidence
                    * vehicle.battery_fraction
                    * vehicle.link_quality
                    * (priority / 100.0)
                    / (1.0 + distance)
                )
                candidates.append(
                    (utility, vehicle.agent_id, frontier.frontier_id, distance, priority)
                )

        assignments: list[TaskAssignment] = []
        routes: list[RouteRequest] = []
        used_agents: set[str] = set()
        used_frontiers: set[str] = set()
        vehicles = {vehicle.agent_id: vehicle for vehicle in handoff.vehicles}
        frontiers = {frontier.frontier_id: frontier for frontier in handoff.frontiers}
        for utility, agent_id, frontier_id, distance, priority in sorted(
            candidates, key=lambda item: (-item[0], item[1], item[2])
        ):
            if agent_id in used_agents or frontier_id in used_frontiers:
                continue
            assignment = TaskAssignment(agent_id, frontier_id, utility, distance)
            identifier = f"{handoff.handoff_id}:{agent_id}:{frontier_id}"
            task_id = f"task-{hashlib.sha256(identifier.encode()).hexdigest()[:20]}"
            assignments.append(assignment)
            routes.append(
                RouteRequest(
                    task_id,
                    identifier,
                    handoff.handoff_id,
                    handoff.context.revision,
                    handoff.mission.revision,
                    agent_id,
                    frontier_id,
                    priority,
                    (vehicles[agent_id].position_m, frontiers[frontier_id].position_m),
                    handoff.expires_at_ns,
                )
            )
            used_agents.add(agent_id)
            used_frontiers.add(frontier_id)

        assignments.sort(key=lambda item: item.agent_id)
        routes.sort(key=lambda item: item.agent_id)
        blocked = constrained_frontiers - feasible_frontiers
        reasons = () if routes else ("no_feasible_routes",)
        result = SkylaPlanningResult(
            handoff.handoff_id,
            "passed" if routes else "failed",
            tuple(assignments),
            tuple(routes),
            tuple(sorted(blocked)),
            excluded,
            reasons,
        )
        self._record(handoff, result)
        LOGGER.info(
            "skyla_plan handoff=%s routes=%d blocked=%d excluded=%d",
            handoff.handoff_id,
            len(routes),
            len(blocked),
            len(excluded),
        )
        return result

    def _record(self, handoff: SkylaHandoff, result: SkylaPlanningResult) -> None:
        self._results[handoff.handoff_id] = result
        self._latest_context_revision = max(
            self._latest_context_revision, handoff.context.revision
        )
        self.metrics["plans"] += 1
        self.metrics["route_requests"] += len(result.route_requests)
        self.metrics["blocked_frontiers"] += len(result.blocked_frontier_ids)
        self.metrics["excluded_vehicles"] += len(result.excluded_vehicle_ids)
