from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from ariadne.common import Timestamp
from ariadne.context import SceneNode, UnifiedContext
from ariadne.context.scene_graph import SceneNodeKind
from ariadne.planning import Frontier
from ariadne.skyla import (
    SKYLA_HANDOFF_SCHEMA,
    MissionGoal,
    MissionIntent,
    NoFlyZone,
    SkylaHandoff,
    SkylaMissionPlanner,
    VehicleState,
)


def _handoff(*, blocked: bool = False) -> SkylaHandoff:
    timestamp = Timestamp(100)
    context = UnifiedContext()
    context.upsert_node(
        SceneNode("wingman_01", SceneNodeKind.WINGMAN, np.zeros(3), 0.9, timestamp)
    )
    context.upsert_node(
        SceneNode(
            "frontier_a",
            SceneNodeKind.FRONTIER,
            np.array([2.0, 0.0, 0.0]),
            0.9,
            timestamp,
        )
    )
    context.upsert_node(
        SceneNode(
            "frontier_b",
            SceneNodeKind.FRONTIER,
            np.array([0.0, 2.0, 0.0]),
            0.8,
            timestamp,
        )
    )
    zones = (
        (NoFlyZone("restricted", np.array([1.0, 0.0, 0.0]), 0.25),)
        if blocked
        else ()
    )
    return SkylaHandoff.create(
        context=context.snapshot(timestamp),
        mission=MissionIntent(
            "mission-1",
            3,
            (MissionGoal("inspect", 90, ("frontier_a", "frontier_b")),),
        ),
        vehicles=(
            VehicleState("wingman_01", np.zeros(3), 0.8, 0.9),
            VehicleState("wingman_02", np.array([3.0, 3.0, 0.0]), 0.1, 0.9),
        ),
        frontiers=(
            Frontier("frontier_a", np.array([2.0, 0.0, 0.0]), 1.0, 0.9),
            Frontier("frontier_b", np.array([0.0, 2.0, 0.0]), 0.8, 0.8),
        ),
        no_fly_zones=zones,
        issued_at=timestamp,
        ttl_ns=1_000,
    )


def test_handoff_is_versioned_deterministic_and_json_serializable() -> None:
    first = _handoff()
    second = _handoff()
    assert first.schema == SKYLA_HANDOFF_SCHEMA
    assert first.handoff_id == second.handoff_id
    payload = json.loads(first.to_json())
    assert payload["global_frame"] == "global"
    assert payload["context"]["revision"] == 3
    assert payload["mission"]["revision"] == 3
    assert len(payload["vehicles"]) == 2
    assert _handoff(blocked=True).handoff_id != first.handoff_id


def test_planner_filters_constraints_and_replays_idempotently() -> None:
    handoff = _handoff(blocked=True)
    planner = SkylaMissionPlanner()
    first = planner.plan(handoff, now=Timestamp(200))
    replay = planner.plan(handoff, now=Timestamp(300))

    assert first.status == "passed"
    assert replay is first
    assert first.blocked_frontier_ids == ("frontier_a",)
    assert first.excluded_vehicle_ids == ("wingman_02",)
    assert len(first.route_requests) == 1
    assert first.route_requests[0].frontier_id == "frontier_b"
    assert first.route_requests[0].requires_local_safety_validation
    assert planner.metrics["idempotent_replays"] == 1


def test_planner_fails_closed_on_degraded_context() -> None:
    handoff = _handoff()
    degraded = replace(
        handoff,
        handoff_id="handoff-degraded",
        context=replace(
            handoff.context,
            degraded=True,
            stale_node_ids=("wingman_01",),
        ),
    )
    result = SkylaMissionPlanner().plan(degraded, now=Timestamp(200))
    assert result.status == "degraded"
    assert result.route_requests == ()
    assert result.reasons == ("context_degraded",)


def test_planner_rejects_expired_and_stale_context() -> None:
    planner = SkylaMissionPlanner()
    first = _handoff()
    planner.plan(first, now=Timestamp(200))
    stale = replace(
        first,
        handoff_id="handoff-stale",
        context=replace(first.context, revision=first.context.revision - 1),
    )
    with pytest.raises(ValueError, match="revision is stale"):
        planner.plan(stale, now=Timestamp(200))
    with pytest.raises(ValueError, match="expired"):
        SkylaMissionPlanner().plan(first, now=Timestamp(first.expires_at_ns))
    with pytest.raises(ValueError, match="expired"):
        planner.plan(first, now=Timestamp(first.expires_at_ns))


def test_handoff_rejects_unknown_mission_frontier() -> None:
    handoff = _handoff()
    mission = MissionIntent(
        "mission-2",
        1,
        (MissionGoal("unknown", 50, ("missing_frontier",)),),
    )
    with pytest.raises(ValueError, match="unknown frontiers"):
        SkylaHandoff.create(
            context=handoff.context,
            mission=mission,
            vehicles=handoff.vehicles,
            frontiers=handoff.frontiers,
            issued_at=handoff.issued_at,
        )
