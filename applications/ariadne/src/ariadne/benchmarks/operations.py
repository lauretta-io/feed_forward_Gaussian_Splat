"""Planning, telemetry, registry, security, deployment, and fault-operation benchmark."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter_ns

import numpy as np

from ariadne.common import Timestamp
from ariadne.context import SceneNode, UnifiedContext
from ariadne.context.scene_graph import SceneNodeKind
from ariadne.datasets import DatasetEvaluation
from ariadne.datasets.simulation import evaluate_simulation
from ariadne.deployment import CapabilityProbe, DeploymentProfile, HardwareCapabilities
from ariadne.planning import Frontier, WingmanPlanningState
from ariadne.registry import RegistryCatalog
from ariadne.security import HmacEnvelopeSecurity
from ariadne.skyla import (
    MissionGoal,
    MissionIntent,
    NoFlyZone,
    SkylaHandoff,
    SkylaMissionPlanner,
    VehicleState,
)
from ariadne.telemetry import ComponentHealth, TelemetryCollector

APP_ROOT = Path(__file__).resolve().parents[3]


def run_operations_benchmark(seed: int = 7) -> DatasetEvaluation:
    start_ns = perf_counter_ns()
    wingmen = (
        WingmanPlanningState("wingman_01", np.array([0.0, 0.0, 0.0]), 0.9),
        WingmanPlanningState("wingman_02", np.array([8.0, 0.0, 0.0]), 0.8),
        WingmanPlanningState("wingman_03", np.array([4.0, 6.0, 0.0]), 0.15),
    )
    frontiers = (
        Frontier("frontier_a", np.array([2.0, 0.0, 0.0]), 1.0, 0.9),
        Frontier("frontier_b", np.array([9.0, 1.0, 0.0]), 0.8, 0.9),
        Frontier("frontier_c", np.array([4.0, 8.0, 0.0]), 1.2, 0.65),
    )
    context = UnifiedContext()
    context_time = Timestamp(1_000)
    for wingman in wingmen:
        context.upsert_node(
            SceneNode(
                wingman.agent_id,
                SceneNodeKind.WINGMAN,
                wingman.position_m,
                0.95,
                context_time,
            )
        )
    for frontier in frontiers:
        context.upsert_node(
            SceneNode(
                frontier.frontier_id,
                SceneNodeKind.FRONTIER,
                frontier.position_m,
                frontier.confidence,
                context_time,
            )
        )
    context_snapshot = context.snapshot(context_time)
    mission = MissionIntent(
        "mission-reference",
        1,
        (
            MissionGoal(
                "inspect-frontiers",
                80,
                tuple(frontier.frontier_id for frontier in frontiers),
            ),
        ),
    )
    vehicles = tuple(
        VehicleState(
            wingman.agent_id,
            wingman.position_m,
            wingman.battery_fraction,
            0.9 if wingman.agent_id != "wingman_03" else 0.5,
            available=wingman.available,
        )
        for wingman in wingmen
    )
    no_fly_zones = (NoFlyZone("restricted_c", frontiers[2].position_m, 0.5),)
    handoff = SkylaHandoff.create(
        context=context_snapshot,
        mission=mission,
        vehicles=vehicles,
        frontiers=frontiers,
        no_fly_zones=no_fly_zones,
        issued_at=context_time,
    )
    planner = SkylaMissionPlanner()
    plan = planner.plan(handoff, now=Timestamp(1_100))
    replayed_plan = planner.plan(handoff, now=Timestamp(1_200))
    assignments = plan.assignments

    telemetry = TelemetryCollector(
        max_distribution_samples=64,
        mission_id=mission.mission_id,
        node_id="intelligence_01",
    )
    for latency in (2.1, 2.4, 2.2, 3.0, 2.5, 2.3, 4.2):
        telemetry.observe("pipeline_latency_ms", latency)
    telemetry.increment("packets_delivered", 12)
    telemetry.gauge("battery_fraction", 0.8)
    telemetry.set_health("mesh", ComponentHealth.HEALTHY)
    telemetry.record_event(
        Timestamp(900),
        "skyla",
        "handoff_created",
        fields={"context_revision": handoff.context.revision, "handoff_id": handoff.handoff_id},
    )
    telemetry_snapshot = telemetry.snapshot(Timestamp(1_000))
    prometheus_text = telemetry.prometheus_text(Timestamp(1_000))

    catalog = RegistryCatalog.load(
        APP_ROOT / "configs/datasets/registry.yaml",
        APP_ROOT / "configs/models/registry.yaml",
    )
    handoff_payload = handoff.to_json().encode()
    security = HmacEnvelopeSecurity({"intelligence_01": b"ariadne-reference-key"})
    envelope = security.sign(
        "intelligence_01",
        "skyla_01",
        handoff.handoff_id,
        context_time,
        handoff.expires_at_ns,
        handoff_payload,
    )
    verified = security.verify(envelope, destination="skyla_01", now=Timestamp(1_200))
    profile = DeploymentProfile("cpu_reference", 1, 512 * 1024 * 1024)
    capabilities = HardwareCapabilities("reference", 4, 8 * 1024**3, (), ())
    profile_validation = CapabilityProbe.validate(capabilities, profile)
    simulation = evaluate_simulation(seed)

    latency_summary = telemetry_snapshot.distributions["pipeline_latency_ms"]
    metrics: dict[str, int | float | str] = {
        "seed": seed,
        "assignment_count": len(assignments),
        "unassigned_frontiers": len(frontiers) - len(assignments),
        "handoff_context_revision": handoff.context.revision,
        "handoff_bytes": len(handoff_payload),
        "route_request_count": len(plan.route_requests),
        "blocked_frontier_count": len(plan.blocked_frontier_ids),
        "excluded_vehicle_count": len(plan.excluded_vehicle_ids),
        "idempotent_replay_count": planner.metrics["idempotent_replays"],
        "telemetry_p50_ms": float(latency_summary["p50"]),
        "telemetry_p95_ms": float(latency_summary["p95"]),
        "telemetry_event_count": len(telemetry_snapshot.events),
        "prometheus_series_count": len(prometheus_text.splitlines()),
        "dataset_registry_count": len(catalog.datasets),
        "model_registry_count": len(catalog.models),
        "security_verified": int(verified == handoff_payload),
        "deployment_compatible": int(profile_validation.compatible),
        "packet_loss_rate": simulation.metrics["packet_loss_rate"],
        "partition_duration_seconds": simulation.metrics["partition_duration_seconds"],
        "recovery_packets": simulation.metrics["recovery_packets"],
        "benchmark_latency_ms": (perf_counter_ns() - start_ns) / 1e6,
    }
    passed = (
        len(assignments) == 2
        and len(plan.route_requests) == 2
        and replayed_plan is plan
        and planner.metrics["idempotent_replays"] == 1
        and profile_validation.compatible
        and verified == handoff_payload
        and int(simulation.metrics["recovery_packets"]) > 0
    )
    return DatasetEvaluation(
        dataset="operations-reference",
        status="passed" if passed else "failed",
        agents=("wingman_01", "wingman_02", "wingman_03", "intelligence_01"),
        modalities=(
            "context_handoff",
            "mission_planning",
            "route_requests",
            "telemetry",
            "registry",
            "security",
            "deployment",
            "faults",
        ),
        metrics=metrics,
        details={
            "wingmen": [
                {
                    "id": wingman.agent_id,
                    "position_m": wingman.position_m.tolist(),
                    "battery": wingman.battery_fraction,
                    "eligible": wingman.battery_fraction >= 0.2,
                }
                for wingman in wingmen
            ],
            "frontiers": [
                {
                    "id": frontier.frontier_id,
                    "position_m": frontier.position_m.tolist(),
                    "information_gain": frontier.information_gain,
                }
                for frontier in frontiers
            ],
            "assignments": [
                {
                    "agent": assignment.agent_id,
                    "frontier": assignment.frontier_id,
                    "utility": assignment.utility,
                    "distance_m": assignment.travel_distance_m,
                }
                for assignment in assignments
            ],
            "handoff": {
                "id": handoff.handoff_id,
                "schema": handoff.schema,
                "frame": str(handoff.global_frame),
                "scene_revision": handoff.context.revision,
                "nodes": len(handoff.context.nodes),
                "edges": len(handoff.context.edges),
                "degraded": handoff.context.degraded,
                "mission_id": handoff.mission.mission_id,
                "mission_revision": handoff.mission.revision,
                "goal_count": len(handoff.mission.goals),
                "vehicle_count": len(handoff.vehicles),
                "frontier_count": len(handoff.frontiers),
                "no_fly_zone_count": len(handoff.no_fly_zones),
                "payload_bytes": len(handoff_payload),
                "gates": {
                    "schema": True,
                    "global_frame": True,
                    "freshness": not handoff.context.degraded,
                    "version": handoff.mission.revision > 0,
                    "idempotency": planner.metrics["idempotent_replays"] == 1,
                },
            },
            "skyla_planning": {
                "status": plan.status,
                "route_requests": [
                    {
                        "task_id": route.task_id,
                        "idempotency_key": route.idempotency_key,
                        "agent": route.agent_id,
                        "frontier": route.frontier_id,
                        "priority": route.priority,
                        "waypoints_m": [
                            waypoint.tolist() for waypoint in route.waypoints_m
                        ],
                        "requires_local_safety_validation": (
                            route.requires_local_safety_validation
                        ),
                    }
                    for route in plan.route_requests
                ],
                "blocked_frontiers": list(plan.blocked_frontier_ids),
                "excluded_vehicles": list(plan.excluded_vehicle_ids),
                "metrics": planner.metrics,
                "stages": [
                    "Mission intent",
                    "Context ingest",
                    "Constraint filtering",
                    "Utility + risk",
                    "Multi-agent allocation",
                    "Route requests",
                ],
            },
            "telemetry": {
                "latencies_ms": [2.1, 2.4, 2.2, 3.0, 2.5, 2.3, 4.2],
                "p50_ms": latency_summary["p50"],
                "p95_ms": latency_summary["p95"],
                "health": {key: value.value for key, value in telemetry_snapshot.health.items()},
                "mission_id": telemetry_snapshot.mission_id,
                "node_id": telemetry_snapshot.node_id,
                "event_count": len(telemetry_snapshot.events),
                "prometheus_series": prometheus_text.splitlines(),
            },
            "registry": {
                "datasets": [entry.name for entry in catalog.datasets],
                "models": [entry.name for entry in catalog.models],
            },
            "security": security.metrics,
            "deployment": {
                "profile": profile.name,
                "compatible": profile_validation.compatible,
                "missing": profile_validation.missing,
            },
            "simulation": {
                "packet_loss_rate": simulation.metrics["packet_loss_rate"],
                "partition_duration_seconds": simulation.metrics["partition_duration_seconds"],
                "recovery_packets": simulation.metrics["recovery_packets"],
                "drift_improvement_percent": simulation.metrics["drift_improvement_percent"],
            },
        },
    )
