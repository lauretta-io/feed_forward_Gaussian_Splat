from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ariadne.common import Timestamp
from ariadne.deployment import CapabilityProbe, DeploymentProfile, HardwareCapabilities
from ariadne.planning import Frontier, FrontierAuctionPlanner, WingmanPlanningState
from ariadne.registry import ExperimentEntry, RegistryCatalog
from ariadne.security import HmacEnvelopeSecurity
from ariadne.telemetry import ComponentHealth, TelemetryCollector

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_frontier_auction_is_deterministic_and_replans_after_failure() -> None:
    wingmen = (
        WingmanPlanningState("wingman_01", np.array([0.0, 0.0, 0.0]), 0.9),
        WingmanPlanningState("wingman_02", np.array([8.0, 0.0, 0.0]), 0.8),
    )
    frontiers = (
        Frontier("frontier_a", np.array([2.0, 0.0, 0.0]), 1.0, 0.9),
        Frontier("frontier_b", np.array([9.0, 0.0, 0.0]), 0.8, 0.9),
    )
    planner = FrontierAuctionPlanner()
    first = planner.plan(wingmen, frontiers)
    assert [(item.agent_id, item.frontier_id) for item in first] == [
        ("wingman_01", "frontier_a"),
        ("wingman_02", "frontier_b"),
    ]
    failed = (wingmen[0], replace(wingmen[1], available=False))
    second = planner.plan(failed, frontiers)
    assert len(second) == 1 and second[0].agent_id == "wingman_01"


def test_telemetry_bounds_and_percentiles() -> None:
    telemetry = TelemetryCollector(max_distribution_samples=3, max_metrics=3)
    telemetry.increment("frames", 2)
    telemetry.gauge("battery", 0.7)
    for value in (1.0, 2.0, 3.0, 4.0):
        telemetry.observe("latency_ms", value)
    telemetry.set_health("vio", ComponentHealth.DEGRADED)
    snapshot = telemetry.snapshot(Timestamp(10))
    assert snapshot.counters["frames"] == 2
    assert snapshot.distributions["latency_ms"]["count"] == 3
    assert snapshot.distributions["latency_ms"]["max"] == 4.0
    with pytest.raises(ValueError, match="cardinality"):
        telemetry.gauge("memory_mb", 20.0)


def test_telemetry_exports_bounded_redacted_trace_and_prometheus(tmp_path: Path) -> None:
    telemetry = TelemetryCollector(
        mission_id="mission-7",
        node_id="wingman_01",
        max_events=2,
    )
    telemetry.increment("frames", 3)
    telemetry.gauge("battery_fraction", 0.75)
    telemetry.observe("latency_ms", 2.5)
    telemetry.set_health("vio", ComponentHealth.RECOVERING)
    telemetry.record_event(Timestamp(10), "mesh", "connected")
    telemetry.record_event(
        Timestamp(11),
        "vio",
        "relocalized",
        fields={"matches": 42, "secret": "must-not-leak"},
    )
    telemetry.record_event(Timestamp(12), "planner", "route_ready")
    snapshot = telemetry.snapshot(Timestamp(20))
    assert len(snapshot.events) == 2
    assert snapshot.events[0].event == "relocalized"
    assert snapshot.events[0].fields["secret"] == "[REDACTED]"

    prometheus = telemetry.prometheus_text(Timestamp(20))
    assert 'mission_id="mission-7",node_id="wingman_01"' in prometheus
    assert "ariadne_frames_total" in prometheus
    assert "ariadne_component_health" in prometheus

    path = tmp_path / "telemetry.json"
    snapshot.write_json(path)
    content = path.read_text(encoding="utf-8")
    assert '"mission_id": "mission-7"' in content
    assert "must-not-leak" not in content


def test_registry_loads_catalog_and_writes_experiment(tmp_path: Path) -> None:
    catalog = RegistryCatalog.load(
        PROJECT_ROOT / "configs/datasets/registry.yaml",
        PROJECT_ROOT / "configs/models/registry.yaml",
    )
    assert {entry.name for entry in catalog.datasets} >= {"d2slam", "miluv", "s3e"}
    assert {entry.name for entry in catalog.models} >= {"openvins", "orbslam3"}
    output = tmp_path / "experiment.json"
    RegistryCatalog.write_experiment(
        ExperimentEntry("exp-1", "global-scene", 7, "sha256:config", "report.json", "passed"),
        output,
    )
    assert '"status": "passed"' in output.read_text()


def test_secure_envelope_rejects_tamper_replay_expiry_and_wrong_destination() -> None:
    security = HmacEnvelopeSecurity({"wingman_01": b"0123456789abcdef"})
    envelope = security.sign(
        "wingman_01", "intelligence_01", "nonce-1", Timestamp(10), 100, b"payload"
    )
    tampered = replace(envelope, payload=b"tampered")
    with pytest.raises(ValueError, match="signature"):
        security.verify(tampered, destination="intelligence_01", now=Timestamp(20))
    assert security.verify(envelope, destination="intelligence_01", now=Timestamp(20)) == b"payload"
    with pytest.raises(ValueError, match="replay"):
        security.verify(envelope, destination="intelligence_01", now=Timestamp(20))
    expiring = security.sign(
        "wingman_01", "intelligence_01", "nonce-2", Timestamp(10), 20, b"payload"
    )
    with pytest.raises(ValueError, match="destination or expiry"):
        security.verify(expiring, destination="intelligence_01", now=Timestamp(20))


def test_deployment_profile_reports_every_missing_capability() -> None:
    capabilities = HardwareCapabilities("x86_64", 2, 1024, (), ())
    profile = DeploymentProfile("wingman", 4, 2048, ("cuda",), 1)
    validation = CapabilityProbe.validate(capabilities, profile)
    assert not validation.compatible
    assert validation.missing == (
        "cpu_cores>=4",
        "memory_bytes>=2048",
        "accelerator:cuda",
        "cameras>=1",
    )
