"""Shared controlled SE(3) rationalization primitives for dataset benchmarks."""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Literal

import numpy as np
import numpy.typing as npt

from ariadne.optimization import (
    RobustSE3PoseGraph,
    SE3PoseConstraint,
    SE3PoseGraphResult,
)
from ariadne.pose_correction import CorrectionCadenceScheduler, CorrectionDemand

PoseMatrix = npt.NDArray[np.float64]
PoseSeries = dict[str, tuple[PoseMatrix, ...]]
CrossAgentFactors = dict[tuple[int, str, str], PoseMatrix]
FactorEvidenceSource = Literal[
    "production",
    "measured",
    "controlled",
    "ground_truth_derived",
    "unavailable",
]


@dataclass(frozen=True)
class PoseSample:
    timestamp_s: float
    matrix: PoseMatrix


@dataclass(frozen=True)
class GraphRun:
    result: SE3PoseGraphResult
    optimized: PoseSeries
    false_loop_index: int
    global_correction_count: int
    correction_count_by_agent: dict[str, int]
    correction_payload_bytes_by_agent: dict[str, int]
    optimization_latency_ms: float


@dataclass(frozen=True)
class GlobalPoseClaimEvidence:
    """Fail-closed evidence gate for deployment-level global-pose claims."""

    odometry_source: FactorEvidenceSource
    position_reference_source: FactorEvidenceSource
    orientation_reference_source: FactorEvidenceSource
    cross_agent_source: FactorEvidenceSource
    uses_ground_truth_in_estimator: bool
    causal: bool
    all_agents_position_target_met: bool
    fleet_position_target_met: bool
    orientation_target_met: bool

    @property
    def position_claim_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.odometry_source != "production":
            reasons.append("production local odometry is unavailable")
        if self.position_reference_source != "measured":
            reasons.append("position factors are not independent measured observations")
        if self.cross_agent_source != "measured":
            reasons.append("cross-agent factors are not independent measured observations")
        if self.uses_ground_truth_in_estimator:
            reasons.append("estimator consumes evaluation ground truth")
        if not self.causal:
            reasons.append("estimator uses future measurements")
        if not self.fleet_position_target_met:
            reasons.append("fleet position target is not met")
        if not self.all_agents_position_target_met:
            reasons.append("position target is not met by every agent")
        return tuple(reasons)

    @property
    def position_claim_eligible(self) -> bool:
        return not self.position_claim_reasons

    @property
    def full_pose_claim_reasons(self) -> tuple[str, ...]:
        reasons = list(self.position_claim_reasons)
        if self.orientation_reference_source != "measured":
            reasons.append("orientation factors are not independent measured observations")
        if not self.orientation_target_met:
            reasons.append("orientation target is not met")
        return tuple(reasons)

    @property
    def full_pose_claim_eligible(self) -> bool:
        return not self.full_pose_claim_reasons

    def as_dict(self) -> dict[str, object]:
        return {
            "odometry_source": self.odometry_source,
            "position_reference_source": self.position_reference_source,
            "orientation_reference_source": self.orientation_reference_source,
            "cross_agent_source": self.cross_agent_source,
            "uses_ground_truth_in_estimator": self.uses_ground_truth_in_estimator,
            "causal": self.causal,
            "all_agents_position_target_met": self.all_agents_position_target_met,
            "fleet_position_target_met": self.fleet_position_target_met,
            "orientation_target_met": self.orientation_target_met,
            "position_claim_eligible": self.position_claim_eligible,
            "full_pose_claim_eligible": self.full_pose_claim_eligible,
            "position_claim_reasons": list(self.position_claim_reasons),
            "full_pose_claim_reasons": list(self.full_pose_claim_reasons),
        }


def rotation_matrix(quaternion_xyzw: npt.ArrayLike) -> npt.NDArray[np.float64]:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must be a finite xyzw vector")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("quaternion norm must be non-zero")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_xyzw(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert a rotation matrix to a normalized xyzw quaternion."""
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        quaternion = np.asarray(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ]
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            quaternion = np.asarray(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            quaternion = np.asarray(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
    return np.asarray(quaternion / np.linalg.norm(quaternion), dtype=np.float64)


def rotation_vector(vector_rad: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Map a compact axis-angle vector into SO(3)."""
    vector = np.asarray(vector_rad, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotation vector must be a finite 3-vector")
    angle = float(np.linalg.norm(vector))
    if angle <= np.finfo(np.float64).eps:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    return np.asarray(
        np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew),
        dtype=np.float64,
    )


def rotation_error_rad(estimated: PoseMatrix, truth: PoseMatrix) -> float:
    difference = truth[:3, :3].T @ estimated[:3, :3]
    return float(np.arccos(np.clip((float(np.trace(difference)) - 1.0) / 2.0, -1.0, 1.0)))


def sample_overlap(
    trajectories: dict[str, tuple[PoseSample, ...]],
    sample_count: int,
) -> tuple[dict[str, tuple[PoseSample, ...]], float, float]:
    if sample_count < 3:
        raise ValueError("global-pose benchmark requires at least three samples")
    start_s = max(samples[0].timestamp_s for samples in trajectories.values())
    end_s = min(samples[-1].timestamp_s for samples in trajectories.values())
    if end_s <= start_s:
        raise ValueError("ground-truth trajectories have no common time overlap")
    target_times = np.linspace(start_s, end_s, sample_count)
    selected: dict[str, tuple[PoseSample, ...]] = {}
    for agent, samples in trajectories.items():
        times = np.asarray([sample.timestamp_s for sample in samples])
        indices = [int(np.argmin(np.abs(times - target_time))) for target_time in target_times]
        if len(set(indices)) != len(indices):
            raise ValueError("requested sample count exceeds distinct overlap poses")
        selected[agent] = tuple(samples[index] for index in indices)
    return selected, start_s, end_s


def pose_constraint(
    source: str,
    destination: str,
    relative: PoseMatrix,
    *,
    information: float,
    kind: str,
    translation_information: float | None = None,
    rotation_information: float | None = None,
) -> SE3PoseConstraint:
    translation_weight = information if translation_information is None else translation_information
    rotation_weight = information if rotation_information is None else rotation_information
    if translation_weight <= 0 or rotation_weight <= 0:
        raise ValueError("constraint translation and rotation information must be positive")
    covariance = np.diag(
        [
            *(1.0 / translation_weight for _ in range(3)),
            *(1.0 / rotation_weight for _ in range(3)),
        ]
    )
    return SE3PoseConstraint(
        source,
        destination,
        relative[:3, 3],
        quaternion_xyzw(relative[:3, :3]),
        covariance,
        information,
        kind,
    )


def translation_ate(estimates: PoseSeries, truth: PoseSeries) -> float:
    errors = [
        np.linalg.norm(estimates[agent][index][:3, 3] - truth[agent][index][:3, 3])
        for agent in truth
        for index in range(len(truth[agent]))
    ]
    return float(np.sqrt(np.mean(np.square(errors))))


def translation_rpe(estimates: PoseSeries, truth: PoseSeries) -> float:
    errors: list[float] = []
    for agent in truth:
        for index in range(1, len(truth[agent])):
            estimated_delta = np.linalg.inv(estimates[agent][index - 1]) @ estimates[agent][index]
            truth_delta = np.linalg.inv(truth[agent][index - 1]) @ truth[agent][index]
            errors.append(float(np.linalg.norm(estimated_delta[:3, 3] - truth_delta[:3, 3])))
    return float(np.sqrt(np.mean(np.square(errors))))


def orientation_rmse(estimates: PoseSeries, truth: PoseSeries) -> float:
    errors = [
        rotation_error_rad(estimates[agent][index], truth[agent][index])
        for agent in truth
        for index in range(len(truth[agent]))
    ]
    return float(np.sqrt(np.mean(np.square(errors))))


def rotation_rpe(estimates: PoseSeries, truth: PoseSeries) -> float:
    errors: list[float] = []
    for agent in truth:
        for index in range(1, len(truth[agent])):
            estimated_delta = np.linalg.inv(estimates[agent][index - 1]) @ estimates[agent][index]
            truth_delta = np.linalg.inv(truth[agent][index - 1]) @ truth[agent][index]
            errors.append(rotation_error_rad(estimated_delta, truth_delta))
    return float(np.sqrt(np.mean(np.square(errors))))


def correction_payload_bytes(agent: str, revision: int, pose: PoseMatrix) -> int:
    payload = {
        "agent": agent,
        "revision": revision,
        "translation_m": np.round(pose[:3, 3], 6).tolist(),
        "quaternion_xyzw": np.round(quaternion_xyzw(pose[:3, :3]), 7).tolist(),
        "confidence": 0.95,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return len(zlib.compress(encoded))


def run_graph(
    agents: tuple[str, ...],
    truth: PoseSeries,
    odometry: PoseSeries,
    cross_agent: CrossAgentFactors,
    global_corrections: dict[tuple[int, str], PoseMatrix],
    *,
    correction_interval: int | None,
    correction_indices: frozenset[tuple[int, str]] | None = None,
    global_graph_information: float = 60.0,
    global_translation_information: float | None = None,
    global_rotation_information: float | None = None,
    global_constraint_kind: str = "global-correction",
) -> GraphRun:
    if len(agents) < 2 or set(agents) != set(truth) or set(agents) != set(odometry):
        raise ValueError("graph agents and pose series do not match")
    sample_count = len(truth[agents[0]])
    graph = RobustSE3PoseGraph("world", max_constraints=sample_count * len(agents) * 5)
    for agent in agents:
        graph.add_constraint(
            pose_constraint(
                "world",
                f"{agent}:0",
                truth[agent][0],
                information=100.0,
                kind="initial-global-anchor",
            )
        )
    correction_payload_bytes_by_agent = dict.fromkeys(agents, 0)
    correction_count_by_agent = dict.fromkeys(agents, 0)
    global_correction_count = 0
    for index in range(1, sample_count):
        for agent in agents:
            graph.add_constraint(
                pose_constraint(
                    f"{agent}:{index - 1}",
                    f"{agent}:{index}",
                    odometry[agent][index - 1],
                    information=20.0,
                    kind="local-vio-odometry",
                )
            )
        for (factor_index, source, destination), relative in cross_agent.items():
            if factor_index == index:
                graph.add_constraint(
                    pose_constraint(
                        f"{source}:{index}",
                        f"{destination}:{index}",
                        relative,
                        information=30.0,
                        kind="controlled-cross-agent-observation",
                    )
                )
        if correction_indices is not None:
            correction_agents = [agent for agent in agents if (index, agent) in correction_indices]
        elif correction_interval is not None and (
            index % correction_interval == 0 or index == sample_count - 1
        ):
            correction_agents = list(agents)
        else:
            correction_agents = []
        for agent in correction_agents:
            corrected_pose = global_corrections[(index, agent)]
            graph.add_constraint(
                pose_constraint(
                    "world",
                    f"{agent}:{index}",
                    corrected_pose,
                    information=global_graph_information,
                    translation_information=global_translation_information,
                    rotation_information=global_rotation_information,
                    kind=global_constraint_kind,
                )
            )
            correction_payload_bytes_by_agent[agent] += correction_payload_bytes(
                agent, index, corrected_pose
            )
            correction_count_by_agent[agent] += 1
            global_correction_count += 1

    false_loop = np.eye(4)
    false_loop[:3, 3] = np.asarray([25.0, -18.0, 6.0])
    false_loop_index = graph.revision
    graph.add_constraint(
        pose_constraint(
            f"{agents[1]}:0",
            f"{agents[-1]}:{sample_count - 1}",
            false_loop,
            information=0.1,
            kind="deliberate-false-loop",
        )
    )
    optimization_start_ns = perf_counter_ns()
    result = graph.optimize(
        translation_gate_m=1.5,
        rotation_gate_rad=0.35,
        rationalize_se3=True,
    )
    optimization_latency_ms = (perf_counter_ns() - optimization_start_ns) / 1e6
    optimized = {
        agent: tuple(result.poses[f"{agent}:{index}"] for index in range(sample_count))
        for agent in agents
    }
    return GraphRun(
        result,
        optimized,
        false_loop_index,
        global_correction_count,
        correction_count_by_agent,
        correction_payload_bytes_by_agent,
        optimization_latency_ms,
    )


def adaptive_correction_schedule(
    agents: tuple[str, ...],
    sampled: dict[str, tuple[PoseSample, ...]],
    truth: PoseSeries,
    baseline: PoseSeries,
    global_corrections: dict[tuple[int, str], PoseMatrix],
    *,
    target_error_m: float,
    target_orientation_error_rad: float,
    sample_interval_s: float,
    network_profiles: dict[str, tuple[float, float]],
    nominal_interval_s: float | None = None,
    maximum_interval_s: float | None = None,
) -> tuple[frozenset[tuple[int, str]], list[dict[str, object]], dict[str, int]]:
    nominal = nominal_interval_s or sample_interval_s
    maximum = maximum_interval_s or sample_interval_s * 8
    scheduler = CorrectionCadenceScheduler(
        target_error_m=target_error_m,
        target_orientation_error_rad=target_orientation_error_rad,
        nominal_interval_s=nominal,
        minimum_interval_s=sample_interval_s,
        maximum_interval_s=maximum,
        evaluation_period_s=sample_interval_s,
        max_corrections_per_cycle=max(2, len(agents) - 1),
    )
    if set(network_profiles) != set(agents):
        raise ValueError("network profiles must cover every graph agent")
    last_correction = dict.fromkeys(agents, 0)
    selected: set[tuple[int, str]] = set()
    trace: list[dict[str, object]] = []
    for index in range(1, len(truth[agents[0]])):
        demands = []
        for agent in agents:
            previous = last_correction[agent]
            alignment = truth[agent][previous] @ np.linalg.inv(baseline[agent][previous])
            predicted = alignment @ baseline[agent][index]
            estimated_error_m = float(np.linalg.norm(predicted[:3, 3] - truth[agent][index][:3, 3]))
            orientation_error = rotation_error_rad(predicted, truth[agent][index])
            age_s = sampled[agent][index].timestamp_s - sampled[agent][previous].timestamp_s
            drift_denominator = max(age_s, sample_interval_s)
            steps = index - previous
            covariance_sigma_m = 0.002 + 0.001 * np.sqrt(steps)
            orientation_covariance_sigma_rad = 0.001 + 0.0005 * np.sqrt(steps)
            link_quality, queue_utilization = network_profiles[agent]
            demands.append(
                CorrectionDemand(
                    agent,
                    estimated_error_m,
                    estimated_error_m / drift_denominator,
                    3.0 * covariance_sigma_m**2,
                    age_s,
                    link_quality,
                    queue_utilization,
                    correction_payload_bytes(
                        agent,
                        index,
                        global_corrections[(index, agent)],
                    ),
                    orientation_error_rad=orientation_error,
                    orientation_drift_rate_rad_s=orientation_error / drift_denominator,
                    orientation_covariance_trace_rad2=(3.0 * orientation_covariance_sigma_rad**2),
                )
            )
        schedule = scheduler.schedule(tuple(demands))
        for agent in schedule.selected_agent_ids:
            selected.add((index, agent))
            last_correction[agent] = index
        trace.append(
            {
                "sample_index": index,
                "selected_agents": list(schedule.selected_agent_ids),
                "selected_payload_bytes": schedule.selected_payload_bytes,
                "capacity_overridden": schedule.capacity_overridden,
                "maximum_predicted_next_error_m": max(
                    decision.predicted_next_error_m for decision in schedule.decisions
                ),
                "maximum_predicted_next_orientation_error_rad": max(
                    decision.predicted_next_orientation_error_rad for decision in schedule.decisions
                ),
            }
        )
    return frozenset(selected), trace, dict(scheduler.metrics)
