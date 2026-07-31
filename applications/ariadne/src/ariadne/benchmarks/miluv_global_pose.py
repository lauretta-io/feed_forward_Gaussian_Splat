"""MILUV multi-agent SE(3) rationalization benchmark."""

from __future__ import annotations

import csv
import io
import zipfile
from itertools import pairwise
from pathlib import Path
from time import perf_counter_ns
from typing import cast

import numpy as np
import numpy.typing as npt

from ariadne.benchmarks.global_pose_rationalization import (
    CrossAgentFactors,
    GlobalPoseClaimEvidence,
    GraphRun,
    PoseMatrix,
    PoseSample,
    PoseSeries,
    adaptive_correction_schedule,
    correction_payload_bytes,
    orientation_rmse,
    rotation_matrix,
    rotation_rpe,
    rotation_vector,
    run_graph,
    sample_overlap,
    translation_ate,
    translation_rpe,
)
from ariadne.benchmarks.miluv_uwb import (
    UwbFixedLagResult,
    UwbPositionEstimate,
    UwbRangeSample,
    batch_rationalize_uwb_positions,
    estimate_uwb_positions,
    fixed_lag_rationalize_uwb_positions,
    read_uwb_ranges,
)
from ariadne.datasets import DatasetEvaluation

AGENTS = ("ifo001", "ifo002", "ifo003")
DEFAULT_SAMPLE_COUNT = 81


def _default_archive() -> Path:
    return (
        Path(__file__).resolve().parents[5]
        / "datasets/ariadne/miluv/archives/default_3_random_0.zip"
    )


def _member_root(names: tuple[str, ...]) -> str:
    matches = [
        name.removesuffix("/ifo001/mocap.csv")
        for name in names
        if name.endswith("/ifo001/mocap.csv")
    ]
    if len(matches) != 1:
        raise ValueError("MILUV archive must contain one ifo001/mocap.csv member")
    return matches[0]


def _pose_factor_inventory(names: tuple[str, ...], root: str) -> dict[str, object]:
    local_pose_suffixes = (
        "vio.csv",
        "odometry.csv",
        "odom.csv",
        "pose.csv",
        "trajectory.csv",
    )
    local_pose_members = {
        agent: [
            name
            for name in names
            if name.startswith(f"{root}/{agent}/")
            and not name.endswith("/mocap.csv")
            and name.rsplit("/", 1)[-1].lower().endswith(local_pose_suffixes)
        ]
        for agent in AGENTS
    }
    imu_members = {
        agent: [
            name
            for name in names
            if name.startswith(f"{root}/{agent}/")
            and name.rsplit("/", 1)[-1].lower().startswith("imu")
            and name.lower().endswith(".csv")
        ]
        for agent in AGENTS
    }
    return {
        "production_local_pose_stream_available": all(local_pose_members.values()),
        "local_pose_members_by_agent": local_pose_members,
        "imu_members_by_agent": imu_members,
        "attitude_stream_available": False,
        "interpretation": (
            "The archive supplies raw IMU measurements but no independent VIO, odometry, "
            "pose, trajectory, or attitude product. Raw IMU is not treated as a pose factor."
        ),
    }


def _read_mocap(
    archive: zipfile.ZipFile,
    member: str,
) -> tuple[PoseSample, ...]:
    samples: list[PoseSample] = []
    with archive.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8") as text:
        for row in csv.DictReader(text):
            timestamp_s = float(row["timestamp"])
            quaternion = np.asarray(
                [
                    row["pose.orientation.x"],
                    row["pose.orientation.y"],
                    row["pose.orientation.z"],
                    row["pose.orientation.w"],
                ],
                dtype=np.float64,
            )
            matrix = np.eye(4)
            matrix[:3, :3] = rotation_matrix(quaternion)
            matrix[:3, 3] = np.asarray(
                [
                    row["pose.position.x"],
                    row["pose.position.y"],
                    row["pose.position.z"],
                ],
                dtype=np.float64,
            )
            samples.append(PoseSample(timestamp_s, matrix))
    if len(samples) < 3 or any(
        samples[index].timestamp_s <= samples[index - 1].timestamp_s
        for index in range(1, len(samples))
    ):
        raise ValueError(f"MILUV mocap member is incomplete or unordered: {member}")
    return tuple(samples)


def _uwb_quality(ranges: tuple[UwbRangeSample, ...]) -> dict[str, int | float]:
    errors = [
        sample.range_m - sample.ground_truth_range_m
        for sample in ranges
        if sample.ground_truth_range_m is not None
    ]
    if not errors:
        raise ValueError("MILUV archive contains no usable UWB ranges")
    values = np.asarray(errors, dtype=np.float64)
    return {
        "unique_range_count": len(ranges),
        "range_bias_m": float(np.mean(values)),
        "range_rmse_m": float(np.sqrt(np.mean(np.square(values)))),
        "range_absolute_error_p95_m": float(np.percentile(np.abs(values), 95)),
        "range_absolute_error_max_m": float(np.max(np.abs(values))),
    }


def _build_controlled_odometry(
    agents: tuple[str, ...],
    truth: PoseSeries,
    *,
    seed: int,
) -> tuple[
    PoseSeries,
    PoseSeries,
    dict[str, float],
    dict[str, npt.NDArray[np.float64]],
]:
    translation_rng = np.random.default_rng(seed)
    rotation_rng = np.random.default_rng(seed ^ 0x61A7)
    drift_scales = dict(zip(agents, (1.006, 1.014, 0.991), strict=True))
    drift_biases = {
        agents[0]: np.asarray([0.0020, -0.0010, 0.0005]),
        agents[1]: np.asarray([0.0045, 0.0025, -0.0008]),
        agents[2]: np.asarray([-0.0035, 0.0030, 0.0010]),
    }
    rotation_drift = {
        agents[0]: np.asarray([0.0008, -0.0005, 0.0015]),
        agents[1]: np.asarray([0.0018, 0.0008, 0.0030]),
        agents[2]: np.asarray([-0.0012, 0.0015, -0.0022]),
    }
    baseline: PoseSeries = {}
    odometry: PoseSeries = {}
    for agent in agents:
        local_poses = [truth[agent][0].copy()]
        relative_poses = []
        for index in range(1, len(truth[agent])):
            relative = np.linalg.inv(truth[agent][index - 1]) @ truth[agent][index]
            perturbed = relative.copy()
            perturbed[:3, 3] = (
                relative[:3, 3] * drift_scales[agent]
                + drift_biases[agent]
                + translation_rng.normal(0.0, 0.001, 3)
            )
            perturbed[:3, :3] = relative[:3, :3] @ rotation_vector(
                rotation_drift[agent] + rotation_rng.normal(0.0, 0.0003, 3)
            )
            relative_poses.append(perturbed)
            local_poses.append(local_poses[-1] @ perturbed)
        baseline[agent] = tuple(local_poses)
        odometry[agent] = tuple(relative_poses)
    return baseline, odometry, drift_scales, rotation_drift


def _controlled_factors(
    agents: tuple[str, ...],
    truth: PoseSeries,
    *,
    seed: int,
) -> tuple[
    CrossAgentFactors,
    dict[tuple[int, str], npt.NDArray[np.float64]],
    dict[tuple[int, str], npt.NDArray[np.float64]],
]:
    translation_rng = np.random.default_rng(seed ^ 0xC2055)
    rotation_rng = np.random.default_rng(seed ^ 0x5E3)
    cross_agent: CrossAgentFactors = {}
    translation_noise: dict[tuple[int, str], npt.NDArray[np.float64]] = {}
    rotation_noise: dict[tuple[int, str], npt.NDArray[np.float64]] = {}
    source = agents[0]
    for index in range(1, len(truth[source])):
        for destination in agents[1:]:
            relative = np.linalg.inv(truth[source][index]) @ truth[destination][index]
            relative = relative.copy()
            relative[:3, 3] += translation_rng.normal(0.0, 0.005, 3)
            relative[:3, :3] = relative[:3, :3] @ rotation_vector(
                rotation_rng.normal(0.0, 0.002, 3)
            )
            cross_agent[(index, source, destination)] = relative
        for agent in agents:
            translation_noise[(index, agent)] = translation_rng.normal(0.0, 1.0, 3)
            rotation_noise[(index, agent)] = rotation_rng.normal(0.0, 1.0, 3)
    return cross_agent, translation_noise, rotation_noise


def _corrections(
    truth: PoseSeries,
    translation_noise: dict[tuple[int, str], npt.NDArray[np.float64]],
    rotation_noise: dict[tuple[int, str], npt.NDArray[np.float64]],
    translation_std_m: float,
    rotation_std_rad: float,
) -> dict[tuple[int, str], PoseMatrix]:
    measurements: dict[tuple[int, str], PoseMatrix] = {}
    for key, noise in translation_noise.items():
        index, agent = key
        pose = truth[agent][index].copy()
        pose[:3, 3] += noise * translation_std_m
        pose[:3, :3] = pose[:3, :3] @ rotation_vector(rotation_noise[key] * rotation_std_rad)
        measurements[key] = pose
    return measurements


def _run_metrics(run: GraphRun, truth: PoseSeries) -> dict[str, float | int]:
    return {
        "global_ate_m": translation_ate(run.optimized, truth),
        "translation_rpe_m": translation_rpe(run.optimized, truth),
        "orientation_rmse_rad": orientation_rmse(run.optimized, truth),
        "rotation_rpe_rmse_rad": rotation_rpe(run.optimized, truth),
        "global_correction_count": run.global_correction_count,
        "correction_payload_bytes_total": sum(run.correction_payload_bytes_by_agent.values()),
        "optimization_latency_ms": run.optimization_latency_ms,
    }


def _uwb_estimate_metrics(
    agents: tuple[str, ...],
    estimates: dict[str, tuple[UwbPositionEstimate, ...]],
    truth: PoseSeries,
) -> dict[str, object]:
    errors_by_agent: dict[str, list[float]] = {agent: [] for agent in agents}
    accepted = 0
    total = 0
    residuals = []
    corrections = []
    rejection_reasons: dict[str, int] = {}
    for agent in agents:
        for index, estimate in enumerate(estimates[agent]):
            total += 1
            if not estimate.accepted:
                rejection_reasons[estimate.rejection_reason] = (
                    rejection_reasons.get(estimate.rejection_reason, 0) + 1
                )
                continue
            accepted += 1
            errors_by_agent[agent].append(
                float(np.linalg.norm(estimate.position_m - truth[agent][index][:3, 3]))
            )
            residuals.append(estimate.residual_rmse_m)
            corrections.append(estimate.correction_delta_m)
    agent_ate = {
        agent: (float(np.sqrt(np.mean(np.square(errors)))) if errors else float("inf"))
        for agent, errors in errors_by_agent.items()
    }
    all_errors = [error for errors in errors_by_agent.values() for error in errors]
    return {
        "accepted_estimate_count": accepted,
        "total_estimate_count": total,
        "coverage_fraction": accepted / total,
        "position_ate_m": (
            float(np.sqrt(np.mean(np.square(all_errors)))) if all_errors else float("inf")
        ),
        "position_ate_m_by_agent": agent_ate,
        "range_residual_rmse_median": (float(np.median(residuals)) if residuals else float("inf")),
        "correction_delta_m_p95": (
            float(np.percentile(corrections, 95)) if corrections else float("inf")
        ),
        "rejection_reasons": rejection_reasons,
    }


def _uwb_corrections(
    agents: tuple[str, ...],
    estimates: dict[str, tuple[UwbPositionEstimate, ...]],
    baseline: PoseSeries,
    *,
    interval_samples: int,
) -> tuple[
    dict[tuple[int, str], PoseMatrix],
    frozenset[tuple[int, str]],
]:
    if interval_samples <= 0:
        raise ValueError("UWB correction interval must be positive")
    corrections: dict[tuple[int, str], PoseMatrix] = {}
    selected: set[tuple[int, str]] = set()
    sample_count = len(baseline[agents[0]])
    for agent in agents:
        for index in range(1, sample_count):
            estimate = estimates[agent][index]
            if not estimate.accepted or (
                index % interval_samples != 0 and index != sample_count - 1
            ):
                continue
            pose = baseline[agent][index].copy()
            pose[:3, 3] = estimate.position_m
            corrections[(index, agent)] = pose
            selected.add((index, agent))
    return corrections, frozenset(selected)


def _position_ate_by_agent(
    agents: tuple[str, ...],
    estimates: PoseSeries,
    truth: PoseSeries,
) -> dict[str, float]:
    return {
        agent: float(
            np.sqrt(
                np.mean(
                    [
                        np.linalg.norm(estimates[agent][index][:3, 3] - truth[agent][index][:3, 3])
                        ** 2
                        for index in range(len(truth[agent]))
                    ]
                )
            )
        )
        for agent in agents
    }


def _position_delta_correction_load(
    agents: tuple[str, ...],
    estimates: PoseSeries,
    baseline: PoseSeries,
    *,
    duration_s: float,
    sample_interval_s: float,
    causal: bool,
    eligible_sample_indices: frozenset[int] | None = None,
    residual_budget_m: float = 0.05,
) -> dict[str, object]:
    """Estimate thresholded position-delta transmission load per Wingman."""
    if duration_s <= 0 or sample_interval_s <= 0 or residual_budget_m <= 0:
        raise ValueError("MILUV batch correction-load bounds must be positive")
    by_agent: dict[str, dict[str, int | float | list[int]]] = {}
    total_correction_count = 0
    total_payload_bytes = 0
    maximum_messages_per_minute = 0.0
    for agent in agents:
        alignment = np.zeros(3, dtype=np.float64)
        selected_indices: list[int] = []
        payload_bytes = 0
        maximum_residual_m = 0.0
        for index in range(1, len(baseline[agent])):
            residual = estimates[agent][index][:3, 3] - baseline[agent][index][:3, 3] - alignment
            residual_m = float(np.linalg.norm(residual))
            maximum_residual_m = max(maximum_residual_m, residual_m)
            if residual_m <= residual_budget_m or (
                eligible_sample_indices is not None and index not in eligible_sample_indices
            ):
                continue
            selected_indices.append(index)
            alignment = estimates[agent][index][:3, 3] - baseline[agent][index][:3, 3]
            payload_bytes += correction_payload_bytes(
                agent,
                len(selected_indices),
                estimates[agent][index],
            )
        intervals_s = [
            (right - left) * sample_interval_s for left, right in pairwise(selected_indices)
        ]
        correction_count = len(selected_indices)
        messages_per_minute = correction_count / duration_s * 60.0
        total_correction_count += correction_count
        total_payload_bytes += payload_bytes
        maximum_messages_per_minute = max(
            maximum_messages_per_minute,
            messages_per_minute,
        )
        by_agent[agent] = {
            "correction_count": correction_count,
            "messages_per_minute": messages_per_minute,
            "payload_bytes": payload_bytes,
            "minimum_interval_seconds": (min(intervals_s) if intervals_s else 0.0),
            "median_interval_seconds": (float(np.median(intervals_s)) if intervals_s else 0.0),
            "maximum_pre_correction_residual_m": maximum_residual_m,
            "selected_sample_indices": selected_indices,
        }
    return {
        "residual_budget_m": residual_budget_m,
        "causal": causal,
        "interpretation": (
            (
                "causal event-triggered position-frame deltas from fixed-lag solves"
                if causal
                else "post-batch lower bound for transmitting position-frame deltas"
            )
            + "; range ingress is excluded"
        ),
        "by_agent": by_agent,
        "total_correction_count": total_correction_count,
        "total_payload_bytes": total_payload_bytes,
        "maximum_messages_per_minute": maximum_messages_per_minute,
    }


def run_miluv_global_pose_benchmark(
    seed: int = 7,
    archive_path: str | Path | None = None,
    *,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> DatasetEvaluation:
    """Measure full-SE(3) rationalization on real MILUV mocap geometry."""
    start_ns = perf_counter_ns()
    path = Path(archive_path) if archive_path is not None else _default_archive()
    if not path.is_file():
        raise FileNotFoundError(f"MILUV archive does not exist: {path}")
    with zipfile.ZipFile(path) as archive:
        names = tuple(archive.namelist())
        root = _member_root(names)
        mocap_members = tuple(f"{root}/{agent}/mocap.csv" for agent in AGENTS)
        uwb_members = tuple(f"{root}/{agent}/uwb_range.csv" for agent in AGENTS)
        missing = [member for member in (*mocap_members, *uwb_members) if member not in names]
        if missing:
            raise ValueError(f"MILUV global-pose evidence is missing: {missing}")
        trajectories = {
            agent: _read_mocap(archive, member)
            for agent, member in zip(AGENTS, mocap_members, strict=True)
        }
        uwb_ranges = read_uwb_ranges(archive, uwb_members)
        uwb_quality = _uwb_quality(uwb_ranges)
        pose_factor_inventory = _pose_factor_inventory(names, root)
        loaded_compressed_bytes = sum(
            archive.getinfo(member).compress_size for member in (*mocap_members, *uwb_members)
        )

    sampled, overlap_start_s, overlap_end_s = sample_overlap(trajectories, sample_count)
    world_origin = sampled[AGENTS[0]][0].matrix[:3, 3]
    truth: PoseSeries = {}
    for agent in AGENTS:
        centered = []
        for sample in sampled[agent]:
            matrix = sample.matrix.copy()
            matrix[:3, 3] -= world_origin
            centered.append(matrix)
        truth[agent] = tuple(centered)

    baseline, odometry, drift_scales, rotation_drift = _build_controlled_odometry(
        AGENTS,
        truth,
        seed=seed,
    )
    cross_agent, translation_noise, correction_rotation_noise = _controlled_factors(
        AGENTS,
        truth,
        seed=seed,
    )
    target_ate_m = 0.1
    target_orientation_rad = 0.05
    sample_interval_s = (overlap_end_s - overlap_start_s) / (sample_count - 1)
    duration_s = overlap_end_s - overlap_start_s
    uwb_window_seconds = 1.5
    uwb_latency_sweep: list[dict[str, object]] = []
    uwb_estimates_by_latency: dict[
        float,
        dict[str, tuple[UwbPositionEstimate, ...]],
    ] = {}
    for latency_seconds in (0.0, 0.25, 0.5, 0.75):
        estimates = estimate_uwb_positions(
            AGENTS,
            sampled,
            baseline,
            uwb_ranges,
            world_origin,
            window_seconds=uwb_window_seconds,
            processing_latency_seconds=latency_seconds,
        )
        uwb_estimates_by_latency[latency_seconds] = estimates
        measurement_metrics = _uwb_estimate_metrics(
            AGENTS,
            estimates,
            truth,
        )
        uwb_corrections, uwb_indices = _uwb_corrections(
            AGENTS,
            estimates,
            baseline,
            interval_samples=1,
        )
        uwb_graph = run_graph(
            AGENTS,
            truth,
            odometry,
            cross_agent,
            uwb_corrections,
            correction_interval=None,
            correction_indices=uwb_indices,
            global_graph_information=10.0,
            global_translation_information=25.0,
            global_rotation_information=1e-6,
            global_constraint_kind="miluv-uwb-position-correction",
        )
        uwb_latency_sweep.append(
            {
                "processing_latency_seconds": latency_seconds,
                **measurement_metrics,
                "graph_global_ate_m": translation_ate(
                    uwb_graph.optimized,
                    truth,
                ),
                "graph_orientation_rmse_rad": orientation_rmse(
                    uwb_graph.optimized,
                    truth,
                ),
                "graph_correction_count": uwb_graph.global_correction_count,
                "graph_optimization_latency_ms": (uwb_graph.optimization_latency_ms),
            }
        )

    best_uwb_latency_result = min(
        uwb_latency_sweep,
        key=lambda values: float(cast(float, values["graph_global_ate_m"])),
    )
    best_uwb_latency_seconds = float(
        cast(float, best_uwb_latency_result["processing_latency_seconds"])
    )
    best_uwb_estimates = uwb_estimates_by_latency[best_uwb_latency_seconds]
    uwb_cadence_sweep: list[dict[str, object]] = []
    uwb_cadence_runs: dict[int, GraphRun] = {}
    uwb_passing_intervals: list[int] = []
    for interval in (16, 8, 4, 2, 1):
        uwb_corrections, uwb_indices = _uwb_corrections(
            AGENTS,
            best_uwb_estimates,
            baseline,
            interval_samples=interval,
        )
        run = run_graph(
            AGENTS,
            truth,
            odometry,
            cross_agent,
            uwb_corrections,
            correction_interval=None,
            correction_indices=uwb_indices,
            global_graph_information=10.0,
            global_translation_information=25.0,
            global_rotation_information=1e-6,
            global_constraint_kind="miluv-uwb-position-correction",
        )
        uwb_cadence_runs[interval] = run
        global_ate_m = translation_ate(run.optimized, truth)
        uwb_cadence_sweep.append(
            {
                "correction_interval_samples": interval,
                "correction_interval_seconds": interval
                * (overlap_end_s - overlap_start_s)
                / (sample_count - 1),
                "global_ate_m": global_ate_m,
                "orientation_rmse_rad": orientation_rmse(
                    run.optimized,
                    truth,
                ),
                "target_ate_met": global_ate_m <= 0.1,
                "global_correction_count": run.global_correction_count,
                "correction_count_by_agent": run.correction_count_by_agent,
                "optimization_latency_ms": run.optimization_latency_ms,
            }
        )
        if global_ate_m <= 0.1:
            uwb_passing_intervals.append(interval)
    uwb_target_reachable = bool(uwb_passing_intervals)
    if uwb_target_reachable:
        best_uwb_interval = max(uwb_passing_intervals)
    else:
        best_uwb_interval = min(
            uwb_cadence_runs,
            key=lambda interval: translation_ate(
                uwb_cadence_runs[interval].optimized,
                truth,
            ),
        )
    best_uwb_graph = uwb_cadence_runs[best_uwb_interval]
    best_uwb_corrections, best_uwb_indices = _uwb_corrections(
        AGENTS,
        best_uwb_estimates,
        baseline,
        interval_samples=best_uwb_interval,
    )
    uwb_without_cross_agent = run_graph(
        AGENTS,
        truth,
        odometry,
        {},
        best_uwb_corrections,
        correction_interval=None,
        correction_indices=best_uwb_indices,
        global_graph_information=10.0,
        global_translation_information=25.0,
        global_rotation_information=1e-6,
        global_constraint_kind="miluv-uwb-position-correction",
    )
    uwb_batch = batch_rationalize_uwb_positions(
        AGENTS,
        sampled,
        baseline,
        uwb_ranges,
        world_origin,
    )
    uwb_batch_ate_m = translation_ate(uwb_batch.optimized, truth)
    uwb_batch_orientation_rmse_rad = orientation_rmse(
        uwb_batch.optimized,
        truth,
    )
    uwb_batch_ate_by_agent = _position_ate_by_agent(
        AGENTS,
        uwb_batch.optimized,
        truth,
    )
    uwb_batch_load = _position_delta_correction_load(
        AGENTS,
        uwb_batch.optimized,
        baseline,
        duration_s=duration_s,
        sample_interval_s=sample_interval_s,
        causal=False,
    )
    uwb_fixed_lag_runs: dict[int, UwbFixedLagResult] = {}
    uwb_fixed_lag_sweep: list[dict[str, object]] = []
    fixed_lag_passing_intervals: list[int] = []
    for interval in (4, 2, 1):
        fixed_lag = fixed_lag_rationalize_uwb_positions(
            AGENTS,
            sampled,
            baseline,
            uwb_ranges,
            world_origin,
            lag_samples=9,
            solve_interval_samples=interval,
            bias_prior_information=2000.0,
            maximum_iterations=6,
        )
        uwb_fixed_lag_runs[interval] = fixed_lag
        fixed_lag_ate_m = translation_ate(
            fixed_lag.optimized,
            truth,
        )
        fixed_lag_orientation_rmse_rad = orientation_rmse(
            fixed_lag.optimized,
            truth,
        )
        fixed_lag_target_met = fixed_lag_ate_m <= target_ate_m
        if fixed_lag_target_met:
            fixed_lag_passing_intervals.append(interval)
        uwb_fixed_lag_sweep.append(
            {
                "solve_interval_samples": interval,
                "solve_interval_seconds": interval * sample_interval_s,
                "position_ate_m": fixed_lag_ate_m,
                "orientation_rmse_rad": fixed_lag_orientation_rmse_rad,
                "position_target_met": fixed_lag_target_met,
                "full_pose_target_met": (
                    fixed_lag_target_met
                    and fixed_lag_orientation_rmse_rad <= target_orientation_rad
                ),
                "solve_count": fixed_lag.solve_count,
                "converged_solve_count": (fixed_lag.converged_solve_count),
                "optimization_latency_ms_total": (fixed_lag.optimization_latency_ms_total),
                "optimization_latency_ms_p95": (fixed_lag.optimization_latency_ms_p95),
                "optimization_latency_ms_max": (fixed_lag.optimization_latency_ms_max),
            }
        )
    best_fixed_lag_interval = max(
        fixed_lag_passing_intervals,
        default=min(
            uwb_fixed_lag_runs,
            key=lambda interval: translation_ate(
                uwb_fixed_lag_runs[interval].optimized,
                truth,
            ),
        ),
    )
    best_fixed_lag = uwb_fixed_lag_runs[best_fixed_lag_interval]
    best_fixed_lag_ate_m = translation_ate(
        best_fixed_lag.optimized,
        truth,
    )
    best_fixed_lag_orientation_rmse_rad = orientation_rmse(
        best_fixed_lag.optimized,
        truth,
    )
    best_fixed_lag_ate_by_agent = _position_ate_by_agent(
        AGENTS,
        best_fixed_lag.optimized,
        truth,
    )
    uwb_fixed_lag_load = _position_delta_correction_load(
        AGENTS,
        best_fixed_lag.optimized,
        baseline,
        duration_s=duration_s,
        sample_interval_s=sample_interval_s,
        causal=True,
        eligible_sample_indices=frozenset(best_fixed_lag.solve_sample_indices),
    )
    fixed_lag_claim_evidence = GlobalPoseClaimEvidence(
        odometry_source="controlled",
        position_reference_source="measured",
        orientation_reference_source="controlled",
        cross_agent_source="measured",
        uses_ground_truth_in_estimator=False,
        causal=True,
        all_agents_position_target_met=(
            max(best_fixed_lag_ate_by_agent.values()) <= target_ate_m
        ),
        fleet_position_target_met=best_fixed_lag_ate_m <= target_ate_m,
        orientation_target_met=(
            best_fixed_lag_orientation_rmse_rad <= target_orientation_rad
        ),
    )
    correction_noise_std_m = 0.005
    correction_rotation_std_rad = 0.003
    global_corrections = _corrections(
        truth,
        translation_noise,
        correction_rotation_noise,
        correction_noise_std_m,
        correction_rotation_std_rad,
    )
    cadence_runs = {
        interval: run_graph(
            AGENTS,
            truth,
            odometry,
            cross_agent,
            global_corrections,
            correction_interval=interval,
        )
        for interval in (None, 16, 8, 4, 2, 1)
    }
    cadence_sweep = []
    passing_intervals: list[int] = []
    for cadence_interval, run in cadence_runs.items():
        values = _run_metrics(run, truth)
        target_met = (
            float(values["global_ate_m"]) <= target_ate_m
            and float(values["orientation_rmse_rad"]) <= target_orientation_rad
        )
        cadence_sweep.append(
            {
                "correction_interval_samples": cadence_interval or 0,
                "correction_interval_seconds": (
                    0.0 if cadence_interval is None else cadence_interval * sample_interval_s
                ),
                **values,
                "target_pose_met": target_met,
            }
        )
        if cadence_interval is not None and target_met:
            passing_intervals.append(cadence_interval)
    fixed_interval = max(passing_intervals, default=1)
    fixed = cadence_runs[fixed_interval]

    adaptive_indices, adaptive_trace, scheduler_metrics = adaptive_correction_schedule(
        AGENTS,
        sampled,
        truth,
        baseline,
        global_corrections,
        target_error_m=0.075,
        target_orientation_error_rad=0.04,
        sample_interval_s=sample_interval_s,
        network_profiles={
            AGENTS[0]: (0.95, 0.10),
            AGENTS[1]: (0.70, 0.45),
            AGENTS[2]: (0.82, 0.25),
        },
        nominal_interval_s=fixed_interval * sample_interval_s,
        maximum_interval_s=32 * sample_interval_s,
    )
    adaptive = run_graph(
        AGENTS,
        truth,
        odometry,
        cross_agent,
        global_corrections,
        correction_interval=None,
        correction_indices=adaptive_indices,
    )
    adaptive_metrics = _run_metrics(adaptive, truth)
    adaptive_target_met = (
        float(adaptive_metrics["global_ate_m"]) <= target_ate_m
        and float(adaptive_metrics["orientation_rmse_rad"]) <= target_orientation_rad
    )
    if adaptive_target_met and adaptive.global_correction_count < fixed.global_correction_count:
        selected_strategy = "adaptive_per_wingman"
        selected_interval = 0
        selected = adaptive
        selected_indices: frozenset[tuple[int, str]] | None = adaptive_indices
    else:
        selected_strategy = "fixed_cadence"
        selected_interval = fixed_interval
        selected = fixed
        selected_indices = None
    selected_metrics = _run_metrics(selected, truth)

    noise_sweep = []
    maximum_translation_noise_m = 0.0
    for noise_std_m in (0.005, 0.025, 0.05, 0.1, 0.2):
        run = run_graph(
            AGENTS,
            truth,
            odometry,
            cross_agent,
            _corrections(
                truth,
                translation_noise,
                correction_rotation_noise,
                noise_std_m,
                correction_rotation_std_rad,
            ),
            correction_interval=selected_interval or None,
            correction_indices=selected_indices,
        )
        values = _run_metrics(run, truth)
        target_met = (
            float(values["global_ate_m"]) <= target_ate_m
            and float(values["orientation_rmse_rad"]) <= target_orientation_rad
        )
        noise_sweep.append(
            {"global_correction_noise_std_m": noise_std_m, **values, "target_pose_met": target_met}
        )
        if target_met:
            maximum_translation_noise_m = noise_std_m

    rotation_noise_sweep = []
    maximum_rotation_noise_rad = 0.0
    for noise_std_rad in (0.003, 0.01, 0.02, 0.05, 0.1):
        run = run_graph(
            AGENTS,
            truth,
            odometry,
            cross_agent,
            _corrections(
                truth,
                translation_noise,
                correction_rotation_noise,
                correction_noise_std_m,
                noise_std_rad,
            ),
            correction_interval=selected_interval or None,
            correction_indices=selected_indices,
        )
        values = _run_metrics(run, truth)
        target_met = (
            float(values["global_ate_m"]) <= target_ate_m
            and float(values["orientation_rmse_rad"]) <= target_orientation_rad
        )
        rotation_noise_sweep.append(
            {
                "global_correction_noise_std_rad": noise_std_rad,
                **values,
                "target_pose_met": target_met,
            }
        )
        if target_met:
            maximum_rotation_noise_rad = noise_std_rad

    baseline_ate_m = translation_ate(baseline, truth)
    baseline_orientation_rad = orientation_rmse(baseline, truth)
    result = selected.result
    optimized_ate_m = float(selected_metrics["global_ate_m"])
    optimized_orientation_rad = float(selected_metrics["orientation_rmse_rad"])
    false_loop_rejected = selected.false_loop_index in result.rejected_constraints
    best_uwb_measurement_metrics = _uwb_estimate_metrics(
        AGENTS,
        best_uwb_estimates,
        truth,
    )
    best_uwb_graph_ate_m = translation_ate(
        best_uwb_graph.optimized,
        truth,
    )
    best_uwb_graph_orientation_rmse_rad = orientation_rmse(
        best_uwb_graph.optimized,
        truth,
    )
    uwb_message_rates = {
        agent: (best_uwb_graph.correction_count_by_agent[agent] / duration_s * 60.0)
        for agent in AGENTS
    }
    message_rates = {
        agent: selected.correction_count_by_agent[agent] / duration_s * 60.0 for agent in AGENTS
    }
    target_met = (
        optimized_ate_m <= target_ate_m and optimized_orientation_rad <= target_orientation_rad
    )
    passed = (
        target_met
        and false_loop_rejected
        and len(set(result.components.values())) == 1
        and float(selected_metrics["translation_rpe_m"]) < translation_rpe(baseline, truth)
        and float(selected_metrics["rotation_rpe_rmse_rad"]) < rotation_rpe(baseline, truth)
    )
    selected_interval_seconds = (
        duration_s * len(AGENTS) / selected.global_correction_count
        if selected_strategy == "adaptive_per_wingman"
        else fixed_interval * sample_interval_s
    )
    metrics: dict[str, int | float | str] = {
        "seed": seed,
        "agent_count": len(AGENTS),
        "sample_count_per_agent": sample_count,
        "ground_truth_input_pose_count": sum(len(value) for value in trajectories.values()),
        "ground_truth_overlap_seconds": duration_s,
        "loaded_compressed_bytes": loaded_compressed_bytes,
        "archive_bytes": path.stat().st_size,
        "loaded_archive_fraction_percent": 100.0 * loaded_compressed_bytes / path.stat().st_size,
        "uwb_unique_range_count": int(uwb_quality["unique_range_count"]),
        "uwb_range_rmse_m": float(uwb_quality["range_rmse_m"]),
        "uwb_range_absolute_error_p95_m": float(uwb_quality["range_absolute_error_p95_m"]),
        "uwb_position_window_seconds": uwb_window_seconds,
        "uwb_position_best_tested_processing_latency_seconds": (best_uwb_latency_seconds),
        "uwb_position_estimate_coverage_fraction": float(
            cast(float, best_uwb_measurement_metrics["coverage_fraction"])
        ),
        "uwb_position_estimate_ate_m": float(
            cast(float, best_uwb_measurement_metrics["position_ate_m"])
        ),
        "uwb_graph_best_tested_global_ate_m": best_uwb_graph_ate_m,
        "uwb_graph_best_tested_orientation_rmse_rad": (best_uwb_graph_orientation_rmse_rad),
        "uwb_graph_target_reachable_with_tested_cadences": int(uwb_target_reachable),
        "uwb_graph_best_tested_correction_interval_samples": best_uwb_interval,
        "uwb_graph_best_tested_correction_interval_seconds": (
            best_uwb_interval * sample_interval_s
        ),
        "uwb_graph_best_tested_global_correction_count": (best_uwb_graph.global_correction_count),
        "uwb_graph_best_tested_messages_per_minute_max": max(uwb_message_rates.values()),
        "uwb_graph_without_cross_agent_ate_m": translation_ate(
            uwb_without_cross_agent.optimized,
            truth,
        ),
        "uwb_batch_global_position_ate_m": uwb_batch_ate_m,
        "uwb_batch_global_orientation_rmse_rad": (uwb_batch_orientation_rmse_rad),
        "uwb_batch_position_target_met": int(uwb_batch_ate_m <= target_ate_m),
        "uwb_batch_full_pose_target_met": int(
            uwb_batch_ate_m <= target_ate_m
            and uwb_batch_orientation_rmse_rad <= target_orientation_rad
        ),
        "uwb_batch_optimization_latency_ms": uwb_batch.optimization_latency_ms,
        "uwb_batch_correction_messages_per_minute_max": float(
            cast(float, uwb_batch_load["maximum_messages_per_minute"])
        ),
        "uwb_fixed_lag_global_position_ate_m": best_fixed_lag_ate_m,
        "uwb_fixed_lag_global_orientation_rmse_rad": (best_fixed_lag_orientation_rmse_rad),
        "uwb_fixed_lag_position_target_met": int(best_fixed_lag_ate_m <= target_ate_m),
        "uwb_fixed_lag_max_agent_position_ate_m": max(best_fixed_lag_ate_by_agent.values()),
        "uwb_fixed_lag_all_agents_position_target_met": int(
            max(best_fixed_lag_ate_by_agent.values()) <= target_ate_m
        ),
        "uwb_fixed_lag_full_pose_target_met": int(
            best_fixed_lag_ate_m <= target_ate_m
            and best_fixed_lag_orientation_rmse_rad <= target_orientation_rad
        ),
        "uwb_fixed_lag_samples": best_fixed_lag.lag_samples,
        "uwb_fixed_lag_duration_seconds": ((best_fixed_lag.lag_samples - 1) * sample_interval_s),
        "uwb_fixed_lag_solve_interval_seconds": (
            best_fixed_lag.solve_interval_samples * sample_interval_s
        ),
        "uwb_fixed_lag_solve_count": best_fixed_lag.solve_count,
        "uwb_fixed_lag_optimization_latency_ms_p95": (best_fixed_lag.optimization_latency_ms_p95),
        "uwb_fixed_lag_optimization_latency_ms_max": (best_fixed_lag.optimization_latency_ms_max),
        "uwb_fixed_lag_p95_deadline_met": int(
            best_fixed_lag.optimization_latency_ms_p95
            <= best_fixed_lag.solve_interval_samples * sample_interval_s * 1000.0
        ),
        "uwb_fixed_lag_correction_messages_per_minute_max": float(
            cast(float, uwb_fixed_lag_load["maximum_messages_per_minute"])
        ),
        "uwb_fixed_lag_position_claim_eligible": int(
            fixed_lag_claim_evidence.position_claim_eligible
        ),
        "uwb_fixed_lag_full_pose_claim_eligible": int(
            fixed_lag_claim_evidence.full_pose_claim_eligible
        ),
        "baseline_global_ate_m": baseline_ate_m,
        "optimized_global_ate_m": optimized_ate_m,
        "target_global_ate_m": target_ate_m,
        "baseline_global_orientation_rmse_rad": baseline_orientation_rad,
        "optimized_global_orientation_rmse_rad": optimized_orientation_rad,
        "target_global_orientation_rmse_rad": target_orientation_rad,
        "target_global_pose_met": int(target_met),
        "baseline_translation_rpe_m": translation_rpe(baseline, truth),
        "optimized_translation_rpe_m": float(selected_metrics["translation_rpe_m"]),
        "baseline_rotation_rpe_rmse_rad": rotation_rpe(baseline, truth),
        "optimized_rotation_rpe_rmse_rad": float(selected_metrics["rotation_rpe_rmse_rad"]),
        "global_ate_improvement_percent": (
            100.0 * (baseline_ate_m - optimized_ate_m) / baseline_ate_m
        ),
        "graph_node_count": len(result.poses),
        "graph_constraint_count": result.constraint_count,
        "graph_component_count": len(set(result.components.values())),
        "false_loop_rejected": int(false_loop_rejected),
        "cross_agent_constraint_count": len(cross_agent),
        "selected_correction_strategy": selected_strategy,
        "selected_correction_interval_samples": selected_interval,
        "selected_correction_interval_seconds": selected_interval_seconds,
        "selected_global_correction_count": selected.global_correction_count,
        "selected_correction_messages_per_minute_max": max(message_rates.values()),
        "selected_correction_payload_bytes_total": int(
            selected_metrics["correction_payload_bytes_total"]
        ),
        "maximum_tested_correction_noise_m_for_target": maximum_translation_noise_m,
        "maximum_tested_correction_rotation_noise_rad_for_target": (maximum_rotation_noise_rad),
        "optimization_latency_ms": selected.optimization_latency_ms,
        "benchmark_latency_ms": (perf_counter_ns() - start_ns) / 1e6,
    }
    return DatasetEvaluation(
        dataset="miluv-global-pose",
        status="passed" if passed else "failed",
        agents=AGENTS,
        modalities=("stereo", "imu", "uwb", "mocap_se3_truth", "global_pose"),
        metrics=metrics,
        warnings=(
            "MILUV mocap supplies real SE(3) truth and the UWB position path uses measured "
            "calibrated ranges plus published anchor/tag geometry without ground truth in the "
            "estimator. The fixed-lag path is causal, but local odometry and orientation remain "
            "deterministic controlled perturbations; this isolates Intelligence-node "
            "rationalization rather than claiming end-to-end VIO, real orientation, or visual "
            "association accuracy.",
        ),
        details={
            "archive_path": str(path),
            "archive_members_loaded": [*mocap_members, *uwb_members],
            "real_pose_factor_inventory": pose_factor_inventory,
            "orientation_reference": "miluv_motion_capture",
            "constraint_source": (
                "MILUV mocap geometry with deterministic local-odometry, relative-pose, "
                "and global-correction perturbations"
            ),
            "uwb_quality": uwb_quality,
            "uwb_position_rationalization": {
                "measurement_source": (
                    "calibrated MILUV ranges to official constellation-0 "
                    "anchor positions with official per-tag body moment arms"
                ),
                "uses_ground_truth_in_estimator": False,
                "orientation_source": "controlled_local_odometry",
                "window_seconds": uwb_window_seconds,
                "best_observed_processing_latency_seconds": (best_uwb_latency_seconds),
                "latency_tuned_against_benchmark_truth": True,
                "best_observed_measurement_metrics": (best_uwb_measurement_metrics),
                "latency_sweep": uwb_latency_sweep,
                "cadence_sweep": uwb_cadence_sweep,
                "target_reachable_with_tested_cadences": (uwb_target_reachable),
                "best_tested_interval_samples": best_uwb_interval,
                "best_tested_interval_seconds": (best_uwb_interval * sample_interval_s),
                "best_tested_graph_global_ate_m": best_uwb_graph_ate_m,
                "best_tested_graph_orientation_rmse_rad": (best_uwb_graph_orientation_rmse_rad),
                "best_tested_graph_correction_count": (best_uwb_graph.global_correction_count),
                "best_tested_graph_correction_count_by_agent": (
                    best_uwb_graph.correction_count_by_agent
                ),
                "best_tested_graph_messages_per_minute_by_agent": (uwb_message_rates),
                "without_cross_agent_global_ate_m": translation_ate(
                    uwb_without_cross_agent.optimized,
                    truth,
                ),
                "translation_information": 25.0,
                "rotation_information": 1e-6,
            },
            "uwb_batch_rationalization": {
                "mode": "bounded_full_batch_non_causal_upper_bound",
                "measurement_source": ("real calibrated MILUV anchor and inter-agent UWB ranges"),
                "uses_ground_truth_in_estimator": False,
                "odometry_source": "controlled_local_odometry_position_deltas",
                "orientation_source": (
                    "fixed controlled local-odometry orientation for tag lever arms"
                ),
                "position_ate_m": uwb_batch_ate_m,
                "position_ate_m_by_agent": uwb_batch_ate_by_agent,
                "orientation_rmse_rad": uwb_batch_orientation_rmse_rad,
                "position_target_met": uwb_batch_ate_m <= target_ate_m,
                "full_pose_target_met": (
                    uwb_batch_ate_m <= target_ate_m
                    and uwb_batch_orientation_rmse_rad <= target_orientation_rad
                ),
                "converged": uwb_batch.converged,
                "iterations": uwb_batch.iterations,
                "range_factor_count": uwb_batch.range_factor_count,
                "anchor_range_factor_count": (uwb_batch.anchor_range_factor_count),
                "inter_agent_range_factor_count": (uwb_batch.inter_agent_range_factor_count),
                "downweighted_range_count": (uwb_batch.downweighted_range_count),
                "range_residual_rmse_m": uwb_batch.range_residual_rmse_m,
                "range_residual_p95_m": uwb_batch.range_residual_p95_m,
                "transceiver_bias_m": uwb_batch.transceiver_bias_m,
                "optimization_latency_ms": uwb_batch.optimization_latency_ms,
                "post_batch_correction_load": uwb_batch_load,
            },
            "uwb_fixed_lag_rationalization": {
                "mode": "bounded_dense_causal_fixed_lag_reference",
                "measurement_source": ("real calibrated MILUV anchor and inter-agent UWB ranges"),
                "uses_ground_truth_in_estimator": False,
                "uses_future_measurements": False,
                "processing_latency_seconds": 0.0,
                "odometry_source": ("controlled_local_odometry_position_deltas"),
                "orientation_source": (
                    "fixed controlled local-odometry orientation for tag lever arms"
                ),
                "position_ate_m": best_fixed_lag_ate_m,
                "position_ate_m_by_agent": (best_fixed_lag_ate_by_agent),
                "orientation_rmse_rad": (best_fixed_lag_orientation_rmse_rad),
                "position_target_met": (best_fixed_lag_ate_m <= target_ate_m),
                "full_pose_target_met": (
                    best_fixed_lag_ate_m <= target_ate_m
                    and best_fixed_lag_orientation_rmse_rad <= target_orientation_rad
                ),
                "lag_samples": best_fixed_lag.lag_samples,
                "lag_duration_seconds": ((best_fixed_lag.lag_samples - 1) * sample_interval_s),
                "selected_solve_interval_samples": (best_fixed_lag.solve_interval_samples),
                "selected_solve_interval_seconds": (
                    best_fixed_lag.solve_interval_samples * sample_interval_s
                ),
                "solve_count": best_fixed_lag.solve_count,
                "converged_solve_count": (best_fixed_lag.converged_solve_count),
                "range_factor_count_total": (best_fixed_lag.range_factor_count_total),
                "downweighted_range_count_total": (best_fixed_lag.downweighted_range_count_total),
                "final_transceiver_bias_m": (best_fixed_lag.final_transceiver_bias_m),
                "optimization_latency_ms_total": (best_fixed_lag.optimization_latency_ms_total),
                "optimization_latency_ms_p95": (best_fixed_lag.optimization_latency_ms_p95),
                "optimization_latency_ms_max": (best_fixed_lag.optimization_latency_ms_max),
                "p95_solve_deadline_met": (
                    best_fixed_lag.optimization_latency_ms_p95
                    <= best_fixed_lag.solve_interval_samples * sample_interval_s * 1000.0
                ),
                "solve_interval_sweep": uwb_fixed_lag_sweep,
                "correction_load": uwb_fixed_lag_load,
                "claim_evidence": fixed_lag_claim_evidence.as_dict(),
            },
            "drift_scale_by_agent": drift_scales,
            "rotation_drift_vector_rad_by_agent": {
                agent: value.tolist() for agent, value in rotation_drift.items()
            },
            "cadence_sweep": cadence_sweep,
            "correction_noise_sweep": noise_sweep,
            "correction_rotation_noise_sweep": rotation_noise_sweep,
            "vision_correction_limits": {
                "cross_agent_only_global_ate_m": translation_ate(
                    cadence_runs[None].optimized,
                    truth,
                ),
                "maximum_tested_noise_std_m_for_target": maximum_translation_noise_m,
                "maximum_tested_rotation_noise_std_rad_for_target": (maximum_rotation_noise_rad),
                "unverified_steps": [
                    "production VIO accuracy and repeatability",
                    "visual cross-agent association precision and recall",
                    "real orientation factors and production VIO/UWB coupling",
                    "sparse marginalization and flight-compute timing",
                ],
            },
            "intelligence_load": {
                "correction_count_by_agent": selected.correction_count_by_agent,
                "messages_per_minute_by_agent": message_rates,
                "payload_bytes_by_agent": selected.correction_payload_bytes_by_agent,
                "payload_bytes_total": int(selected_metrics["correction_payload_bytes_total"]),
                "optimization_latency_ms": selected.optimization_latency_ms,
            },
            "adaptive_scheduler": {
                **adaptive_metrics,
                "target_pose_met": adaptive_target_met,
                "reduces_load_vs_fixed": (
                    adaptive.global_correction_count < fixed.global_correction_count
                ),
                "correction_count_by_agent": adaptive.correction_count_by_agent,
                "messages_per_minute_by_agent": {
                    agent: adaptive.correction_count_by_agent[agent] / duration_s * 60.0
                    for agent in AGENTS
                },
                "scheduler_metrics": scheduler_metrics,
                "trace": adaptive_trace,
            },
            "fixed_cadence_reference": {
                "interval_samples": fixed_interval,
                "interval_seconds": fixed_interval * sample_interval_s,
                **_run_metrics(fixed, truth),
            },
            "trajectories": {
                agent: [
                    {
                        "timestamp_s": sampled[agent][index].timestamp_s,
                        "truth_m": truth[agent][index][:3, 3].tolist(),
                        "baseline_m": baseline[agent][index][:3, 3].tolist(),
                        "optimized_m": selected.optimized[agent][index][:3, 3].tolist(),
                    }
                    for index in range(sample_count)
                ]
                for agent in AGENTS
            },
        },
    )
