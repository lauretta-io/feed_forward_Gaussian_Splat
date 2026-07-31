"""Dataset-backed S3E global-pose regression benchmark."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import numpy.typing as npt

from ariadne.benchmarks.global_pose_rationalization import GlobalPoseClaimEvidence
from ariadne.benchmarks.global_pose_rationalization import (
    GraphRun as _GraphRun,
)
from ariadne.benchmarks.global_pose_rationalization import (
    PoseSample as _PoseSample,
)
from ariadne.benchmarks.global_pose_rationalization import (
    adaptive_correction_schedule as _shared_adaptive_correction_schedule,
)
from ariadne.benchmarks.global_pose_rationalization import (
    orientation_rmse as _orientation_rmse,
)
from ariadne.benchmarks.global_pose_rationalization import (
    rotation_matrix as _rotation,
)
from ariadne.benchmarks.global_pose_rationalization import (
    rotation_rpe as _rotation_rpe,
)
from ariadne.benchmarks.global_pose_rationalization import (
    rotation_vector as _rotation_vector,
)
from ariadne.benchmarks.global_pose_rationalization import (
    run_graph as _shared_run_graph,
)
from ariadne.benchmarks.global_pose_rationalization import (
    sample_overlap as _sample_overlap,
)
from ariadne.benchmarks.global_pose_rationalization import (
    translation_ate as _translation_ate,
)
from ariadne.benchmarks.global_pose_rationalization import (
    translation_rpe as _translation_rpe,
)
from ariadne.datasets import DatasetEvaluation
from ariadne.datasets.s3e import evaluate_s3e

AGENTS = ("Alpha", "Bob", "Carol")
DEFAULT_SAMPLE_COUNT = 24


@dataclass(frozen=True)
class _AdaptiveSweepRun:
    demand_target_error_m: float
    correction_indices: frozenset[tuple[int, str]]
    graph: _GraphRun
    scheduler_metrics: dict[str, int]
    fleet_ate_m: float
    fleet_orientation_rmse_rad: float
    per_agent_ate_m: dict[str, float]
    per_agent_orientation_rmse_rad: dict[str, float]
    all_agents_target_met: bool


def _default_s3e_root() -> Path:
    return Path(__file__).resolve().parents[5] / "datasets/ariadne/s3e/S3Ev1"


def _read_ground_truth(path: Path) -> tuple[_PoseSample, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"S3E ground truth does not exist: {path}")
    samples: list[_PoseSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = [float(value) for value in line.replace(",", " ").split()]
        if len(values) < 8:
            continue
        matrix = np.eye(4)
        matrix[:3, :3] = _rotation(values[4:8])
        matrix[:3, 3] = values[1:4]
        samples.append(_PoseSample(values[0], matrix))
    if len(samples) < 2:
        raise ValueError(f"S3E ground truth must contain at least two poses: {path}")
    if any(
        samples[index].timestamp_s <= samples[index - 1].timestamp_s
        for index in range(1, len(samples))
    ):
        raise ValueError(f"S3E ground-truth timestamps must be strictly increasing: {path}")
    return tuple(samples)


def _validate_calibration(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"S3E calibration does not exist: {path}")
    content = path.read_text(encoding="utf-8")
    required = ("Camera.type:", "Camera.bf:", "IMU.Frequency:", "Tic:")
    missing = [field for field in required if field not in content]
    if missing:
        raise ValueError(f"S3E calibration {path} is missing fields: {missing}")
    return hashlib.sha256(content.encode()).hexdigest()


def _run_graph(
    truth: dict[str, tuple[npt.NDArray[np.float64], ...]],
    odometry: dict[str, tuple[npt.NDArray[np.float64], ...]],
    cross_agent: dict[tuple[int, str], npt.NDArray[np.float64]],
    global_corrections: dict[tuple[int, str], npt.NDArray[np.float64]],
    *,
    correction_interval: int | None,
    correction_indices: frozenset[tuple[int, str]] | None = None,
) -> _GraphRun:
    shared_cross_agent = {
        (index, "Alpha", destination): relative
        for (index, destination), relative in cross_agent.items()
    }
    return _shared_run_graph(
        AGENTS,
        truth,
        odometry,
        shared_cross_agent,
        global_corrections,
        correction_interval=correction_interval,
        correction_indices=correction_indices,
    )


def _adaptive_correction_schedule(
    sampled: dict[str, tuple[_PoseSample, ...]],
    truth: dict[str, tuple[npt.NDArray[np.float64], ...]],
    baseline: dict[str, tuple[npt.NDArray[np.float64], ...]],
    global_corrections: dict[tuple[int, str], npt.NDArray[np.float64]],
    *,
    target_error_m: float,
    target_orientation_error_rad: float,
    sample_interval_s: float,
) -> tuple[frozenset[tuple[int, str]], list[dict[str, object]], dict[str, int]]:
    return _shared_adaptive_correction_schedule(
        AGENTS,
        sampled,
        truth,
        baseline,
        global_corrections,
        target_error_m=target_error_m,
        target_orientation_error_rad=target_orientation_error_rad,
        sample_interval_s=sample_interval_s,
        nominal_interval_s=sample_interval_s,
        maximum_interval_s=sample_interval_s * 8,
        network_profiles={
            "Alpha": (0.95, 0.10),
            "Bob": (0.65, 0.55),
            "Carol": (0.80, 0.30),
        },
    )


def _cross_agent_translation_rmse(
    estimates: dict[str, tuple[npt.NDArray[np.float64], ...]],
    truth: dict[str, tuple[npt.NDArray[np.float64], ...]],
) -> float:
    errors_m: list[float] = []
    for destination in ("Bob", "Carol"):
        for index in range(len(truth["Alpha"])):
            estimated_relative = (
                np.linalg.inv(estimates["Alpha"][index]) @ estimates[destination][index]
            )
            truth_relative = np.linalg.inv(truth["Alpha"][index]) @ truth[destination][index]
            errors_m.append(
                float(np.linalg.norm(estimated_relative[:3, 3] - truth_relative[:3, 3]))
            )
    return float(np.sqrt(np.mean(np.square(errors_m))))


def _per_agent_pose_metrics(
    estimates: dict[str, tuple[npt.NDArray[np.float64], ...]],
    truth: dict[str, tuple[npt.NDArray[np.float64], ...]],
) -> tuple[dict[str, float], dict[str, float]]:
    ate_m = {
        agent: _translation_ate(
            {agent: estimates[agent]},
            {agent: truth[agent]},
        )
        for agent in AGENTS
    }
    orientation_rmse_rad = {
        agent: _orientation_rmse(
            {agent: estimates[agent]},
            {agent: truth[agent]},
        )
        for agent in AGENTS
    }
    return ate_m, orientation_rmse_rad


def run_s3e_global_pose_benchmark(
    seed: int = 7,
    s3e_root: str | Path | None = None,
    *,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> DatasetEvaluation:
    """Measure global-pose recovery on real S3E geometry with controlled drift."""
    start_ns = perf_counter_ns()
    root = Path(s3e_root) if s3e_root is not None else _default_s3e_root()
    playground = root / "S3E_Playground_2"
    bag_path = playground / "S3E_Playground_2.db3"
    calibration_root = root / "Calibration"
    dataset_inventory = evaluate_s3e(bag_path)
    if tuple(dataset_inventory.agents) != AGENTS:
        raise ValueError(f"S3E global-pose benchmark requires agents {AGENTS}")

    trajectories = {
        agent: _read_ground_truth(playground / f"{agent.lower()}_gt.txt") for agent in AGENTS
    }
    calibration_hashes = {
        agent: _validate_calibration(calibration_root / f"{agent.lower()}.yaml") for agent in AGENTS
    }
    sampled, overlap_start_s, overlap_end_s = _sample_overlap(trajectories, sample_count)

    world_origin = sampled["Alpha"][0].matrix[:3, 3]
    truth: dict[str, tuple[npt.NDArray[np.float64], ...]] = {}
    for agent in AGENTS:
        centered = []
        for sample in sampled[agent]:
            matrix = sample.matrix.copy()
            matrix[:3, 3] -= world_origin
            centered.append(matrix)
        truth[agent] = tuple(centered)

    translation_rng = np.random.default_rng(seed)
    rotation_rng = np.random.default_rng(seed ^ 0x5E3)
    drift_scales = {"Alpha": 1.008, "Bob": 1.035, "Carol": 0.968}
    drift_biases = {
        "Alpha": np.asarray([0.004, -0.002, 0.001]),
        "Bob": np.asarray([0.018, 0.010, -0.003]),
        "Carol": np.asarray([-0.014, 0.016, 0.004]),
    }
    rotation_drift_vectors_rad = {
        "Alpha": np.asarray([0.005, -0.003, 0.010]),
        "Bob": np.asarray([0.012, 0.006, 0.024]),
        "Carol": np.asarray([-0.009, 0.012, -0.018]),
    }
    baseline: dict[str, tuple[npt.NDArray[np.float64], ...]] = {}
    odometry: dict[str, tuple[npt.NDArray[np.float64], ...]] = {}
    for agent in AGENTS:
        local_poses = [truth[agent][0].copy()]
        relative_poses = []
        for index in range(1, sample_count):
            relative = np.linalg.inv(truth[agent][index - 1]) @ truth[agent][index]
            perturbed = relative.copy()
            perturbed[:3, 3] = (
                relative[:3, 3] * drift_scales[agent]
                + drift_biases[agent]
                + translation_rng.normal(0.0, 0.002, 3)
            )
            rotation_perturbation = rotation_drift_vectors_rad[agent] + rotation_rng.normal(
                0.0, 0.0005, 3
            )
            perturbed[:3, :3] = relative[:3, :3] @ _rotation_vector(rotation_perturbation)
            relative_poses.append(perturbed)
            local_poses.append(local_poses[-1] @ perturbed)
        baseline[agent] = tuple(local_poses)
        odometry[agent] = tuple(relative_poses)

    cross_agent_translation_noise: dict[
        tuple[int, str],
        npt.NDArray[np.float64],
    ] = {}
    cross_agent_rotation_noise: dict[
        tuple[int, str],
        npt.NDArray[np.float64],
    ] = {}
    correction_noise: dict[tuple[int, str], npt.NDArray[np.float64]] = {}
    correction_rotation_noise: dict[tuple[int, str], npt.NDArray[np.float64]] = {}
    for index in range(1, sample_count):
        for destination_agent in ("Bob", "Carol"):
            key = (index, destination_agent)
            cross_agent_translation_noise[key] = translation_rng.normal(
                0.0,
                1.0,
                3,
            )
            cross_agent_rotation_noise[key] = rotation_rng.normal(
                0.0,
                1.0,
                3,
            )
        for agent in AGENTS:
            correction_noise[(index, agent)] = translation_rng.normal(0.0, 1.0, 3)
            correction_rotation_noise[(index, agent)] = rotation_rng.normal(
                0.0,
                1.0,
                3,
            )

    def cross_agent_measurements(
        translation_noise_std_m: float,
        rotation_noise_std_rad: float = 0.002,
    ) -> dict[tuple[int, str], npt.NDArray[np.float64]]:
        measurements: dict[tuple[int, str], npt.NDArray[np.float64]] = {}
        for key, translation_noise in cross_agent_translation_noise.items():
            index, destination_agent = key
            relative = np.linalg.inv(truth["Alpha"][index]) @ truth[destination_agent][index]
            relative = relative.copy()
            relative[:3, 3] += translation_noise * translation_noise_std_m
            relative[:3, :3] = relative[:3, :3] @ _rotation_vector(
                cross_agent_rotation_noise[key] * rotation_noise_std_rad
            )
            measurements[key] = relative
        return measurements

    cross_agent_translation_noise_std_m = 0.005
    cross_agent_rotation_noise_std_rad = 0.002
    cross_agent = cross_agent_measurements(
        cross_agent_translation_noise_std_m,
        cross_agent_rotation_noise_std_rad,
    )

    def corrections(
        noise_std_m: float,
        rotation_noise_std_rad: float = 0.005,
    ) -> dict[tuple[int, str], npt.NDArray[np.float64]]:
        measurements: dict[tuple[int, str], npt.NDArray[np.float64]] = {}
        for key, noise in correction_noise.items():
            index, agent = key
            pose = truth[agent][index].copy()
            pose[:3, 3] += noise * noise_std_m
            pose[:3, :3] = pose[:3, :3] @ _rotation_vector(
                correction_rotation_noise[key] * rotation_noise_std_rad
            )
            measurements[key] = pose
        return measurements

    correction_noise_std_m = 0.005
    correction_rotation_noise_std_rad = 0.005
    global_corrections = corrections(
        correction_noise_std_m,
        correction_rotation_noise_std_rad,
    )
    target_ate_m = 0.1
    target_orientation_error_rad = 0.05
    sample_interval_s = (overlap_end_s - overlap_start_s) / (sample_count - 1)
    cadence_runs = {
        interval: _run_graph(
            truth,
            odometry,
            cross_agent,
            global_corrections,
            correction_interval=interval,
        )
        for interval in (None, 8, 4, 2, 1)
    }
    baseline_cross_agent_rmse_m = _cross_agent_translation_rmse(
        baseline,
        truth,
    )
    cross_agent_cadence_sweep: list[dict[str, int | float]] = []
    dense_cross_agent_run = cadence_runs[None]
    for cross_interval in (4, 2, 1):
        selected_cross_agent = {
            key: relative
            for key, relative in cross_agent.items()
            if key[0] % cross_interval == 0 or key[0] == sample_count - 1
        }
        run = (
            dense_cross_agent_run
            if cross_interval == 1
            else _run_graph(
                truth,
                odometry,
                selected_cross_agent,
                global_corrections,
                correction_interval=None,
            )
        )
        cross_agent_cadence_sweep.append(
            {
                "interval_samples": cross_interval,
                "interval_seconds": sample_interval_s * cross_interval,
                "factor_count": len(selected_cross_agent),
                "factors_per_minute": (
                    len(selected_cross_agent) / (overlap_end_s - overlap_start_s) * 60.0
                ),
                "global_ate_m": _translation_ate(run.optimized, truth),
                "relative_translation_rmse_m": _cross_agent_translation_rmse(
                    run.optimized,
                    truth,
                ),
                "optimization_latency_ms": run.optimization_latency_ms,
            }
        )
    dense_cross_agent_ate_m = _translation_ate(
        dense_cross_agent_run.optimized,
        truth,
    )
    dense_cross_agent_relative_rmse_m = _cross_agent_translation_rmse(
        dense_cross_agent_run.optimized,
        truth,
    )
    dense_cross_agent_factor_rate = float(cross_agent_cadence_sweep[-1]["factors_per_minute"])
    cross_agent_translation_noise_sweep: list[dict[str, float]] = []
    for noise_std_m in (0.005, 0.05, 0.2):
        run = (
            dense_cross_agent_run
            if noise_std_m == cross_agent_translation_noise_std_m
            else _run_graph(
                truth,
                odometry,
                cross_agent_measurements(
                    noise_std_m,
                    cross_agent_rotation_noise_std_rad,
                ),
                global_corrections,
                correction_interval=None,
            )
        )
        cross_agent_translation_noise_sweep.append(
            {
                "translation_noise_std_m": noise_std_m,
                "global_ate_m": _translation_ate(run.optimized, truth),
                "relative_translation_rmse_m": _cross_agent_translation_rmse(
                    run.optimized,
                    truth,
                ),
            }
        )
    cross_agent_relative_rmse_at_0_05m_noise = cross_agent_translation_noise_sweep[1][
        "relative_translation_rmse_m"
    ]
    cross_agent_relative_rmse_at_0_2m_noise = cross_agent_translation_noise_sweep[-1][
        "relative_translation_rmse_m"
    ]

    baseline_ate_m = _translation_ate(baseline, truth)
    baseline_rpe_m = _translation_rpe(baseline, truth)
    baseline_orientation_rmse_rad = _orientation_rmse(baseline, truth)
    baseline_rotation_rpe_rad = _rotation_rpe(baseline, truth)
    cadence_sweep: list[dict[str, object]] = []
    target_intervals: list[int] = []
    for interval, run in cadence_runs.items():
        ate_m = _translation_ate(run.optimized, truth)
        rpe_m = _translation_rpe(run.optimized, truth)
        orientation_rmse_rad = _orientation_rmse(run.optimized, truth)
        rotation_rpe_rad = _rotation_rpe(run.optimized, truth)
        per_agent_ate_m, per_agent_orientation_rmse_rad = _per_agent_pose_metrics(
            run.optimized, truth
        )
        interval_samples = interval or 0
        target_met = (
            ate_m <= target_ate_m
            and orientation_rmse_rad <= target_orientation_error_rad
            and max(per_agent_ate_m.values()) <= target_ate_m
            and max(per_agent_orientation_rmse_rad.values()) <= target_orientation_error_rad
        )
        cadence_sweep.append(
            {
                "correction_interval_samples": interval_samples,
                "correction_interval_seconds": (
                    0.0
                    if interval is None
                    else (overlap_end_s - overlap_start_s) / (sample_count - 1) * interval
                ),
                "global_ate_m": ate_m,
                "translation_rpe_m": rpe_m,
                "orientation_rmse_rad": orientation_rmse_rad,
                "rotation_rpe_rmse_rad": rotation_rpe_rad,
                "target_ate_met": target_met,
                "maximum_agent_ate_m": max(per_agent_ate_m.values()),
                "maximum_agent_orientation_rmse_rad": max(per_agent_orientation_rmse_rad.values()),
                "global_correction_count": run.global_correction_count,
                "correction_payload_bytes_total": sum(
                    run.correction_payload_bytes_by_agent.values()
                ),
                "correction_payload_bytes_by_agent": run.correction_payload_bytes_by_agent,
                "optimization_latency_ms": run.optimization_latency_ms,
            }
        )
        if target_met and interval_samples > 0:
            target_intervals.append(interval_samples)
    fixed_interval: int = max(target_intervals) if target_intervals else 1
    fixed = cadence_runs[fixed_interval]
    adaptive_sweep_runs: list[_AdaptiveSweepRun] = []
    for demand_target_error_m in (0.05, 0.5, 0.75, 1.0):
        adaptive_indices, _, scheduler_metrics = _adaptive_correction_schedule(
            sampled,
            truth,
            baseline,
            global_corrections,
            target_error_m=demand_target_error_m,
            target_orientation_error_rad=target_orientation_error_rad,
            sample_interval_s=sample_interval_s,
        )
        adaptive_graph = _run_graph(
            truth,
            odometry,
            cross_agent,
            global_corrections,
            correction_interval=None,
            correction_indices=adaptive_indices,
        )
        adaptive_ate_m = _translation_ate(adaptive_graph.optimized, truth)
        adaptive_orientation_rmse_rad = _orientation_rmse(
            adaptive_graph.optimized,
            truth,
        )
        per_agent_ate_m, per_agent_orientation_rmse_rad = _per_agent_pose_metrics(
            adaptive_graph.optimized, truth
        )
        all_agents_target_met = (
            adaptive_ate_m <= target_ate_m
            and adaptive_orientation_rmse_rad <= target_orientation_error_rad
            and max(per_agent_ate_m.values()) <= target_ate_m
            and max(per_agent_orientation_rmse_rad.values()) <= target_orientation_error_rad
        )
        adaptive_sweep_runs.append(
            _AdaptiveSweepRun(
                demand_target_error_m,
                adaptive_indices,
                adaptive_graph,
                scheduler_metrics,
                adaptive_ate_m,
                adaptive_orientation_rmse_rad,
                per_agent_ate_m,
                per_agent_orientation_rmse_rad,
                all_agents_target_met,
            )
        )
    passing_adaptive_runs = [
        run
        for run in adaptive_sweep_runs
        if run.all_agents_target_met
        and run.graph.global_correction_count < fixed.global_correction_count
    ]
    adaptive_run = min(
        passing_adaptive_runs or adaptive_sweep_runs[:1],
        key=lambda run: (
            run.graph.global_correction_count,
            run.demand_target_error_m,
        ),
    )
    adaptive_indices = adaptive_run.correction_indices
    scheduler_metrics = adaptive_run.scheduler_metrics
    adaptive = adaptive_run.graph
    adaptive_ate_m = adaptive_run.fleet_ate_m
    adaptive_rpe_m = _translation_rpe(adaptive.optimized, truth)
    adaptive_orientation_rmse_rad = adaptive_run.fleet_orientation_rmse_rad
    adaptive_rotation_rpe_rad = _rotation_rpe(adaptive.optimized, truth)
    adaptive_target_met = adaptive_run.all_agents_target_met
    adaptive_reduces_load = adaptive.global_correction_count < fixed.global_correction_count
    if adaptive_target_met and adaptive_reduces_load:
        selected_strategy = "adaptive_per_wingman"
        selected_interval = 0
        selected_interval_s = (
            (overlap_end_s - overlap_start_s) * len(AGENTS) / adaptive.global_correction_count
        )
        selected = adaptive
        selected_correction_indices = adaptive_indices
    else:
        selected_strategy = "fixed_cadence"
        selected_interval = fixed_interval
        selected_interval_s = sample_interval_s * fixed_interval
        selected = fixed
        selected_correction_indices = None
    result = selected.result
    optimized = selected.optimized
    optimized_ate_m = _translation_ate(optimized, truth)
    optimized_rpe_m = _translation_rpe(optimized, truth)
    optimized_orientation_rmse_rad = _orientation_rmse(optimized, truth)
    optimized_rotation_rpe_rad = _rotation_rpe(optimized, truth)
    optimized_per_agent_ate_m, optimized_per_agent_orientation_rmse_rad = _per_agent_pose_metrics(
        optimized, truth
    )
    all_agents_position_target_met = max(optimized_per_agent_ate_m.values()) <= target_ate_m
    all_agents_orientation_target_met = (
        max(optimized_per_agent_orientation_rmse_rad.values()) <= target_orientation_error_rad
    )
    improvement_percent = 100.0 * (baseline_ate_m - optimized_ate_m) / baseline_ate_m
    false_loop_rejected = selected.false_loop_index in result.rejected_constraints
    component_count = len(set(result.components.values()))
    payload_bytes_total = sum(selected.correction_payload_bytes_by_agent.values())
    messages_per_minute_per_agent = (
        selected.global_correction_count / len(AGENTS) / (overlap_end_s - overlap_start_s) * 60.0
    )

    noise_sweep: list[dict[str, object]] = []
    tolerated_noise: list[float] = []
    for noise_std_m in (0.005, 0.025, 0.05, 0.1, 0.2):
        run = _run_graph(
            truth,
            odometry,
            cross_agent,
            corrections(noise_std_m),
            correction_interval=selected_interval if selected_interval > 0 else None,
            correction_indices=selected_correction_indices,
        )
        ate_m = _translation_ate(run.optimized, truth)
        orientation_rmse_rad = _orientation_rmse(run.optimized, truth)
        per_agent_ate_m, per_agent_orientation_rmse_rad = _per_agent_pose_metrics(
            run.optimized, truth
        )
        target_met = (
            ate_m <= target_ate_m
            and orientation_rmse_rad <= target_orientation_error_rad
            and max(per_agent_ate_m.values()) <= target_ate_m
            and max(per_agent_orientation_rmse_rad.values()) <= target_orientation_error_rad
        )
        noise_sweep.append(
            {
                "global_correction_noise_std_m": noise_std_m,
                "global_ate_m": ate_m,
                "orientation_rmse_rad": orientation_rmse_rad,
                "maximum_agent_ate_m": max(per_agent_ate_m.values()),
                "target_ate_met": target_met,
            }
        )
        if target_met:
            tolerated_noise.append(noise_std_m)
    maximum_tested_correction_noise_m = max(tolerated_noise, default=0.0)
    rotation_noise_sweep: list[dict[str, object]] = []
    tolerated_rotation_noise: list[float] = []
    for noise_std_rad in (0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.1, 0.2):
        run = _run_graph(
            truth,
            odometry,
            cross_agent,
            corrections(correction_noise_std_m, noise_std_rad),
            correction_interval=selected_interval if selected_interval > 0 else None,
            correction_indices=selected_correction_indices,
        )
        ate_m = _translation_ate(run.optimized, truth)
        orientation_rmse_rad = _orientation_rmse(run.optimized, truth)
        per_agent_ate_m, per_agent_orientation_rmse_rad = _per_agent_pose_metrics(
            run.optimized, truth
        )
        target_met = (
            ate_m <= target_ate_m
            and orientation_rmse_rad <= target_orientation_error_rad
            and max(per_agent_ate_m.values()) <= target_ate_m
            and max(per_agent_orientation_rmse_rad.values()) <= target_orientation_error_rad
        )
        rotation_noise_sweep.append(
            {
                "global_correction_noise_std_rad": noise_std_rad,
                "global_ate_m": ate_m,
                "orientation_rmse_rad": orientation_rmse_rad,
                "maximum_agent_ate_m": max(per_agent_ate_m.values()),
                "maximum_agent_orientation_rmse_rad": max(per_agent_orientation_rmse_rad.values()),
                "target_pose_met": target_met,
            }
        )
        if target_met:
            tolerated_rotation_noise.append(noise_std_rad)
    maximum_tested_rotation_noise_rad = max(
        tolerated_rotation_noise,
        default=0.0,
    )
    passed = (
        dataset_inventory.status == "passed"
        and component_count == 1
        and false_loop_rejected
        and optimized_ate_m <= target_ate_m
        and optimized_orientation_rmse_rad <= target_orientation_error_rad
        and all_agents_position_target_met
        and all_agents_orientation_target_met
        and optimized_rpe_m < baseline_rpe_m
        and optimized_rotation_rpe_rad < baseline_rotation_rpe_rad
    )
    claim_evidence = GlobalPoseClaimEvidence(
        odometry_source="controlled",
        position_reference_source="ground_truth_derived",
        orientation_reference_source="controlled",
        cross_agent_source="controlled",
        uses_ground_truth_in_estimator=True,
        causal=True,
        all_agents_position_target_met=all_agents_position_target_met,
        fleet_position_target_met=optimized_ate_m <= target_ate_m,
        orientation_target_met=all_agents_orientation_target_met,
    )
    metrics: dict[str, int | float | str] = {
        "seed": seed,
        "agent_count": len(AGENTS),
        "sample_count_per_agent": sample_count,
        "ground_truth_input_pose_count": sum(len(samples) for samples in trajectories.values()),
        "ground_truth_overlap_seconds": overlap_end_s - overlap_start_s,
        "bag_duration_seconds": dataset_inventory.metrics["duration_seconds"],
        "bag_vision_message_count": dataset_inventory.metrics["vision_message_count"],
        "bag_imu_message_count": dataset_inventory.metrics["imu_message_count"],
        "graph_node_count": len(result.poses),
        "graph_constraint_count": result.constraint_count,
        "graph_component_count": component_count,
        "graph_rejected_constraint_count": len(result.rejected_constraints),
        "rationalization_constraint_count": result.rationalization_constraint_count,
        "rotation_rationalization_constraint_count": (
            result.rotation_rationalization_constraint_count
        ),
        "rotation_rationalization_iterations": (result.rotation_rationalization_iterations),
        "optimized_constraint_rotation_rmse_rad": result.rotation_rmse_rad,
        "cross_agent_constraint_count": len(cross_agent),
        "baseline_cross_agent_relative_translation_rmse_m": (baseline_cross_agent_rmse_m),
        "dense_cross_agent_only_global_ate_m": dense_cross_agent_ate_m,
        "dense_cross_agent_only_global_target_met": int(dense_cross_agent_ate_m <= target_ate_m),
        "dense_cross_agent_relative_translation_rmse_m": (dense_cross_agent_relative_rmse_m),
        "dense_cross_agent_relative_improvement_percent": (
            100.0
            * (baseline_cross_agent_rmse_m - dense_cross_agent_relative_rmse_m)
            / baseline_cross_agent_rmse_m
        ),
        "dense_cross_agent_factor_rate_per_minute": (dense_cross_agent_factor_rate),
        "cross_agent_relative_rmse_at_0_05m_translation_noise_m": (
            cross_agent_relative_rmse_at_0_05m_noise
        ),
        "cross_agent_relative_rmse_at_0_2m_translation_noise_m": (
            cross_agent_relative_rmse_at_0_2m_noise
        ),
        "false_loop_rejected": int(false_loop_rejected),
        "baseline_global_ate_m": baseline_ate_m,
        "optimized_global_ate_m": optimized_ate_m,
        "target_global_ate_m": target_ate_m,
        "target_global_ate_met": int(optimized_ate_m <= target_ate_m),
        "maximum_agent_global_ate_m": max(optimized_per_agent_ate_m.values()),
        "all_agents_global_position_target_met": int(all_agents_position_target_met),
        "baseline_global_orientation_rmse_rad": baseline_orientation_rmse_rad,
        "optimized_global_orientation_rmse_rad": optimized_orientation_rmse_rad,
        "target_global_orientation_rmse_rad": target_orientation_error_rad,
        "orientation_reference_is_controlled": 1,
        "s3e_orientation_ground_truth_available": 0,
        "target_global_orientation_met": int(
            optimized_orientation_rmse_rad <= target_orientation_error_rad
        ),
        "maximum_agent_global_orientation_rmse_rad": max(
            optimized_per_agent_orientation_rmse_rad.values()
        ),
        "all_agents_global_orientation_target_met": int(all_agents_orientation_target_met),
        "target_global_pose_met": int(
            optimized_ate_m <= target_ate_m
            and optimized_orientation_rmse_rad <= target_orientation_error_rad
            and all_agents_position_target_met
            and all_agents_orientation_target_met
        ),
        "position_claim_eligible": int(claim_evidence.position_claim_eligible),
        "full_pose_claim_eligible": int(claim_evidence.full_pose_claim_eligible),
        "global_ate_improvement_percent": improvement_percent,
        "baseline_translation_rpe_m": baseline_rpe_m,
        "optimized_translation_rpe_m": optimized_rpe_m,
        "baseline_rotation_rpe_rmse_rad": baseline_rotation_rpe_rad,
        "optimized_rotation_rpe_rmse_rad": optimized_rotation_rpe_rad,
        "selected_correction_interval_samples": selected_interval,
        "selected_correction_interval_seconds": selected_interval_s,
        "selected_correction_strategy": selected_strategy,
        "selected_scheduler_demand_error_m": (
            adaptive_run.demand_target_error_m
            if selected_strategy == "adaptive_per_wingman"
            else 0.0
        ),
        "selected_global_correction_count": selected.global_correction_count,
        "fixed_global_correction_count": fixed.global_correction_count,
        "selected_correction_load_reduction_percent": (
            100.0
            * (fixed.global_correction_count - selected.global_correction_count)
            / fixed.global_correction_count
        ),
        "selected_capacity_override_cycle_count": (
            scheduler_metrics["capacity_overrides"]
            if selected_strategy == "adaptive_per_wingman"
            else 0
        ),
        "selected_correction_payload_bytes_total": payload_bytes_total,
        "selected_correction_payload_bytes_per_agent_max": max(
            selected.correction_payload_bytes_by_agent.values()
        ),
        "selected_correction_messages_per_minute_per_agent": messages_per_minute_per_agent,
        "maximum_tested_correction_noise_m_for_target": maximum_tested_correction_noise_m,
        "maximum_tested_correction_rotation_noise_rad_for_target": (
            maximum_tested_rotation_noise_rad
        ),
        "optimization_latency_ms": selected.optimization_latency_ms,
        "benchmark_latency_ms": (perf_counter_ns() - start_ns) / 1e6,
    }
    return DatasetEvaluation(
        dataset="s3e-global-pose",
        status="passed" if passed else "failed",
        agents=AGENTS,
        modalities=(
            "stereo",
            "imu",
            "rtk_position_truth",
            "controlled_orientation",
            "global_pose",
        ),
        metrics=metrics,
        warnings=(
            "Translation constraints use S3E RTK positions; rotation uses a controlled "
            "identity-frame reference because the dataset files contain identity quaternion "
            "placeholders. This isolates pose rationalization and is not an end-to-end visual "
            "localization or real orientation score.",
        ),
        details={
            "dataset_root": str(root),
            "bag_path": str(bag_path),
            "calibration_sha256": calibration_hashes,
            "drift_scale_by_agent": drift_scales,
            "rotation_drift_vector_rad_by_agent": {
                agent: value.tolist() for agent, value in rotation_drift_vectors_rad.items()
            },
            "constraint_source": (
                "S3E RTK positions plus a controlled identity-frame orientation "
                "reference and deterministic perturbations"
            ),
            "orientation_reference": "controlled_identity_frame",
            "claim_evidence": claim_evidence.as_dict(),
            "selected_global_correction_noise_std_m": correction_noise_std_m,
            "selected_global_correction_rotation_noise_std_rad": (
                correction_rotation_noise_std_rad
            ),
            "deliberate_false_loop_index": selected.false_loop_index,
            "rejected_constraint_indices": list(result.rejected_constraints),
            "cadence_sweep": cadence_sweep,
            "correction_noise_sweep": noise_sweep,
            "correction_rotation_noise_sweep": rotation_noise_sweep,
            "cross_agent_cadence_sweep": cross_agent_cadence_sweep,
            "cross_agent_translation_noise_sweep": (cross_agent_translation_noise_sweep),
            "adaptive_scheduler_demand_sweep": [
                {
                    "demand_target_error_m": run.demand_target_error_m,
                    "global_ate_m": run.fleet_ate_m,
                    "maximum_agent_ate_m": max(run.per_agent_ate_m.values()),
                    "maximum_agent_orientation_rmse_rad": max(
                        run.per_agent_orientation_rmse_rad.values()
                    ),
                    "all_agents_target_met": run.all_agents_target_met,
                    "global_correction_count": run.graph.global_correction_count,
                    "capacity_override_cycle_count": run.scheduler_metrics["capacity_overrides"],
                }
                for run in adaptive_sweep_runs
            ],
            "vision_correction_limits": {
                "cross_agent_factors_add_absolute_global_information": False,
                "selected_strategy": selected_strategy,
                "fleet_average_can_mask_weak_wingman": any(
                    run.fleet_ate_m <= target_ate_m and not run.all_agents_target_met
                    for run in adaptive_sweep_runs
                ),
                "target_requires_interval_samples": selected_interval,
                "target_requires_interval_seconds": selected_interval_s,
                "maximum_tested_noise_std_m_for_target": maximum_tested_correction_noise_m,
                "maximum_tested_rotation_noise_std_rad_for_target": (
                    maximum_tested_rotation_noise_rad
                ),
                "unverified_steps": [
                    "production visual association precision and recall",
                    "S3E backend-specific VIO calibration",
                    "real correction covariance calibration",
                ],
            },
            "intelligence_load": {
                "global_correction_count": selected.global_correction_count,
                "correction_count_by_agent": selected.correction_count_by_agent,
                "messages_per_minute_by_agent": {
                    agent: (
                        selected.correction_count_by_agent[agent]
                        / (overlap_end_s - overlap_start_s)
                        * 60.0
                    )
                    for agent in AGENTS
                },
                "payload_bytes_total": payload_bytes_total,
                "payload_bytes_by_agent": selected.correction_payload_bytes_by_agent,
                "messages_per_minute_per_agent": messages_per_minute_per_agent,
                "optimization_latency_ms": selected.optimization_latency_ms,
                "load_reduction_percent_vs_fixed": (
                    100.0
                    * (fixed.global_correction_count - selected.global_correction_count)
                    / fixed.global_correction_count
                ),
                "capacity_override_cycle_count": (
                    scheduler_metrics["capacity_overrides"]
                    if selected_strategy == "adaptive_per_wingman"
                    else 0
                ),
            },
            "adaptive_scheduler": {
                "demand_target_error_m": adaptive_run.demand_target_error_m,
                "demand_target_orientation_error_rad": (target_orientation_error_rad),
                "global_ate_m": adaptive_ate_m,
                "per_agent_ate_m": adaptive_run.per_agent_ate_m,
                "translation_rpe_m": adaptive_rpe_m,
                "orientation_rmse_rad": adaptive_orientation_rmse_rad,
                "per_agent_orientation_rmse_rad": (adaptive_run.per_agent_orientation_rmse_rad),
                "rotation_rpe_rmse_rad": adaptive_rotation_rpe_rad,
                "target_ate_met": adaptive_target_met,
                "target_pose_met": adaptive_target_met,
                "reduces_load_vs_fixed": adaptive_reduces_load,
                "global_correction_count": adaptive.global_correction_count,
                "payload_bytes_total": sum(adaptive.correction_payload_bytes_by_agent.values()),
                "scheduler_metrics": scheduler_metrics,
            },
            "fixed_cadence_reference": {
                "interval_samples": fixed_interval,
                "interval_seconds": sample_interval_s * fixed_interval,
                "global_ate_m": _translation_ate(fixed.optimized, truth),
                "orientation_rmse_rad": _orientation_rmse(
                    fixed.optimized,
                    truth,
                ),
                "global_correction_count": fixed.global_correction_count,
                "payload_bytes_total": sum(fixed.correction_payload_bytes_by_agent.values()),
            },
            "data_load_policy": {
                "shape": "aggregate_metrics_and_bounded_sweeps",
                "omitted_redundant_fields": [
                    "per_pose_trajectories",
                    "adaptive_scheduler_trace",
                    "duplicate_cross_agent_metrics",
                ],
            },
        },
    )
