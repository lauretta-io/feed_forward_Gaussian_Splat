"""Deterministic integrated benchmark for ARIADNE work-order steps one through five."""

from __future__ import annotations

from time import perf_counter_ns

import numpy as np
import numpy.typing as npt

from ariadne.common import Timestamp
from ariadne.datasets import DatasetEvaluation
from ariadne.models.features import (
    GradientPatchExtractor,
    GridPatchExtractor,
    IntensityHistogramEmbedder,
    SpatialPyramidEmbedder,
    benchmark_feature_pair,
)
from ariadne.models.vio import (
    ImuDeadReckoningVio,
    VisualInertialComplementaryVio,
    trajectory_metrics,
)
from ariadne.optimization import IncrementalPoseGraph, PoseConstraint
from ariadne.replay import ImageFrame, ImuSample, ReplaySynchronizer
from ariadne.tracking import (
    CrossAgentAssociator,
    StaticTrackState,
    TemporalStaticFilter,
    TrackObservation,
)


def _synthetic_replay(
    seed: int,
) -> tuple[list[ImageFrame], list[ImuSample], npt.NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    count = 50
    dt = 0.1
    times = np.arange(count, dtype=np.float64) * dt
    truth = np.column_stack((0.5 * 0.3 * times**2, 0.2 * times, np.zeros(count)))
    base_image = np.zeros((32, 32), dtype=np.float64)
    base_image[5:13, 4:11] = 0.8
    base_image[18:27, 20:29] = 1.0
    images: list[ImageFrame] = []
    imu: list[ImuSample] = []
    for index, time_s in enumerate(times):
        timestamp_ns = int(time_s * 1e9)
        visual_delta = truth[index] - truth[index - 1] if index else np.zeros(3)
        visual_delta = visual_delta + rng.normal(0.0, 0.0015, 3)
        image = np.roll(base_image, index % 3, axis=1) + rng.normal(0.0, 0.005, base_image.shape)
        images.append(ImageFrame(Timestamp(timestamp_ns), "wingman_01", image, index, visual_delta))
        for substep in range(10):
            sample_ns = timestamp_ns - 90_000_000 + substep * 10_000_000
            if sample_ns < 0:
                sample_ns = 0
            acceleration = np.array([0.315, 0.003, 0.0]) + rng.normal(0.0, 0.004, 3)
            imu.append(ImuSample(Timestamp(sample_ns), "wingman_01", acceleration, np.zeros(3)))
    return images, imu, truth


def _feature_benchmarks(seed: int) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(seed)
    reference_image = np.zeros((32, 32), dtype=np.float64)
    reference_image[4:12, 5:14] = 0.9
    reference_image[19:28, 21:27] = 0.6
    positive_image = np.roll(reference_image, 1, axis=1) + rng.normal(0.0, 0.01, (32, 32))
    negative_image = rng.random((32, 32))
    frames = [
        ImageFrame(Timestamp(index), "wingman_01", image, index)
        for index, image in enumerate((reference_image, positive_image, negative_image))
    ]
    pairs = (
        (GradientPatchExtractor(), SpatialPyramidEmbedder()),
        (GridPatchExtractor(), IntensityHistogramEmbedder()),
    )
    return [
        benchmark_feature_pair(geometric, semantic, frames[0], frames[1], frames[2])
        for geometric, semantic in pairs
    ]


def _tracking_and_association() -> tuple[dict[str, float | int], list[dict[str, object]]]:
    static_filter = TemporalStaticFilter()
    associator = CrossAgentAssociator(max_distance_m=1.0, min_cosine_similarity=0.85)
    confirmed_static = 0
    false_static = 0
    states: list[dict[str, object]] = []
    static_embedding = np.array([1.0, 0.0, 0.0])
    dynamic_embedding = np.array([0.0, 1.0, 0.0])
    for step in range(5):
        for agent, offset in (("wingman_01", 0.0), ("wingman_02", 0.08)):
            observation = TrackObservation(
                Timestamp(step * 100_000_000),
                agent,
                "static_landmark",
                np.array([4.0 + offset, 2.0, 0.0]),
                static_embedding,
                0.01,
                0.98,
                0.97,
            )
            state = static_filter.update(observation)
            associated = associator.associate(state)
            confirmed_static += int(state.state is StaticTrackState.STATIC_CONFIRMED)
            states.append(
                {
                    "agent": agent,
                    "track": observation.track_id,
                    "state": state.state.value,
                    "global_id": associated.global_id if associated else None,
                }
            )
        dynamic = TrackObservation(
            Timestamp(step * 100_000_000),
            "wingman_01",
            "moving_object",
            np.array([float(step), 1.0, 0.0]),
            dynamic_embedding,
            1.5,
            0.1,
            0.2,
        )
        dynamic_state = static_filter.update(dynamic)
        false_static += int(dynamic_state.state is StaticTrackState.STATIC_CONFIRMED)
    return (
        {
            "confirmed_static_observations": confirmed_static,
            "false_static_insertions": false_static,
            "global_object_count": len(associator.objects),
        },
        states,
    )


def _pose_graph_benchmark() -> tuple[dict[str, float | int], dict[str, list[float]]]:
    graph = IncrementalPoseGraph("wingman_01_t0")
    constraints = (
        PoseConstraint("wingman_01_t0", "wingman_01_t1", np.array([1.0, 0.0, 0.0]), 10.0),
        PoseConstraint("wingman_02_t0", "wingman_02_t1", np.array([1.0, 0.0, 0.0]), 10.0),
        PoseConstraint(
            "wingman_01_t0", "wingman_02_t0", np.array([0.0, 2.0, 0.0]), 8.0, "association"
        ),
        PoseConstraint(
            "wingman_01_t1", "wingman_02_t1", np.array([0.0, 2.0, 0.0]), 8.0, "association"
        ),
        PoseConstraint(
            "wingman_01_t0", "wingman_02_t1", np.array([8.0, -5.0, 0.0]), 0.3, "outlier"
        ),
    )
    for constraint in constraints:
        graph.add_constraint(constraint)
    result = graph.optimize(huber_delta_m=0.35)
    truth = {
        "wingman_01_t0": np.array([0.0, 0.0, 0.0]),
        "wingman_01_t1": np.array([1.0, 0.0, 0.0]),
        "wingman_02_t0": np.array([0.0, 2.0, 0.0]),
        "wingman_02_t1": np.array([1.0, 2.0, 0.0]),
    }
    errors = [
        np.linalg.norm(result.positions_m[node] - position) for node, position in truth.items()
    ]
    return (
        {
            "pose_graph_position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
            "pose_graph_constraint_rmse_m": result.rmse_m,
            "pose_graph_rejected_constraints": len(result.rejected_constraints),
            "pose_graph_iterations": result.iterations,
        },
        {node: position.tolist() for node, position in result.positions_m.items()},
    )


def run_phase1_benchmark(seed: int = 7) -> DatasetEvaluation:
    start_ns = perf_counter_ns()
    images, imu, truth = _synthetic_replay(seed)
    synchronization = ReplaySynchronizer(max_sync_error_ms=12.0).synchronize(images, imu)
    vio_results: dict[str, dict[str, float]] = {}
    for backend in (ImuDeadReckoningVio(), VisualInertialComplementaryVio()):
        estimates = [backend.process(packet) for packet in synchronization.packets]
        vio_results[backend.version.name] = trajectory_metrics(estimates, truth[: len(estimates)])
    feature_results = _feature_benchmarks(seed)
    tracking_metrics, tracking_states = _tracking_and_association()
    graph_metrics, graph_positions = _pose_graph_benchmark()
    imu_ate = vio_results["imu-dead-reckoning-reference"]["ate_rmse_m"]
    fused_ate = vio_results["visual-inertial-complementary-reference"]["ate_rmse_m"]
    best_feature_separation = max(float(item["semantic_separation"]) for item in feature_results)
    metrics: dict[str, int | float | str] = {
        "seed": seed,
        "synchronized_packets": len(synchronization.packets),
        "dropped_frames": synchronization.dropped_frames,
        "sync_median_ms": synchronization.median_error_ms,
        "sync_p95_ms": synchronization.p95_error_ms,
        "imu_ate_rmse_m": imu_ate,
        "fused_ate_rmse_m": fused_ate,
        "vio_ate_improvement_percent": (1.0 - fused_ate / imu_ate) * 100.0,
        "best_semantic_separation": best_feature_separation,
        **tracking_metrics,
        **graph_metrics,
        "benchmark_latency_ms": (perf_counter_ns() - start_ns) / 1e6,
    }
    passed = (
        synchronization.dropped_frames == 0
        and fused_ate < imu_ate
        and best_feature_separation > 0
        and tracking_metrics["false_static_insertions"] == 0
        and tracking_metrics["global_object_count"] == 1
        and graph_metrics["pose_graph_position_rmse_m"] < 0.25
        and graph_metrics["pose_graph_rejected_constraints"] >= 1
    )
    return DatasetEvaluation(
        dataset="phase1-reference",
        status="passed" if passed else "failed",
        agents=("wingman_01", "wingman_02"),
        modalities=("synthetic_vision", "synthetic_imu", "object_tracks", "pose_constraints"),
        metrics=metrics,
        warnings=(
            "NumPy backends validate interfaces and metrics; they are not production VIO "
            "or vision models.",
        ),
        details={
            "vio_backends": vio_results,
            "feature_pairs": feature_results,
            "tracking_states": tracking_states,
            "pose_graph_positions_m": graph_positions,
        },
    )
