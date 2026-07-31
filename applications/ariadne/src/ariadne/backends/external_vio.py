"""Process-isolated OpenVINS and ORB-SLAM3 adapters."""

from __future__ import annotations

import importlib
import re
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

from ariadne.replay import GroundTruthPose, ReplayBatch


@dataclass(frozen=True)
class TrajectoryPose:
    timestamp_ns: int
    position_m: npt.NDArray[np.float64]
    quaternion_xyzw: npt.NDArray[np.float64]


@dataclass(frozen=True)
class ExternalVioResult:
    backend: str
    status: str
    return_code: int | None
    elapsed_seconds: float
    trajectory: tuple[TrajectoryPose, ...]
    metrics: dict[str, int | float | str]
    command: tuple[str, ...]
    stdout_path: Path
    stderr_path: Path
    trajectory_path: Path
    detail: str = ""


@dataclass(frozen=True)
class EurocExportResult:
    times_path: Path
    stereo_pair_count: int
    imu_sample_count: int
    start_timestamp_ns: int
    end_timestamp_ns: int
    compressed_image_bytes: int


@dataclass(frozen=True)
class OrientationReference:
    timestamp_ns: int
    quaternion_xyzw: npt.NDArray[np.float64]
    covariance_available: bool

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("orientation-reference timestamp must be non-negative")
        quaternion = _normalize_quaternion(self.quaternion_xyzw)
        object.__setattr__(self, "quaternion_xyzw", quaternion)


def parse_trajectory(path: Path) -> tuple[TrajectoryPose, ...]:
    poses: list[TrajectoryPose] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = [float(value) for value in line.replace(",", " ").split()]
        if len(values) < 8:
            continue
        timestamp_ns = int(values[0]) if values[0] > 1e12 else int(values[0] * 1e9)
        poses.append(TrajectoryPose(timestamp_ns, np.asarray(values[1:4]), np.asarray(values[4:8])))
    return tuple(poses)


def _normalize_quaternion(
    quaternion_xyzw: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)) or norm <= 1e-12:
        raise ValueError("quaternion must be a finite non-zero vector")
    return np.asarray(quaternion / norm, dtype=np.float64)


def _interpolate_quaternion(
    before_xyzw: npt.ArrayLike,
    after_xyzw: npt.ArrayLike,
    fraction: float,
) -> npt.NDArray[np.float64]:
    before = _normalize_quaternion(before_xyzw)
    after = _normalize_quaternion(after_xyzw)
    dot = float(np.dot(before, after))
    if dot < 0:
        after = -after
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalize_quaternion(before + fraction * (after - before))
    angle = float(np.arccos(dot))
    denominator = float(np.sin(angle))
    return _normalize_quaternion(
        np.sin((1.0 - fraction) * angle) / denominator * before
        + np.sin(fraction * angle) / denominator * after
    )


def _quaternion_rotation(
    quaternion_xyzw: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    x, y, z, w = _normalize_quaternion(quaternion_xyzw)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_angles(
    rotations: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    cosines = np.clip(
        (np.trace(rotations, axis1=1, axis2=2) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return np.asarray(np.arccos(cosines), dtype=np.float64)


def _orientation_reference_available(
    truth: tuple[GroundTruthPose, ...],
) -> bool:
    return bool(truth) and all(pose.orientation_available for pose in truth)


def _matched_pose_samples(
    estimates: tuple[TrajectoryPose, ...],
    truth: tuple[GroundTruthPose, ...],
    tolerance_seconds: float,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    if not estimates or not truth:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.empty((0, 4)),
            np.empty((0, 4)),
        )
    truth_times = np.asarray([pose.timestamp.monotonic_ns for pose in truth], dtype=np.int64)
    matched_times: list[int] = []
    estimated_positions: list[npt.NDArray[np.float64]] = []
    truth_positions: list[npt.NDArray[np.float64]] = []
    estimated_quaternions: list[npt.NDArray[np.float64]] = []
    truth_quaternions: list[npt.NDArray[np.float64]] = []
    tolerance_ns = int(tolerance_seconds * 1e9)
    for estimate in estimates:
        insertion = int(np.searchsorted(truth_times, estimate.timestamp_ns))
        if 0 < insertion < len(truth):
            before = insertion - 1
            after = insertion
            before_ns = int(truth_times[before])
            after_ns = int(truth_times[after])
            if (
                min(
                    estimate.timestamp_ns - before_ns,
                    after_ns - estimate.timestamp_ns,
                )
                > tolerance_ns
                or after_ns - before_ns > 2 * tolerance_ns
            ):
                continue
            fraction = (estimate.timestamp_ns - before_ns) / (after_ns - before_ns)
            interpolated = (
                truth[before].position_m * (1.0 - fraction)
                + truth[after].position_m * fraction
            )
            interpolated_quaternion = _interpolate_quaternion(
                truth[before].quaternion_xyzw,
                truth[after].quaternion_xyzw,
                fraction,
            )
        else:
            nearest = min(max(insertion, 0), len(truth) - 1)
            if abs(int(truth_times[nearest]) - estimate.timestamp_ns) > tolerance_ns:
                continue
            interpolated = truth[nearest].position_m
            interpolated_quaternion = _normalize_quaternion(
                truth[nearest].quaternion_xyzw
            )
        try:
            estimated_quaternion = _normalize_quaternion(estimate.quaternion_xyzw)
        except ValueError:
            continue
        matched_times.append(estimate.timestamp_ns)
        estimated_positions.append(estimate.position_m)
        truth_positions.append(interpolated)
        estimated_quaternions.append(estimated_quaternion)
        truth_quaternions.append(interpolated_quaternion)
    return (
        np.asarray(matched_times, dtype=np.int64),
        np.asarray(estimated_positions),
        np.asarray(truth_positions),
        np.asarray(estimated_quaternions),
        np.asarray(truth_quaternions),
    )


def _matched_samples(
    estimates: tuple[TrajectoryPose, ...],
    truth: tuple[GroundTruthPose, ...],
    tolerance_seconds: float,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    timestamps, estimated, matched_truth, _, _ = _matched_pose_samples(
        estimates, truth, tolerance_seconds
    )
    return timestamps, estimated, matched_truth


def _matched_positions(
    estimates: tuple[TrajectoryPose, ...],
    truth: tuple[GroundTruthPose, ...],
    tolerance_seconds: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    _, estimated, matched_truth = _matched_samples(estimates, truth, tolerance_seconds)
    return estimated, matched_truth


def _rigid_alignment(
    estimated: npt.NDArray[np.float64], truth: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    estimated_center = np.mean(estimated, axis=0)
    truth_center = np.mean(truth, axis=0)
    covariance = (estimated - estimated_center).T @ (truth - truth_center)
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transpose[-1] *= -1
        rotation = right_transpose.T @ left.T
    aligned = (rotation @ (estimated - estimated_center).T).T + truth_center
    return np.asarray(aligned, dtype=np.float64), np.asarray(rotation, dtype=np.float64)


def _rigid_align(
    estimated: npt.NDArray[np.float64], truth: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return _rigid_alignment(estimated, truth)[0]


def _aligned_orientations(
    estimated_quaternions: npt.NDArray[np.float64],
    truth_quaternions: npt.NDArray[np.float64],
    position_alignment_rotation: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    estimated_rotations = np.stack(
        [_quaternion_rotation(value) for value in estimated_quaternions]
    )
    truth_rotations = np.stack(
        [_quaternion_rotation(value) for value in truth_quaternions]
    )
    globally_aligned = np.einsum(
        "ij,njk->nik",
        position_alignment_rotation,
        estimated_rotations,
    )
    cross_covariance = np.sum(
        np.einsum("nji,njk->nik", globally_aligned, truth_rotations),
        axis=0,
    )
    left, _, right_transpose = np.linalg.svd(cross_covariance)
    body_alignment = left @ right_transpose
    if np.linalg.det(body_alignment) < 0:
        left[:, -1] *= -1
        body_alignment = left @ right_transpose
    aligned = np.einsum("nij,jk->nik", globally_aligned, body_alignment)
    return np.asarray(aligned), np.asarray(truth_rotations)


def _matched_orientation_samples(
    estimates: tuple[TrajectoryPose, ...],
    reference: tuple[OrientationReference, ...],
    tolerance_seconds: float,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    if not estimates or not reference:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, 4)),
            np.empty((0, 4)),
        )
    reference_times = np.asarray(
        [sample.timestamp_ns for sample in reference],
        dtype=np.int64,
    )
    tolerance_ns = int(tolerance_seconds * 1e9)
    matched_times: list[int] = []
    estimated_quaternions: list[npt.NDArray[np.float64]] = []
    reference_quaternions: list[npt.NDArray[np.float64]] = []
    for estimate in estimates:
        insertion = int(np.searchsorted(reference_times, estimate.timestamp_ns))
        try:
            estimated_quaternion = _normalize_quaternion(
                estimate.quaternion_xyzw
            )
            if (
                insertion < len(reference)
                and int(reference_times[insertion]) == estimate.timestamp_ns
            ):
                reference_quaternion = reference[insertion].quaternion_xyzw
            else:
                if not 0 < insertion < len(reference):
                    continue
                before = insertion - 1
                after = insertion
                before_ns = int(reference_times[before])
                after_ns = int(reference_times[after])
                if (
                    min(
                        estimate.timestamp_ns - before_ns,
                        after_ns - estimate.timestamp_ns,
                    )
                    > tolerance_ns
                    or after_ns - before_ns > 2 * tolerance_ns
                ):
                    continue
                fraction = (estimate.timestamp_ns - before_ns) / (
                    after_ns - before_ns
                )
                reference_quaternion = _interpolate_quaternion(
                    reference[before].quaternion_xyzw,
                    reference[after].quaternion_xyzw,
                    fraction,
                )
        except ValueError:
            continue
        matched_times.append(estimate.timestamp_ns)
        estimated_quaternions.append(estimated_quaternion)
        reference_quaternions.append(reference_quaternion)
    return (
        np.asarray(matched_times, dtype=np.int64),
        np.asarray(estimated_quaternions),
        np.asarray(reference_quaternions),
    )


def evaluate_orientation_proxy(
    estimates: tuple[TrajectoryPose, ...],
    position_truth: tuple[GroundTruthPose, ...],
    orientation_reference: tuple[OrientationReference, ...],
    *,
    position_tolerance_seconds: float = 0.6,
    orientation_tolerance_seconds: float = 0.03,
    independent_of_vio: bool = False,
) -> dict[str, int | float | str]:
    """Compare estimated rotation with an IMU/AHRS orientation proxy."""
    if position_tolerance_seconds <= 0 or orientation_tolerance_seconds <= 0:
        raise ValueError("orientation-proxy tolerances must be positive")
    _, estimated_positions, matched_positions = _matched_samples(
        estimates,
        position_truth,
        position_tolerance_seconds,
    )
    if len(estimated_positions) < 3:
        return {
            "orientation_proxy_source": "s3e_imu_ahrs_proxy",
            "orientation_proxy_is_ground_truth": 0,
            "orientation_proxy_independent_of_vio": int(independent_of_vio),
            "orientation_proxy_matched_pose_count": 0,
        }
    _, position_alignment_rotation = _rigid_alignment(
        estimated_positions,
        matched_positions,
    )
    (
        timestamps_ns,
        estimated_quaternions,
        reference_quaternions,
    ) = _matched_orientation_samples(
        estimates,
        orientation_reference,
        orientation_tolerance_seconds,
    )
    if len(estimated_quaternions) < 3:
        return {
            "orientation_proxy_source": "s3e_imu_ahrs_proxy",
            "orientation_proxy_is_ground_truth": 0,
            "orientation_proxy_independent_of_vio": int(independent_of_vio),
            "orientation_proxy_matched_pose_count": len(estimated_quaternions),
        }
    aligned_rotations, reference_rotations = _aligned_orientations(
        estimated_quaternions,
        reference_quaternions,
        position_alignment_rotation,
    )
    errors_rad = _rotation_angles(
        np.einsum(
            "nji,njk->nik",
            reference_rotations,
            aligned_rotations,
        )
    )
    estimated_deltas = np.einsum(
        "nji,njk->nik",
        aligned_rotations[:-1],
        aligned_rotations[1:],
    )
    reference_deltas = np.einsum(
        "nji,njk->nik",
        reference_rotations[:-1],
        reference_rotations[1:],
    )
    rpe_rad = _rotation_angles(
        np.einsum(
            "nji,njk->nik",
            reference_deltas,
            estimated_deltas,
        )
    )
    timestamp_deltas_s = np.diff(timestamps_ns).astype(np.float64) / 1e9
    return {
        "orientation_proxy_source": "s3e_imu_ahrs_proxy",
        "orientation_proxy_is_ground_truth": 0,
        "orientation_proxy_independent_of_vio": int(independent_of_vio),
        "orientation_proxy_covariance_available": int(
            bool(orientation_reference)
            and all(sample.covariance_available for sample in orientation_reference)
        ),
        "orientation_proxy_reference_sample_count": len(orientation_reference),
        "orientation_proxy_matched_pose_count": len(estimated_quaternions),
        "orientation_proxy_rmse_rad": float(np.sqrt(np.mean(errors_rad**2))),
        "orientation_proxy_p95_rad": float(np.percentile(errors_rad, 95)),
        "orientation_proxy_max_rad": float(np.max(errors_rad)),
        "orientation_proxy_rpe_rmse_rad": float(np.sqrt(np.mean(rpe_rad**2))),
        "orientation_proxy_rpe_p95_rad": float(np.percentile(rpe_rad, 95)),
        "orientation_proxy_tracking_gap_p95_ms": float(
            np.percentile(timestamp_deltas_s, 95) * 1e3
        ),
    }


def evaluate_time_offset_sensitivity(
    estimates: tuple[TrajectoryPose, ...],
    truth: tuple[GroundTruthPose, ...],
    *,
    tolerance_seconds: float = 0.6,
    minimum_offset_ms: int = -500,
    maximum_offset_ms: int = 500,
    offset_step_ms: int = 25,
) -> dict[str, int | float]:
    """Measure whether one bounded timestamp shift explains trajectory error."""
    if (
        tolerance_seconds <= 0
        or minimum_offset_ms > 0
        or maximum_offset_ms < 0
        or minimum_offset_ms >= maximum_offset_ms
        or offset_step_ms <= 0
        or (maximum_offset_ms - minimum_offset_ms) % offset_step_ms
    ):
        raise ValueError("time-offset sweep controls are invalid")
    orientation_reference_available = _orientation_reference_available(truth)
    candidates: list[tuple[int, int, float, float]] = []
    for offset_ms in range(
        minimum_offset_ms,
        maximum_offset_ms + 1,
        offset_step_ms,
    ):
        offset_ns = offset_ms * 1_000_000
        shifted = tuple(
            TrajectoryPose(
                pose.timestamp_ns + offset_ns,
                pose.position_m,
                pose.quaternion_xyzw,
            )
            for pose in estimates
        )
        (
            _,
            estimated,
            matched_truth,
            estimated_quaternions,
            truth_quaternions,
        ) = _matched_pose_samples(shifted, truth, tolerance_seconds)
        if len(estimated) < 3:
            continue
        aligned, position_rotation = _rigid_alignment(estimated, matched_truth)
        translation_errors = np.linalg.norm(aligned - matched_truth, axis=1)
        aligned_rotations, truth_rotations = _aligned_orientations(
            estimated_quaternions,
            truth_quaternions,
            position_rotation,
        )
        orientation_errors = _rotation_angles(
            np.einsum(
                "nji,njk->nik",
                truth_rotations,
                aligned_rotations,
            )
        )
        candidates.append(
            (
                offset_ms,
                len(estimated),
                float(np.sqrt(np.mean(translation_errors**2))),
                float(np.sqrt(np.mean(orientation_errors**2))),
            )
        )
    zero = next((item for item in candidates if item[0] == 0), None)
    if zero is None:
        return {
            "timing_offset_candidate_count": len(candidates),
            "timing_offset_is_dominant": 0,
        }
    best_translation = min(candidates, key=lambda item: (item[2], abs(item[0])))
    ate_improvement_percent = (
        100.0 * (zero[2] - best_translation[2]) / zero[2]
        if zero[2] > np.finfo(np.float64).eps
        else 0.0
    )
    metrics: dict[str, int | float] = {
        "timing_offset_candidate_count": len(candidates),
        "timing_offset_minimum_ms": minimum_offset_ms,
        "timing_offset_maximum_ms": maximum_offset_ms,
        "timing_offset_step_ms": offset_step_ms,
        "timing_zero_ate_rmse_m": zero[2],
        "timing_best_ate_rmse_m": best_translation[2],
        "timing_best_ate_offset_ms": best_translation[0],
        "timing_best_ate_matched_pose_count": best_translation[1],
        "timing_ate_improvement_percent": ate_improvement_percent,
        "timing_best_ate_at_boundary": int(
            best_translation[0] in (minimum_offset_ms, maximum_offset_ms)
        ),
        "timing_offset_is_dominant": int(ate_improvement_percent >= 10.0),
    }
    if orientation_reference_available:
        best_orientation = min(candidates, key=lambda item: (item[3], abs(item[0])))
        orientation_improvement_percent = (
            100.0 * (zero[3] - best_orientation[3]) / zero[3]
            if zero[3] > np.finfo(np.float64).eps
            else 0.0
        )
        metrics.update(
            {
                "timing_zero_orientation_rmse_rad": zero[3],
                "timing_best_orientation_rmse_rad": best_orientation[3],
                "timing_best_orientation_offset_ms": best_orientation[0],
                "timing_orientation_improvement_percent": (
                    orientation_improvement_percent
                ),
                "timing_best_orientation_at_boundary": int(
                    best_orientation[0] in (minimum_offset_ms, maximum_offset_ms)
                ),
                "timing_offset_is_dominant": int(
                    max(ate_improvement_percent, orientation_improvement_percent)
                    >= 10.0
                ),
            }
        )
    return metrics


def evaluate_trajectory(
    estimates: tuple[TrajectoryPose, ...],
    truth: tuple[GroundTruthPose, ...],
    *,
    tolerance_seconds: float = 0.6,
) -> dict[str, int | float | str]:
    orientation_reference_available = _orientation_reference_available(truth)
    (
        timestamps_ns,
        estimated,
        matched_truth,
        estimated_quaternions,
        truth_quaternions,
    ) = _matched_pose_samples(
        estimates, truth, tolerance_seconds
    )
    if len(estimated) < 3:
        return {
            "trajectory_pose_count": len(estimates),
            "matched_pose_count": len(estimated),
            "ate_rmse_m": float("nan"),
            "rpe_rmse_m": float("nan"),
            "orientation_rmse_rad": float("nan"),
            "rotation_rpe_rmse_rad": float("nan"),
            "final_drift_m": float("nan"),
            "orientation_reference_available": int(
                orientation_reference_available
            ),
        }
    aligned, position_alignment_rotation = _rigid_alignment(
        estimated, matched_truth
    )
    errors = np.linalg.norm(aligned - matched_truth, axis=1)
    aligned_centered = aligned - np.mean(aligned, axis=0)
    truth_centered = matched_truth - np.mean(matched_truth, axis=0)
    scale_denominator = float(np.sum(aligned_centered**2))
    metric_scale_correction = (
        float(np.sum(aligned_centered * truth_centered) / scale_denominator)
        if scale_denominator > np.finfo(np.float64).eps
        else 1.0
    )
    similarity_aligned = (
        np.mean(matched_truth, axis=0)
        + metric_scale_correction * aligned_centered
    )
    similarity_errors = np.linalg.norm(similarity_aligned - matched_truth, axis=1)
    relative_errors = np.linalg.norm(
        np.diff(aligned, axis=0) - np.diff(matched_truth, axis=0), axis=1
    )
    aligned_rotations, truth_rotations = _aligned_orientations(
        estimated_quaternions,
        truth_quaternions,
        position_alignment_rotation,
    )
    orientation_error_rotations = np.einsum(
        "nji,njk->nik",
        truth_rotations,
        aligned_rotations,
    )
    orientation_errors_rad = _rotation_angles(orientation_error_rotations)
    estimated_rotation_deltas = np.einsum(
        "nji,njk->nik",
        aligned_rotations[:-1],
        aligned_rotations[1:],
    )
    truth_rotation_deltas = np.einsum(
        "nji,njk->nik",
        truth_rotations[:-1],
        truth_rotations[1:],
    )
    rotation_rpe_rotations = np.einsum(
        "nji,njk->nik",
        truth_rotation_deltas,
        estimated_rotation_deltas,
    )
    rotation_rpe_rad = _rotation_angles(rotation_rpe_rotations)
    timestamp_deltas_s = np.diff(timestamps_ns).astype(np.float64) / 1e9
    valid_deltas = timestamp_deltas_s > 0
    error_growth = np.maximum(np.diff(errors), 0.0)[valid_deltas] / timestamp_deltas_s[
        valid_deltas
    ]
    correction_profile: list[tuple[float, float]] = []
    vector_errors = aligned - matched_truth
    timestamps_s = timestamps_ns.astype(np.float64) / 1e9
    for interval_s in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
        corrected_errors: list[float] = []
        anchor_error = vector_errors[0]
        last_correction_s = timestamps_s[0]
        for timestamp_s, error in zip(timestamps_s, vector_errors, strict=True):
            if timestamp_s - last_correction_s >= interval_s:
                anchor_error = error
                last_correction_s = timestamp_s
            corrected_errors.append(float(np.linalg.norm(error - anchor_error)))
        corrected_ate_m = float(np.sqrt(np.mean(np.square(corrected_errors))))
        correction_profile.append((interval_s, corrected_ate_m))
    passing_intervals = [
        (interval_s, ate_m)
        for interval_s, ate_m in correction_profile
        if ate_m <= 0.1
    ]
    correction_target_reachable = bool(passing_intervals)
    if correction_target_reachable:
        selected_interval_s, selected_corrected_ate_m = max(passing_intervals)
    else:
        selected_interval_s = 0.0
        selected_corrected_ate_m = correction_profile[0][1]
    duration_s = float(timestamps_s[-1] - timestamps_s[0])
    quarter_size = max(len(errors) // 4, 1)
    first_quarter_errors = errors[:quarter_size]
    last_quarter_errors = errors[-quarter_size:]
    middle_half_errors = errors[quarter_size:-quarter_size]
    middle_half_ate_m = float(np.sqrt(np.mean(np.square(middle_half_errors))))
    first_quarter_ate_m = float(
        np.sqrt(np.mean(np.square(first_quarter_errors)))
    )
    last_quarter_ate_m = float(
        np.sqrt(np.mean(np.square(last_quarter_errors)))
    )
    peak_error_index = int(np.argmax(errors))
    event_candidates: list[
        tuple[int, float, float, float, float, float, float, int]
    ] = []
    for trigger_m in np.linspace(0.1, 1.0, 91):
        anchor_error = vector_errors[0]
        correction_times_s: list[float] = []
        corrected_errors = []
        for timestamp_s, error in zip(timestamps_s, vector_errors, strict=True):
            residual_m = float(np.linalg.norm(error - anchor_error))
            if residual_m > trigger_m:
                anchor_error = error
                correction_times_s.append(float(timestamp_s))
                residual_m = 0.0
            corrected_errors.append(residual_m)
        corrected_ate_m = float(np.sqrt(np.mean(np.square(corrected_errors))))
        if corrected_ate_m <= 0.1:
            intervals_s = np.diff(
                np.asarray([timestamps_s[0], *correction_times_s], dtype=np.float64)
            )
            peak_corrections_per_second = 0
            left = 0
            for right, correction_time_s in enumerate(correction_times_s):
                while correction_time_s - correction_times_s[left] >= 1.0:
                    left += 1
                peak_corrections_per_second = max(
                    peak_corrections_per_second,
                    right - left + 1,
                )
            event_candidates.append(
                (
                    len(correction_times_s),
                    corrected_ate_m,
                    float(trigger_m),
                    float(np.mean(intervals_s)) if len(intervals_s) else duration_s,
                    (
                        float(np.percentile(intervals_s, 95))
                        if len(intervals_s)
                        else duration_s
                    ),
                    float(np.min(intervals_s)) if len(intervals_s) else duration_s,
                    (
                        float(np.percentile(intervals_s, 5))
                        if len(intervals_s)
                        else duration_s
                    ),
                    peak_corrections_per_second,
                )
            )
    (
        event_count,
        event_corrected_ate_m,
        event_trigger_m,
        event_mean_interval_s,
        event_p95_interval_s,
        event_min_interval_s,
        event_p05_interval_s,
        event_peak_corrections_per_second,
    ) = min(event_candidates, key=lambda item: (item[0], item[1]))
    event_messages_per_minute = (
        event_count * 60.0 / duration_s if duration_s > 0 else 0.0
    )
    event_indices: set[int] = set()
    anchor_error = vector_errors[0]
    for index, error in enumerate(vector_errors):
        if float(np.linalg.norm(error - anchor_error)) > event_trigger_m:
            anchor_error = error
            event_indices.add(index)
    orientation_correction = truth_rotations[0] @ aligned_rotations[0].T
    corrected_event_rotations: list[npt.NDArray[np.float64]] = []
    rotation_correction_updates_rad: list[float] = []
    for index, (estimated_rotation, truth_rotation) in enumerate(
        zip(aligned_rotations, truth_rotations, strict=True)
    ):
        if index in event_indices:
            next_correction = truth_rotation @ estimated_rotation.T
            correction_update = next_correction @ orientation_correction.T
            rotation_correction_updates_rad.append(
                float(_rotation_angles(correction_update[None, ...])[0])
            )
            orientation_correction = next_correction
        corrected_event_rotations.append(orientation_correction @ estimated_rotation)
    corrected_orientation_errors_rad = _rotation_angles(
        np.einsum(
            "nji,njk->nik",
            truth_rotations,
            np.stack(corrected_event_rotations),
        )
    )
    se3_rotation_target_rad = 0.05
    se3_anchor_error = vector_errors[0]
    se3_orientation_correction = truth_rotations[0] @ aligned_rotations[0].T
    se3_correction_times_s: list[float] = []
    se3_corrected_translation_errors_m: list[float] = []
    se3_corrected_orientation_errors_rad: list[float] = []
    se3_rotation_updates_rad: list[float] = []
    for timestamp_s, error, estimated_rotation, truth_rotation in zip(
        timestamps_s,
        vector_errors,
        aligned_rotations,
        truth_rotations,
        strict=True,
    ):
        translation_residual_m = float(np.linalg.norm(error - se3_anchor_error))
        corrected_rotation = se3_orientation_correction @ estimated_rotation
        orientation_residual_rad = float(
            _rotation_angles(
                (truth_rotation.T @ corrected_rotation)[None, ...]
            )[0]
        )
        if (
            translation_residual_m > event_trigger_m
            or orientation_residual_rad > se3_rotation_target_rad
        ):
            se3_anchor_error = error
            next_correction = truth_rotation @ estimated_rotation.T
            se3_rotation_updates_rad.append(
                float(
                    _rotation_angles(
                        (next_correction @ se3_orientation_correction.T)[None, ...]
                    )[0]
                )
            )
            se3_orientation_correction = next_correction
            se3_correction_times_s.append(float(timestamp_s))
            translation_residual_m = 0.0
            orientation_residual_rad = 0.0
        se3_corrected_translation_errors_m.append(translation_residual_m)
        se3_corrected_orientation_errors_rad.append(orientation_residual_rad)
    se3_count = len(se3_correction_times_s)
    se3_messages_per_minute = (
        se3_count * 60.0 / duration_s if duration_s > 0 else 0.0
    )
    se3_intervals_s = np.diff(
        np.asarray(
            [timestamps_s[0], *se3_correction_times_s],
            dtype=np.float64,
        )
    )
    se3_peak_corrections_per_second = 0
    left = 0
    for right, correction_time_s in enumerate(se3_correction_times_s):
        while correction_time_s - se3_correction_times_s[left] >= 1.0:
            left += 1
        se3_peak_corrections_per_second = max(
            se3_peak_corrections_per_second,
            right - left + 1,
        )
    periodic_messages_per_minute = (
        60.0 / selected_interval_s if selected_interval_s > 0 else 0.0
    )
    metrics: dict[str, int | float | str] = {
        "trajectory_pose_count": len(estimates),
        "matched_pose_count": len(estimated),
        "ate_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "sim3_ate_rmse_m": float(np.sqrt(np.mean(similarity_errors**2))),
        "metric_scale_correction_to_truth": metric_scale_correction,
        "rpe_rmse_m": float(np.sqrt(np.mean(relative_errors**2))),
        "orientation_rmse_rad": float(
            np.sqrt(np.mean(orientation_errors_rad**2))
        ),
        "orientation_p95_rad": float(np.percentile(orientation_errors_rad, 95)),
        "orientation_max_rad": float(np.max(orientation_errors_rad)),
        "rotation_rpe_rmse_rad": float(np.sqrt(np.mean(rotation_rpe_rad**2))),
        "rotation_rpe_p95_rad": float(np.percentile(rotation_rpe_rad, 95)),
        "final_drift_m": float(errors[-1]),
        "error_p95_m": float(np.percentile(errors, 95)),
        "error_max_m": float(np.max(errors)),
        "positive_error_growth_p95_m_s": (
            float(np.percentile(error_growth, 95)) if len(error_growth) else 0.0
        ),
        "tracking_gap_p95_ms": float(np.percentile(timestamp_deltas_s, 95) * 1e3),
        "evaluated_duration_seconds": duration_s,
        "ate_first_quarter_m": first_quarter_ate_m,
        "ate_middle_half_m": middle_half_ate_m,
        "ate_last_quarter_m": last_quarter_ate_m,
        "ate_last_to_first_ratio": (
            last_quarter_ate_m / first_quarter_ate_m
            if first_quarter_ate_m > np.finfo(np.float64).eps
            else 0.0
        ),
        "peak_error_time_offset_seconds": float(
            timestamps_s[peak_error_index] - timestamps_s[0]
        ),
        "peak_error_trajectory_fraction": (
            peak_error_index / (len(errors) - 1) if len(errors) > 1 else 0.0
        ),
        "orientation_reference_available": int(
            orientation_reference_available
        ),
        "maximum_correction_interval_seconds_for_0_1m": selected_interval_s,
        "corrected_ate_at_selected_interval_m": selected_corrected_ate_m,
        "correction_target_reachable_with_tested_intervals": int(
            correction_target_reachable
        ),
        "correction_messages_per_minute_for_0_1m": periodic_messages_per_minute,
        "minimum_tested_correction_interval_seconds": correction_profile[0][0],
        "maximum_tested_correction_messages_per_minute": (
            60.0 / correction_profile[0][0]
        ),
        "event_triggered_correction_count_for_0_1m": event_count,
        "event_triggered_corrected_ate_m": event_corrected_ate_m,
        "event_triggered_error_threshold_m": event_trigger_m,
        "event_triggered_messages_per_minute_for_0_1m": event_messages_per_minute,
        "event_triggered_mean_interval_seconds": event_mean_interval_s,
        "event_triggered_p95_interval_seconds": event_p95_interval_s,
        "event_triggered_min_interval_seconds": event_min_interval_s,
        "event_triggered_p05_interval_seconds": event_p05_interval_s,
        "event_triggered_peak_corrections_per_second": (
            event_peak_corrections_per_second
        ),
        "event_triggered_orientation_rmse_rad": float(
            np.sqrt(np.mean(corrected_orientation_errors_rad**2))
        ),
        "event_triggered_orientation_p95_rad": float(
            np.percentile(corrected_orientation_errors_rad, 95)
        ),
        "event_triggered_rotation_update_p95_rad": (
            float(np.percentile(rotation_correction_updates_rad, 95))
            if rotation_correction_updates_rad
            else 0.0
        ),
        "event_triggered_rotation_update_max_rad": (
            float(np.max(rotation_correction_updates_rad))
            if rotation_correction_updates_rad
            else 0.0
        ),
        "se3_event_triggered_correction_count": se3_count,
        "se3_event_triggered_messages_per_minute": se3_messages_per_minute,
        "se3_event_triggered_additional_messages_per_minute": (
            se3_messages_per_minute - event_messages_per_minute
        ),
        "se3_event_triggered_corrected_ate_m": float(
            np.sqrt(np.mean(np.square(se3_corrected_translation_errors_m)))
        ),
        "se3_event_triggered_orientation_target_rad": se3_rotation_target_rad,
        "se3_event_triggered_orientation_rmse_rad": float(
            np.sqrt(np.mean(np.square(se3_corrected_orientation_errors_rad)))
        ),
        "se3_event_triggered_min_interval_seconds": (
            float(np.min(se3_intervals_s)) if len(se3_intervals_s) else duration_s
        ),
        "se3_event_triggered_peak_corrections_per_second": (
            se3_peak_corrections_per_second
        ),
        "se3_event_triggered_rotation_update_p95_rad": (
            float(np.percentile(se3_rotation_updates_rad, 95))
            if se3_rotation_updates_rad
            else 0.0
        ),
        "se3_event_triggered_rotation_update_max_rad": (
            float(np.max(se3_rotation_updates_rad))
            if se3_rotation_updates_rad
            else 0.0
        ),
        "event_triggered_rate_reduction_vs_periodic_percent": (
            100.0
            * (periodic_messages_per_minute - event_messages_per_minute)
            / periodic_messages_per_minute
            if periodic_messages_per_minute > 0
            else 0.0
        ),
    }
    metrics.update(
        evaluate_time_offset_sensitivity(
            estimates,
            truth,
            tolerance_seconds=tolerance_seconds,
        )
    )
    if not orientation_reference_available:
        unavailable_keys = tuple(
            key
            for key in metrics
            if (
                "orientation" in key
                or "rotation" in key
                or key.startswith("se3_event_triggered_")
            )
            and key != "orientation_reference_available"
        )
        for key in unavailable_keys:
            metrics.pop(key)
    return metrics


def _result_from_artifacts(
    *,
    backend: str,
    trajectory_path: Path,
    truth: tuple[GroundTruthPose, ...],
    stdout_path: Path,
    stderr_path: Path,
    return_code: int | None,
    elapsed_seconds: float,
    command: tuple[str, ...],
    detail: str,
) -> ExternalVioResult:
    trajectory = parse_trajectory(trajectory_path) if trajectory_path.is_file() else ()
    metrics = evaluate_trajectory(trajectory, truth)
    metrics["elapsed_seconds"] = elapsed_seconds
    metrics["return_code"] = return_code if return_code is not None else -1
    stdout = stdout_path.read_text(encoding="utf-8") if stdout_path.is_file() else ""
    lost_frame_groups = re.findall(r"(\d+) Frames set to lost", stdout)
    metrics["lost_frame_count"] = max(
        (int(value) for value in lost_frame_groups),
        default=0,
    )
    metrics["map_reset_count"] = max(stdout.count("New Map created") - 1, 0)
    metrics["insufficient_acceleration_count"] = stdout.count("not enough acceleration")
    metrics["initialization_failure_count"] = stdout.count("failed static init")
    metrics["initialization_success_count"] = stdout.count(
        "successful initialization"
    )
    scale_metric = metrics.get("metric_scale_correction_to_truth")
    metric_scale_plausible = int(metrics["matched_pose_count"]) < 3 or (
        scale_metric is not None and 0.25 <= float(scale_metric) <= 4.0
    )
    metrics["metric_scale_plausible"] = int(metric_scale_plausible)
    metrics["geometric_divergence_detected"] = int(not metric_scale_plausible)
    tracking_healthy = (
        metrics["lost_frame_count"] == 0 and metrics["map_reset_count"] == 0
        and metric_scale_plausible
    )
    metrics["tracking_healthy"] = int(tracking_healthy)
    metrics["event_triggered_correction_applicable"] = int(tracking_healthy)
    metrics["recommended_intelligence_action"] = (
        "schedule_event_triggered_corrections"
        if tracking_healthy
        else "relocalize"
    )
    accuracy_metrics = ("ate_rmse_m", "rpe_rmse_m", "final_drift_m")
    has_valid_accuracy = int(metrics["matched_pose_count"]) >= 3 and all(
        np.isfinite(float(metrics[name])) for name in accuracy_metrics
    )
    status = (
        "passed"
        if return_code == 0 and len(trajectory) >= 3 and has_valid_accuracy
        else "failed"
    )
    return ExternalVioResult(
        backend,
        status,
        return_code,
        elapsed_seconds,
        trajectory,
        metrics,
        command,
        stdout_path,
        stderr_path,
        trajectory_path,
        detail,
    )


def reanalyze_vio_artifacts(
    *,
    backend: str,
    trajectory_path: Path,
    truth: tuple[GroundTruthPose, ...],
    stdout_path: Path,
    stderr_path: Path,
    return_code: int = 0,
    elapsed_seconds: float = 0.0,
    command: tuple[str, ...] = (),
) -> ExternalVioResult:
    """Recompute VIO evidence without rerunning or re-exporting the dataset."""
    if not backend or return_code < 0 or elapsed_seconds < 0:
        raise ValueError("existing VIO artifact metadata is invalid")
    return _result_from_artifacts(
        backend=backend,
        trajectory_path=trajectory_path,
        truth=truth,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        return_code=return_code,
        elapsed_seconds=elapsed_seconds,
        command=command,
        detail="reanalyzed existing trajectory and backend logs",
    )


def _save_png(image: npt.NDArray[np.generic], path: Path) -> None:
    image_module = importlib.import_module("PIL.Image")
    image_module.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def export_euroc(batch: ReplayBatch, output: Path, *, stereo_tolerance_ms: float = 20.0) -> Path:
    left = batch.camera_streams.get("left") or batch.camera_streams.get("color") or ()
    right = batch.camera_streams.get("right") or ()
    if not left:
        raise ValueError("EuRoC export requires a primary camera stream")
    if not right:
        raise ValueError("stereo-inertial EuRoC export requires a right camera stream")
    cam0 = output / "mav0/cam0/data"
    cam1 = output / "mav0/cam1/data"
    imu_root = output / "mav0/imu0"
    cam0.mkdir(parents=True, exist_ok=True)
    cam1.mkdir(parents=True, exist_ok=True)
    imu_root.mkdir(parents=True, exist_ok=True)
    right_times = np.asarray([frame.timestamp.monotonic_ns for frame in right], dtype=np.int64)
    exported_timestamps: list[int] = []
    tolerance_ns = int(stereo_tolerance_ms * 1e6)
    for frame in left:
        insertion = int(np.searchsorted(right_times, frame.timestamp.monotonic_ns))
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(right)]
        nearest = min(
            candidates,
            key=lambda index: abs(int(right_times[index]) - frame.timestamp.monotonic_ns),
        )
        if abs(int(right_times[nearest]) - frame.timestamp.monotonic_ns) > tolerance_ns:
            continue
        filename = f"{frame.timestamp.monotonic_ns}.png"
        _save_png(frame.image, cam0 / filename)
        _save_png(right[nearest].image, cam1 / filename)
        exported_timestamps.append(frame.timestamp.monotonic_ns)
    if not exported_timestamps:
        raise ValueError("no stereo pairs satisfied the synchronization tolerance")
    times_path = output / "times.txt"
    times_path.write_text("".join(f"{value}\n" for value in exported_timestamps), encoding="utf-8")
    imu_lines = [
        "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
        "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
    ]
    for sample in batch.imu_samples:
        gyro = sample.angular_velocity_rps
        acceleration = sample.acceleration_mps2
        imu_lines.append(
            f"{sample.timestamp.monotonic_ns},{gyro[0]},{gyro[1]},{gyro[2]},"
            f"{acceleration[0]},{acceleration[1]},{acceleration[2]}\n"
        )
    (imu_root / "data.csv").write_text("".join(imu_lines), encoding="utf-8")
    return times_path


def prepare_orbslam3_s3e_settings(
    source: Path,
    output: Path,
    *,
    stereo_baseline_scale: float = 1.0,
    imu_fast_init: bool = False,
    orb_feature_profile: Literal["balanced", "high-recall"] = "balanced",
) -> Path:
    """Adapt the upstream S3E calibration to ORB-SLAM3's legacy settings reader."""
    if not np.isfinite(stereo_baseline_scale) or stereo_baseline_scale <= 0:
        raise ValueError("stereo baseline scale must be finite and positive")
    if orb_feature_profile not in {"balanced", "high-recall"}:
        raise ValueError("ORB feature profile must be balanced or high-recall")
    feature_parameters = {
        "balanced": (1600, 20, 7),
        "high-recall": (2400, 12, 5),
    }
    feature_count, initial_fast_threshold, minimum_fast_threshold = (
        feature_parameters[orb_feature_profile]
    )
    content = source.read_text(encoding="utf-8")
    required = (
        "Camera.type:",
        "Camera.fx:",
        "Camera.bf:",
        "IMU.Frequency:",
        "Tic:",
        "LEFT.K:",
        "RIGHT.K:",
    )
    missing = [field for field in required if field not in content]
    if missing:
        raise ValueError(f"S3E calibration is missing required fields: {missing}")
    baseline_match = re.search(
        r"(?m)^(Camera\.bf:\s*)([-+0-9.eE]+)(.*)$",
        content,
    )
    if baseline_match is None:
        raise ValueError("S3E calibration Camera.bf is not numeric")
    scaled_baseline = float(baseline_match.group(2)) * stereo_baseline_scale
    content = (
        content[: baseline_match.start()]
        + f"{baseline_match.group(1)}{scaled_baseline:.15g}{baseline_match.group(3)}"
        + content[baseline_match.end() :]
    )
    content = content.replace("Tic:", "Tbc:", 1)
    # ORB-SLAM3's legacy IMU parser reads the OpenCV matrix buffer as float
    # without converting a double matrix first.
    content = content.replace("dt: d", "dt: f")
    additions = """

# ARIADNE ORB-SLAM3 runtime parameters.
Camera.RGB: 1
ThDepth: 40.0
InsertKFsWhenLost: 0
IMU.fastInit: {imu_fast_init}
ORBextractor.nFeatures: {feature_count}
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: {initial_fast_threshold}
ORBextractor.minThFAST: {minimum_fast_threshold}
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        content.rstrip()
        + additions.format(
            imu_fast_init=int(imu_fast_init),
            feature_count=feature_count,
            initial_fast_threshold=initial_fast_threshold,
            minimum_fast_threshold=minimum_fast_threshold,
        ),
        encoding="utf-8",
    )
    return output


def _compressed_payload(message: Any) -> bytes:
    data = cast(Any, message.data)
    return data.tobytes() if hasattr(data, "tobytes") else bytes(data)


def swap_euroc_stereo_files(
    cam0: Path,
    cam1: Path,
    timestamps_ns: list[int],
) -> None:
    for timestamp_ns in timestamps_ns:
        left_path = cam0 / f"{timestamp_ns}.png"
        right_path = cam1 / f"{timestamp_ns}.png"
        temporary_path = cam0 / f".{timestamp_ns}.swap"
        left_path.replace(temporary_path)
        right_path.replace(left_path)
        temporary_path.replace(right_path)


def apply_euroc_stereo_row_correction(
    cam1: Path,
    timestamps_ns: list[int],
    *,
    x_slope: float = 0.0,
    y_slope: float = 0.0,
    intercept_px: float = 0.0,
) -> None:
    coefficients = np.asarray(
        [x_slope, y_slope, intercept_px],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("stereo row-correction coefficients must be finite")
    if (
        abs(x_slope) > 0.1
        or abs(y_slope) > 0.1
        or abs(intercept_px) > 100.0
    ):
        raise ValueError("stereo row correction exceeds bounded affine limits")
    if np.all(np.abs(coefficients) < 1e-9):
        return
    cv2 = importlib.import_module("cv2")
    transform = np.asarray(
        [[1.0, 0.0, 0.0], [x_slope, 1.0 + y_slope, intercept_px]],
        dtype=np.float32,
    )
    for timestamp_ns in timestamps_ns:
        path = cam1 / f"{timestamp_ns}.png"
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"right stereo image cannot be decoded at {timestamp_ns}")
        shifted = cv2.warpAffine(
            image,
            transform,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        success, payload = cv2.imencode(
            ".jpg",
            shifted,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        if not success:
            raise RuntimeError(f"right stereo image cannot be encoded at {timestamp_ns}")
        path.write_bytes(bytes(payload))


def read_s3e_imu_orientation_reference(
    bag: Path,
    agent_id: str,
    *,
    start_timestamp_ns: int | None = None,
    end_timestamp_ns: int | None = None,
) -> tuple[OrientationReference, ...]:
    """Stream the S3E IMU/AHRS quaternion signal for consistency diagnostics."""
    if not bag.is_file():
        raise FileNotFoundError(f"S3E bag does not exist: {bag}")
    normalized_agent = agent_id.capitalize()
    if normalized_agent not in {"Alpha", "Bob", "Carol"}:
        raise ValueError("S3E agent must be Alpha, Bob, or Carol")
    if (
        start_timestamp_ns is not None
        and end_timestamp_ns is not None
        and start_timestamp_ns > end_timestamp_ns
    ):
        raise ValueError("orientation-reference time window is invalid")
    topic = f"/{normalized_agent}/imu/data"
    highlevel = importlib.import_module("rosbags.highlevel")
    typesys = importlib.import_module("rosbags.typesys")
    typestore = typesys.get_typestore(typesys.Stores.ROS2_HUMBLE)
    reference: list[OrientationReference] = []
    with highlevel.AnyReader([bag], default_typestore=typestore) as reader:
        connections = [
            connection for connection in reader.connections if connection.topic == topic
        ]
        if len(connections) != 1:
            raise ValueError(f"S3E IMU orientation topic is unavailable: {topic}")
        connection = connections[0]
        for _, timestamp_ns, raw in reader.messages(connections=connections):
            if start_timestamp_ns is not None and timestamp_ns < start_timestamp_ns:
                continue
            if end_timestamp_ns is not None and timestamp_ns > end_timestamp_ns:
                break
            message = reader.deserialize(raw, connection.msgtype)
            orientation = message.orientation
            covariance = np.asarray(message.orientation_covariance, dtype=np.float64)
            try:
                reference.append(
                    OrientationReference(
                        timestamp_ns,
                        np.asarray(
                            [
                                orientation.x,
                                orientation.y,
                                orientation.z,
                                orientation.w,
                            ],
                            dtype=np.float64,
                        ),
                        bool(np.any(covariance)),
                    )
                )
            except ValueError:
                continue
    if any(
        reference[index].timestamp_ns <= reference[index - 1].timestamp_ns
        for index in range(1, len(reference))
    ):
        raise ValueError("S3E IMU orientation timestamps must be strictly increasing")
    return tuple(reference)


def export_s3e_euroc_window(
    bag: Path,
    agent_id: str,
    output: Path,
    *,
    start_frame: int = 0,
    max_frames: int = 500,
    stereo_tolerance_ms: float = 20.0,
    swap_stereo: bool = False,
    right_image_vertical_shift_px: float = 0.0,
) -> EurocExportResult:
    """Stream one S3E Wingman's compressed stereo/IMU window to EuRoC layout."""
    if start_frame < 0 or max_frames <= 0 or stereo_tolerance_ms <= 0:
        raise ValueError("S3E EuRoC window controls are invalid")
    if not bag.is_file():
        raise FileNotFoundError(f"S3E bag does not exist: {bag}")
    normalized_agent = agent_id.capitalize()
    if normalized_agent not in {"Alpha", "Bob", "Carol"}:
        raise ValueError("S3E agent must be Alpha, Bob, or Carol")

    cam0 = output / "mav0/cam0/data"
    cam1 = output / "mav0/cam1/data"
    imu_root = output / "mav0/imu0"
    cam0.mkdir(parents=True, exist_ok=True)
    cam1.mkdir(parents=True, exist_ok=True)
    imu_root.mkdir(parents=True, exist_ok=True)
    topics = {
        "left": f"/{normalized_agent}/left_camera/compressed",
        "right": f"/{normalized_agent}/right_camera/compressed",
        "imu": f"/{normalized_agent}/imu/data",
    }
    tolerance_ns = int(stereo_tolerance_ms * 1e6)
    imu_margin_ns = 100_000_000
    left_seen = 0
    left_timestamps: list[int] = []
    right_messages: list[tuple[int, bytes]] = []
    recent_right: deque[tuple[int, bytes]] = deque()
    recent_imu: deque[tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64]]] = (
        deque()
    )
    imu_messages: list[
        tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ] = []
    compressed_bytes = 0
    first_left_ns: int | None = None
    last_left_ns: int | None = None

    highlevel = importlib.import_module("rosbags.highlevel")
    typesys = importlib.import_module("rosbags.typesys")
    typestore = typesys.get_typestore(typesys.Stores.ROS2_HUMBLE)
    with highlevel.AnyReader([bag], default_typestore=typestore) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in set(topics.values())
        ]
        for connection, timestamp_ns, raw in reader.messages(connections=connections):
            if last_left_ns is not None and len(left_timestamps) == max_frames:
                if timestamp_ns > last_left_ns + tolerance_ns:
                    break
            topic = connection.topic
            if topic == topics["left"]:
                if left_seen < start_frame:
                    left_seen += 1
                    continue
                if len(left_timestamps) >= max_frames:
                    continue
                message = reader.deserialize(raw, connection.msgtype)
                payload = _compressed_payload(message)
                if first_left_ns is None:
                    first_left_ns = timestamp_ns
                    right_messages.extend(recent_right)
                    imu_messages.extend(recent_imu)
                left_seen += 1
                left_timestamps.append(timestamp_ns)
                last_left_ns = timestamp_ns
                (cam0 / f"{timestamp_ns}.png").write_bytes(payload)
                compressed_bytes += len(payload)
            elif topic == topics["right"]:
                message = reader.deserialize(raw, connection.msgtype)
                right_record = (timestamp_ns, _compressed_payload(message))
                if first_left_ns is None:
                    recent_right.append(right_record)
                    while (
                        recent_right
                        and timestamp_ns - recent_right[0][0] > tolerance_ns
                    ):
                        recent_right.popleft()
                elif last_left_ns is None or timestamp_ns <= last_left_ns + tolerance_ns:
                    right_messages.append(right_record)
            elif topic == topics["imu"]:
                message = reader.deserialize(raw, connection.msgtype)
                imu_record = (
                    timestamp_ns,
                    np.asarray(
                        [
                            message.angular_velocity.x,
                            message.angular_velocity.y,
                            message.angular_velocity.z,
                        ],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        [
                            message.linear_acceleration.x,
                            message.linear_acceleration.y,
                            message.linear_acceleration.z,
                        ],
                        dtype=np.float64,
                    ),
                )
                if first_left_ns is None:
                    recent_imu.append(imu_record)
                    while recent_imu and timestamp_ns - recent_imu[0][0] > imu_margin_ns:
                        recent_imu.popleft()
                else:
                    imu_messages.append(imu_record)

    if first_left_ns is None or last_left_ns is None or not left_timestamps:
        raise ValueError(f"no S3E frames found for {normalized_agent}")
    right_times = np.asarray([timestamp for timestamp, _ in right_messages], dtype=np.int64)
    matched_timestamps: list[int] = []
    used_right: set[int] = set()
    for left_timestamp in left_timestamps:
        insertion = int(np.searchsorted(right_times, left_timestamp))
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < len(right_messages) and index not in used_right
        ]
        if not candidates:
            continue
        nearest = min(
            candidates,
            key=lambda index: abs(int(right_times[index]) - left_timestamp),
        )
        if abs(int(right_times[nearest]) - left_timestamp) > tolerance_ns:
            continue
        payload = right_messages[nearest][1]
        (cam1 / f"{left_timestamp}.png").write_bytes(payload)
        compressed_bytes += len(payload)
        used_right.add(nearest)
        matched_timestamps.append(left_timestamp)
    matched = set(matched_timestamps)
    for timestamp_ns in left_timestamps:
        if timestamp_ns not in matched:
            (cam0 / f"{timestamp_ns}.png").unlink(missing_ok=True)
    if not matched_timestamps:
        raise ValueError("no S3E stereo pairs satisfied the synchronization tolerance")
    if swap_stereo:
        swap_euroc_stereo_files(cam0, cam1, matched_timestamps)
    apply_euroc_stereo_row_correction(
        cam1,
        matched_timestamps,
        intercept_px=right_image_vertical_shift_px,
    )

    times_path = output / "times.txt"
    times_path.write_text(
        "".join(f"{timestamp_ns}\n" for timestamp_ns in matched_timestamps),
        encoding="utf-8",
    )
    imu_lines = [
        "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
        "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],"
        "a_RS_S_z [m s^-2]\n"
    ]
    selected_imu = [
        record
        for record in imu_messages
        if first_left_ns - imu_margin_ns <= record[0] <= last_left_ns
    ]
    for timestamp_ns, gyro, acceleration in selected_imu:
        imu_lines.append(
            f"{timestamp_ns},{gyro[0]},{gyro[1]},{gyro[2]},"
            f"{acceleration[0]},{acceleration[1]},{acceleration[2]}\n"
        )
    (imu_root / "data.csv").write_text("".join(imu_lines), encoding="utf-8")
    return EurocExportResult(
        times_path,
        len(matched_timestamps),
        len(selected_imu),
        matched_timestamps[0],
        matched_timestamps[-1],
        compressed_bytes,
    )


class _ExternalProcessAdapter:
    backend_name: str

    def _run(
        self,
        command: tuple[str, ...],
        output_dir: Path,
        trajectory_path: Path,
        truth: tuple[GroundTruthPose, ...],
        timeout_seconds: float,
    ) -> ExternalVioResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"
        start = perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=output_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return_code: int | None = completed.returncode
            stdout = completed.stdout
            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            detail = ""
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            return_code = None
            stdout = ""
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(str(error), encoding="utf-8")
            detail = str(error)
        elapsed = perf_counter() - start
        return _result_from_artifacts(
            backend=self.backend_name,
            trajectory_path=trajectory_path,
            truth=truth,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            return_code=return_code,
            elapsed_seconds=elapsed,
            command=command,
            detail=detail,
        )


class OpenVinsAdapter(_ExternalProcessAdapter):
    backend_name = "openvins"

    def run(
        self,
        *,
        bag: Path,
        config: Path,
        truth: tuple[GroundTruthPose, ...],
        output_dir: Path,
        launcher: tuple[str, ...] = ("roslaunch",),
        launch_target: tuple[str, ...] = ("ov_msckf", "serial.launch"),
        timeout_seconds: float = 3600.0,
    ) -> ExternalVioResult:
        trajectory = output_dir / "trajectory.txt"
        timing = output_dir / "timing.txt"
        command = (
            *launcher,
            *launch_target,
            f"config_path:={config.resolve()}",
            f"bag:={bag.resolve()}",
            "dosave:=true",
            f"path_est:={trajectory.resolve()}",
            "dotime:=true",
            f"path_time:={timing.resolve()}",
            "dolivetraj:=false",
        )
        return self._run(command, output_dir, trajectory, truth, timeout_seconds)


class OrbSlam3Adapter(_ExternalProcessAdapter):
    backend_name = "orbslam3"

    def run(
        self,
        *,
        batch: ReplayBatch,
        executable: Path,
        vocabulary: Path,
        settings: Path,
        output_dir: Path,
        mode: Literal["stereo", "stereo-inertial"] = "stereo-inertial",
        deterministic_runtime: bool = False,
        sync_local_mapping: bool = False,
        timeout_seconds: float = 3600.0,
    ) -> ExternalVioResult:
        sequence = output_dir / "euroc"
        times = export_euroc(batch, sequence)
        return self.run_euroc(
            sequence=sequence,
            times=times,
            truth=batch.ground_truth,
            executable=executable,
            vocabulary=vocabulary,
            settings=settings,
            output_dir=output_dir,
            mode=mode,
            deterministic_runtime=deterministic_runtime,
            sync_local_mapping=sync_local_mapping,
            timeout_seconds=timeout_seconds,
        )

    def run_euroc(
        self,
        *,
        sequence: Path,
        times: Path,
        truth: tuple[GroundTruthPose, ...],
        executable: Path,
        vocabulary: Path,
        settings: Path,
        output_dir: Path,
        mode: Literal["stereo", "stereo-inertial"] = "stereo-inertial",
        deterministic_runtime: bool = False,
        sync_local_mapping: bool = False,
        timeout_seconds: float = 3600.0,
    ) -> ExternalVioResult:
        """Run a previously streamed EuRoC-layout sequence."""
        run_name = "ariadne"
        trajectory = output_dir / f"f_{run_name}.txt"
        command = (
            str(executable.resolve()),
            *(("--deterministic-runtime",) if deterministic_runtime else ()),
            *(("--sync-local-mapping",) if sync_local_mapping else ()),
            "--mode",
            mode,
            str(vocabulary.resolve()),
            str(settings.resolve()),
            str(sequence.resolve()),
            str(times.resolve()),
            run_name,
        )
        if shutil.which(command[0]) is None and not executable.is_file():
            output_dir.mkdir(parents=True, exist_ok=True)
        return self._run(command, output_dir, trajectory, truth, timeout_seconds)
