"""Optimistic trajectory diagnostics that never replace scored VIO metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ariadne.backends.external_vio import (
    OrientationReference,
    TrajectoryPose,
    _matched_orientation_samples,
    _matched_samples,
    _quaternion_rotation,
    _rigid_align,
    _rigid_alignment,
)
from ariadne.replay import GroundTruthPose

LOCAL_ALIGNMENT_INTERVALS_S = (0.5, 1.0, 2.0, 5.0, 10.0)
CAUSAL_ALIGNMENT_CADENCES_S = (0.1, 0.2, 0.5, 1.0, 2.0)
CAUSAL_REFERENCE_CADENCE_S = 0.1
CAUSAL_REFERENCE_CORRECTION_THRESHOLD_M = 0.17
CAUSAL_LOW_INGRESS_CADENCE_S = 0.2
CAUSAL_LOW_INGRESS_CORRECTION_THRESHOLD_M = 0.15
FIXED_LAG_SCALE_MINIMUM = 0.5
FIXED_LAG_SCALE_MAXIMUM = 2.0
FIXED_LAG_ADAPTIVE_LOG_SCALE_THRESHOLD = 0.1
FIXED_LAG_ADAPTIVE_ROTATION_THRESHOLD_RAD = 0.1
FIXED_LAG_ADAPTIVE_MAX_INTERVAL_SECONDS = 2.0
CAUSAL_SEGMENT_HOLD_HISTORY = 10
CAUSAL_SEGMENT_HOLD_DECAY = 0.75
CAUSAL_SEGMENT_HOLD_HORIZONS_S = (0.1, 0.2, 0.5, 1.0)
CAUSAL_CORRECTION_THRESHOLDS_M = (
    0.0,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.125,
    0.15,
    0.16,
    0.17,
    0.18,
    0.19,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
)


@dataclass(frozen=True)
class _CausalAlignmentProfile:
    ate_m: float
    pose_coverage_fraction: float
    anchor_count: int
    fit_update_count: int
    anchor_messages_per_minute: float
    fit_updates_per_minute: float
    correction_count: int
    correction_messages_per_minute: float
    correction_interval_min_s: float
    correction_interval_p95_s: float
    correction_interval_max_s: float
    correction_burst_per_second_max: int
    first_fit_delay_s: float
    correction_jump_p95_m: float
    correction_jump_max_m: float
    scale_p05: float
    scale_p95: float
    scale_update_p95_log: float


@dataclass(frozen=True)
class _PositionAnchor:
    timestamp_ns: int
    application_index: int
    estimated_position_m: npt.NDArray[np.float64]
    truth_position_m: npt.NDArray[np.float64]


@dataclass(frozen=True)
class _FixedLagAlignmentProfile:
    ate_m: float
    pose_coverage_fraction: float
    segment_count: int
    finalization_updates_per_minute: float
    latency_mean_s: float
    latency_p95_s: float
    latency_max_s: float
    scale_min: float
    scale_p05: float
    scale_p95: float
    scale_max: float
    scale_plausible_fraction: float


@dataclass(frozen=True)
class _CausalSegmentHoldProfile:
    ate_m: float
    pose_coverage_fraction: float
    prediction_updates_per_minute: float
    scale_p05: float
    scale_p95: float
    scale_plausible_fraction: float
    horizon_ate_m: tuple[float, ...]


def _rigid_transform(
    estimated: npt.NDArray[np.float64],
    truth: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    _, rotation = _rigid_alignment(estimated, truth)
    translation = np.mean(truth, axis=0) - rotation @ np.mean(estimated, axis=0)
    return rotation, np.asarray(translation, dtype=np.float64)


def _similarity_transform(
    estimated: npt.NDArray[np.float64],
    truth: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    rotation, _ = _rigid_transform(estimated, truth)
    rotated = (rotation @ estimated.T).T
    rotated_centered = rotated - np.mean(rotated, axis=0)
    truth_centered = truth - np.mean(truth, axis=0)
    denominator = float(np.sum(rotated_centered**2))
    scale = (
        float(np.sum(rotated_centered * truth_centered) / denominator)
        if denominator > np.finfo(np.float64).eps
        else 1.0
    )
    translation = np.mean(truth, axis=0) - scale * rotation @ np.mean(
        estimated,
        axis=0,
    )
    return rotation, np.asarray(translation, dtype=np.float64), scale


def _rotation_between_vectors(
    source: npt.NDArray[np.float64],
    destination: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], float]:
    source_norm = float(np.linalg.norm(source))
    destination_norm = float(np.linalg.norm(destination))
    if source_norm <= 1e-9 or destination_norm <= 1e-9:
        return np.eye(3, dtype=np.float64), 1.0
    source_unit = source / source_norm
    destination_unit = destination / destination_norm
    cross = np.cross(source_unit, destination_unit)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source_unit, destination_unit), -1.0, 1.0))
    if sine <= 1e-9:
        if cosine > 0:
            return np.eye(3, dtype=np.float64), destination_norm / source_norm
        axis = np.cross(source_unit, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(axis)) <= 1e-6:
            axis = np.cross(source_unit, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float64,
        )
        return (
            np.eye(3, dtype=np.float64) + 2.0 * skew @ skew,
            destination_norm / source_norm,
        )
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    rotation = (
        np.eye(3, dtype=np.float64)
        + skew
        + skew @ skew * ((1.0 - cosine) / (sine * sine))
    )
    return rotation, destination_norm / source_norm


def _rotation_angle(rotation: npt.NDArray[np.float64]) -> float:
    return float(
        np.arccos(
            np.clip(
                (float(np.trace(rotation)) - 1.0) / 2.0,
                -1.0,
                1.0,
            )
        )
    )


def _weighted_rotation_average(
    rotations: list[npt.NDArray[np.float64]],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    weighted = np.sum(
        np.stack(rotations) * weights[:, np.newaxis, np.newaxis],
        axis=0,
    )
    left, _, right_transpose = np.linalg.svd(weighted)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transpose
    return np.asarray(rotation, dtype=np.float64)


def _bounded_least_squares(
    design: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    maximum_norm: float,
) -> tuple[npt.NDArray[np.float64], float, bool]:
    unconstrained = np.asarray(
        np.linalg.lstsq(design, target, rcond=None)[0],
        dtype=np.float64,
    )
    unconstrained_norm = float(np.linalg.norm(unconstrained))
    if unconstrained_norm <= maximum_norm:
        return unconstrained, unconstrained_norm, False
    normal = design.T @ design
    right_hand_side = design.T @ target
    identity = np.eye(normal.shape[0], dtype=np.float64)
    lower = 0.0
    upper = 1.0
    while (
        float(
            np.linalg.norm(
                np.linalg.solve(normal + upper * identity, right_hand_side)
            )
        )
        > maximum_norm
    ):
        upper *= 2.0
    bounded = unconstrained
    for _ in range(64):
        regularization = (lower + upper) / 2.0
        bounded = np.asarray(
            np.linalg.solve(
                normal + regularization * identity,
                right_hand_side,
            ),
            dtype=np.float64,
        )
        if float(np.linalg.norm(bounded)) > maximum_norm:
            lower = regularization
        else:
            upper = regularization
    return bounded, unconstrained_norm, True


def _fit_rtk_lever_arm_model(
    estimated: npt.NDArray[np.float64],
    truth: npt.NDArray[np.float64],
    reference_rotations: npt.NDArray[np.float64],
    maximum_lever_arm_m: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float,
    bool,
    int,
]:
    lever_arm = np.zeros(3, dtype=np.float64)
    unconstrained_norm = 0.0
    bound_active = False
    iterations = 0
    for _ in range(50):
        iterations += 1
        body_truth = truth - np.einsum(
            "nij,j->ni",
            reference_rotations,
            lever_arm,
        )
        rotation, translation = _rigid_transform(estimated, body_truth)
        predicted_body = (rotation @ estimated.T).T + translation
        residual = truth - predicted_body
        centered_rotations = reference_rotations - np.mean(
            reference_rotations,
            axis=0,
        )
        centered_residual = residual - np.mean(residual, axis=0)
        next_lever_arm, unconstrained_norm, bound_active = _bounded_least_squares(
            centered_rotations.reshape(-1, 3),
            centered_residual.reshape(-1),
            maximum_lever_arm_m,
        )
        if float(np.linalg.norm(next_lever_arm - lever_arm)) <= 1e-9:
            lever_arm = next_lever_arm
            break
        lever_arm = next_lever_arm
    body_truth = truth - np.einsum(
        "nij,j->ni",
        reference_rotations,
        lever_arm,
    )
    rotation, translation = _rigid_transform(estimated, body_truth)
    return (
        rotation,
        translation,
        lever_arm,
        unconstrained_norm,
        bound_active,
        iterations,
    )


def evaluate_rtk_lever_arm_sensitivity(
    estimates: tuple[TrajectoryPose, ...],
    position_truth: tuple[GroundTruthPose, ...],
    orientation_reference: tuple[OrientationReference, ...],
    *,
    position_tolerance_seconds: float = 0.6,
    orientation_tolerance_seconds: float = 0.03,
    maximum_lever_arm_m: float = 1.0,
    orientation_independent_of_vio: bool = False,
) -> dict[str, int | float | str]:
    """Bound the ATE that a fitted rotating RTK antenna offset could explain."""
    controls = (
        position_tolerance_seconds,
        orientation_tolerance_seconds,
        maximum_lever_arm_m,
    )
    if not all(np.isfinite(value) and value > 0 for value in controls):
        raise ValueError("lever-arm sensitivity controls must be finite and positive")
    position_times, estimated, truth = _matched_samples(
        estimates,
        position_truth,
        position_tolerance_seconds,
    )
    orientation_times, _, reference_quaternions = _matched_orientation_samples(
        estimates,
        orientation_reference,
        orientation_tolerance_seconds,
    )
    common_times, position_indices, orientation_indices = np.intersect1d(
        position_times,
        orientation_times,
        assume_unique=False,
        return_indices=True,
    )
    metrics: dict[str, int | float | str] = {
        "lever_arm_sensitivity_source": "s3e_rtk_plus_imu_ahrs_proxy",
        "lever_arm_sensitivity_is_calibration": 0,
        "lever_arm_sensitivity_is_ground_truth": 0,
        "lever_arm_sensitivity_uses_fitted_evaluation_data": 1,
        "lever_arm_sensitivity_orientation_independent_of_vio": int(
            orientation_independent_of_vio
        ),
        "lever_arm_sensitivity_orientation_covariance_available": int(
            bool(orientation_reference)
            and all(sample.covariance_available for sample in orientation_reference)
        ),
        "lever_arm_sensitivity_matched_pose_count": len(common_times),
        "lever_arm_sensitivity_maximum_norm_m": maximum_lever_arm_m,
    }
    if len(common_times) < 10:
        return metrics
    estimated = estimated[position_indices]
    truth = truth[position_indices]
    reference_rotations = np.stack(
        [_quaternion_rotation(value) for value in reference_quaternions[orientation_indices]]
    )
    baseline_aligned = _rigid_align(estimated, truth)
    baseline_errors = np.linalg.norm(baseline_aligned - truth, axis=1)
    (
        rotation,
        translation,
        lever_arm,
        unconstrained_norm,
        bound_active,
        iterations,
    ) = _fit_rtk_lever_arm_model(
        estimated,
        truth,
        reference_rotations,
        maximum_lever_arm_m,
    )
    predicted_antenna = (
        (rotation @ estimated.T).T
        + translation
        + np.einsum("nij,j->ni", reference_rotations, lever_arm)
    )
    adjusted_errors = np.linalg.norm(predicted_antenna - truth, axis=1)
    split = len(common_times) // 2
    baseline_rotation, baseline_translation = _rigid_transform(
        estimated[:split],
        truth[:split],
    )
    (
        holdout_rotation,
        holdout_translation,
        holdout_lever_arm,
        _,
        holdout_bound_active,
        _,
    ) = _fit_rtk_lever_arm_model(
        estimated[:split],
        truth[:split],
        reference_rotations[:split],
        maximum_lever_arm_m,
    )
    baseline_holdout = (
        baseline_rotation @ estimated[split:].T
    ).T + baseline_translation
    adjusted_holdout = (
        (holdout_rotation @ estimated[split:].T).T
        + holdout_translation
        + np.einsum(
            "nij,j->ni",
            reference_rotations[split:],
            holdout_lever_arm,
        )
    )
    baseline_holdout_errors = np.linalg.norm(
        baseline_holdout - truth[split:],
        axis=1,
    )
    adjusted_holdout_errors = np.linalg.norm(
        adjusted_holdout - truth[split:],
        axis=1,
    )
    baseline_ate_m = float(np.sqrt(np.mean(baseline_errors**2)))
    adjusted_ate_m = float(np.sqrt(np.mean(adjusted_errors**2)))
    baseline_holdout_ate_m = float(
        np.sqrt(np.mean(baseline_holdout_errors**2))
    )
    adjusted_holdout_ate_m = float(
        np.sqrt(np.mean(adjusted_holdout_errors**2))
    )
    metrics.update(
        {
            "lever_arm_sensitivity_fitted_x_m": float(lever_arm[0]),
            "lever_arm_sensitivity_fitted_y_m": float(lever_arm[1]),
            "lever_arm_sensitivity_fitted_z_m": float(lever_arm[2]),
            "lever_arm_sensitivity_fitted_norm_m": float(
                np.linalg.norm(lever_arm)
            ),
            "lever_arm_sensitivity_unconstrained_norm_m": unconstrained_norm,
            "lever_arm_sensitivity_bound_active": int(bound_active),
            "lever_arm_sensitivity_fit_iterations": iterations,
            "lever_arm_sensitivity_baseline_ate_m": baseline_ate_m,
            "lever_arm_sensitivity_adjusted_ate_m": adjusted_ate_m,
            "lever_arm_sensitivity_adjusted_p95_m": float(
                np.percentile(adjusted_errors, 95)
            ),
            "lever_arm_sensitivity_full_fit_improvement_percent": (
                100.0 * (baseline_ate_m - adjusted_ate_m) / baseline_ate_m
                if baseline_ate_m > np.finfo(np.float64).eps
                else 0.0
            ),
            "lever_arm_sensitivity_holdout_pose_count": len(common_times) - split,
            "lever_arm_sensitivity_holdout_baseline_ate_m": (
                baseline_holdout_ate_m
            ),
            "lever_arm_sensitivity_holdout_adjusted_ate_m": (
                adjusted_holdout_ate_m
            ),
            "lever_arm_sensitivity_holdout_improvement_percent": (
                100.0
                * (baseline_holdout_ate_m - adjusted_holdout_ate_m)
                / baseline_holdout_ate_m
                if baseline_holdout_ate_m > np.finfo(np.float64).eps
                else 0.0
            ),
            "lever_arm_sensitivity_holdout_bound_active": int(
                holdout_bound_active
            ),
        }
    )
    return metrics


def _interval_metric_token(interval_s: float) -> str:
    return f"{interval_s:g}".replace(".", "_")


def _piecewise_alignment_profile(
    timestamps_ns: npt.NDArray[np.int64],
    estimated: npt.NDArray[np.float64],
    truth: npt.NDArray[np.float64],
    interval_s: float,
) -> tuple[float, float, float, int, npt.NDArray[np.float64]]:
    elapsed_ns = timestamps_ns - timestamps_ns[0]
    segment_ids = np.floor(elapsed_ns / (interval_s * 1e9)).astype(np.int64)
    rigid_errors: list[npt.NDArray[np.float64]] = []
    similarity_errors: list[npt.NDArray[np.float64]] = []
    scales: list[float] = []
    evaluated_pose_count = 0
    segment_count = 0
    for segment_id in np.unique(segment_ids):
        indices = np.flatnonzero(segment_ids == segment_id)
        if len(indices) < 3:
            continue
        segment_count += 1
        evaluated_pose_count += len(indices)
        segment_truth = truth[indices]
        rotation, translation = _rigid_transform(
            estimated[indices],
            segment_truth,
        )
        aligned = (rotation @ estimated[indices].T).T + translation
        rigid_errors.append(np.linalg.norm(aligned - segment_truth, axis=1))
        similarity_rotation, similarity_translation, scale = _similarity_transform(
            estimated[indices],
            segment_truth,
        )
        scales.append(scale)
        similarity_aligned = (
            scale * (similarity_rotation @ estimated[indices].T).T
            + similarity_translation
        )
        similarity_errors.append(
            np.linalg.norm(similarity_aligned - segment_truth, axis=1)
        )
    if not rigid_errors:
        return float("nan"), float("nan"), 0.0, 0, np.empty(0)
    rigid_values = np.concatenate(rigid_errors)
    similarity_values = np.concatenate(similarity_errors)
    return (
        float(np.sqrt(np.mean(rigid_values**2))),
        float(np.sqrt(np.mean(similarity_values**2))),
        evaluated_pose_count / len(timestamps_ns),
        segment_count,
        np.asarray(scales, dtype=np.float64),
    )


def _causal_alignment_profile(
    timestamps_ns: npt.NDArray[np.int64],
    estimated: npt.NDArray[np.float64],
    truth: npt.NDArray[np.float64],
    cadence_s: float,
    *,
    similarity: bool,
    history_s: float | None = None,
    correction_threshold_m: float = 0.0,
    explicit_anchors: tuple[_PositionAnchor, ...] | None = None,
) -> _CausalAlignmentProfile:
    cadence_ns = int(cadence_s * 1e9)
    selected_history_s = (
        max(0.5, 2.0 * cadence_s) if history_s is None else history_s
    )
    if (
        not np.isfinite(selected_history_s)
        or selected_history_s <= 0
        or selected_history_s < 2.0 * cadence_s
        or not np.isfinite(correction_threshold_m)
        or correction_threshold_m < 0
    ):
        raise ValueError(
            "causal alignment history and correction threshold are invalid"
        )
    history_ns = int(selected_history_s * 1e9)
    if explicit_anchors is None:
        anchor_indices = [0]
        last_anchor_ns = int(timestamps_ns[0])
        for index, timestamp_ns in enumerate(timestamps_ns[1:], start=1):
            if int(timestamp_ns) - last_anchor_ns >= cadence_ns:
                anchor_indices.append(index)
                last_anchor_ns = int(timestamp_ns)
        anchors = tuple(
            _PositionAnchor(
                int(timestamps_ns[index]),
                index,
                estimated[index],
                truth[index],
            )
            for index in anchor_indices
        )
    else:
        anchors = explicit_anchors
        anchor_indices = [anchor.application_index for anchor in anchors]
        anchor_timestamps_ns = [anchor.timestamp_ns for anchor in anchors]
        if (
            not anchors
            or anchor_indices != sorted(set(anchor_indices))
            or anchor_timestamps_ns != sorted(anchor_timestamps_ns)
            or anchor_indices[0] < 0
            or anchor_indices[-1] >= len(timestamps_ns)
            or any(
                anchor.estimated_position_m.shape != (3,)
                or anchor.truth_position_m.shape != (3,)
                or not np.all(np.isfinite(anchor.estimated_position_m))
                or not np.all(np.isfinite(anchor.truth_position_m))
                for anchor in anchors
            )
        ):
            raise ValueError("explicit causal anchors are invalid")
    anchor_cursor = 0
    rotation: npt.NDArray[np.float64] | None = None
    translation: npt.NDArray[np.float64] | None = None
    scale = 1.0
    errors_m: list[float] = []
    correction_jumps_m: list[float] = []
    scale_factors: list[float] = []
    scale_updates_log: list[float] = []
    correction_timestamps_ns: list[int] = []
    fit_update_count = 0
    first_fit_timestamp_ns: int | None = None
    for index, (timestamp_ns, estimated_position, truth_position) in enumerate(
        zip(timestamps_ns, estimated, truth, strict=True)
    ):
        if (
            anchor_cursor < len(anchors)
            and index == anchors[anchor_cursor].application_index
        ):
            available_anchors = anchors[: anchor_cursor + 1]
            window_start_ns = anchors[anchor_cursor].timestamp_ns - history_ns
            fit_anchors = [
                anchor
                for anchor in available_anchors
                if anchor.timestamp_ns >= window_start_ns
            ]
            if len(fit_anchors) < 3 and len(available_anchors) >= 3:
                fit_anchors = list(available_anchors[-3:])
            if len(fit_anchors) >= 3:
                fit_estimated = np.stack(
                    [anchor.estimated_position_m for anchor in fit_anchors]
                )
                fit_truth = np.stack(
                    [anchor.truth_position_m for anchor in fit_anchors]
                )
                if similarity:
                    candidate_rotation, candidate_translation, candidate_scale = (
                        _similarity_transform(
                            fit_estimated,
                            fit_truth,
                        )
                    )
                else:
                    candidate_rotation, candidate_translation = _rigid_transform(
                        fit_estimated,
                        fit_truth,
                    )
                    candidate_scale = 1.0
                transmit = rotation is None or translation is None
                correction_jump_m = 0.0
                if rotation is not None and translation is not None:
                    previous_prediction = (
                        scale * (rotation @ estimated_position) + translation
                    )
                    next_prediction = (
                        candidate_scale
                        * (candidate_rotation @ estimated_position)
                        + candidate_translation
                    )
                    correction_jump_m = float(
                        np.linalg.norm(next_prediction - previous_prediction)
                    )
                    transmit = correction_jump_m >= correction_threshold_m
                if transmit:
                    if rotation is not None and translation is not None:
                        correction_jumps_m.append(correction_jump_m)
                        if similarity and scale > 0 and candidate_scale > 0:
                            scale_updates_log.append(
                                float(abs(np.log(candidate_scale / scale)))
                            )
                    rotation = candidate_rotation
                    translation = candidate_translation
                    scale = candidate_scale
                    correction_timestamps_ns.append(int(timestamp_ns))
                scale_factors.append(candidate_scale)
                fit_update_count += 1
                if first_fit_timestamp_ns is None:
                    first_fit_timestamp_ns = int(timestamp_ns)
            anchor_cursor += 1
        if rotation is None or translation is None:
            continue
        corrected_position = scale * (rotation @ estimated_position) + translation
        errors_m.append(float(np.linalg.norm(corrected_position - truth_position)))
    duration_s = float(timestamps_ns[-1] - timestamps_ns[0]) / 1e9
    first_fit_delay_s = (
        float(first_fit_timestamp_ns - int(timestamps_ns[0])) / 1e9
        if first_fit_timestamp_ns is not None
        else duration_s
    )
    correction_intervals_s = np.diff(
        np.asarray(correction_timestamps_ns, dtype=np.int64)
    ).astype(np.float64) / 1e9
    correction_burst_per_second_max = 0
    if correction_timestamps_ns:
        second_bins = (
            np.asarray(correction_timestamps_ns, dtype=np.int64)
            - int(timestamps_ns[0])
        ) // 1_000_000_000
        correction_burst_per_second_max = int(
            np.max(np.unique(second_bins, return_counts=True)[1])
        )
    return _CausalAlignmentProfile(
        ate_m=(
            float(np.sqrt(np.mean(np.square(errors_m))))
            if errors_m
            else float("nan")
        ),
        pose_coverage_fraction=len(errors_m) / len(timestamps_ns),
        anchor_count=len(anchors),
        fit_update_count=fit_update_count,
        anchor_messages_per_minute=(
            len(anchors) * 60.0 / duration_s if duration_s > 0 else 0.0
        ),
        fit_updates_per_minute=(
            fit_update_count * 60.0 / duration_s if duration_s > 0 else 0.0
        ),
        correction_count=len(correction_timestamps_ns),
        correction_messages_per_minute=(
            len(correction_timestamps_ns) * 60.0 / duration_s
            if duration_s > 0
            else 0.0
        ),
        correction_interval_min_s=(
            float(np.min(correction_intervals_s))
            if len(correction_intervals_s)
            else 0.0
        ),
        correction_interval_p95_s=(
            float(np.percentile(correction_intervals_s, 95))
            if len(correction_intervals_s)
            else 0.0
        ),
        correction_interval_max_s=(
            float(np.max(correction_intervals_s))
            if len(correction_intervals_s)
            else 0.0
        ),
        correction_burst_per_second_max=correction_burst_per_second_max,
        first_fit_delay_s=first_fit_delay_s,
        correction_jump_p95_m=(
            float(np.percentile(correction_jumps_m, 95))
            if correction_jumps_m
            else 0.0
        ),
        correction_jump_max_m=(
            float(np.max(correction_jumps_m)) if correction_jumps_m else 0.0
        ),
        scale_p05=(
            float(np.percentile(scale_factors, 5)) if scale_factors else 1.0
        ),
        scale_p95=(
            float(np.percentile(scale_factors, 95)) if scale_factors else 1.0
        ),
        scale_update_p95_log=(
            float(np.percentile(scale_updates_log, 95))
            if scale_updates_log
            else 0.0
        ),
    )


def _causal_load_metrics(
    model: str,
    selected: tuple[float, float, _CausalAlignmentProfile] | None,
) -> dict[str, int | float | str]:
    prefix = f"causal_{model}_load_selected"
    if selected is None:
        return {
            f"{prefix}_anchor_cadence_seconds": 0.0,
            f"{prefix}_correction_threshold_m": 0.0,
            f"{prefix}_ate_m": 0.0,
            f"{prefix}_correction_count": 0,
            f"{prefix}_correction_messages_per_minute": 0.0,
            f"{prefix}_anchor_messages_per_minute": 0.0,
            f"{prefix}_fit_updates_per_minute": 0.0,
            f"{prefix}_correction_interval_min_seconds": 0.0,
            f"{prefix}_correction_interval_p95_seconds": 0.0,
            f"{prefix}_correction_interval_max_seconds": 0.0,
            f"{prefix}_correction_burst_per_second_max": 0,
            f"{prefix}_correction_jump_p95_m": 0.0,
            f"{prefix}_correction_jump_max_m": 0.0,
        }
    cadence_s, correction_threshold_m, profile = selected
    return {
        f"{prefix}_anchor_cadence_seconds": cadence_s,
        f"{prefix}_correction_threshold_m": correction_threshold_m,
        f"{prefix}_ate_m": profile.ate_m,
        f"{prefix}_correction_count": profile.correction_count,
        f"{prefix}_correction_messages_per_minute": (
            profile.correction_messages_per_minute
        ),
        f"{prefix}_anchor_messages_per_minute": profile.anchor_messages_per_minute,
        f"{prefix}_fit_updates_per_minute": profile.fit_updates_per_minute,
        f"{prefix}_correction_interval_min_seconds": (
            profile.correction_interval_min_s
        ),
        f"{prefix}_correction_interval_p95_seconds": (
            profile.correction_interval_p95_s
        ),
        f"{prefix}_correction_interval_max_seconds": (
            profile.correction_interval_max_s
        ),
        f"{prefix}_correction_burst_per_second_max": (
            profile.correction_burst_per_second_max
        ),
        f"{prefix}_correction_jump_p95_m": profile.correction_jump_p95_m,
        f"{prefix}_correction_jump_max_m": profile.correction_jump_max_m,
    }


def _causal_reference_metrics(
    name: str,
    cadence_s: float,
    correction_threshold_m: float,
    profile: _CausalAlignmentProfile,
    target_ate_m: float,
) -> dict[str, int | float | str]:
    prefix = f"causal_sim3_{name}"
    return {
        f"{prefix}_anchor_cadence_seconds": cadence_s,
        f"{prefix}_correction_threshold_m": correction_threshold_m,
        f"{prefix}_ate_m": profile.ate_m,
        f"{prefix}_target_met": int(
            profile.pose_coverage_fraction >= 0.9
            and np.isfinite(profile.ate_m)
            and profile.ate_m <= target_ate_m
        ),
        f"{prefix}_correction_count": profile.correction_count,
        f"{prefix}_correction_messages_per_minute": (
            profile.correction_messages_per_minute
        ),
        f"{prefix}_anchor_messages_per_minute": (
            profile.anchor_messages_per_minute
        ),
        f"{prefix}_correction_interval_min_seconds": (
            profile.correction_interval_min_s
        ),
        f"{prefix}_correction_interval_p95_seconds": (
            profile.correction_interval_p95_s
        ),
        f"{prefix}_correction_interval_max_seconds": (
            profile.correction_interval_max_s
        ),
        f"{prefix}_correction_burst_per_second_max": (
            profile.correction_burst_per_second_max
        ),
    }


def _native_position_anchors(
    timestamps_ns: npt.NDArray[np.int64],
    estimated: npt.NDArray[np.float64],
    position_truth: tuple[GroundTruthPose, ...],
    tolerance_seconds: float,
) -> tuple[_PositionAnchor, ...]:
    tolerance_ns = int(tolerance_seconds * 1e9)
    relative_timestamps_ns = np.asarray(
        timestamps_ns - timestamps_ns[0],
        dtype=np.float64,
    )
    selected: list[_PositionAnchor] = []
    for pose in position_truth:
        timestamp_ns = pose.timestamp.monotonic_ns
        if timestamp_ns < int(timestamps_ns[0]) or timestamp_ns > int(
            timestamps_ns[-1]
        ):
            continue
        insertion = int(np.searchsorted(timestamps_ns, timestamp_ns))
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < len(timestamps_ns)
        ]
        if not candidates:
            continue
        nearest = min(
            candidates,
            key=lambda index: abs(int(timestamps_ns[index]) - timestamp_ns),
        )
        if abs(int(timestamps_ns[nearest]) - timestamp_ns) > tolerance_ns:
            continue
        application_index = int(
            np.searchsorted(timestamps_ns, timestamp_ns, side="left")
        )
        anchor_relative_ns = float(timestamp_ns - int(timestamps_ns[0]))
        estimated_position = np.asarray(
            [
                np.interp(
                    anchor_relative_ns,
                    relative_timestamps_ns,
                    estimated[:, dimension],
                )
                for dimension in range(3)
            ],
            dtype=np.float64,
        )
        selected.append(
            _PositionAnchor(
                timestamp_ns,
                application_index,
                estimated_position,
                np.asarray(pose.position_m, dtype=np.float64),
            )
        )
    return tuple(selected)


def _adaptive_fixed_lag_anchors(
    anchors: tuple[_PositionAnchor, ...],
) -> tuple[_PositionAnchor, ...]:
    if len(anchors) < 2:
        return anchors
    selected = [anchors[0]]
    previous_rotation: npt.NDArray[np.float64] | None = None
    previous_scale: float | None = None
    for index, anchor in enumerate(anchors[1:], start=1):
        start = selected[-1]
        rotation, scale = _rotation_between_vectors(
            anchor.estimated_position_m - start.estimated_position_m,
            anchor.truth_position_m - start.truth_position_m,
        )
        duration_s = float(anchor.timestamp_ns - start.timestamp_ns) / 1e9
        transform_changed = previous_rotation is None or previous_scale is None
        if previous_rotation is not None and previous_scale is not None:
            transform_changed = (
                abs(float(np.log(scale / previous_scale)))
                >= FIXED_LAG_ADAPTIVE_LOG_SCALE_THRESHOLD
                or _rotation_angle(rotation @ previous_rotation.T)
                >= FIXED_LAG_ADAPTIVE_ROTATION_THRESHOLD_RAD
            )
        maximum_interval_reached = (
            duration_s >= FIXED_LAG_ADAPTIVE_MAX_INTERVAL_SECONDS - 0.1
        )
        if transform_changed or maximum_interval_reached or index == len(anchors) - 1:
            selected.append(anchor)
            previous_rotation = rotation
            previous_scale = scale
    return tuple(selected)


def _native_fixed_lag_alignment_profile(
    timestamps_ns: npt.NDArray[np.int64],
    estimated: npt.NDArray[np.float64],
    truth: npt.NDArray[np.float64],
    anchors: tuple[_PositionAnchor, ...],
    *,
    similarity: bool,
) -> _FixedLagAlignmentProfile:
    if len(anchors) < 2:
        return _FixedLagAlignmentProfile(
            float("nan"),
            0.0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
        )
    anchor_timestamps_ns = np.asarray(
        [anchor.timestamp_ns for anchor in anchors],
        dtype=np.int64,
    )
    errors_m: list[float] = []
    latencies_s: list[float] = []
    finalized_segments: set[int] = set()
    segment_transforms: list[
        tuple[npt.NDArray[np.float64], float]
    ] = []
    segment_scales: list[float] = []
    for start, end in zip(anchors[:-1], anchors[1:], strict=True):
        rotation, scale = _rotation_between_vectors(
            end.estimated_position_m - start.estimated_position_m,
            end.truth_position_m - start.truth_position_m,
        )
        segment_transforms.append((rotation, scale))
        segment_scales.append(scale)
    for timestamp_ns, estimated_position, truth_position in zip(
        timestamps_ns,
        estimated,
        truth,
        strict=True,
    ):
        end_index = int(
            np.searchsorted(anchor_timestamps_ns, timestamp_ns, side="left")
        )
        if end_index <= 0 or end_index >= len(anchors):
            continue
        start = anchors[end_index - 1]
        end = anchors[end_index]
        rotation, scale = segment_transforms[end_index - 1]
        if not similarity:
            scale = 1.0
        corrected_position = start.truth_position_m + scale * (
            rotation @ (estimated_position - start.estimated_position_m)
        )
        errors_m.append(float(np.linalg.norm(corrected_position - truth_position)))
        latencies_s.append(float(end.timestamp_ns - int(timestamp_ns)) / 1e9)
        finalized_segments.add(end_index)
    duration_s = float(timestamps_ns[-1] - timestamps_ns[0]) / 1e9
    return _FixedLagAlignmentProfile(
        (
            float(np.sqrt(np.mean(np.square(errors_m))))
            if errors_m
            else float("nan")
        ),
        len(errors_m) / len(timestamps_ns),
        len(finalized_segments),
        (
            len(finalized_segments) * 60.0 / duration_s
            if duration_s > 0
            else 0.0
        ),
        float(np.mean(latencies_s)) if latencies_s else 0.0,
        float(np.percentile(latencies_s, 95)) if latencies_s else 0.0,
        float(np.max(latencies_s)) if latencies_s else 0.0,
        float(np.min(segment_scales)),
        float(np.percentile(segment_scales, 5)),
        float(np.percentile(segment_scales, 95)),
        float(np.max(segment_scales)),
        float(
            np.mean(
                [
                    FIXED_LAG_SCALE_MINIMUM <= scale <= FIXED_LAG_SCALE_MAXIMUM
                    for scale in segment_scales
                ]
            )
        ),
    )


def _native_segment_hold_alignment_profile(
    timestamps_ns: npt.NDArray[np.int64],
    estimated: npt.NDArray[np.float64],
    truth: npt.NDArray[np.float64],
    anchors: tuple[_PositionAnchor, ...],
) -> _CausalSegmentHoldProfile:
    if len(anchors) < 2:
        return _CausalSegmentHoldProfile(
            float("nan"),
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            tuple(float("nan") for _ in CAUSAL_SEGMENT_HOLD_HORIZONS_S),
        )
    anchor_timestamps_ns = np.asarray(
        [anchor.timestamp_ns for anchor in anchors],
        dtype=np.int64,
    )
    observed_rotations: list[npt.NDArray[np.float64]] = []
    observed_scales: list[float] = []
    for start, end in zip(anchors[:-1], anchors[1:], strict=True):
        rotation, scale = _rotation_between_vectors(
            end.estimated_position_m - start.estimated_position_m,
            end.truth_position_m - start.truth_position_m,
        )
        observed_rotations.append(rotation)
        observed_scales.append(scale)

    held_transforms: list[tuple[npt.NDArray[np.float64], float]] = []
    held_scales: list[float] = []
    for anchor_index in range(1, len(anchors)):
        history_start = max(0, anchor_index - CAUSAL_SEGMENT_HOLD_HISTORY)
        history_indices = range(history_start, anchor_index)
        weights = np.asarray(
            [
                CAUSAL_SEGMENT_HOLD_DECAY ** (anchor_index - 1 - index)
                for index in history_indices
            ],
            dtype=np.float64,
        )
        weights /= np.sum(weights)
        rotation = _weighted_rotation_average(
            [observed_rotations[index] for index in history_indices],
            weights,
        )
        scale = float(
            np.exp(
                np.sum(
                    weights
                    * np.log(
                        np.maximum(
                            np.asarray(
                                [
                                    observed_scales[index]
                                    for index in history_indices
                                ],
                                dtype=np.float64,
                            ),
                            np.finfo(np.float64).eps,
                        )
                    )
                )
            )
        )
        held_transforms.append((rotation, scale))
        held_scales.append(scale)

    errors_m: list[float] = []
    ages_s: list[float] = []
    for timestamp_ns, estimated_position, truth_position in zip(
        timestamps_ns,
        estimated,
        truth,
        strict=True,
    ):
        anchor_index = int(
            np.searchsorted(anchor_timestamps_ns, timestamp_ns, side="right")
        ) - 1
        if anchor_index < 1:
            continue
        anchor = anchors[anchor_index]
        rotation, scale = held_transforms[anchor_index - 1]
        corrected_position = anchor.truth_position_m + scale * (
            rotation @ (estimated_position - anchor.estimated_position_m)
        )
        errors_m.append(float(np.linalg.norm(corrected_position - truth_position)))
        ages_s.append(float(int(timestamp_ns) - anchor.timestamp_ns) / 1e9)

    duration_s = float(timestamps_ns[-1] - timestamps_ns[0]) / 1e9
    error_values = np.asarray(errors_m, dtype=np.float64)
    age_values = np.asarray(ages_s, dtype=np.float64)
    horizon_ate_m = tuple(
        (
            float(np.sqrt(np.mean(np.square(error_values[age_values <= horizon_s]))))
            if np.any(age_values <= horizon_s)
            else float("nan")
        )
        for horizon_s in CAUSAL_SEGMENT_HOLD_HORIZONS_S
    )
    return _CausalSegmentHoldProfile(
        (
            float(np.sqrt(np.mean(np.square(errors_m))))
            if errors_m
            else float("nan")
        ),
        len(errors_m) / len(timestamps_ns),
        (
            len(held_transforms) * 60.0 / duration_s
            if duration_s > 0
            else 0.0
        ),
        float(np.percentile(held_scales, 5)),
        float(np.percentile(held_scales, 95)),
        float(
            np.mean(
                [
                    FIXED_LAG_SCALE_MINIMUM <= scale <= FIXED_LAG_SCALE_MAXIMUM
                    for scale in held_scales
                ]
            )
        ),
        horizon_ate_m,
    )


def evaluate_local_alignment_sensitivity(
    estimates: tuple[TrajectoryPose, ...],
    position_truth: tuple[GroundTruthPose, ...],
    *,
    tolerance_seconds: float = 0.6,
    target_ate_m: float = 0.1,
) -> dict[str, int | float | str]:
    """Measure optimistic piecewise frame/scale correction floors by cadence."""
    if (
        not np.isfinite(tolerance_seconds)
        or tolerance_seconds <= 0
        or not np.isfinite(target_ate_m)
        or target_ate_m <= 0
    ):
        raise ValueError("local-alignment controls must be finite and positive")
    timestamps_ns, estimated, truth = _matched_samples(
        estimates,
        position_truth,
        tolerance_seconds,
    )
    unique_times, unique_indices = np.unique(timestamps_ns, return_index=True)
    estimated = estimated[unique_indices]
    truth = truth[unique_indices]
    metrics: dict[str, int | float | str] = {
        "local_alignment_sensitivity_model": "piecewise_se3_and_sim3",
        "local_alignment_sensitivity_is_causal": 0,
        "local_alignment_sensitivity_uses_fitted_evaluation_data": 1,
        "local_alignment_sensitivity_changes_scored_ate": 0,
        "local_alignment_sensitivity_target_ate_m": target_ate_m,
        "local_alignment_sensitivity_input_pose_count": len(timestamps_ns),
        "local_alignment_sensitivity_unique_pose_count": len(unique_times),
        "local_alignment_sensitivity_duplicate_pose_count": (
            len(timestamps_ns) - len(unique_times)
        ),
    }
    if len(unique_times) < 3:
        return metrics
    global_errors = np.linalg.norm(_rigid_align(estimated, truth) - truth, axis=1)
    metrics["local_alignment_sensitivity_global_rigid_ate_m"] = float(
        np.sqrt(np.mean(global_errors**2))
    )
    rigid_passing: list[tuple[float, float]] = []
    similarity_passing: list[tuple[float, float]] = []
    for interval_s in LOCAL_ALIGNMENT_INTERVALS_S:
        rigid_ate_m, similarity_ate_m, coverage, window_count, scales = (
            _piecewise_alignment_profile(
                unique_times,
                estimated,
                truth,
                interval_s,
            )
        )
        token = _interval_metric_token(interval_s)
        metrics[f"local_alignment_{token}s_rigid_ate_m"] = rigid_ate_m
        metrics[f"local_alignment_{token}s_sim3_ate_m"] = similarity_ate_m
        metrics[f"local_alignment_{token}s_pose_coverage_fraction"] = coverage
        metrics[f"local_alignment_{token}s_window_count"] = window_count
        if len(scales):
            metrics[f"local_alignment_{token}s_scale_median"] = float(
                np.median(scales)
            )
            metrics[f"local_alignment_{token}s_scale_p05"] = float(
                np.percentile(scales, 5)
            )
            metrics[f"local_alignment_{token}s_scale_p95"] = float(
                np.percentile(scales, 95)
            )
        if coverage >= 0.9 and np.isfinite(rigid_ate_m) and rigid_ate_m <= target_ate_m:
            rigid_passing.append((interval_s, rigid_ate_m))
        if (
            coverage >= 0.9
            and np.isfinite(similarity_ate_m)
            and similarity_ate_m <= target_ate_m
        ):
            similarity_passing.append((interval_s, similarity_ate_m))
    rigid_interval_s, rigid_ate_m = (
        max(rigid_passing) if rigid_passing else (0.0, 0.0)
    )
    similarity_interval_s, similarity_ate_m = (
        max(similarity_passing)
        if similarity_passing
        else (0.0, 0.0)
    )
    metrics.update(
        {
            "local_rigid_target_reachable_with_tested_intervals": int(
                bool(rigid_passing)
            ),
            "local_rigid_maximum_passing_interval_seconds": rigid_interval_s,
            "local_rigid_ate_at_selected_interval_m": rigid_ate_m,
            "local_rigid_optimistic_anchor_messages_per_minute": (
                60.0 / rigid_interval_s if rigid_interval_s > 0 else 0.0
            ),
            "local_sim3_target_reachable_with_tested_intervals": int(
                bool(similarity_passing)
            ),
            "local_sim3_maximum_passing_interval_seconds": similarity_interval_s,
            "local_sim3_ate_at_selected_interval_m": similarity_ate_m,
            "local_sim3_optimistic_anchor_messages_per_minute": (
                60.0 / similarity_interval_s if similarity_interval_s > 0 else 0.0
            ),
        }
    )
    metrics.update(
        {
            "causal_alignment_sensitivity_model": (
                "trailing_position_anchor_se3_and_sim3"
            ),
            "causal_alignment_sensitivity_is_causal": 1,
            "causal_alignment_sensitivity_uses_future_samples": 0,
            "causal_alignment_sensitivity_uses_fitted_evaluation_data": 1,
            "causal_alignment_sensitivity_uses_ideal_position_truth_anchors": 1,
            "causal_alignment_sensitivity_assumes_zero_latency": 1,
            "causal_alignment_sensitivity_is_deployable": 0,
            "causal_alignment_sensitivity_changes_scored_ate": 0,
            "causal_alignment_sensitivity_evaluates_orientation": 0,
            "causal_alignment_sensitivity_target_ate_m": target_ate_m,
            "causal_alignment_sensitivity_minimum_history_seconds": 0.5,
            "causal_alignment_sensitivity_history_cadence_multiplier": 2.0,
            "causal_correction_load_uses_ideal_position_truth_anchors": 1,
            "causal_correction_load_claim_eligible": 0,
        }
    )
    causal_se3_passing: list[
        tuple[float, _CausalAlignmentProfile]
    ] = []
    causal_sim3_passing: list[
        tuple[float, _CausalAlignmentProfile]
    ] = []
    causal_se3_load_candidates: list[
        tuple[float, float, _CausalAlignmentProfile]
    ] = []
    causal_sim3_load_candidates: list[
        tuple[float, float, _CausalAlignmentProfile]
    ] = []
    for cadence_s in CAUSAL_ALIGNMENT_CADENCES_S:
        se3_profile = _causal_alignment_profile(
            unique_times,
            estimated,
            truth,
            cadence_s,
            similarity=False,
        )
        sim3_profile = _causal_alignment_profile(
            unique_times,
            estimated,
            truth,
            cadence_s,
            similarity=True,
        )
        token = _interval_metric_token(cadence_s)
        metrics.update(
            {
                f"causal_alignment_{token}s_se3_ate_m": se3_profile.ate_m,
                f"causal_alignment_{token}s_sim3_ate_m": sim3_profile.ate_m,
                f"causal_alignment_{token}s_se3_pose_coverage_fraction": (
                    se3_profile.pose_coverage_fraction
                ),
                f"causal_alignment_{token}s_sim3_pose_coverage_fraction": (
                    sim3_profile.pose_coverage_fraction
                ),
                f"causal_alignment_{token}s_anchor_messages_per_minute": (
                    se3_profile.anchor_messages_per_minute
                ),
                f"causal_alignment_{token}s_fit_updates_per_minute": (
                    se3_profile.fit_updates_per_minute
                ),
                f"causal_alignment_{token}s_first_fit_delay_seconds": (
                    se3_profile.first_fit_delay_s
                ),
                f"causal_alignment_{token}s_se3_correction_jump_p95_m": (
                    se3_profile.correction_jump_p95_m
                ),
                f"causal_alignment_{token}s_se3_correction_jump_max_m": (
                    se3_profile.correction_jump_max_m
                ),
                f"causal_alignment_{token}s_sim3_correction_jump_p95_m": (
                    sim3_profile.correction_jump_p95_m
                ),
                f"causal_alignment_{token}s_sim3_correction_jump_max_m": (
                    sim3_profile.correction_jump_max_m
                ),
                f"causal_alignment_{token}s_sim3_scale_p05": (
                    sim3_profile.scale_p05
                ),
                f"causal_alignment_{token}s_sim3_scale_p95": (
                    sim3_profile.scale_p95
                ),
                f"causal_alignment_{token}s_sim3_scale_update_p95_log": (
                    sim3_profile.scale_update_p95_log
                ),
            }
        )
        if (
            se3_profile.pose_coverage_fraction >= 0.9
            and np.isfinite(se3_profile.ate_m)
            and se3_profile.ate_m <= target_ate_m
        ):
            causal_se3_passing.append((cadence_s, se3_profile))
        if (
            sim3_profile.pose_coverage_fraction >= 0.9
            and np.isfinite(sim3_profile.ate_m)
            and sim3_profile.ate_m <= target_ate_m
        ):
            causal_sim3_passing.append((cadence_s, sim3_profile))
        for correction_threshold_m in CAUSAL_CORRECTION_THRESHOLDS_M:
            thresholded_se3 = (
                se3_profile
                if correction_threshold_m == 0.0
                else _causal_alignment_profile(
                    unique_times,
                    estimated,
                    truth,
                    cadence_s,
                    similarity=False,
                    correction_threshold_m=correction_threshold_m,
                )
            )
            thresholded_sim3 = (
                sim3_profile
                if correction_threshold_m == 0.0
                else _causal_alignment_profile(
                    unique_times,
                    estimated,
                    truth,
                    cadence_s,
                    similarity=True,
                    correction_threshold_m=correction_threshold_m,
                )
            )
            if (
                thresholded_se3.pose_coverage_fraction >= 0.9
                and np.isfinite(thresholded_se3.ate_m)
                and thresholded_se3.ate_m <= target_ate_m
            ):
                causal_se3_load_candidates.append(
                    (cadence_s, correction_threshold_m, thresholded_se3)
                )
            if (
                thresholded_sim3.pose_coverage_fraction >= 0.9
                and np.isfinite(thresholded_sim3.ate_m)
                and thresholded_sim3.ate_m <= target_ate_m
            ):
                causal_sim3_load_candidates.append(
                    (cadence_s, correction_threshold_m, thresholded_sim3)
                )
    selected_se3 = max(causal_se3_passing) if causal_se3_passing else None
    selected_sim3 = max(causal_sim3_passing) if causal_sim3_passing else None
    selected_se3_load = (
        min(
            causal_se3_load_candidates,
            key=lambda candidate: (
                candidate[2].correction_messages_per_minute,
                candidate[2].ate_m,
                -candidate[0],
            ),
        )
        if causal_se3_load_candidates
        else None
    )
    selected_sim3_load = (
        min(
            causal_sim3_load_candidates,
            key=lambda candidate: (
                candidate[2].correction_messages_per_minute,
                candidate[2].ate_m,
                -candidate[0],
            ),
        )
        if causal_sim3_load_candidates
        else None
    )
    reference_sim3_load = _causal_alignment_profile(
        unique_times,
        estimated,
        truth,
        CAUSAL_REFERENCE_CADENCE_S,
        similarity=True,
        correction_threshold_m=CAUSAL_REFERENCE_CORRECTION_THRESHOLD_M,
    )
    low_ingress_sim3_load = _causal_alignment_profile(
        unique_times,
        estimated,
        truth,
        CAUSAL_LOW_INGRESS_CADENCE_S,
        similarity=True,
        correction_threshold_m=CAUSAL_LOW_INGRESS_CORRECTION_THRESHOLD_M,
    )
    native_anchors = _native_position_anchors(
        unique_times,
        estimated,
        position_truth,
        tolerance_seconds,
    )
    native_rtk_se3 = _causal_alignment_profile(
        unique_times,
        estimated,
        truth,
        1.0,
        similarity=False,
        explicit_anchors=native_anchors,
    )
    native_rtk_sim3 = _causal_alignment_profile(
        unique_times,
        estimated,
        truth,
        1.0,
        similarity=True,
        explicit_anchors=native_anchors,
    )
    native_segment_hold = _native_segment_hold_alignment_profile(
        unique_times,
        estimated,
        truth,
        native_anchors,
    )
    native_fixed_lag_se3 = _native_fixed_lag_alignment_profile(
        unique_times,
        estimated,
        truth,
        native_anchors,
        similarity=False,
    )
    native_fixed_lag_sim3 = _native_fixed_lag_alignment_profile(
        unique_times,
        estimated,
        truth,
        native_anchors,
        similarity=True,
    )
    adaptive_fixed_lag_anchors = _adaptive_fixed_lag_anchors(native_anchors)
    adaptive_fixed_lag_se3 = _native_fixed_lag_alignment_profile(
        unique_times,
        estimated,
        truth,
        adaptive_fixed_lag_anchors,
        similarity=False,
    )
    adaptive_fixed_lag_sim3 = _native_fixed_lag_alignment_profile(
        unique_times,
        estimated,
        truth,
        adaptive_fixed_lag_anchors,
        similarity=True,
    )
    metrics.update(
        {
            "causal_se3_target_reachable_with_tested_cadences": int(
                selected_se3 is not None
            ),
            "causal_se3_maximum_passing_cadence_seconds": (
                selected_se3[0] if selected_se3 is not None else 0.0
            ),
            "causal_se3_ate_at_selected_cadence_m": (
                selected_se3[1].ate_m if selected_se3 is not None else 0.0
            ),
            "causal_se3_anchor_messages_per_minute": (
                selected_se3[1].anchor_messages_per_minute
                if selected_se3 is not None
                else 0.0
            ),
            "causal_se3_correction_jump_p95_m": (
                selected_se3[1].correction_jump_p95_m
                if selected_se3 is not None
                else 0.0
            ),
            "causal_sim3_target_reachable_with_tested_cadences": int(
                selected_sim3 is not None
            ),
            "causal_sim3_maximum_passing_cadence_seconds": (
                selected_sim3[0] if selected_sim3 is not None else 0.0
            ),
            "causal_sim3_ate_at_selected_cadence_m": (
                selected_sim3[1].ate_m if selected_sim3 is not None else 0.0
            ),
            "causal_sim3_anchor_messages_per_minute": (
                selected_sim3[1].anchor_messages_per_minute
                if selected_sim3 is not None
                else 0.0
            ),
            "causal_sim3_correction_jump_p95_m": (
                selected_sim3[1].correction_jump_p95_m
                if selected_sim3 is not None
                else 0.0
            ),
            "causal_sim3_scale_update_p95_log": (
                selected_sim3[1].scale_update_p95_log
                if selected_sim3 is not None
                else 0.0
            ),
            "causal_correction_load_thresholds_m": ",".join(
                f"{threshold:g}"
                for threshold in CAUSAL_CORRECTION_THRESHOLDS_M
            ),
            **_causal_load_metrics("se3", selected_se3_load),
            **_causal_load_metrics("sim3", selected_sim3_load),
            **_causal_reference_metrics(
                "reference",
                CAUSAL_REFERENCE_CADENCE_S,
                CAUSAL_REFERENCE_CORRECTION_THRESHOLD_M,
                reference_sim3_load,
                target_ate_m,
            ),
            **_causal_reference_metrics(
                "low_ingress",
                CAUSAL_LOW_INGRESS_CADENCE_S,
                CAUSAL_LOW_INGRESS_CORRECTION_THRESHOLD_M,
                low_ingress_sim3_load,
                target_ate_m,
            ),
            "causal_native_rtk_anchor_source": "observed_s3e_rtk_samples",
            "causal_native_rtk_uses_interpolated_anchors": 0,
            "causal_native_rtk_truth_position_interpolated": 0,
            "causal_native_rtk_estimate_position_interpolated_to_observation": 1,
            "causal_native_rtk_anchor_count": native_rtk_sim3.anchor_count,
            "causal_native_rtk_anchor_messages_per_minute": (
                native_rtk_sim3.anchor_messages_per_minute
            ),
            "causal_native_rtk_se3_ate_m": native_rtk_se3.ate_m,
            "causal_native_rtk_se3_target_met": int(
                native_rtk_se3.pose_coverage_fraction >= 0.9
                and np.isfinite(native_rtk_se3.ate_m)
                and native_rtk_se3.ate_m <= target_ate_m
            ),
            "causal_native_rtk_sim3_ate_m": native_rtk_sim3.ate_m,
            "causal_native_rtk_sim3_target_met": int(
                native_rtk_sim3.pose_coverage_fraction >= 0.9
                and np.isfinite(native_rtk_sim3.ate_m)
                and native_rtk_sim3.ate_m <= target_ate_m
            ),
            "causal_native_rtk_sim3_correction_messages_per_minute": (
                native_rtk_sim3.correction_messages_per_minute
            ),
            "causal_native_rtk_sim3_correction_interval_min_seconds": (
                native_rtk_sim3.correction_interval_min_s
            ),
            "causal_native_rtk_sim3_correction_interval_p95_seconds": (
                native_rtk_sim3.correction_interval_p95_s
            ),
            "causal_native_rtk_sim3_correction_interval_max_seconds": (
                native_rtk_sim3.correction_interval_max_s
            ),
            "causal_native_rtk_sim3_correction_burst_per_second_max": (
                native_rtk_sim3.correction_burst_per_second_max
            ),
            "causal_native_rtk_claim_eligible": 0,
            "causal_segment_hold_native_rtk_model": (
                "past_native_segment_transform_exponential_hold"
            ),
            "causal_segment_hold_native_rtk_history_segments": (
                CAUSAL_SEGMENT_HOLD_HISTORY
            ),
            "causal_segment_hold_native_rtk_decay": CAUSAL_SEGMENT_HOLD_DECAY,
            "causal_segment_hold_native_rtk_uses_future_pose_time_observation": 0,
            "causal_segment_hold_native_rtk_uses_all_native_anchor_ingress": 1,
            "causal_segment_hold_native_rtk_live_pose_capable": 1,
            "causal_segment_hold_native_rtk_position_only": 1,
            "causal_segment_hold_native_rtk_sim3_ate_m": native_segment_hold.ate_m,
            "causal_segment_hold_native_rtk_sim3_target_met": int(
                native_segment_hold.pose_coverage_fraction >= 0.9
                and np.isfinite(native_segment_hold.ate_m)
                and native_segment_hold.ate_m <= target_ate_m
                and native_segment_hold.scale_plausible_fraction >= 0.95
            ),
            "causal_segment_hold_native_rtk_pose_coverage_fraction": (
                native_segment_hold.pose_coverage_fraction
            ),
            "causal_segment_hold_native_rtk_prediction_updates_per_minute": (
                native_segment_hold.prediction_updates_per_minute
            ),
            "causal_segment_hold_native_rtk_scale_p05": (
                native_segment_hold.scale_p05
            ),
            "causal_segment_hold_native_rtk_scale_p95": (
                native_segment_hold.scale_p95
            ),
            "causal_segment_hold_native_rtk_scale_plausible_fraction": (
                native_segment_hold.scale_plausible_fraction
            ),
            "causal_segment_hold_native_rtk_claim_eligible": 0,
            "fixed_lag_native_rtk_model": (
                "native_endpoint_relative_vio_se3_and_sim3"
            ),
            "fixed_lag_native_rtk_is_causal_at_emission": 1,
            "fixed_lag_native_rtk_uses_future_pose_time_observation": 1,
            "fixed_lag_native_rtk_uses_interpolated_truth_anchors": 0,
            "fixed_lag_native_rtk_changes_scored_ate": 0,
            "fixed_lag_native_rtk_live_pose_capable": 0,
            "fixed_lag_native_rtk_position_only": 1,
            "fixed_lag_native_rtk_se3_ate_m": native_fixed_lag_se3.ate_m,
            "fixed_lag_native_rtk_se3_target_met": int(
                native_fixed_lag_se3.pose_coverage_fraction >= 0.9
                and np.isfinite(native_fixed_lag_se3.ate_m)
                and native_fixed_lag_se3.ate_m <= target_ate_m
            ),
            "fixed_lag_native_rtk_sim3_ate_m": native_fixed_lag_sim3.ate_m,
            "fixed_lag_native_rtk_sim3_target_met": int(
                native_fixed_lag_sim3.pose_coverage_fraction >= 0.9
                and np.isfinite(native_fixed_lag_sim3.ate_m)
                and native_fixed_lag_sim3.ate_m <= target_ate_m
                and native_fixed_lag_sim3.scale_plausible_fraction >= 0.95
            ),
            "fixed_lag_native_rtk_pose_coverage_fraction": (
                native_fixed_lag_sim3.pose_coverage_fraction
            ),
            "fixed_lag_native_rtk_segment_count": (
                native_fixed_lag_sim3.segment_count
            ),
            "fixed_lag_native_rtk_finalization_updates_per_minute": (
                native_fixed_lag_sim3.finalization_updates_per_minute
            ),
            "fixed_lag_native_rtk_latency_mean_seconds": (
                native_fixed_lag_sim3.latency_mean_s
            ),
            "fixed_lag_native_rtk_latency_p95_seconds": (
                native_fixed_lag_sim3.latency_p95_s
            ),
            "fixed_lag_native_rtk_latency_max_seconds": (
                native_fixed_lag_sim3.latency_max_s
            ),
            "fixed_lag_native_rtk_scale_minimum_allowed": (
                FIXED_LAG_SCALE_MINIMUM
            ),
            "fixed_lag_native_rtk_scale_maximum_allowed": (
                FIXED_LAG_SCALE_MAXIMUM
            ),
            "fixed_lag_native_rtk_scale_min": native_fixed_lag_sim3.scale_min,
            "fixed_lag_native_rtk_scale_p05": native_fixed_lag_sim3.scale_p05,
            "fixed_lag_native_rtk_scale_p95": native_fixed_lag_sim3.scale_p95,
            "fixed_lag_native_rtk_scale_max": native_fixed_lag_sim3.scale_max,
            "fixed_lag_native_rtk_scale_plausible_fraction": (
                native_fixed_lag_sim3.scale_plausible_fraction
            ),
            "fixed_lag_native_rtk_claim_eligible": 0,
            "adaptive_fixed_lag_native_rtk_model": (
                "bounded_transform_change_or_two_interval_finalization"
            ),
            "adaptive_fixed_lag_native_rtk_is_causal_at_emission": 1,
            "adaptive_fixed_lag_native_rtk_uses_all_native_anchor_ingress": 1,
            "adaptive_fixed_lag_native_rtk_live_pose_capable": 0,
            "adaptive_fixed_lag_native_rtk_log_scale_threshold": (
                FIXED_LAG_ADAPTIVE_LOG_SCALE_THRESHOLD
            ),
            "adaptive_fixed_lag_native_rtk_rotation_threshold_rad": (
                FIXED_LAG_ADAPTIVE_ROTATION_THRESHOLD_RAD
            ),
            "adaptive_fixed_lag_native_rtk_maximum_interval_seconds": (
                FIXED_LAG_ADAPTIVE_MAX_INTERVAL_SECONDS
            ),
            "adaptive_fixed_lag_native_rtk_selected_anchor_count": len(
                adaptive_fixed_lag_anchors
            ),
            "adaptive_fixed_lag_native_rtk_se3_ate_m": (
                adaptive_fixed_lag_se3.ate_m
            ),
            "adaptive_fixed_lag_native_rtk_sim3_ate_m": (
                adaptive_fixed_lag_sim3.ate_m
            ),
            "adaptive_fixed_lag_native_rtk_sim3_target_met": int(
                adaptive_fixed_lag_sim3.pose_coverage_fraction >= 0.9
                and np.isfinite(adaptive_fixed_lag_sim3.ate_m)
                and adaptive_fixed_lag_sim3.ate_m <= target_ate_m
                and adaptive_fixed_lag_sim3.scale_plausible_fraction >= 0.95
            ),
            "adaptive_fixed_lag_native_rtk_pose_coverage_fraction": (
                adaptive_fixed_lag_sim3.pose_coverage_fraction
            ),
            "adaptive_fixed_lag_native_rtk_finalization_updates_per_minute": (
                adaptive_fixed_lag_sim3.finalization_updates_per_minute
            ),
            "adaptive_fixed_lag_native_rtk_update_reduction_percent": (
                100.0
                * (
                    1.0
                    - adaptive_fixed_lag_sim3.finalization_updates_per_minute
                    / native_fixed_lag_sim3.finalization_updates_per_minute
                )
                if native_fixed_lag_sim3.finalization_updates_per_minute > 0
                else 0.0
            ),
            "adaptive_fixed_lag_native_rtk_latency_mean_seconds": (
                adaptive_fixed_lag_sim3.latency_mean_s
            ),
            "adaptive_fixed_lag_native_rtk_latency_p95_seconds": (
                adaptive_fixed_lag_sim3.latency_p95_s
            ),
            "adaptive_fixed_lag_native_rtk_latency_max_seconds": (
                adaptive_fixed_lag_sim3.latency_max_s
            ),
            "adaptive_fixed_lag_native_rtk_scale_p05": (
                adaptive_fixed_lag_sim3.scale_p05
            ),
            "adaptive_fixed_lag_native_rtk_scale_p95": (
                adaptive_fixed_lag_sim3.scale_p95
            ),
            "adaptive_fixed_lag_native_rtk_scale_plausible_fraction": (
                adaptive_fixed_lag_sim3.scale_plausible_fraction
            ),
            "adaptive_fixed_lag_native_rtk_claim_eligible": 0,
        }
    )
    for horizon_s, ate_m in zip(
        CAUSAL_SEGMENT_HOLD_HORIZONS_S,
        native_segment_hold.horizon_ate_m,
        strict=True,
    ):
        token = f"{horizon_s:g}".replace(".", "_")
        metrics[
            f"causal_segment_hold_native_rtk_horizon_{token}s_ate_m"
        ] = ate_m
    passing_horizons = [
        horizon_s
        for horizon_s, ate_m in zip(
            CAUSAL_SEGMENT_HOLD_HORIZONS_S,
            native_segment_hold.horizon_ate_m,
            strict=True,
        )
        if native_segment_hold.scale_plausible_fraction >= 0.95
        and np.isfinite(ate_m)
        and ate_m <= target_ate_m
    ]
    maximum_target_horizon_s = max(passing_horizons, default=0.0)
    metrics["causal_segment_hold_native_rtk_target_horizon_reachable"] = int(
        maximum_target_horizon_s > 0
    )
    metrics[
        "causal_segment_hold_native_rtk_maximum_target_horizon_seconds"
    ] = maximum_target_horizon_s
    metrics[
        "causal_segment_hold_native_rtk_minimum_observation_rate_per_minute"
    ] = (
        60.0 / maximum_target_horizon_s
        if maximum_target_horizon_s > 0
        else 0.0
    )
    return metrics
