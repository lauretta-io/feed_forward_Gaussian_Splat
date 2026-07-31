"""Bounded robust MILUV UWB position estimation primitives."""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
import numpy.typing as npt

from ariadne.benchmarks.global_pose_rationalization import (
    PoseMatrix,
    PoseSample,
    PoseSeries,
    quaternion_xyzw,
    rotation_matrix,
)

# MILUV development-kit configuration for anchor constellation 0 and the default
# two-tag-per-robot layout:
# https://github.com/decargroup/miluv/blob/main/config/uwb/anchors.yaml
# https://github.com/decargroup/miluv/blob/main/config/uwb/tags.yaml
ANCHOR_POSITIONS_M: dict[int, npt.NDArray[np.float64]] = {
    0: np.asarray([3.273827392578125, 3.46404736328125, 1.8093309326171875]),
    1: np.asarray([3.186386962890625, 0.2739448547363281, 1.5884853515625]),
    2: np.asarray([2.850500244140625, -2.923056884765625, 1.89742041015625]),
    3: np.asarray([-2.497634521484375, -3.5018203125, 1.7730911865234375]),
    4: np.asarray([-2.95793310546875, 0.6128419189453125, 1.65714208984375]),
    5: np.asarray([-2.734676513671875, 3.65854248046875, 1.890254638671875]),
}
TAG_MOMENT_ARMS_M: dict[int, npt.NDArray[np.float64]] = {
    10: np.asarray([0.13189, -0.17245, -0.05249]),
    11: np.asarray([-0.17542, 0.15712, -0.05307]),
    20: np.asarray([0.16544, -0.15085, -0.03456]),
    21: np.asarray([-0.15467, 0.16972, -0.01680]),
    30: np.asarray([0.16685, -0.18113, -0.05576]),
    31: np.asarray([-0.13485, 0.15468, -0.05164]),
}
TAG_AGENT: dict[int, str] = {
    10: "ifo001",
    11: "ifo001",
    20: "ifo002",
    21: "ifo002",
    30: "ifo003",
    31: "ifo003",
}


@dataclass(frozen=True)
class UwbRangeSample:
    timestamp_s: float
    from_id: int
    to_id: int
    range_m: float
    std_m: float
    ground_truth_range_m: float | None = None

    @property
    def tag_id(self) -> int | None:
        if self.from_id in TAG_MOMENT_ARMS_M:
            return self.from_id
        if self.to_id in TAG_MOMENT_ARMS_M:
            return self.to_id
        return None

    @property
    def anchor_id(self) -> int | None:
        if self.from_id in ANCHOR_POSITIONS_M:
            return self.from_id
        if self.to_id in ANCHOR_POSITIONS_M:
            return self.to_id
        return None


@dataclass(frozen=True)
class UwbPositionEstimate:
    timestamp_s: float
    position_m: npt.NDArray[np.float64]
    accepted: bool
    rejection_reason: str
    range_count: int
    inlier_count: int
    unique_anchor_count: int
    residual_rmse_m: float
    residual_p95_m: float
    geometry_condition: float
    correction_delta_m: float
    iterations: int


@dataclass(frozen=True)
class UwbBatchResult:
    optimized: PoseSeries
    converged: bool
    iterations: int
    range_factor_count: int
    anchor_range_factor_count: int
    inter_agent_range_factor_count: int
    downweighted_range_count: int
    range_residual_rmse_m: float
    range_residual_p95_m: float
    transceiver_bias_m: dict[int, float]
    optimization_latency_ms: float


@dataclass(frozen=True)
class UwbFixedLagResult:
    optimized: PoseSeries
    lag_samples: int
    solve_interval_samples: int
    solve_sample_indices: tuple[int, ...]
    solve_count: int
    converged_solve_count: int
    range_factor_count_total: int
    downweighted_range_count_total: int
    final_transceiver_bias_m: dict[int, float]
    optimization_latency_ms_total: float
    optimization_latency_ms_p95: float
    optimization_latency_ms_max: float


@dataclass(frozen=True)
class _BatchRangeFactor:
    position_terms: tuple[tuple[int, float], ...]
    constant_m: npt.NDArray[np.float64]
    range_m: float
    sigma_m: float
    from_bias_index: int
    to_bias_index: int
    anchor_range: bool


def read_uwb_ranges(
    archive: zipfile.ZipFile,
    members: tuple[str, ...],
) -> tuple[UwbRangeSample, ...]:
    """Read and deduplicate the calibrated active ranges needed by localization."""
    samples: list[UwbRangeSample] = []
    seen: set[tuple[int, int, int, int]] = set()
    for member in members:
        with archive.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8") as text:
            for row in csv.DictReader(text):
                timestamp_s = float(row["timestamp"])
                from_id = int(row["from_id"])
                to_id = int(row["to_id"])
                range_m = float(row["range"])
                std_m = float(row["std"])
                values = np.asarray((timestamp_s, range_m, std_m), dtype=np.float64)
                if not np.all(np.isfinite(values)) or range_m <= 0 or std_m <= 0:
                    continue
                key = (
                    round(timestamp_s * 1e6),
                    min(from_id, to_id),
                    max(from_id, to_id),
                    round(range_m * 1e6),
                )
                if key in seen:
                    continue
                seen.add(key)
                truth_text = row.get("gt_range", "")
                truth = float(truth_text) if truth_text else None
                samples.append(
                    UwbRangeSample(
                        timestamp_s,
                        from_id,
                        to_id,
                        range_m,
                        std_m,
                        truth,
                    )
                )
    if not samples:
        raise ValueError("MILUV archive contains no usable calibrated UWB ranges")
    return tuple(sorted(samples, key=lambda sample: sample.timestamp_s))


def _slerp(
    first: npt.NDArray[np.float64],
    second: npt.NDArray[np.float64],
    fraction: float,
) -> npt.NDArray[np.float64]:
    left = quaternion_xyzw(first)
    right = quaternion_xyzw(second)
    dot = float(np.dot(left, right))
    if dot < 0:
        right = -right
        dot = -dot
    if dot > 0.9995:
        quaternion = left + fraction * (right - left)
        return rotation_matrix(quaternion / np.linalg.norm(quaternion))
    angle = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    scale = np.sin(angle)
    quaternion = (
        np.sin((1.0 - fraction) * angle) / scale * left + np.sin(fraction * angle) / scale * right
    )
    return rotation_matrix(quaternion)


def _interpolate_pose(
    timestamps_s: npt.NDArray[np.float64],
    poses: tuple[PoseMatrix, ...],
    timestamp_s: float,
) -> PoseMatrix:
    upper = int(np.searchsorted(timestamps_s, timestamp_s))
    if upper <= 0:
        return poses[0]
    if upper >= len(poses):
        return poses[-1]
    lower = upper - 1
    duration = timestamps_s[upper] - timestamps_s[lower]
    fraction = float((timestamp_s - timestamps_s[lower]) / duration)
    pose = np.eye(4)
    pose[:3, 3] = (1.0 - fraction) * poses[lower][:3, 3] + fraction * poses[upper][:3, 3]
    pose[:3, :3] = _slerp(
        poses[lower][:3, :3],
        poses[upper][:3, :3],
        fraction,
    )
    return pose


def _range_linearization(
    position_m: npt.NDArray[np.float64],
    target_baseline_position_m: npt.NDArray[np.float64],
    observations: tuple[UwbRangeSample, ...],
    agent: str,
    timestamps_s: npt.NDArray[np.float64],
    baseline: PoseSeries,
    anchors_m: dict[int, npt.NDArray[np.float64]],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    jacobian: list[npt.NDArray[np.float64]] = []
    residuals: list[float] = []
    sigmas: list[float] = []
    for observation in observations:
        tag_id = observation.tag_id
        anchor_id = observation.anchor_id
        if tag_id is None or anchor_id is None:
            continue
        measurement_pose = _interpolate_pose(
            timestamps_s,
            baseline[agent],
            observation.timestamp_s,
        )
        body_position = position_m + measurement_pose[:3, 3] - target_baseline_position_m
        tag_position = body_position + measurement_pose[:3, :3] @ TAG_MOMENT_ARMS_M[tag_id]
        difference = tag_position - anchors_m[anchor_id]
        predicted_range = float(np.linalg.norm(difference))
        if predicted_range <= np.finfo(np.float64).eps:
            continue
        jacobian.append(difference / predicted_range)
        residuals.append(predicted_range - observation.range_m)
        sigmas.append(float(np.clip(observation.std_m, 0.05, 0.30)))
    return (
        np.asarray(jacobian, dtype=np.float64),
        np.asarray(residuals, dtype=np.float64),
        np.asarray(sigmas, dtype=np.float64),
    )


def estimate_uwb_positions(
    agents: tuple[str, ...],
    sampled: dict[str, tuple[PoseSample, ...]],
    baseline: PoseSeries,
    ranges: tuple[UwbRangeSample, ...],
    world_origin_m: npt.NDArray[np.float64],
    *,
    window_seconds: float = 1.5,
    processing_latency_seconds: float = 0.75,
    maximum_ranges: int = 48,
    minimum_ranges: int = 8,
    retain_fraction: float = 0.85,
    huber_sigma: float = 1.0,
    maximum_iterations: int = 15,
    maximum_residual_rmse_m: float = 0.35,
    maximum_correction_delta_m: float = 1.5,
    maximum_geometry_condition: float = 1e4,
) -> dict[str, tuple[UwbPositionEstimate, ...]]:
    """Estimate body positions from fixed anchors with bounded delayed smoothing."""
    if (
        window_seconds <= 0
        or processing_latency_seconds < 0
        or maximum_ranges < minimum_ranges
        or minimum_ranges < 4
        or not 0.5 <= retain_fraction <= 1.0
        or huber_sigma <= 0
        or maximum_iterations <= 0
    ):
        raise ValueError("MILUV UWB estimator bounds are invalid")
    if set(agents) != set(sampled) or set(agents) != set(baseline):
        raise ValueError("MILUV UWB agents and pose series do not match")

    anchors_m = {
        anchor_id: position - world_origin_m for anchor_id, position in ANCHOR_POSITIONS_M.items()
    }
    ranges_by_agent = {
        agent: tuple(
            sample
            for sample in ranges
            if sample.tag_id is not None
            and TAG_AGENT.get(sample.tag_id) == agent
            and sample.anchor_id is not None
        )
        for agent in agents
    }
    estimates: dict[str, tuple[UwbPositionEstimate, ...]] = {}
    for agent in agents:
        timestamps_s = np.asarray(
            [sample.timestamp_s for sample in sampled[agent]],
            dtype=np.float64,
        )
        agent_estimates: list[UwbPositionEstimate] = []
        for index, target in enumerate(sampled[agent]):
            available_until_s = target.timestamp_s + processing_latency_seconds
            window_start_s = available_until_s - window_seconds
            observations = tuple(
                sorted(
                    (
                        sample
                        for sample in ranges_by_agent[agent]
                        if window_start_s <= sample.timestamp_s <= available_until_s
                        and timestamps_s[0] <= sample.timestamp_s <= timestamps_s[-1]
                    ),
                    key=lambda sample: abs(sample.timestamp_s - target.timestamp_s),
                )[:maximum_ranges]
            )
            baseline_position = baseline[agent][index][:3, 3]
            position = baseline_position.copy()
            converged = False
            iterations = 0
            inliers = np.arange(len(observations))
            geometry_condition = float("inf")
            for iteration in range(1, maximum_iterations + 1):
                jacobian, residuals, sigmas = _range_linearization(
                    position,
                    baseline_position,
                    observations,
                    agent,
                    timestamps_s,
                    baseline,
                    anchors_m,
                )
                if len(residuals) < minimum_ranges:
                    break
                normalized = np.abs(residuals) / sigmas
                retained = max(minimum_ranges, int(np.ceil(retain_fraction * len(residuals))))
                inliers = np.argsort(normalized)[:retained]
                inlier_normalized = normalized[inliers]
                huber_weights = np.minimum(
                    1.0,
                    huber_sigma / np.maximum(inlier_normalized, 1e-12),
                )
                weights = huber_weights / np.square(sigmas[inliers])
                weighted_jacobian = weights[:, None] * jacobian[inliers]
                hessian = jacobian[inliers].T @ weighted_jacobian
                geometry_condition = float(np.linalg.cond(hessian))
                gradient = jacobian[inliers].T @ (weights * residuals[inliers])
                try:
                    step = np.linalg.solve(
                        hessian + np.eye(3) * 1e-4,
                        gradient,
                    )
                except np.linalg.LinAlgError:
                    break
                step_norm = float(np.linalg.norm(step))
                if step_norm > 1.0:
                    step /= step_norm
                position -= step
                iterations = iteration
                if step_norm <= 1e-5:
                    converged = True
                    break

            jacobian, residuals, sigmas = _range_linearization(
                position,
                baseline_position,
                observations,
                agent,
                timestamps_s,
                baseline,
                anchors_m,
            )
            if len(residuals):
                normalized = np.abs(residuals) / sigmas
                retained = max(
                    min(minimum_ranges, len(residuals)),
                    int(np.ceil(retain_fraction * len(residuals))),
                )
                inliers = np.argsort(normalized)[:retained]
                selected_residuals = residuals[inliers]
                residual_rmse_m = float(np.sqrt(np.mean(np.square(selected_residuals))))
                residual_p95_m = float(np.percentile(np.abs(selected_residuals), 95))
            else:
                inliers = np.asarray([], dtype=np.int64)
                residual_rmse_m = float("inf")
                residual_p95_m = float("inf")
            unique_anchor_count = len(
                {
                    observation.anchor_id
                    for observation in observations
                    if observation.anchor_id is not None
                }
            )
            correction_delta_m = float(np.linalg.norm(position - baseline_position))
            rejection_reason = ""
            if len(residuals) < minimum_ranges:
                rejection_reason = "insufficient_ranges"
            elif unique_anchor_count < 4:
                rejection_reason = "insufficient_anchor_geometry"
            elif not converged:
                rejection_reason = "not_converged"
            elif not np.isfinite(geometry_condition) or (
                geometry_condition > maximum_geometry_condition
            ):
                rejection_reason = "ill_conditioned_geometry"
            elif residual_rmse_m > maximum_residual_rmse_m:
                rejection_reason = "residual_gate"
            elif correction_delta_m > maximum_correction_delta_m:
                rejection_reason = "correction_delta_gate"
            agent_estimates.append(
                UwbPositionEstimate(
                    target.timestamp_s,
                    position,
                    not rejection_reason,
                    rejection_reason,
                    len(residuals),
                    len(inliers),
                    unique_anchor_count,
                    residual_rmse_m,
                    residual_p95_m,
                    geometry_condition,
                    correction_delta_m,
                    iterations,
                )
            )
        estimates[agent] = tuple(agent_estimates)
    return estimates


def _interpolation_terms(
    timestamps_s: npt.NDArray[np.float64],
    timestamp_s: float,
    node_offset: int,
) -> tuple[tuple[int, float], ...]:
    upper = int(np.searchsorted(timestamps_s, timestamp_s))
    if upper <= 0:
        return ((node_offset, 1.0),)
    if upper >= len(timestamps_s):
        return ((node_offset + len(timestamps_s) - 1, 1.0),)
    lower = upper - 1
    fraction = float(
        (timestamp_s - timestamps_s[lower]) / (timestamps_s[upper] - timestamps_s[lower])
    )
    return (
        (node_offset + lower, 1.0 - fraction),
        (node_offset + upper, fraction),
    )


def batch_rationalize_uwb_positions(
    agents: tuple[str, ...],
    sampled: dict[str, tuple[PoseSample, ...]],
    baseline: PoseSeries,
    ranges: tuple[UwbRangeSample, ...],
    world_origin_m: npt.NDArray[np.float64],
    *,
    odometry_information: float = 1000.0,
    initial_position_information: float = 1000.0,
    bias_information: float = 25.0,
    huber_sigma: float = 0.5,
    maximum_iterations: int = 12,
    convergence_tolerance_m: float = 2.5e-4,
    maximum_range_factors: int = 8_000,
    position_prior_m_by_agent: dict[str, npt.NDArray[np.float64]] | None = None,
    transceiver_bias_prior_m: dict[int, float] | None = None,
) -> UwbBatchResult:
    """Jointly smooth all positions and online range-bias states in one bounded batch."""
    if (
        set(agents) != set(sampled)
        or set(agents) != set(baseline)
        or any(len(sampled[agent]) != len(baseline[agent]) for agent in agents)
    ):
        raise ValueError("MILUV UWB batch agents and pose series do not match")
    if (
        len(agents) < 1
        or odometry_information <= 0
        or initial_position_information <= 0
        or bias_information <= 0
        or huber_sigma <= 0
        or maximum_iterations <= 0
        or convergence_tolerance_m <= 0
        or maximum_range_factors <= 0
    ):
        raise ValueError("MILUV UWB batch bounds must be positive")
    if position_prior_m_by_agent is not None and set(position_prior_m_by_agent) != set(agents):
        raise ValueError("MILUV UWB batch position priors do not match agents")

    start_ns = perf_counter_ns()
    sample_count = len(sampled[agents[0]])
    if any(len(sampled[agent]) != sample_count for agent in agents):
        raise ValueError("MILUV UWB batch requires equal per-agent sample counts")
    timestamps = {
        agent: np.asarray(
            [sample.timestamp_s for sample in sampled[agent]],
            dtype=np.float64,
        )
        for agent in agents
    }
    overlap_start_s = max(values[0] for values in timestamps.values())
    overlap_end_s = min(values[-1] for values in timestamps.values())
    agent_index = {agent: index for index, agent in enumerate(agents)}
    anchors_m = {
        anchor_id: position - world_origin_m for anchor_id, position in ANCHOR_POSITIONS_M.items()
    }
    transceiver_ids = tuple(sorted((*ANCHOR_POSITIONS_M, *TAG_MOMENT_ARMS_M)))
    bias_index = {transceiver_id: index for index, transceiver_id in enumerate(transceiver_ids)}
    if transceiver_bias_prior_m is not None and set(transceiver_bias_prior_m) != set(
        transceiver_ids
    ):
        raise ValueError("MILUV UWB batch bias priors do not match transceivers")

    def endpoint(
        transceiver_id: int,
        timestamp_s: float,
    ) -> tuple[tuple[tuple[int, float], ...], npt.NDArray[np.float64]]:
        if transceiver_id in anchors_m:
            return (), anchors_m[transceiver_id]
        agent = TAG_AGENT[transceiver_id]
        node_offset = agent_index[agent] * sample_count
        pose = _interpolate_pose(
            timestamps[agent],
            baseline[agent],
            timestamp_s,
        )
        return (
            _interpolation_terms(
                timestamps[agent],
                timestamp_s,
                node_offset,
            ),
            pose[:3, :3] @ TAG_MOMENT_ARMS_M[transceiver_id],
        )

    candidate_ranges = tuple(
        sample
        for sample in ranges
        if overlap_start_s <= sample.timestamp_s <= overlap_end_s
        and sample.from_id in bias_index
        and sample.to_id in bias_index
    )
    if len(candidate_ranges) > maximum_range_factors:
        retained_indices = np.linspace(
            0,
            len(candidate_ranges) - 1,
            maximum_range_factors,
            dtype=np.int64,
        )
        candidate_ranges = tuple(candidate_ranges[index] for index in retained_indices)

    factors: list[_BatchRangeFactor] = []
    for sample in candidate_ranges:
        left_terms, left_constant = endpoint(
            sample.from_id,
            sample.timestamp_s,
        )
        right_terms, right_constant = endpoint(
            sample.to_id,
            sample.timestamp_s,
        )
        combined: dict[int, float] = {}
        for node, coefficient in left_terms:
            combined[node] = combined.get(node, 0.0) + coefficient
        for node, coefficient in right_terms:
            combined[node] = combined.get(node, 0.0) - coefficient
        factors.append(
            _BatchRangeFactor(
                tuple(sorted(combined.items())),
                left_constant - right_constant,
                sample.range_m,
                float(np.clip(sample.std_m, 0.05, 0.30)),
                bias_index[sample.from_id],
                bias_index[sample.to_id],
                sample.anchor_id is not None,
            )
        )
    if len(factors) < sample_count:
        raise ValueError("MILUV UWB batch has insufficient range factors")

    position_count = len(agents) * sample_count
    odometry_reference_positions = np.vstack(
        [baseline[agent][index][:3, 3] for agent in agents for index in range(sample_count)]
    )
    position_priors = np.vstack(
        [
            (
                baseline[agent][0][:3, 3]
                if position_prior_m_by_agent is None
                else position_prior_m_by_agent[agent]
            )
            for agent in agents
        ]
    )
    if position_priors.shape != (len(agents), 3) or not np.all(np.isfinite(position_priors)):
        raise ValueError("MILUV UWB batch position priors must be finite 3-vectors")
    positions = odometry_reference_positions.copy()
    for agent_offset in range(len(agents)):
        start = agent_offset * sample_count
        stop = start + sample_count
        positions[start:stop] += position_priors[agent_offset] - positions[start]
    bias_priors = np.asarray(
        [
            (0.0 if transceiver_bias_prior_m is None else transceiver_bias_prior_m[transceiver_id])
            for transceiver_id in transceiver_ids
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(bias_priors)):
        raise ValueError("MILUV UWB batch bias priors must be finite")
    biases = bias_priors.copy()
    position_dimension = position_count * 3
    dimension = position_dimension + len(biases)
    converged = False
    iterations = 0
    final_residuals = np.asarray([], dtype=np.float64)
    final_downweighted = 0

    for iteration in range(1, maximum_iterations + 1):
        hessian = np.eye(dimension, dtype=np.float64) * 1e-6
        gradient = np.zeros(dimension, dtype=np.float64)
        for agent_offset in range(len(agents)):
            node = agent_offset * sample_count
            residual = positions[node] - position_priors[agent_offset]
            block = slice(node * 3, node * 3 + 3)
            hessian[block, block] += np.eye(3) * initial_position_information
            gradient[block] += initial_position_information * residual
        bias_block = slice(position_dimension, dimension)
        hessian[bias_block, bias_block] += np.eye(len(biases)) * bias_information
        gradient[bias_block] += bias_information * (biases - bias_priors)

        for agent_offset in range(len(agents)):
            for index in range(1, sample_count):
                source = agent_offset * sample_count + index - 1
                destination = source + 1
                measured_delta = (
                    odometry_reference_positions[destination] - odometry_reference_positions[source]
                )
                residual = positions[destination] - positions[source] - measured_delta
                source_block = slice(source * 3, source * 3 + 3)
                destination_block = slice(
                    destination * 3,
                    destination * 3 + 3,
                )
                information = np.eye(3) * odometry_information
                hessian[source_block, source_block] += information
                hessian[destination_block, destination_block] += information
                hessian[source_block, destination_block] -= information
                hessian[destination_block, source_block] -= information
                gradient[source_block] -= odometry_information * residual
                gradient[destination_block] += odometry_information * residual

        residual_values: list[float] = []
        downweighted = 0
        for factor in factors:
            difference = factor.constant_m.copy()
            for node, coefficient in factor.position_terms:
                difference += coefficient * positions[node]
            predicted_range = float(np.linalg.norm(difference))
            if predicted_range <= np.finfo(np.float64).eps:
                continue
            residual = (
                predicted_range
                + biases[factor.from_bias_index]
                + biases[factor.to_bias_index]
                - factor.range_m
            )
            normalized = abs(residual) / factor.sigma_m
            robust_scale = min(
                1.0,
                huber_sigma / max(normalized, 1e-12),
            )
            if robust_scale < 1.0:
                downweighted += 1
            weight = robust_scale / factor.sigma_m**2
            direction = difference / predicted_range
            jacobian: list[tuple[int, float]] = []
            for node, coefficient in factor.position_terms:
                jacobian.extend(
                    (
                        node * 3 + axis,
                        coefficient * direction[axis],
                    )
                    for axis in range(3)
                )
            jacobian.extend(
                (
                    (position_dimension + factor.from_bias_index, 1.0),
                    (position_dimension + factor.to_bias_index, 1.0),
                )
            )
            jacobian_indices = np.fromiter(
                (item[0] for item in jacobian),
                dtype=np.int64,
            )
            jacobian_values = np.fromiter(
                (item[1] for item in jacobian),
                dtype=np.float64,
            )
            gradient[jacobian_indices] += weight * jacobian_values * residual
            hessian[np.ix_(jacobian_indices, jacobian_indices)] += weight * np.outer(
                jacobian_values, jacobian_values
            )
            residual_values.append(residual)

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise ValueError("MILUV UWB batch normal equations are singular") from error
        position_step = step[:position_dimension].reshape(position_count, 3)
        positions -= position_step
        biases -= step[position_dimension:]
        iterations = iteration
        final_residuals = np.asarray(residual_values, dtype=np.float64)
        final_downweighted = downweighted
        if (
            max(
                float(np.max(np.linalg.norm(position_step, axis=1))),
                float(np.max(np.abs(step[position_dimension:]))),
            )
            <= convergence_tolerance_m
        ):
            converged = True
            break

    optimized: PoseSeries = {}
    for agent_offset, agent in enumerate(agents):
        agent_poses = []
        for index in range(sample_count):
            pose = baseline[agent][index].copy()
            pose[:3, 3] = positions[agent_offset * sample_count + index]
            agent_poses.append(pose)
        optimized[agent] = tuple(agent_poses)
    return UwbBatchResult(
        optimized,
        converged,
        iterations,
        len(factors),
        sum(factor.anchor_range for factor in factors),
        sum(not factor.anchor_range for factor in factors),
        final_downweighted,
        float(np.sqrt(np.mean(np.square(final_residuals)))),
        float(np.percentile(np.abs(final_residuals), 95)),
        {
            transceiver_id: float(biases[index])
            for index, transceiver_id in enumerate(transceiver_ids)
        },
        (perf_counter_ns() - start_ns) / 1e6,
    )


def fixed_lag_rationalize_uwb_positions(
    agents: tuple[str, ...],
    sampled: dict[str, tuple[PoseSample, ...]],
    baseline: PoseSeries,
    ranges: tuple[UwbRangeSample, ...],
    world_origin_m: npt.NDArray[np.float64],
    *,
    lag_samples: int = 9,
    solve_interval_samples: int = 1,
    odometry_information: float = 1000.0,
    position_prior_information: float = 1000.0,
    bias_prior_information: float = 2000.0,
    huber_sigma: float = 0.5,
    maximum_iterations: int = 6,
    convergence_tolerance_m: float = 5e-4,
    maximum_range_factors_per_solve: int = 2_000,
) -> UwbFixedLagResult:
    """Run a delayed-free causal fixed-lag UWB position and bias rationalizer."""
    if (
        set(agents) != set(sampled)
        or set(agents) != set(baseline)
        or lag_samples < 2
        or solve_interval_samples <= 0
    ):
        raise ValueError("MILUV UWB fixed-lag agents and bounds are invalid")
    sample_count = len(sampled[agents[0]])
    if sample_count < 3 or any(
        len(sampled[agent]) != sample_count or len(baseline[agent]) != sample_count
        for agent in agents
    ):
        raise ValueError("MILUV UWB fixed-lag pose series must have equal lengths")

    optimized: PoseSeries = {
        agent: tuple(pose.copy() for pose in baseline[agent]) for agent in agents
    }
    mutable_optimized = {agent: list(optimized[agent]) for agent in agents}
    alignment_m = {agent: np.zeros(3, dtype=np.float64) for agent in agents}
    transceiver_ids = tuple(sorted((*ANCHOR_POSITIONS_M, *TAG_MOMENT_ARMS_M)))
    bias_prior = dict.fromkeys(transceiver_ids, 0.0)
    solve_indices = tuple(
        index
        for index in range(1, sample_count)
        if index % solve_interval_samples == 0 or index == sample_count - 1
    )
    latencies_ms: list[float] = []
    converged_solve_count = 0
    range_factor_count_total = 0
    downweighted_range_count_total = 0

    for index in range(1, sample_count):
        for agent in agents:
            pose = baseline[agent][index].copy()
            pose[:3, 3] += alignment_m[agent]
            mutable_optimized[agent][index] = pose
        if index not in solve_indices:
            continue

        start_index = max(0, index - lag_samples + 1)
        window_sampled = {agent: sampled[agent][start_index : index + 1] for agent in agents}
        window_baseline = {agent: baseline[agent][start_index : index + 1] for agent in agents}
        position_priors = {
            agent: mutable_optimized[agent][start_index][:3, 3].copy() for agent in agents
        }
        result = batch_rationalize_uwb_positions(
            agents,
            window_sampled,
            window_baseline,
            ranges,
            world_origin_m,
            odometry_information=odometry_information,
            initial_position_information=position_prior_information,
            bias_information=bias_prior_information,
            huber_sigma=huber_sigma,
            maximum_iterations=maximum_iterations,
            convergence_tolerance_m=convergence_tolerance_m,
            maximum_range_factors=maximum_range_factors_per_solve,
            position_prior_m_by_agent=position_priors,
            transceiver_bias_prior_m=bias_prior,
        )
        if result.converged:
            converged_solve_count += 1
        range_factor_count_total += result.range_factor_count
        downweighted_range_count_total += result.downweighted_range_count
        latencies_ms.append(result.optimization_latency_ms)
        bias_prior = result.transceiver_bias_m
        for agent in agents:
            current_pose = result.optimized[agent][-1]
            mutable_optimized[agent][index] = current_pose
            alignment_m[agent] = current_pose[:3, 3] - baseline[agent][index][:3, 3]

    latency_values = np.asarray(latencies_ms, dtype=np.float64)
    return UwbFixedLagResult(
        {agent: tuple(mutable_optimized[agent]) for agent in agents},
        lag_samples,
        solve_interval_samples,
        solve_indices,
        len(solve_indices),
        converged_solve_count,
        range_factor_count_total,
        downweighted_range_count_total,
        bias_prior,
        float(np.sum(latency_values)),
        float(np.percentile(latency_values, 95)),
        float(np.max(latency_values)),
    )
