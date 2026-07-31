"""Repeated-run stability metrics for production VIO evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import numpy.typing as npt


def _values(
    replicates: tuple[Mapping[str, object], ...],
    key: str,
) -> npt.NDArray[np.float64]:
    try:
        values = np.asarray(
            [float(cast(float | int | str, metrics[key])) for metrics in replicates]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"VIO replicate is missing numeric metric {key}") from error
    if not np.all(np.isfinite(values)):
        raise ValueError(f"VIO replicate metric {key} must be finite")
    return values


def _coefficient_of_variation(values: npt.NDArray[np.float64]) -> float:
    mean = float(np.mean(values))
    return float(np.std(values) / abs(mean)) if abs(mean) > 1e-12 else 0.0


def summarize_vio_replicates(
    replicates: tuple[Mapping[str, object], ...],
    *,
    target_ate_m: float = 0.1,
) -> dict[str, float | int]:
    """Summarize identical-input VIO replicates and gate global-pose claims."""
    if len(replicates) < 3:
        raise ValueError("VIO reproducibility requires at least three replicates")
    if not np.isfinite(target_ate_m) or target_ate_m <= 0:
        raise ValueError("VIO reproducibility target must be finite and positive")
    ate = _values(replicates, "ate_rmse_m")
    sim3 = _values(replicates, "sim3_ate_rmse_m")
    rpe = _values(replicates, "rpe_rmse_m")
    drift = _values(replicates, "final_drift_m")
    poses = _values(replicates, "trajectory_pose_count")
    lost = _values(replicates, "lost_frame_count")
    resets = _values(replicates, "map_reset_count")
    scale = _values(replicates, "metric_scale_correction_to_truth")
    correction_rate = _values(
        replicates,
        "event_triggered_messages_per_minute_for_0_1m",
    )
    correction_interval = _values(
        replicates,
        "event_triggered_min_interval_seconds",
    )
    correction_peak = _values(
        replicates,
        "event_triggered_peak_corrections_per_second",
    )
    causal_load_reachable = _values(
        replicates,
        "causal_sim3_target_reachable_with_tested_cadences",
    )
    causal_load_ate = _values(
        replicates,
        "causal_sim3_load_selected_ate_m",
    )
    causal_load_rate = _values(
        replicates,
        "causal_sim3_load_selected_correction_messages_per_minute",
    )
    causal_anchor_rate = _values(
        replicates,
        "causal_sim3_load_selected_anchor_messages_per_minute",
    )
    causal_threshold = _values(
        replicates,
        "causal_sim3_load_selected_correction_threshold_m",
    )
    causal_interval_min = _values(
        replicates,
        "causal_sim3_load_selected_correction_interval_min_seconds",
    )
    causal_interval_p95 = _values(
        replicates,
        "causal_sim3_load_selected_correction_interval_p95_seconds",
    )
    causal_interval_max = _values(
        replicates,
        "causal_sim3_load_selected_correction_interval_max_seconds",
    )
    causal_burst = _values(
        replicates,
        "causal_sim3_load_selected_correction_burst_per_second_max",
    )
    causal_reference_target_met = _values(
        replicates,
        "causal_sim3_reference_target_met",
    )
    causal_reference_ate = _values(
        replicates,
        "causal_sim3_reference_ate_m",
    )
    causal_reference_rate = _values(
        replicates,
        "causal_sim3_reference_correction_messages_per_minute",
    )
    causal_reference_anchor_rate = _values(
        replicates,
        "causal_sim3_reference_anchor_messages_per_minute",
    )
    causal_reference_threshold = _values(
        replicates,
        "causal_sim3_reference_correction_threshold_m",
    )
    causal_reference_interval_min = _values(
        replicates,
        "causal_sim3_reference_correction_interval_min_seconds",
    )
    causal_reference_interval_p95 = _values(
        replicates,
        "causal_sim3_reference_correction_interval_p95_seconds",
    )
    causal_reference_interval_max = _values(
        replicates,
        "causal_sim3_reference_correction_interval_max_seconds",
    )
    causal_reference_burst = _values(
        replicates,
        "causal_sim3_reference_correction_burst_per_second_max",
    )
    causal_low_ingress_target_met = _values(
        replicates,
        "causal_sim3_low_ingress_target_met",
    )
    causal_low_ingress_ate = _values(
        replicates,
        "causal_sim3_low_ingress_ate_m",
    )
    causal_low_ingress_rate = _values(
        replicates,
        "causal_sim3_low_ingress_correction_messages_per_minute",
    )
    causal_low_ingress_anchor_rate = _values(
        replicates,
        "causal_sim3_low_ingress_anchor_messages_per_minute",
    )
    causal_low_ingress_cadence = _values(
        replicates,
        "causal_sim3_low_ingress_anchor_cadence_seconds",
    )
    causal_low_ingress_threshold = _values(
        replicates,
        "causal_sim3_low_ingress_correction_threshold_m",
    )
    causal_low_ingress_interval_min = _values(
        replicates,
        "causal_sim3_low_ingress_correction_interval_min_seconds",
    )
    causal_low_ingress_interval_p95 = _values(
        replicates,
        "causal_sim3_low_ingress_correction_interval_p95_seconds",
    )
    causal_low_ingress_interval_max = _values(
        replicates,
        "causal_sim3_low_ingress_correction_interval_max_seconds",
    )
    causal_low_ingress_burst = _values(
        replicates,
        "causal_sim3_low_ingress_correction_burst_per_second_max",
    )
    causal_native_rtk_target_met = _values(
        replicates,
        "causal_native_rtk_sim3_target_met",
    )
    causal_native_rtk_se3_ate = _values(
        replicates,
        "causal_native_rtk_se3_ate_m",
    )
    causal_native_rtk_sim3_ate = _values(
        replicates,
        "causal_native_rtk_sim3_ate_m",
    )
    causal_native_rtk_anchor_rate = _values(
        replicates,
        "causal_native_rtk_anchor_messages_per_minute",
    )
    causal_native_rtk_correction_rate = _values(
        replicates,
        "causal_native_rtk_sim3_correction_messages_per_minute",
    )
    causal_native_rtk_interval_min = _values(
        replicates,
        "causal_native_rtk_sim3_correction_interval_min_seconds",
    )
    causal_native_rtk_interval_p95 = _values(
        replicates,
        "causal_native_rtk_sim3_correction_interval_p95_seconds",
    )
    causal_native_rtk_interval_max = _values(
        replicates,
        "causal_native_rtk_sim3_correction_interval_max_seconds",
    )
    causal_native_rtk_burst = _values(
        replicates,
        "causal_native_rtk_sim3_correction_burst_per_second_max",
    )
    causal_native_rtk_uses_interpolated_anchors = _values(
        replicates,
        "causal_native_rtk_uses_interpolated_anchors",
    )
    causal_native_rtk_claim_eligible = _values(
        replicates,
        "causal_native_rtk_claim_eligible",
    )
    causal_segment_hold_target_met = _values(
        replicates,
        "causal_segment_hold_native_rtk_sim3_target_met",
    )
    causal_segment_hold_ate = _values(
        replicates,
        "causal_segment_hold_native_rtk_sim3_ate_m",
    )
    causal_segment_hold_coverage = _values(
        replicates,
        "causal_segment_hold_native_rtk_pose_coverage_fraction",
    )
    causal_segment_hold_updates = _values(
        replicates,
        "causal_segment_hold_native_rtk_prediction_updates_per_minute",
    )
    causal_segment_hold_scale_p05 = _values(
        replicates,
        "causal_segment_hold_native_rtk_scale_p05",
    )
    causal_segment_hold_scale_p95 = _values(
        replicates,
        "causal_segment_hold_native_rtk_scale_p95",
    )
    causal_segment_hold_scale_plausible = _values(
        replicates,
        "causal_segment_hold_native_rtk_scale_plausible_fraction",
    )
    causal_segment_hold_live_pose = _values(
        replicates,
        "causal_segment_hold_native_rtk_live_pose_capable",
    )
    causal_segment_hold_claim_eligible = _values(
        replicates,
        "causal_segment_hold_native_rtk_claim_eligible",
    )
    causal_segment_hold_horizon_ate = {
        token: _values(
            replicates,
            f"causal_segment_hold_native_rtk_horizon_{token}s_ate_m",
        )
        for token in ("0_1", "0_2", "0_5", "1")
    }
    causal_segment_hold_target_horizon_reachable = _values(
        replicates,
        "causal_segment_hold_native_rtk_target_horizon_reachable",
    )
    causal_segment_hold_maximum_target_horizon = _values(
        replicates,
        "causal_segment_hold_native_rtk_maximum_target_horizon_seconds",
    )
    causal_segment_hold_minimum_observation_rate = _values(
        replicates,
        "causal_segment_hold_native_rtk_minimum_observation_rate_per_minute",
    )
    fixed_lag_native_rtk_target_met = _values(
        replicates,
        "fixed_lag_native_rtk_sim3_target_met",
    )
    fixed_lag_native_rtk_se3_ate = _values(
        replicates,
        "fixed_lag_native_rtk_se3_ate_m",
    )
    fixed_lag_native_rtk_sim3_ate = _values(
        replicates,
        "fixed_lag_native_rtk_sim3_ate_m",
    )
    fixed_lag_native_rtk_coverage = _values(
        replicates,
        "fixed_lag_native_rtk_pose_coverage_fraction",
    )
    fixed_lag_native_rtk_updates = _values(
        replicates,
        "fixed_lag_native_rtk_finalization_updates_per_minute",
    )
    fixed_lag_native_rtk_latency_mean = _values(
        replicates,
        "fixed_lag_native_rtk_latency_mean_seconds",
    )
    fixed_lag_native_rtk_latency_p95 = _values(
        replicates,
        "fixed_lag_native_rtk_latency_p95_seconds",
    )
    fixed_lag_native_rtk_latency_max = _values(
        replicates,
        "fixed_lag_native_rtk_latency_max_seconds",
    )
    fixed_lag_native_rtk_scale_min = _values(
        replicates,
        "fixed_lag_native_rtk_scale_min",
    )
    fixed_lag_native_rtk_scale_p05 = _values(
        replicates,
        "fixed_lag_native_rtk_scale_p05",
    )
    fixed_lag_native_rtk_scale_p95 = _values(
        replicates,
        "fixed_lag_native_rtk_scale_p95",
    )
    fixed_lag_native_rtk_scale_max = _values(
        replicates,
        "fixed_lag_native_rtk_scale_max",
    )
    fixed_lag_native_rtk_scale_plausible_fraction = _values(
        replicates,
        "fixed_lag_native_rtk_scale_plausible_fraction",
    )
    fixed_lag_native_rtk_live_pose_capable = _values(
        replicates,
        "fixed_lag_native_rtk_live_pose_capable",
    )
    fixed_lag_native_rtk_claim_eligible = _values(
        replicates,
        "fixed_lag_native_rtk_claim_eligible",
    )
    adaptive_fixed_lag_target_met = _values(
        replicates,
        "adaptive_fixed_lag_native_rtk_sim3_target_met",
    )
    adaptive_fixed_lag_se3_ate = _values(
        replicates,
        "adaptive_fixed_lag_native_rtk_se3_ate_m",
    )
    adaptive_fixed_lag_sim3_ate = _values(
        replicates,
        "adaptive_fixed_lag_native_rtk_sim3_ate_m",
    )
    adaptive_fixed_lag_updates = _values(
        replicates,
        "adaptive_fixed_lag_native_rtk_finalization_updates_per_minute",
    )
    adaptive_fixed_lag_update_reduction = _values(
        replicates,
        "adaptive_fixed_lag_native_rtk_update_reduction_percent",
    )
    adaptive_fixed_lag_latency_mean = _values(
        replicates,
        "adaptive_fixed_lag_native_rtk_latency_mean_seconds",
    )
    adaptive_fixed_lag_latency_p95 = _values(
        replicates,
        "adaptive_fixed_lag_native_rtk_latency_p95_seconds",
    )
    adaptive_fixed_lag_latency_max = _values(
        replicates,
        "adaptive_fixed_lag_native_rtk_latency_max_seconds",
    )
    adaptive_fixed_lag_scale_p05 = _values(
        replicates,
        "adaptive_fixed_lag_native_rtk_scale_p05",
    )
    adaptive_fixed_lag_scale_p95 = _values(
        replicates,
        "adaptive_fixed_lag_native_rtk_scale_p95",
    )
    adaptive_fixed_lag_scale_plausible_fraction = _values(
        replicates,
        "adaptive_fixed_lag_native_rtk_scale_plausible_fraction",
    )
    adaptive_fixed_lag_live_pose_capable = _values(
        replicates,
        "adaptive_fixed_lag_native_rtk_live_pose_capable",
    )
    adaptive_fixed_lag_claim_eligible = _values(
        replicates,
        "adaptive_fixed_lag_native_rtk_claim_eligible",
    )
    tracking = _values(replicates, "tracking_healthy")
    target_met = ate <= target_ate_m

    ate_cv = _coefficient_of_variation(ate)
    pose_cv = _coefficient_of_variation(poses)
    sim3_range = float(np.ptp(sim3))
    reproducible = (
        ate_cv <= 0.1
        and pose_cv <= 0.05
        and sim3_range <= target_ate_m
        and float(np.ptp(lost)) == 0.0
        and float(np.ptp(resets)) == 0.0
    )
    claim_eligible = reproducible and bool(np.all(target_met)) and bool(np.all(tracking == 1))

    summary: dict[str, float | int] = {
        "replicate_count": len(replicates),
        "target_ate_m": target_ate_m,
        "target_pass_count": int(np.count_nonzero(target_met)),
        "target_pass_fraction": float(np.mean(target_met)),
        "ate_rmse_m_min": float(np.min(ate)),
        "ate_rmse_m_median": float(np.median(ate)),
        "ate_rmse_m_max": float(np.max(ate)),
        "ate_rmse_m_range": float(np.ptp(ate)),
        "ate_rmse_m_coefficient_of_variation": ate_cv,
        "sim3_ate_rmse_m_min": float(np.min(sim3)),
        "sim3_ate_rmse_m_median": float(np.median(sim3)),
        "sim3_ate_rmse_m_max": float(np.max(sim3)),
        "sim3_ate_rmse_m_range": sim3_range,
        "rpe_rmse_m_median": float(np.median(rpe)),
        "rpe_rmse_m_max": float(np.max(rpe)),
        "final_drift_m_median": float(np.median(drift)),
        "final_drift_m_max": float(np.max(drift)),
        "trajectory_pose_count_min": int(np.min(poses)),
        "trajectory_pose_count_median": float(np.median(poses)),
        "trajectory_pose_count_max": int(np.max(poses)),
        "trajectory_pose_count_coefficient_of_variation": pose_cv,
        "lost_frame_count_min": int(np.min(lost)),
        "lost_frame_count_max": int(np.max(lost)),
        "map_reset_count_min": int(np.min(resets)),
        "map_reset_count_max": int(np.max(resets)),
        "metric_scale_correction_min": float(np.min(scale)),
        "metric_scale_correction_median": float(np.median(scale)),
        "metric_scale_correction_max": float(np.max(scale)),
        "event_correction_messages_per_minute_min": float(np.min(correction_rate)),
        "event_correction_messages_per_minute_median": float(np.median(correction_rate)),
        "event_correction_messages_per_minute_max": float(np.max(correction_rate)),
        "event_correction_min_interval_seconds_min": float(np.min(correction_interval)),
        "event_correction_peak_per_second_max": int(np.max(correction_peak)),
        "causal_sim3_load_target_pass_count": int(
            np.count_nonzero(
                (causal_load_reachable == 1) & (causal_load_ate <= target_ate_m)
            )
        ),
        "causal_sim3_load_ate_m_min": float(np.min(causal_load_ate)),
        "causal_sim3_load_ate_m_median": float(np.median(causal_load_ate)),
        "causal_sim3_load_ate_m_max": float(np.max(causal_load_ate)),
        "causal_sim3_load_correction_messages_per_minute_min": float(
            np.min(causal_load_rate)
        ),
        "causal_sim3_load_correction_messages_per_minute_median": float(
            np.median(causal_load_rate)
        ),
        "causal_sim3_load_correction_messages_per_minute_max": float(
            np.max(causal_load_rate)
        ),
        "causal_sim3_load_anchor_messages_per_minute_max": float(
            np.max(causal_anchor_rate)
        ),
        "causal_sim3_load_correction_threshold_m_min": float(
            np.min(causal_threshold)
        ),
        "causal_sim3_load_correction_threshold_m_max": float(
            np.max(causal_threshold)
        ),
        "causal_sim3_load_correction_interval_min_seconds_min": float(
            np.min(causal_interval_min)
        ),
        "causal_sim3_load_correction_interval_p95_seconds_max": float(
            np.max(causal_interval_p95)
        ),
        "causal_sim3_load_correction_interval_max_seconds_max": float(
            np.max(causal_interval_max)
        ),
        "causal_sim3_load_correction_burst_per_second_max": int(
            np.max(causal_burst)
        ),
        "causal_sim3_reference_target_pass_count": int(
            np.count_nonzero(causal_reference_target_met == 1)
        ),
        "causal_sim3_reference_ate_m_min": float(
            np.min(causal_reference_ate)
        ),
        "causal_sim3_reference_ate_m_median": float(
            np.median(causal_reference_ate)
        ),
        "causal_sim3_reference_ate_m_max": float(
            np.max(causal_reference_ate)
        ),
        "causal_sim3_reference_correction_messages_per_minute_min": float(
            np.min(causal_reference_rate)
        ),
        "causal_sim3_reference_correction_messages_per_minute_median": float(
            np.median(causal_reference_rate)
        ),
        "causal_sim3_reference_correction_messages_per_minute_max": float(
            np.max(causal_reference_rate)
        ),
        "causal_sim3_reference_anchor_messages_per_minute_max": float(
            np.max(causal_reference_anchor_rate)
        ),
        "causal_sim3_reference_correction_threshold_m": float(
            causal_reference_threshold[0]
        ),
        "causal_sim3_reference_correction_interval_min_seconds_min": float(
            np.min(causal_reference_interval_min)
        ),
        "causal_sim3_reference_correction_interval_p95_seconds_max": float(
            np.max(causal_reference_interval_p95)
        ),
        "causal_sim3_reference_correction_interval_max_seconds_max": float(
            np.max(causal_reference_interval_max)
        ),
        "causal_sim3_reference_correction_burst_per_second_max": int(
            np.max(causal_reference_burst)
        ),
        "causal_sim3_low_ingress_target_pass_count": int(
            np.count_nonzero(causal_low_ingress_target_met == 1)
        ),
        "causal_sim3_low_ingress_ate_m_max": float(
            np.max(causal_low_ingress_ate)
        ),
        "causal_sim3_low_ingress_ate_m_min": float(
            np.min(causal_low_ingress_ate)
        ),
        "causal_sim3_low_ingress_ate_m_median": float(
            np.median(causal_low_ingress_ate)
        ),
        "causal_sim3_low_ingress_correction_messages_per_minute_min": float(
            np.min(causal_low_ingress_rate)
        ),
        "causal_sim3_low_ingress_correction_messages_per_minute_median": float(
            np.median(causal_low_ingress_rate)
        ),
        "causal_sim3_low_ingress_correction_messages_per_minute_max": float(
            np.max(causal_low_ingress_rate)
        ),
        "causal_sim3_low_ingress_anchor_messages_per_minute_max": float(
            np.max(causal_low_ingress_anchor_rate)
        ),
        "causal_sim3_low_ingress_anchor_cadence_seconds": float(
            causal_low_ingress_cadence[0]
        ),
        "causal_sim3_low_ingress_correction_threshold_m": float(
            causal_low_ingress_threshold[0]
        ),
        "causal_sim3_low_ingress_correction_interval_min_seconds_min": float(
            np.min(causal_low_ingress_interval_min)
        ),
        "causal_sim3_low_ingress_correction_interval_p95_seconds_max": float(
            np.max(causal_low_ingress_interval_p95)
        ),
        "causal_sim3_low_ingress_correction_interval_max_seconds_max": float(
            np.max(causal_low_ingress_interval_max)
        ),
        "causal_sim3_low_ingress_correction_burst_per_second_max": int(
            np.max(causal_low_ingress_burst)
        ),
        "causal_native_rtk_target_pass_count": int(
            np.count_nonzero(causal_native_rtk_target_met == 1)
        ),
        "causal_native_rtk_se3_ate_m_min": float(
            np.min(causal_native_rtk_se3_ate)
        ),
        "causal_native_rtk_se3_ate_m_median": float(
            np.median(causal_native_rtk_se3_ate)
        ),
        "causal_native_rtk_se3_ate_m_max": float(
            np.max(causal_native_rtk_se3_ate)
        ),
        "causal_native_rtk_sim3_ate_m_min": float(
            np.min(causal_native_rtk_sim3_ate)
        ),
        "causal_native_rtk_sim3_ate_m_median": float(
            np.median(causal_native_rtk_sim3_ate)
        ),
        "causal_native_rtk_sim3_ate_m_max": float(
            np.max(causal_native_rtk_sim3_ate)
        ),
        "causal_native_rtk_anchor_messages_per_minute_min": float(
            np.min(causal_native_rtk_anchor_rate)
        ),
        "causal_native_rtk_anchor_messages_per_minute_max": float(
            np.max(causal_native_rtk_anchor_rate)
        ),
        "causal_native_rtk_correction_messages_per_minute_min": float(
            np.min(causal_native_rtk_correction_rate)
        ),
        "causal_native_rtk_correction_messages_per_minute_max": float(
            np.max(causal_native_rtk_correction_rate)
        ),
        "causal_native_rtk_correction_interval_min_seconds_min": float(
            np.min(causal_native_rtk_interval_min)
        ),
        "causal_native_rtk_correction_interval_p95_seconds_max": float(
            np.max(causal_native_rtk_interval_p95)
        ),
        "causal_native_rtk_correction_interval_max_seconds_max": float(
            np.max(causal_native_rtk_interval_max)
        ),
        "causal_native_rtk_correction_burst_per_second_max": int(
            np.max(causal_native_rtk_burst)
        ),
        "causal_native_rtk_interpolated_anchor_run_count": int(
            np.count_nonzero(causal_native_rtk_uses_interpolated_anchors)
        ),
        "causal_native_rtk_claim_eligible_count": int(
            np.count_nonzero(causal_native_rtk_claim_eligible)
        ),
        "causal_segment_hold_native_rtk_target_pass_count": int(
            np.count_nonzero(causal_segment_hold_target_met == 1)
        ),
        "causal_segment_hold_native_rtk_sim3_ate_m_min": float(
            np.min(causal_segment_hold_ate)
        ),
        "causal_segment_hold_native_rtk_sim3_ate_m_max": float(
            np.max(causal_segment_hold_ate)
        ),
        "causal_segment_hold_native_rtk_pose_coverage_fraction_min": float(
            np.min(causal_segment_hold_coverage)
        ),
        "causal_segment_hold_native_rtk_updates_per_minute_max": float(
            np.max(causal_segment_hold_updates)
        ),
        "causal_segment_hold_native_rtk_scale_p05_min": float(
            np.min(causal_segment_hold_scale_p05)
        ),
        "causal_segment_hold_native_rtk_scale_p95_max": float(
            np.max(causal_segment_hold_scale_p95)
        ),
        "causal_segment_hold_native_rtk_scale_plausible_fraction_min": float(
            np.min(causal_segment_hold_scale_plausible)
        ),
        "causal_segment_hold_native_rtk_live_pose_capable_count": int(
            np.count_nonzero(causal_segment_hold_live_pose)
        ),
        "causal_segment_hold_native_rtk_claim_eligible_count": int(
            np.count_nonzero(causal_segment_hold_claim_eligible)
        ),
        "causal_segment_hold_native_rtk_target_horizon_reachable_count": int(
            np.count_nonzero(causal_segment_hold_target_horizon_reachable)
        ),
        "causal_segment_hold_native_rtk_maximum_target_horizon_seconds_min": float(
            np.min(causal_segment_hold_maximum_target_horizon)
        ),
        "causal_segment_hold_native_rtk_minimum_observation_rate_per_minute_max": float(
            np.max(causal_segment_hold_minimum_observation_rate)
        ),
        "fixed_lag_native_rtk_target_pass_count": int(
            np.count_nonzero(fixed_lag_native_rtk_target_met == 1)
        ),
        "fixed_lag_native_rtk_se3_ate_m_min": float(
            np.min(fixed_lag_native_rtk_se3_ate)
        ),
        "fixed_lag_native_rtk_se3_ate_m_median": float(
            np.median(fixed_lag_native_rtk_se3_ate)
        ),
        "fixed_lag_native_rtk_se3_ate_m_max": float(
            np.max(fixed_lag_native_rtk_se3_ate)
        ),
        "fixed_lag_native_rtk_sim3_ate_m_min": float(
            np.min(fixed_lag_native_rtk_sim3_ate)
        ),
        "fixed_lag_native_rtk_sim3_ate_m_median": float(
            np.median(fixed_lag_native_rtk_sim3_ate)
        ),
        "fixed_lag_native_rtk_sim3_ate_m_max": float(
            np.max(fixed_lag_native_rtk_sim3_ate)
        ),
        "fixed_lag_native_rtk_pose_coverage_fraction_min": float(
            np.min(fixed_lag_native_rtk_coverage)
        ),
        "fixed_lag_native_rtk_finalization_updates_per_minute_max": float(
            np.max(fixed_lag_native_rtk_updates)
        ),
        "fixed_lag_native_rtk_latency_mean_seconds_max": float(
            np.max(fixed_lag_native_rtk_latency_mean)
        ),
        "fixed_lag_native_rtk_latency_p95_seconds_max": float(
            np.max(fixed_lag_native_rtk_latency_p95)
        ),
        "fixed_lag_native_rtk_latency_max_seconds_max": float(
            np.max(fixed_lag_native_rtk_latency_max)
        ),
        "fixed_lag_native_rtk_scale_min": float(
            np.min(fixed_lag_native_rtk_scale_min)
        ),
        "fixed_lag_native_rtk_scale_p05_min": float(
            np.min(fixed_lag_native_rtk_scale_p05)
        ),
        "fixed_lag_native_rtk_scale_p95_max": float(
            np.max(fixed_lag_native_rtk_scale_p95)
        ),
        "fixed_lag_native_rtk_scale_max": float(
            np.max(fixed_lag_native_rtk_scale_max)
        ),
        "fixed_lag_native_rtk_scale_plausible_fraction_min": float(
            np.min(fixed_lag_native_rtk_scale_plausible_fraction)
        ),
        "fixed_lag_native_rtk_live_pose_capable_count": int(
            np.count_nonzero(fixed_lag_native_rtk_live_pose_capable)
        ),
        "fixed_lag_native_rtk_claim_eligible_count": int(
            np.count_nonzero(fixed_lag_native_rtk_claim_eligible)
        ),
        "adaptive_fixed_lag_native_rtk_target_pass_count": int(
            np.count_nonzero(adaptive_fixed_lag_target_met == 1)
        ),
        "adaptive_fixed_lag_native_rtk_se3_ate_m_max": float(
            np.max(adaptive_fixed_lag_se3_ate)
        ),
        "adaptive_fixed_lag_native_rtk_sim3_ate_m_min": float(
            np.min(adaptive_fixed_lag_sim3_ate)
        ),
        "adaptive_fixed_lag_native_rtk_sim3_ate_m_median": float(
            np.median(adaptive_fixed_lag_sim3_ate)
        ),
        "adaptive_fixed_lag_native_rtk_sim3_ate_m_max": float(
            np.max(adaptive_fixed_lag_sim3_ate)
        ),
        "adaptive_fixed_lag_native_rtk_updates_per_minute_min": float(
            np.min(adaptive_fixed_lag_updates)
        ),
        "adaptive_fixed_lag_native_rtk_updates_per_minute_median": float(
            np.median(adaptive_fixed_lag_updates)
        ),
        "adaptive_fixed_lag_native_rtk_updates_per_minute_max": float(
            np.max(adaptive_fixed_lag_updates)
        ),
        "adaptive_fixed_lag_native_rtk_update_reduction_percent_min": float(
            np.min(adaptive_fixed_lag_update_reduction)
        ),
        "adaptive_fixed_lag_native_rtk_update_reduction_percent_max": float(
            np.max(adaptive_fixed_lag_update_reduction)
        ),
        "adaptive_fixed_lag_native_rtk_latency_mean_seconds_max": float(
            np.max(adaptive_fixed_lag_latency_mean)
        ),
        "adaptive_fixed_lag_native_rtk_latency_p95_seconds_max": float(
            np.max(adaptive_fixed_lag_latency_p95)
        ),
        "adaptive_fixed_lag_native_rtk_latency_max_seconds_max": float(
            np.max(adaptive_fixed_lag_latency_max)
        ),
        "adaptive_fixed_lag_native_rtk_scale_p05_min": float(
            np.min(adaptive_fixed_lag_scale_p05)
        ),
        "adaptive_fixed_lag_native_rtk_scale_p95_max": float(
            np.max(adaptive_fixed_lag_scale_p95)
        ),
        "adaptive_fixed_lag_native_rtk_scale_plausible_fraction_min": float(
            np.min(adaptive_fixed_lag_scale_plausible_fraction)
        ),
        "adaptive_fixed_lag_native_rtk_live_pose_capable_count": int(
            np.count_nonzero(adaptive_fixed_lag_live_pose_capable)
        ),
        "adaptive_fixed_lag_native_rtk_claim_eligible_count": int(
            np.count_nonzero(adaptive_fixed_lag_claim_eligible)
        ),
        "tracking_healthy_count": int(np.count_nonzero(tracking)),
        "trajectory_reproducible": int(reproducible),
        "global_pose_claim_eligible": int(claim_eligible),
    }
    for token, values in causal_segment_hold_horizon_ate.items():
        summary[
            f"causal_segment_hold_native_rtk_horizon_{token}s_ate_m_max"
        ] = float(np.max(values))
    return summary
