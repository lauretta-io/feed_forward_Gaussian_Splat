from __future__ import annotations

import unittest

from ariadne.backends import summarize_vio_replicates


def _metrics(ate_m: float, *, poses: int = 100) -> dict[str, float | int]:
    return {
        "ate_rmse_m": ate_m,
        "sim3_ate_rmse_m": ate_m * 0.8,
        "rpe_rmse_m": ate_m * 0.1,
        "final_drift_m": ate_m * 1.5,
        "trajectory_pose_count": poses,
        "lost_frame_count": 0,
        "map_reset_count": 0,
        "metric_scale_correction_to_truth": 1.0,
        "event_triggered_messages_per_minute_for_0_1m": 20.0,
        "event_triggered_min_interval_seconds": 0.2,
        "event_triggered_peak_corrections_per_second": 3,
        "causal_sim3_target_reachable_with_tested_cadences": 1,
        "causal_sim3_load_selected_ate_m": 0.08,
        "causal_sim3_load_selected_correction_messages_per_minute": 18.0,
        "causal_sim3_load_selected_anchor_messages_per_minute": 300.0,
        "causal_sim3_load_selected_correction_threshold_m": 0.15,
        "causal_sim3_load_selected_correction_interval_min_seconds": 0.2,
        "causal_sim3_load_selected_correction_interval_p95_seconds": 2.0,
        "causal_sim3_load_selected_correction_interval_max_seconds": 4.0,
        "causal_sim3_load_selected_correction_burst_per_second_max": 2,
        "causal_sim3_reference_target_met": 1,
        "causal_sim3_reference_ate_m": 0.085,
        "causal_sim3_reference_correction_messages_per_minute": 19.0,
        "causal_sim3_reference_anchor_messages_per_minute": 300.0,
        "causal_sim3_reference_correction_threshold_m": 0.17,
        "causal_sim3_reference_correction_interval_min_seconds": 0.2,
        "causal_sim3_reference_correction_interval_p95_seconds": 2.2,
        "causal_sim3_reference_correction_interval_max_seconds": 4.5,
        "causal_sim3_reference_correction_burst_per_second_max": 3,
        "causal_sim3_low_ingress_target_met": 1,
        "causal_sim3_low_ingress_ate_m": 0.09,
        "causal_sim3_low_ingress_correction_messages_per_minute": 22.0,
        "causal_sim3_low_ingress_anchor_messages_per_minute": 150.0,
        "causal_sim3_low_ingress_anchor_cadence_seconds": 0.2,
        "causal_sim3_low_ingress_correction_threshold_m": 0.15,
        "causal_sim3_low_ingress_correction_interval_min_seconds": 0.2,
        "causal_sim3_low_ingress_correction_interval_p95_seconds": 2.5,
        "causal_sim3_low_ingress_correction_interval_max_seconds": 5.0,
        "causal_sim3_low_ingress_correction_burst_per_second_max": 2,
        "causal_native_rtk_sim3_target_met": 0,
        "causal_native_rtk_se3_ate_m": 0.30,
        "causal_native_rtk_sim3_ate_m": 0.20,
        "causal_native_rtk_anchor_messages_per_minute": 60.0,
        "causal_native_rtk_sim3_correction_messages_per_minute": 58.0,
        "causal_native_rtk_sim3_correction_interval_min_seconds": 1.0,
        "causal_native_rtk_sim3_correction_interval_p95_seconds": 1.1,
        "causal_native_rtk_sim3_correction_interval_max_seconds": 1.2,
        "causal_native_rtk_sim3_correction_burst_per_second_max": 1,
        "causal_native_rtk_uses_interpolated_anchors": 0,
        "causal_native_rtk_claim_eligible": 0,
        "causal_segment_hold_native_rtk_sim3_target_met": 0,
        "causal_segment_hold_native_rtk_sim3_ate_m": 0.17,
        "causal_segment_hold_native_rtk_pose_coverage_fraction": 0.98,
        "causal_segment_hold_native_rtk_prediction_updates_per_minute": 59.5,
        "causal_segment_hold_native_rtk_scale_p05": 0.7,
        "causal_segment_hold_native_rtk_scale_p95": 1.3,
        "causal_segment_hold_native_rtk_scale_plausible_fraction": 1.0,
        "causal_segment_hold_native_rtk_live_pose_capable": 1,
        "causal_segment_hold_native_rtk_claim_eligible": 0,
        "causal_segment_hold_native_rtk_horizon_0_1s_ate_m": 0.01,
        "causal_segment_hold_native_rtk_horizon_0_2s_ate_m": 0.05,
        "causal_segment_hold_native_rtk_horizon_0_5s_ate_m": 0.11,
        "causal_segment_hold_native_rtk_horizon_1s_ate_m": 0.17,
        "causal_segment_hold_native_rtk_target_horizon_reachable": 1,
        "causal_segment_hold_native_rtk_maximum_target_horizon_seconds": 0.2,
        "causal_segment_hold_native_rtk_minimum_observation_rate_per_minute": 300.0,
        "fixed_lag_native_rtk_sim3_target_met": 1,
        "fixed_lag_native_rtk_se3_ate_m": 0.15,
        "fixed_lag_native_rtk_sim3_ate_m": 0.08,
        "fixed_lag_native_rtk_pose_coverage_fraction": 0.99,
        "fixed_lag_native_rtk_finalization_updates_per_minute": 59.5,
        "fixed_lag_native_rtk_latency_mean_seconds": 0.54,
        "fixed_lag_native_rtk_latency_p95_seconds": 0.99,
        "fixed_lag_native_rtk_latency_max_seconds": 1.0,
        "fixed_lag_native_rtk_scale_min": 0.6,
        "fixed_lag_native_rtk_scale_p05": 0.7,
        "fixed_lag_native_rtk_scale_p95": 1.3,
        "fixed_lag_native_rtk_scale_max": 1.9,
        "fixed_lag_native_rtk_scale_plausible_fraction": 1.0,
        "fixed_lag_native_rtk_live_pose_capable": 0,
        "fixed_lag_native_rtk_claim_eligible": 0,
        "adaptive_fixed_lag_native_rtk_sim3_target_met": 1,
        "adaptive_fixed_lag_native_rtk_se3_ate_m": 0.16,
        "adaptive_fixed_lag_native_rtk_sim3_ate_m": 0.095,
        "adaptive_fixed_lag_native_rtk_finalization_updates_per_minute": 38.0,
        "adaptive_fixed_lag_native_rtk_update_reduction_percent": 36.0,
        "adaptive_fixed_lag_native_rtk_latency_mean_seconds": 0.9,
        "adaptive_fixed_lag_native_rtk_latency_p95_seconds": 1.9,
        "adaptive_fixed_lag_native_rtk_latency_max_seconds": 2.0,
        "adaptive_fixed_lag_native_rtk_scale_p05": 0.8,
        "adaptive_fixed_lag_native_rtk_scale_p95": 1.3,
        "adaptive_fixed_lag_native_rtk_scale_plausible_fraction": 1.0,
        "adaptive_fixed_lag_native_rtk_live_pose_capable": 0,
        "adaptive_fixed_lag_native_rtk_claim_eligible": 0,
        "tracking_healthy": 1,
    }


class VioReproducibilityTest(unittest.TestCase):
    def test_stable_target_passing_replicates_enable_claim(self) -> None:
        summary = summarize_vio_replicates((_metrics(0.05), _metrics(0.052), _metrics(0.048)))

        self.assertEqual(summary["trajectory_reproducible"], 1)
        self.assertEqual(summary["global_pose_claim_eligible"], 1)
        self.assertEqual(summary["target_pass_count"], 3)
        self.assertEqual(summary["event_correction_min_interval_seconds_min"], 0.2)
        self.assertEqual(summary["event_correction_peak_per_second_max"], 3)
        self.assertEqual(summary["causal_sim3_load_target_pass_count"], 3)
        self.assertEqual(
            summary["causal_sim3_load_correction_messages_per_minute_max"],
            18.0,
        )
        self.assertEqual(
            summary["causal_sim3_load_correction_burst_per_second_max"],
            2,
        )
        self.assertEqual(summary["causal_sim3_reference_target_pass_count"], 3)
        self.assertEqual(
            summary["causal_sim3_reference_correction_messages_per_minute_max"],
            19.0,
        )
        self.assertEqual(summary["causal_sim3_low_ingress_target_pass_count"], 3)
        self.assertEqual(
            summary[
                "causal_sim3_low_ingress_anchor_messages_per_minute_max"
            ],
            150.0,
        )
        self.assertEqual(
            summary["causal_sim3_low_ingress_anchor_cadence_seconds"],
            0.2,
        )
        self.assertEqual(summary["causal_native_rtk_target_pass_count"], 0)
        self.assertEqual(summary["causal_native_rtk_sim3_ate_m_max"], 0.2)
        self.assertEqual(
            summary["causal_native_rtk_anchor_messages_per_minute_max"],
            60.0,
        )
        self.assertEqual(
            summary["causal_native_rtk_correction_burst_per_second_max"],
            1,
        )
        self.assertEqual(
            summary["causal_native_rtk_correction_interval_min_seconds_min"],
            1.0,
        )
        self.assertEqual(summary["causal_native_rtk_interpolated_anchor_run_count"], 0)
        self.assertEqual(summary["causal_native_rtk_claim_eligible_count"], 0)
        self.assertEqual(
            summary["causal_segment_hold_native_rtk_target_pass_count"],
            0,
        )
        self.assertEqual(
            summary["causal_segment_hold_native_rtk_sim3_ate_m_max"],
            0.17,
        )
        self.assertEqual(
            summary["causal_segment_hold_native_rtk_live_pose_capable_count"],
            3,
        )
        self.assertEqual(
            summary["causal_segment_hold_native_rtk_claim_eligible_count"],
            0,
        )
        self.assertEqual(
            summary[
                "causal_segment_hold_native_rtk_target_horizon_reachable_count"
            ],
            3,
        )
        self.assertEqual(
            summary[
                "causal_segment_hold_native_rtk_maximum_target_horizon_seconds_min"
            ],
            0.2,
        )
        self.assertEqual(
            summary[
                "causal_segment_hold_native_rtk_minimum_observation_rate_per_minute_max"
            ],
            300.0,
        )
        self.assertEqual(
            summary[
                "causal_segment_hold_native_rtk_horizon_0_5s_ate_m_max"
            ],
            0.11,
        )
        self.assertEqual(summary["fixed_lag_native_rtk_target_pass_count"], 3)
        self.assertEqual(summary["fixed_lag_native_rtk_sim3_ate_m_max"], 0.08)
        self.assertEqual(
            summary["fixed_lag_native_rtk_latency_p95_seconds_max"],
            0.99,
        )
        self.assertEqual(
            summary["fixed_lag_native_rtk_scale_plausible_fraction_min"],
            1.0,
        )
        self.assertEqual(summary["fixed_lag_native_rtk_live_pose_capable_count"], 0)
        self.assertEqual(summary["fixed_lag_native_rtk_claim_eligible_count"], 0)
        self.assertEqual(
            summary["adaptive_fixed_lag_native_rtk_target_pass_count"],
            3,
        )
        self.assertEqual(
            summary["adaptive_fixed_lag_native_rtk_updates_per_minute_max"],
            38.0,
        )
        self.assertEqual(
            summary[
                "adaptive_fixed_lag_native_rtk_scale_plausible_fraction_min"
            ],
            1.0,
        )
        self.assertEqual(
            summary["adaptive_fixed_lag_native_rtk_live_pose_capable_count"],
            0,
        )

    def test_unstable_replicates_reject_claim(self) -> None:
        summary = summarize_vio_replicates(
            (_metrics(1.0, poses=100), _metrics(4.0, poses=70), _metrics(8.0, poses=130))
        )

        self.assertEqual(summary["trajectory_reproducible"], 0)
        self.assertEqual(summary["global_pose_claim_eligible"], 0)
        self.assertGreater(summary["ate_rmse_m_coefficient_of_variation"], 0.1)

    def test_at_least_three_replicates_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least three"):
            summarize_vio_replicates((_metrics(0.05), _metrics(0.05)))
