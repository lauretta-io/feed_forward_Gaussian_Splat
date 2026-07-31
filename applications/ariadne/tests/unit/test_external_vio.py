from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ariadne.backends import (
    OpenVinsAdapter,
    OrbSlam3Adapter,
    OrientationReference,
    TrajectoryPose,
    evaluate_local_alignment_sensitivity,
    evaluate_orientation_proxy,
    evaluate_rtk_lever_arm_sensitivity,
    evaluate_time_offset_sensitivity,
    evaluate_trajectory,
    export_euroc,
    prepare_orbslam3_s3e_settings,
    reanalyze_vio_artifacts,
)
from ariadne.backends.external_vio import (
    apply_euroc_stereo_row_correction,
    swap_euroc_stereo_files,
)
from ariadne.backends.trajectory_diagnostics import _native_position_anchors
from ariadne.common import Timestamp
from ariadne.replay import (
    GroundTruthPose,
    ImageFrame,
    ImuSample,
    ReplayBatch,
    read_ground_truth_poses,
)


class ExternalVioTest(unittest.TestCase):
    def test_s3e_stereo_file_swap_preserves_both_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "cam0"
            right = root / "cam1"
            left.mkdir()
            right.mkdir()
            (left / "123.png").write_bytes(b"left")
            (right / "123.png").write_bytes(b"right")

            swap_euroc_stereo_files(left, right, [123])

            self.assertEqual((left / "123.png").read_bytes(), b"right")
            self.assertEqual((right / "123.png").read_bytes(), b"left")

    def test_s3e_right_image_vertical_shift_moves_content_without_png_expansion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            camera = Path(directory)
            image = np.zeros((20, 30, 3), dtype=np.uint8)
            image[10, 15] = 255
            import cv2

            success, payload = cv2.imencode(".jpg", image)
            self.assertTrue(success)
            (camera / "123.png").write_bytes(bytes(payload))

            apply_euroc_stereo_row_correction(
                camera,
                [123],
                intercept_px=-4.0,
            )

            shifted = cv2.imread(str(camera / "123.png"))
            self.assertGreater(int(shifted[6, 15].max()), 100)
            self.assertLess((camera / "123.png").stat().st_size, 10_000)

    def test_evaluation_removes_rigid_frame_offset(self) -> None:
        truth = tuple(
            GroundTruthPose(
                Timestamp(index * 1_000_000_000), np.array([index, 0, 0]), np.array([0, 0, 0, 1])
            )
            for index in range(4)
        )
        estimated = tuple(
            TrajectoryPose(index * 1_000_000_000, np.array([10, index, 2]), np.array([0, 0, 0, 1]))
            for index in range(4)
        )
        metrics = evaluate_trajectory(estimated, truth)
        self.assertAlmostEqual(float(metrics["ate_rmse_m"]), 0.0, places=10)
        self.assertEqual(metrics["matched_pose_count"], 4)

    def test_evaluation_interpolates_sparse_ground_truth(self) -> None:
        def yaw_quaternion(angle_rad: float) -> np.ndarray:
            return np.array(
                [0.0, 0.0, np.sin(angle_rad / 2.0), np.cos(angle_rad / 2.0)]
            )

        truth = tuple(
            GroundTruthPose(
                Timestamp(index * 1_000_000_000),
                np.array([2.0 * index, 0, 0]),
                yaw_quaternion(0.2 * index),
            )
            for index in range(4)
        )
        estimated = tuple(
            TrajectoryPose(
                int(index * 0.5e9),
                np.array([10.0, float(index), 2.0]),
                yaw_quaternion(0.1 * index),
            )
            for index in range(7)
        )

        metrics = evaluate_trajectory(estimated, truth)

        self.assertEqual(metrics["matched_pose_count"], 7)
        self.assertAlmostEqual(float(metrics["ate_rmse_m"]), 0.0, places=10)
        self.assertAlmostEqual(float(metrics["rpe_rmse_m"]), 0.0, places=10)
        self.assertAlmostEqual(float(metrics["orientation_rmse_rad"]), 0.0, places=10)

    def test_evaluation_reports_metric_scale_bias_separately(self) -> None:
        truth = tuple(
            GroundTruthPose(
                Timestamp(index * 1_000_000_000),
                np.array([float(index), float(index % 2), 0.0]),
                np.array([0, 0, 0, 1]),
            )
            for index in range(5)
        )
        estimated = tuple(
            TrajectoryPose(
                index * 1_000_000_000,
                pose.position_m * 0.5,
                np.array([0, 0, 0, 1]),
            )
            for index, pose in enumerate(truth)
        )

        metrics = evaluate_trajectory(estimated, truth)

        self.assertGreater(float(metrics["ate_rmse_m"]), 0.1)
        self.assertAlmostEqual(
            float(metrics["metric_scale_correction_to_truth"]), 2.0, places=10
        )
        self.assertAlmostEqual(float(metrics["sim3_ate_rmse_m"]), 0.0, places=10)

    def test_event_triggered_corrections_reduce_periodic_load(self) -> None:
        truth = tuple(
            GroundTruthPose(
                Timestamp(index * 100_000_000),
                np.array([index * 0.1, 0.0, 0.0]),
                np.array([0, 0, 0, 1]),
            )
            for index in range(100)
        )
        estimated = tuple(
            TrajectoryPose(
                index * 100_000_000,
                np.array([index * 0.1, 0.02 * (index / 10) ** 2, 0.0]),
                np.array([0, 0, 0, 1]),
            )
            for index in range(100)
        )

        metrics = evaluate_trajectory(estimated, truth)

        self.assertLessEqual(float(metrics["event_triggered_corrected_ate_m"]), 0.1)
        self.assertGreater(
            float(metrics["event_triggered_rate_reduction_vs_periodic_percent"]),
            50.0,
        )

    def test_se3_trigger_catches_rotation_drift_without_translation_error(self) -> None:
        def yaw_quaternion(angle_rad: float) -> np.ndarray:
            return np.array(
                [0.0, 0.0, np.sin(angle_rad / 2.0), np.cos(angle_rad / 2.0)]
            )

        truth = tuple(
            GroundTruthPose(
                Timestamp(index * 1_000_000_000),
                np.array([float(index), float(index % 2), 0.0]),
                yaw_quaternion(0.0),
            )
            for index in range(8)
        )
        estimated = tuple(
            TrajectoryPose(
                index * 1_000_000_000,
                pose.position_m,
                yaw_quaternion(index * 0.1),
            )
            for index, pose in enumerate(truth)
        )

        metrics = evaluate_trajectory(estimated, truth)

        self.assertAlmostEqual(float(metrics["rotation_rpe_rmse_rad"]), 0.1)
        self.assertEqual(metrics["event_triggered_correction_count_for_0_1m"], 0)
        self.assertGreater(float(metrics["event_triggered_orientation_rmse_rad"]), 0.2)
        self.assertGreater(metrics["se3_event_triggered_correction_count"], 0)
        self.assertLess(float(metrics["se3_event_triggered_orientation_rmse_rad"]), 1e-6)

    def test_position_only_truth_excludes_orientation_and_se3_load(self) -> None:
        truth = tuple(
            GroundTruthPose(
                Timestamp(index * 1_000_000_000),
                np.array([float(index), float(index % 2), 0.0]),
                np.array([0, 0, 0, 1]),
                orientation_available=False,
            )
            for index in range(8)
        )
        estimated = tuple(
            TrajectoryPose(
                pose.timestamp.monotonic_ns,
                pose.position_m + np.array([0.0, 0.03 * index**2, 0.0]),
                np.array(
                    [0.0, 0.0, np.sin(index * 0.05), np.cos(index * 0.05)]
                ),
            )
            for index, pose in enumerate(truth)
        )

        metrics = evaluate_trajectory(estimated, truth)

        self.assertEqual(metrics["orientation_reference_available"], 0)
        self.assertNotIn("orientation_rmse_rad", metrics)
        self.assertNotIn("rotation_rpe_rmse_rad", metrics)
        self.assertNotIn("se3_event_triggered_messages_per_minute", metrics)
        self.assertNotIn("timing_best_orientation_offset_ms", metrics)
        self.assertGreater(
            float(metrics["ate_last_quarter_m"]),
            float(metrics["ate_first_quarter_m"]),
        )

    def test_imu_orientation_proxy_reports_non_independent_rotation_drift(self) -> None:
        def yaw_quaternion(angle_rad: float) -> np.ndarray:
            return np.array(
                [0.0, 0.0, np.sin(angle_rad / 2.0), np.cos(angle_rad / 2.0)]
            )

        truth = tuple(
            GroundTruthPose(
                Timestamp(index * 1_000_000_000),
                np.array([float(index), float(index % 2), 0.0]),
                np.array([0, 0, 0, 1]),
                orientation_available=False,
            )
            for index in range(8)
        )
        estimated = tuple(
            TrajectoryPose(
                pose.timestamp.monotonic_ns,
                pose.position_m,
                yaw_quaternion(index * 0.05),
            )
            for index, pose in enumerate(truth)
        )
        reference = tuple(
            OrientationReference(
                pose.timestamp.monotonic_ns,
                yaw_quaternion(0.0),
                covariance_available=False,
            )
            for pose in truth
        )

        metrics = evaluate_orientation_proxy(estimated, truth, reference)

        self.assertEqual(
            metrics["orientation_proxy_source"],
            "s3e_imu_ahrs_proxy",
        )
        self.assertEqual(metrics["orientation_proxy_is_ground_truth"], 0)
        self.assertEqual(metrics["orientation_proxy_independent_of_vio"], 0)
        self.assertEqual(metrics["orientation_proxy_covariance_available"], 0)
        self.assertEqual(metrics["orientation_proxy_matched_pose_count"], 8)
        self.assertAlmostEqual(
            float(metrics["orientation_proxy_rpe_rmse_rad"]),
            0.05,
        )
        independent_metrics = evaluate_orientation_proxy(
            estimated,
            truth,
            reference,
            independent_of_vio=True,
        )
        self.assertEqual(
            independent_metrics["orientation_proxy_independent_of_vio"],
            1,
        )

    def test_rtk_lever_arm_sensitivity_recovers_rotating_offset(self) -> None:
        lever_arm = np.array([0.45, -0.2, 0.0])
        estimated: list[TrajectoryPose] = []
        truth: list[GroundTruthPose] = []
        reference: list[OrientationReference] = []
        for index in range(30):
            timestamp_ns = index * 100_000_000
            yaw = index * 0.17
            quaternion = np.array(
                [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)]
            )
            rotation = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw), np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            body_position = np.array(
                [
                    index * 0.4,
                    np.sin(index * 0.31),
                    np.cos(index * 0.23) * 0.3,
                ]
            )
            estimated.append(
                TrajectoryPose(timestamp_ns, body_position, quaternion)
            )
            truth.append(
                GroundTruthPose(
                    Timestamp(timestamp_ns),
                    body_position + rotation @ lever_arm,
                    np.array([0.0, 0.0, 0.0, 1.0]),
                    orientation_available=False,
                )
            )
            reference.append(
                OrientationReference(
                    timestamp_ns,
                    quaternion,
                    covariance_available=False,
                )
            )
        estimated.append(estimated[-1])

        metrics = evaluate_rtk_lever_arm_sensitivity(
            tuple(estimated),
            tuple(truth),
            tuple(reference),
            orientation_independent_of_vio=True,
        )

        self.assertEqual(metrics["lever_arm_sensitivity_is_calibration"], 0)
        self.assertEqual(metrics["lever_arm_sensitivity_is_ground_truth"], 0)
        self.assertEqual(metrics["lever_arm_sensitivity_matched_pose_count"], 30)
        self.assertEqual(
            metrics["lever_arm_sensitivity_orientation_independent_of_vio"],
            1,
        )
        self.assertEqual(
            metrics["lever_arm_sensitivity_orientation_covariance_available"],
            0,
        )
        self.assertAlmostEqual(
            float(metrics["lever_arm_sensitivity_fitted_norm_m"]),
            float(np.linalg.norm(lever_arm)),
            places=6,
        )
        self.assertLess(
            float(metrics["lever_arm_sensitivity_adjusted_ate_m"]),
            1e-8,
        )
        self.assertLess(
            float(metrics["lever_arm_sensitivity_holdout_adjusted_ate_m"]),
            1e-7,
        )
        bounded_metrics = evaluate_rtk_lever_arm_sensitivity(
            tuple(estimated),
            tuple(truth),
            tuple(reference),
            maximum_lever_arm_m=0.1,
        )
        self.assertEqual(bounded_metrics["lever_arm_sensitivity_bound_active"], 1)
        self.assertAlmostEqual(
            float(bounded_metrics["lever_arm_sensitivity_fitted_norm_m"]),
            0.1,
        )
        self.assertGreater(
            float(bounded_metrics["lever_arm_sensitivity_adjusted_ate_m"]),
            float(metrics["lever_arm_sensitivity_adjusted_ate_m"]),
        )

    def test_local_alignment_sensitivity_separates_scale_from_rigid_drift(
        self,
    ) -> None:
        estimated: list[TrajectoryPose] = []
        truth: list[GroundTruthPose] = []
        for index in range(100):
            timestamp_ns = index * 100_000_000
            truth_position = np.array(
                [
                    index * 0.3,
                    np.sin(index * 0.4),
                    np.cos(index * 0.27),
                ]
            )
            scale = 0.7 if (index // 10) % 2 == 0 else 1.3
            estimated.append(
                TrajectoryPose(
                    timestamp_ns,
                    scale * truth_position,
                    np.array([0.0, 0.0, 0.0, 1.0]),
                )
            )
            truth.append(
                GroundTruthPose(
                    Timestamp(timestamp_ns),
                    truth_position,
                    np.array([0.0, 0.0, 0.0, 1.0]),
                    orientation_available=False,
                )
            )
        estimated.append(estimated[-1])

        metrics = evaluate_local_alignment_sensitivity(
            tuple(estimated),
            tuple(truth),
        )

        self.assertEqual(metrics["local_alignment_sensitivity_is_causal"], 0)
        self.assertEqual(
            metrics["local_alignment_sensitivity_changes_scored_ate"],
            0,
        )
        self.assertEqual(
            metrics["local_alignment_sensitivity_duplicate_pose_count"],
            1,
        )
        self.assertGreater(
            float(metrics["local_alignment_1s_rigid_ate_m"]),
            0.1,
        )
        self.assertLess(
            float(metrics["local_alignment_1s_sim3_ate_m"]),
            1e-8,
        )
        self.assertEqual(
            metrics["local_sim3_maximum_passing_interval_seconds"],
            1.0,
        )
        self.assertEqual(
            metrics["local_sim3_optimistic_anchor_messages_per_minute"],
            60.0,
        )

    def test_causal_alignment_uses_trailing_anchors_without_changing_score(
        self,
    ) -> None:
        estimated: list[TrajectoryPose] = []
        truth: list[GroundTruthPose] = []
        for index in range(100):
            timestamp_ns = index * 100_000_000
            truth_position = np.array(
                [
                    index * 0.25,
                    np.sin(index * 0.21),
                    np.cos(index * 0.13),
                ]
            )
            estimated.append(
                TrajectoryPose(
                    timestamp_ns,
                    0.8 * truth_position,
                    np.array([0.0, 0.0, 0.0, 1.0]),
                )
            )
            truth.append(
                GroundTruthPose(
                    Timestamp(timestamp_ns),
                    truth_position,
                    np.array([0.0, 0.0, 0.0, 1.0]),
                    orientation_available=False,
                )
            )

        metrics = evaluate_local_alignment_sensitivity(
            tuple(estimated),
            tuple(truth),
        )

        self.assertEqual(metrics["causal_alignment_sensitivity_is_causal"], 1)
        self.assertEqual(
            metrics["causal_alignment_sensitivity_uses_future_samples"],
            0,
        )
        self.assertEqual(
            metrics["causal_alignment_sensitivity_changes_scored_ate"],
            0,
        )
        self.assertEqual(
            metrics["causal_se3_target_reachable_with_tested_cadences"],
            0,
        )
        self.assertEqual(
            metrics["causal_sim3_target_reachable_with_tested_cadences"],
            1,
        )
        self.assertEqual(
            metrics["causal_sim3_maximum_passing_cadence_seconds"],
            0.5,
        )
        self.assertLess(
            float(metrics["causal_sim3_ate_at_selected_cadence_m"]),
            1e-8,
        )
        self.assertGreater(
            float(metrics["causal_sim3_anchor_messages_per_minute"]),
            115.0,
        )
        self.assertLess(
            float(metrics["causal_sim3_correction_jump_p95_m"]),
            1e-8,
        )
        self.assertEqual(
            metrics["causal_sim3_load_selected_correction_count"],
            1,
        )
        self.assertGreater(
            float(metrics["causal_sim3_load_selected_correction_threshold_m"]),
            0.0,
        )
        self.assertLess(
            float(
                metrics[
                    "causal_sim3_load_selected_correction_messages_per_minute"
                ]
            ),
            float(
                metrics["causal_sim3_load_selected_anchor_messages_per_minute"]
            ),
        )
        self.assertEqual(
            metrics["causal_sim3_load_selected_correction_burst_per_second_max"],
            1,
        )
        self.assertEqual(metrics["causal_sim3_reference_target_met"], 1)
        self.assertEqual(
            metrics["causal_sim3_reference_correction_threshold_m"],
            0.17,
        )
        self.assertEqual(metrics["causal_sim3_reference_correction_count"], 1)
        self.assertEqual(metrics["causal_correction_load_claim_eligible"], 0)
        self.assertEqual(metrics["causal_sim3_low_ingress_target_met"], 1)
        self.assertLess(
            float(metrics["causal_sim3_low_ingress_anchor_messages_per_minute"]),
            float(metrics["causal_sim3_reference_anchor_messages_per_minute"]),
        )

    def test_causal_native_rtk_uses_only_observed_truth_timestamps(self) -> None:
        estimated: list[TrajectoryPose] = []
        truth: list[GroundTruthPose] = []
        for index in range(301):
            timestamp_ns = index * 100_000_000
            truth_position = np.array(
                [
                    index * 0.04,
                    np.sin(index * 0.04),
                    np.cos(index * 0.03),
                ]
            )
            estimated.append(
                TrajectoryPose(
                    timestamp_ns,
                    0.8 * truth_position,
                    np.array([0.0, 0.0, 0.0, 1.0]),
                )
            )
            if index % 10 == 0:
                truth.append(
                    GroundTruthPose(
                        Timestamp(timestamp_ns),
                        truth_position,
                        np.array([0.0, 0.0, 0.0, 1.0]),
                        orientation_available=False,
                    )
                )

        metrics = evaluate_local_alignment_sensitivity(
            tuple(estimated),
            tuple(truth),
        )

        self.assertEqual(
            metrics["causal_native_rtk_anchor_source"],
            "observed_s3e_rtk_samples",
        )
        self.assertEqual(metrics["causal_native_rtk_uses_interpolated_anchors"], 0)
        self.assertEqual(metrics["causal_native_rtk_truth_position_interpolated"], 0)
        self.assertEqual(
            metrics[
                "causal_native_rtk_estimate_position_interpolated_to_observation"
            ],
            1,
        )
        self.assertEqual(metrics["causal_native_rtk_anchor_count"], len(truth))
        self.assertLess(
            float(metrics["causal_native_rtk_anchor_messages_per_minute"]),
            70.0,
        )
        self.assertLess(
            float(metrics["causal_native_rtk_anchor_messages_per_minute"]),
            float(metrics["causal_sim3_low_ingress_anchor_messages_per_minute"]),
        )
        self.assertEqual(metrics["causal_native_rtk_sim3_target_met"], 1)
        self.assertEqual(metrics["causal_native_rtk_claim_eligible"], 0)
        self.assertEqual(
            metrics[
                "causal_segment_hold_native_rtk_uses_future_pose_time_observation"
            ],
            0,
        )
        self.assertEqual(
            metrics["causal_segment_hold_native_rtk_live_pose_capable"],
            1,
        )
        self.assertEqual(
            metrics["causal_segment_hold_native_rtk_sim3_target_met"],
            1,
        )
        self.assertLessEqual(
            float(
                metrics[
                    "causal_segment_hold_native_rtk_prediction_updates_per_minute"
                ]
            ),
            float(metrics["causal_native_rtk_anchor_messages_per_minute"]),
        )
        self.assertEqual(
            metrics["causal_segment_hold_native_rtk_scale_plausible_fraction"],
            1.0,
        )
        self.assertEqual(
            metrics["causal_segment_hold_native_rtk_claim_eligible"],
            0,
        )
        self.assertEqual(
            metrics[
                "causal_segment_hold_native_rtk_target_horizon_reachable"
            ],
            1,
        )
        self.assertGreater(
            float(
                metrics[
                    "causal_segment_hold_native_rtk_maximum_target_horizon_seconds"
                ]
            ),
            0.0,
        )
        self.assertGreaterEqual(
            float(
                metrics[
                    "causal_segment_hold_native_rtk_minimum_observation_rate_per_minute"
                ]
            ),
            60.0,
        )
        self.assertEqual(metrics["fixed_lag_native_rtk_is_causal_at_emission"], 1)
        self.assertEqual(
            metrics["fixed_lag_native_rtk_uses_future_pose_time_observation"],
            1,
        )
        self.assertEqual(metrics["fixed_lag_native_rtk_live_pose_capable"], 0)
        self.assertEqual(metrics["fixed_lag_native_rtk_sim3_target_met"], 1)
        self.assertLess(
            float(metrics["fixed_lag_native_rtk_latency_max_seconds"]),
            1.0,
        )
        self.assertEqual(
            metrics["fixed_lag_native_rtk_scale_plausible_fraction"],
            1.0,
        )
        self.assertEqual(metrics["fixed_lag_native_rtk_claim_eligible"], 0)
        self.assertEqual(
            metrics["adaptive_fixed_lag_native_rtk_sim3_target_met"],
            1,
        )
        self.assertLess(
            float(
                metrics[
                    "adaptive_fixed_lag_native_rtk_finalization_updates_per_minute"
                ]
            ),
            float(metrics["fixed_lag_native_rtk_finalization_updates_per_minute"]),
        )
        self.assertGreater(
            float(metrics["adaptive_fixed_lag_native_rtk_update_reduction_percent"]),
            40.0,
        )
        self.assertLessEqual(
            float(metrics["adaptive_fixed_lag_native_rtk_latency_max_seconds"]),
            2.0,
        )
        self.assertEqual(metrics["adaptive_fixed_lag_native_rtk_claim_eligible"], 0)

    def test_native_rtk_anchor_keeps_measured_truth_and_interpolates_vio(self) -> None:
        timestamps_ns = np.array(
            [50_000_000, 150_000_000, 250_000_000],
            dtype=np.int64,
        )
        estimated = np.array(
            [[0.05, 0.0, 0.0], [0.15, 0.0, 0.0], [0.25, 0.0, 0.0]]
        )
        truth = (
            GroundTruthPose(
                Timestamp(100_000_000),
                np.array([7.0, 8.0, 9.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                orientation_available=False,
            ),
        )

        anchors = _native_position_anchors(
            timestamps_ns,
            estimated,
            truth,
            0.1,
        )

        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0].timestamp_ns, 100_000_000)
        self.assertEqual(anchors[0].application_index, 1)
        np.testing.assert_allclose(
            anchors[0].estimated_position_m,
            np.array([0.1, 0.0, 0.0]),
        )
        np.testing.assert_array_equal(
            anchors[0].truth_position_m,
            truth[0].position_m,
        )

    def test_fixed_lag_native_rtk_rejects_implausible_segment_scale(self) -> None:
        estimated = tuple(
            TrajectoryPose(
                index * 100_000_000,
                np.array([index * 0.01, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
            )
            for index in range(301)
        )
        truth = tuple(
            GroundTruthPose(
                Timestamp(index * 1_000_000_000),
                np.array([index * 1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                orientation_available=False,
            )
            for index in range(31)
        )

        metrics = evaluate_local_alignment_sensitivity(estimated, truth)

        self.assertLess(float(metrics["fixed_lag_native_rtk_sim3_ate_m"]), 1e-9)
        self.assertEqual(
            metrics["fixed_lag_native_rtk_scale_plausible_fraction"],
            0.0,
        )
        self.assertEqual(metrics["fixed_lag_native_rtk_sim3_target_met"], 0)
        self.assertEqual(
            metrics["adaptive_fixed_lag_native_rtk_sim3_target_met"],
            0,
        )
        self.assertEqual(
            metrics["causal_segment_hold_native_rtk_sim3_target_met"],
            0,
        )
        self.assertEqual(
            metrics["causal_segment_hold_native_rtk_scale_plausible_fraction"],
            0.0,
        )
        self.assertEqual(
            metrics[
                "causal_segment_hold_native_rtk_target_horizon_reachable"
            ],
            0,
        )
        self.assertEqual(
            metrics[
                "causal_segment_hold_native_rtk_maximum_target_horizon_seconds"
            ],
            0.0,
        )

    def test_ground_truth_reader_marks_position_only_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.txt"
            path.write_text(
                "1.0 10.0 20.0 30.0 0.0 0.0 0.0 1.0\n",
                encoding="utf-8",
            )
            truth = read_ground_truth_poses(path, orientation_available=False)

        self.assertEqual(len(truth), 1)
        self.assertFalse(truth[0].orientation_available)

    def test_time_offset_sweep_recovers_known_shift_on_nonlinear_motion(self) -> None:
        def yaw_quaternion(angle_rad: float) -> np.ndarray:
            return np.array(
                [0.0, 0.0, np.sin(angle_rad / 2.0), np.cos(angle_rad / 2.0)]
            )

        truth = tuple(
            GroundTruthPose(
                Timestamp(index * 100_000_000),
                np.array(
                    [
                        index * 0.1,
                        (index * 0.1) ** 2,
                        np.sin(index * 0.2),
                    ]
                ),
                yaw_quaternion((index * 0.08) ** 2),
            )
            for index in range(20)
        )
        estimated = tuple(
            TrajectoryPose(
                pose.timestamp.monotonic_ns + 100_000_000,
                pose.position_m,
                pose.quaternion_xyzw,
            )
            for pose in truth[2:-2]
        )

        metrics = evaluate_time_offset_sensitivity(
            estimated,
            truth,
            tolerance_seconds=0.15,
        )

        self.assertEqual(metrics["timing_best_ate_offset_ms"], -100)
        self.assertEqual(metrics["timing_best_orientation_offset_ms"], -100)
        self.assertGreater(float(metrics["timing_ate_improvement_percent"]), 90.0)
        self.assertEqual(metrics["timing_best_ate_at_boundary"], 0)
        self.assertEqual(metrics["timing_offset_is_dominant"], 1)

    def test_euroc_export_writes_stereo_and_imu(self) -> None:
        timestamp = Timestamp(1_000_000_000)
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        batch = ReplayBatch(
            "fixture",
            "agent",
            {
                "left": (ImageFrame(timestamp, "agent", image, 0),),
                "right": (ImageFrame(timestamp, "agent", image, 0),),
            },
            (ImuSample(timestamp, "agent", np.zeros(3), np.zeros(3)),),
            (),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = export_euroc(batch, root)
            self.assertEqual(times.read_text(encoding="utf-8"), "1000000000\n")
            self.assertTrue((root / "mav0/cam0/data/1000000000.png").is_file())
            self.assertTrue((root / "mav0/cam1/data/1000000000.png").is_file())
            self.assertIn("1000000000", (root / "mav0/imu0/data.csv").read_text())

    def test_s3e_settings_normalize_orbslam3_matrix_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "alpha.yaml"
            source.write_text(
                "\n".join(
                    (
                        "%YAML:1.0",
                        'Camera.type: "PinHole"',
                        "Camera.fx: 100.0",
                        "Camera.bf: 20.0",
                        "IMU.Frequency: 100",
                        "Tic: !!opencv-matrix",
                        "   rows: 4",
                        "   cols: 4",
                        "   dt: d",
                        "   data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]",
                        "LEFT.K: !!opencv-matrix",
                        "RIGHT.K: !!opencv-matrix",
                    )
                ),
                encoding="utf-8",
            )
            output = prepare_orbslam3_s3e_settings(
                source,
                root / "settings.yaml",
                stereo_baseline_scale=1.25,
                imu_fast_init=True,
                orb_feature_profile="high-recall",
            )
            content = output.read_text(encoding="utf-8")

        self.assertIn("Tbc: !!opencv-matrix", content)
        self.assertNotIn("Tic:", content)
        self.assertIn("dt: f", content)
        self.assertIn("ORBextractor.nFeatures: 2400", content)
        self.assertIn("ORBextractor.iniThFAST: 12", content)
        self.assertIn("ORBextractor.minThFAST: 5", content)
        self.assertIn("Camera.bf: 25", content)
        self.assertIn("IMU.fastInit: 1", content)

    def test_successful_process_without_matched_ground_truth_fails(self) -> None:
        truth = tuple(
            GroundTruthPose(
                Timestamp(index * 1_000_000_000),
                np.array([index, 0, 0]),
                np.array([0, 0, 0, 1]),
            )
            for index in range(3)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trajectory.txt").write_text(
                "\n".join(
                    f"{100 + index}.0 {index} 0 0 0 0 0 1"
                    for index in range(3)
                ),
                encoding="utf-8",
            )
            with patch("ariadne.backends.external_vio.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = ""
                run.return_value.stderr = ""
                result = OpenVinsAdapter().run(
                    bag=root / "input.bag",
                    config=root / "config.yaml",
                    truth=truth,
                    output_dir=root,
                )

        self.assertEqual(result.metrics["matched_pose_count"], 0)
        self.assertEqual(result.status, "failed")

    def test_existing_artifact_reanalysis_marks_reset_tracking_unhealthy(self) -> None:
        truth = tuple(
            GroundTruthPose(
                Timestamp(index * 1_000_000_000),
                np.array([float(index), 0.0, 0.0]),
                np.array([0, 0, 0, 1]),
            )
            for index in range(4)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "trajectory.txt"
            trajectory.write_text(
                "\n".join(
                    f"{index}.0 {index} 0 0 0 0 0 1"
                    for index in range(4)
                ),
                encoding="utf-8",
            )
            stdout = root / "stdout.log"
            stdout.write_text(
                "New Map created\nNew Map created\n7 Frames set to lost\n",
                encoding="utf-8",
            )
            stderr = root / "stderr.log"
            stderr.write_text("", encoding="utf-8")

            result = reanalyze_vio_artifacts(
                backend="orbslam3",
                trajectory_path=trajectory,
                truth=truth,
                stdout_path=stdout,
                stderr_path=stderr,
            )

        self.assertEqual(result.metrics["tracking_healthy"], 0)
        self.assertEqual(result.metrics["recommended_intelligence_action"], "relocalize")
        self.assertEqual(result.metrics["event_triggered_correction_applicable"], 0)

    def test_existing_artifact_reanalysis_rejects_implausible_metric_scale(self) -> None:
        truth = tuple(
            GroundTruthPose(
                Timestamp(index * 1_000_000_000),
                np.array([float(index), float(index % 2), 0.0]),
                np.array([0, 0, 0, 1]),
            )
            for index in range(5)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "trajectory.txt"
            trajectory.write_text(
                "\n".join(
                    f"{index}.0 {100 * index} {100 * (index % 2)} 0 0 0 0 1"
                    for index in range(5)
                ),
                encoding="utf-8",
            )
            stdout = root / "stdout.log"
            stdout.write_text("", encoding="utf-8")
            stderr = root / "stderr.log"
            stderr.write_text("", encoding="utf-8")

            result = reanalyze_vio_artifacts(
                backend="openvins",
                trajectory_path=trajectory,
                truth=truth,
                stdout_path=stdout,
                stderr_path=stderr,
            )

        self.assertEqual(result.metrics["metric_scale_plausible"], 0)
        self.assertEqual(result.metrics["geometric_divergence_detected"], 1)
        self.assertEqual(result.metrics["tracking_healthy"], 0)
        self.assertEqual(result.metrics["recommended_intelligence_action"], "relocalize")

    def test_orbslam3_stereo_mode_is_forwarded_to_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("ariadne.backends.external_vio.subprocess.run") as run:
                run.return_value.returncode = 1
                run.return_value.stdout = ""
                run.return_value.stderr = ""
                result = OrbSlam3Adapter().run_euroc(
                    sequence=root / "euroc",
                    times=root / "times.txt",
                    truth=(),
                    executable=root / "run_orbslam3.sh",
                    vocabulary=root / "ORBvoc.txt",
                    settings=root / "settings.yaml",
                    output_dir=root / "output",
                    mode="stereo",
                )

        self.assertEqual(result.command[1:3], ("--mode", "stereo"))

    def test_orbslam3_deterministic_runtime_is_forwarded_to_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("ariadne.backends.external_vio.subprocess.run") as run:
                run.return_value.returncode = 1
                run.return_value.stdout = ""
                run.return_value.stderr = ""
                result = OrbSlam3Adapter().run_euroc(
                    sequence=root / "euroc",
                    times=root / "times.txt",
                    truth=(),
                    executable=root / "run_orbslam3.sh",
                    vocabulary=root / "ORBvoc.txt",
                    settings=root / "settings.yaml",
                    output_dir=root / "output",
                    deterministic_runtime=True,
                )

        self.assertEqual(
            result.command[1:4],
            ("--deterministic-runtime", "--mode", "stereo-inertial"),
        )

    def test_orbslam3_local_mapping_sync_is_forwarded_to_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("ariadne.backends.external_vio.subprocess.run") as run:
                run.return_value.returncode = 1
                run.return_value.stdout = ""
                run.return_value.stderr = ""
                result = OrbSlam3Adapter().run_euroc(
                    sequence=root / "euroc",
                    times=root / "times.txt",
                    truth=(),
                    executable=root / "run_orbslam3.sh",
                    vocabulary=root / "ORBvoc.txt",
                    settings=root / "settings.yaml",
                    output_dir=root / "output",
                    sync_local_mapping=True,
                )

        self.assertEqual(
            result.command[1:4],
            ("--sync-local-mapping", "--mode", "stereo-inertial"),
        )


if __name__ == "__main__":
    unittest.main()
