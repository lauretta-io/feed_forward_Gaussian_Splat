from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from ariadne.benchmarks import run_miluv_global_pose_benchmark
from ariadne.benchmarks.global_pose_rationalization import PoseSample
from ariadne.benchmarks.miluv_uwb import (
    ANCHOR_POSITIONS_M,
    TAG_MOMENT_ARMS_M,
    UwbRangeSample,
    fixed_lag_rationalize_uwb_positions,
)
from ariadne.cli import main


class MiluvGlobalPoseBenchmarkTest(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        archive_path = root / "miluv.zip"
        fieldnames = (
            "timestamp",
            "pose.position.x",
            "pose.position.y",
            "pose.position.z",
            "pose.orientation.x",
            "pose.orientation.y",
            "pose.orientation.z",
            "pose.orientation.w",
        )
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for agent_index, agent in enumerate(("ifo001", "ifo002", "ifo003")):
                text = io.StringIO()
                writer = csv.DictWriter(text, fieldnames=fieldnames)
                writer.writeheader()
                for index in range(101):
                    yaw = 0.008 * index + 0.1 * agent_index
                    writer.writerow(
                        {
                            "timestamp": index,
                            "pose.position.x": agent_index * 2 + 0.03 * index,
                            "pose.position.y": -agent_index + 0.002 * index**2,
                            "pose.position.z": 0.2 + 0.01 * agent_index,
                            "pose.orientation.x": 0.0,
                            "pose.orientation.y": 0.0,
                            "pose.orientation.z": np.sin(yaw / 2),
                            "pose.orientation.w": np.cos(yaw / 2),
                        }
                    )
                archive.writestr(
                    f"default_3_random_0/{agent}/mocap.csv",
                    text.getvalue(),
                )
                uwb = io.StringIO()
                range_writer = csv.DictWriter(
                    uwb,
                    fieldnames=(
                        "range",
                        "from_id",
                        "to_id",
                        "std",
                        "gt_range",
                        "pair",
                        "timestamp",
                    ),
                )
                range_writer.writeheader()
                tag_ids = (10 + 10 * agent_index, 11 + 10 * agent_index)
                for index in range(101):
                    yaw = 0.008 * index + 0.1 * agent_index
                    body_position = np.asarray(
                        [
                            agent_index * 2 + 0.03 * index,
                            -agent_index + 0.002 * index**2,
                            0.2 + 0.01 * agent_index,
                        ]
                    )
                    rotation = np.asarray(
                        [
                            [np.cos(yaw), -np.sin(yaw), 0.0],
                            [np.sin(yaw), np.cos(yaw), 0.0],
                            [0.0, 0.0, 1.0],
                        ]
                    )
                    tag_id = tag_ids[index % 2]
                    tag_position = body_position + rotation @ TAG_MOMENT_ARMS_M[tag_id]
                    for anchor_id, anchor_position in ANCHOR_POSITIONS_M.items():
                        ground_truth_range = float(np.linalg.norm(tag_position - anchor_position))
                        measured_range = ground_truth_range + 0.005 * np.sin(
                            index + anchor_id + agent_index
                        )
                        range_writer.writerow(
                            {
                                "range": measured_range,
                                "from_id": tag_id,
                                "to_id": anchor_id,
                                "std": 0.08,
                                "gt_range": ground_truth_range,
                                "pair": f"({tag_id},{anchor_id})",
                                "timestamp": index,
                            }
                        )
                    if agent_index == 0:
                        inter_agent_tags: dict[int, np.ndarray] = {}
                        for peer_index, peer_tag_id in enumerate((10, 20, 30)):
                            peer_yaw = 0.008 * index + 0.1 * peer_index
                            peer_rotation = np.asarray(
                                [
                                    [np.cos(peer_yaw), -np.sin(peer_yaw), 0.0],
                                    [np.sin(peer_yaw), np.cos(peer_yaw), 0.0],
                                    [0.0, 0.0, 1.0],
                                ]
                            )
                            inter_agent_tags[peer_tag_id] = (
                                np.asarray(
                                    [
                                        peer_index * 2 + 0.03 * index,
                                        -peer_index + 0.002 * index**2,
                                        0.2 + 0.01 * peer_index,
                                    ]
                                )
                                + peer_rotation @ TAG_MOMENT_ARMS_M[peer_tag_id]
                            )
                        for pair_index, (from_id, to_id) in enumerate(
                            ((10, 20), (20, 30), (10, 30))
                        ):
                            ground_truth_range = float(
                                np.linalg.norm(inter_agent_tags[from_id] - inter_agent_tags[to_id])
                            )
                            range_writer.writerow(
                                {
                                    "range": (
                                        ground_truth_range + 0.003 * np.cos(index + pair_index)
                                    ),
                                    "from_id": from_id,
                                    "to_id": to_id,
                                    "std": 0.08,
                                    "gt_range": ground_truth_range,
                                    "pair": f"({from_id},{to_id})",
                                    "timestamp": index,
                                }
                            )
                archive.writestr(
                    f"default_3_random_0/{agent}/uwb_range.csv",
                    uwb.getvalue(),
                )
        return archive_path

    def test_real_se3_truth_rationalization_meets_global_pose_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = self._fixture(Path(directory))
            result = run_miluv_global_pose_benchmark(
                17,
                archive,
                sample_count=41,
            )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.agents, ("ifo001", "ifo002", "ifo003"))
        self.assertEqual(result.metrics["target_global_pose_met"], 1)
        self.assertLessEqual(result.metrics["optimized_global_ate_m"], 0.1)
        self.assertLessEqual(
            result.metrics["optimized_global_orientation_rmse_rad"],
            0.05,
        )
        self.assertEqual(result.metrics["false_loop_rejected"], 1)
        self.assertEqual(
            result.details["orientation_reference"],
            "miluv_motion_capture",
        )
        self.assertGreater(result.metrics["uwb_unique_range_count"], 0)
        self.assertLess(
            result.metrics["uwb_batch_global_position_ate_m"],
            result.metrics["baseline_global_ate_m"],
        )
        self.assertFalse(
            result.details["uwb_batch_rationalization"]["uses_ground_truth_in_estimator"]
        )
        self.assertEqual(
            set(
                result.details["uwb_batch_rationalization"]["post_batch_correction_load"][
                    "by_agent"
                ]
            ),
            {"ifo001", "ifo002", "ifo003"},
        )
        fixed_lag = result.details["uwb_fixed_lag_rationalization"]
        self.assertFalse(fixed_lag["uses_ground_truth_in_estimator"])
        self.assertFalse(fixed_lag["uses_future_measurements"])
        self.assertTrue(fixed_lag["correction_load"]["causal"])
        self.assertFalse(fixed_lag["claim_evidence"]["position_claim_eligible"])
        self.assertFalse(fixed_lag["claim_evidence"]["full_pose_claim_eligible"])
        self.assertEqual(result.metrics["uwb_fixed_lag_position_claim_eligible"], 0)
        self.assertEqual(result.metrics["uwb_fixed_lag_full_pose_claim_eligible"], 0)
        inventory = result.details["real_pose_factor_inventory"]
        self.assertFalse(inventory["production_local_pose_stream_available"])
        self.assertFalse(inventory["attitude_stream_available"])
        self.assertEqual(len(fixed_lag["solve_interval_sweep"]), 3)
        self.assertEqual(len(result.details["archive_members_loaded"]), 6)
        self.assertGreater(len(result.details["cadence_sweep"]), 4)
        self.assertIn("deterministic controlled", result.warnings[0])

    def test_cli_runs_miluv_global_pose_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = self._fixture(Path(directory))
            report = Path(directory) / "global-pose.json"
            standard_output = io.StringIO()
            with redirect_stdout(standard_output):
                return_code = main(
                    [
                        "benchmark",
                        "--suite",
                        "miluv-global-pose",
                        "--miluv-archive",
                        str(archive),
                        "--output",
                        str(report),
                    ]
                )
            payload = json.loads(standard_output.getvalue())
            report_exists = report.is_file()

        self.assertEqual(return_code, 0)
        self.assertEqual(payload["dataset"], "miluv-global-pose")
        self.assertEqual(payload["status"], "passed")
        self.assertTrue(report_exists)

    def test_fixed_lag_estimates_do_not_change_when_future_ranges_change(
        self,
    ) -> None:
        agents = ("ifo001", "ifo002", "ifo003")
        sampled: dict[str, tuple[PoseSample, ...]] = {}
        baseline: dict[str, tuple[np.ndarray, ...]] = {}
        ranges: list[UwbRangeSample] = []
        future_outlier_ranges: list[UwbRangeSample] = []
        for agent_index, agent in enumerate(agents):
            truth_poses = []
            baseline_poses = []
            tag_id = 10 + 10 * agent_index
            for index in range(5):
                truth_pose = np.eye(4)
                truth_pose[:3, 3] = np.asarray([0.1 * index, 0.4 * agent_index, 0.5])
                baseline_pose = truth_pose.copy()
                baseline_pose[0, 3] += 0.02 * index
                truth_poses.append(PoseSample(float(index), truth_pose))
                baseline_poses.append(baseline_pose)
                tag_position = truth_pose[:3, 3] + TAG_MOMENT_ARMS_M[tag_id]
                for anchor_id, anchor_position in ANCHOR_POSITIONS_M.items():
                    measured_range = float(np.linalg.norm(tag_position - anchor_position))
                    sample = UwbRangeSample(
                        float(index),
                        tag_id,
                        anchor_id,
                        measured_range,
                        0.08,
                        measured_range,
                    )
                    ranges.append(sample)
                    future_outlier_ranges.append(
                        UwbRangeSample(
                            sample.timestamp_s,
                            sample.from_id,
                            sample.to_id,
                            (sample.range_m + 2.0 if index == 4 else sample.range_m),
                            sample.std_m,
                            sample.ground_truth_range_m,
                        )
                    )
            sampled[agent] = tuple(truth_poses)
            baseline[agent] = tuple(baseline_poses)

        reference = fixed_lag_rationalize_uwb_positions(
            agents,
            sampled,
            baseline,
            tuple(ranges),
            np.zeros(3),
            lag_samples=3,
            maximum_iterations=3,
            maximum_range_factors_per_solve=200,
        )
        changed_future = fixed_lag_rationalize_uwb_positions(
            agents,
            sampled,
            baseline,
            tuple(future_outlier_ranges),
            np.zeros(3),
            lag_samples=3,
            maximum_iterations=3,
            maximum_range_factors_per_solve=200,
        )

        self.assertEqual(reference.solve_sample_indices, (1, 2, 3, 4))
        for agent in agents:
            for index in range(4):
                np.testing.assert_allclose(
                    reference.optimized[agent][index],
                    changed_future.optimized[agent][index],
                    atol=1e-12,
                )


if __name__ == "__main__":
    unittest.main()
