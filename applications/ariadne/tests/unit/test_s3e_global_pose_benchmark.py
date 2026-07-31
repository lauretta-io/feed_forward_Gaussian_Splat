import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ariadne.benchmarks import run_s3e_global_pose_benchmark
from ariadne.cli import main


class S3EGlobalPoseBenchmarkTest(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        s3e_root = root / "S3Ev1"
        calibration_root = s3e_root / "Calibration"
        playground = s3e_root / "S3E_Playground_2"
        calibration_root.mkdir(parents=True)
        playground.mkdir()
        for agent in ("alpha", "bob", "carol"):
            (calibration_root / f"{agent}.yaml").write_text(
                'Camera.type: "PinHole"\nCamera.bf: 10.0\nIMU.Frequency: 100\nTic: identity\n',
                encoding="utf-8",
            )
        offsets = {"alpha": (0.0, 0.0), "bob": (4.0, -2.0), "carol": (-3.0, 5.0)}
        for agent, (offset_x, offset_y) in offsets.items():
            rows = []
            for index in range(30):
                x = offset_x + index * 0.8
                y = offset_y + 0.04 * index * index
                rows.append(f"{1000.0 + index} {x} {y} 0 0 0 0 1")
            (playground / f"{agent}_gt.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

        connection = sqlite3.connect(playground / "S3E_Playground_2.db3")
        connection.executescript(
            """
            CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT);
            CREATE TABLE messages(
                id INTEGER PRIMARY KEY,
                topic_id INTEGER,
                timestamp INTEGER,
                data BLOB
            );
            """
        )
        topic_id = 1
        message_id = 1
        for agent in ("Alpha", "Bob", "Carol"):
            for suffix, topic_type in (
                ("imu/data", "sensor_msgs/msg/Imu"),
                ("left_camera/compressed", "sensor_msgs/msg/CompressedImage"),
            ):
                connection.execute(
                    "INSERT INTO topics VALUES (?, ?, ?)",
                    (topic_id, f"/{agent}/{suffix}", topic_type),
                )
                for sample in range(3):
                    connection.execute(
                        "INSERT INTO messages VALUES (?, ?, ?, ?)",
                        (message_id, topic_id, 1_000_000_000 + sample * 10_000_000, b"data"),
                    )
                    message_id += 1
                topic_id += 1
        connection.commit()
        connection.close()
        return s3e_root

    def test_real_geometry_regression_reduces_global_pose_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._fixture(Path(directory))
            result = run_s3e_global_pose_benchmark(17, root, sample_count=12)

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.agents, ("Alpha", "Bob", "Carol"))
        self.assertGreater(result.metrics["global_ate_improvement_percent"], 30.0)
        self.assertLessEqual(result.metrics["optimized_global_ate_m"], 0.1)
        self.assertEqual(result.metrics["target_global_ate_met"], 1)
        self.assertEqual(
            result.metrics["all_agents_global_position_target_met"],
            1,
        )
        self.assertLessEqual(
            result.metrics["maximum_agent_global_ate_m"],
            result.metrics["target_global_ate_m"],
        )
        self.assertLessEqual(
            result.metrics["optimized_global_orientation_rmse_rad"],
            result.metrics["target_global_orientation_rmse_rad"],
        )
        self.assertEqual(
            result.metrics["all_agents_global_orientation_target_met"],
            1,
        )
        self.assertEqual(result.metrics["target_global_pose_met"], 1)
        self.assertEqual(result.metrics["orientation_reference_is_controlled"], 1)
        self.assertEqual(result.metrics["s3e_orientation_ground_truth_available"], 0)
        self.assertEqual(result.metrics["position_claim_eligible"], 0)
        self.assertEqual(result.metrics["full_pose_claim_eligible"], 0)
        self.assertTrue(result.details["claim_evidence"]["uses_ground_truth_in_estimator"])
        self.assertEqual(
            result.details["orientation_reference"],
            "controlled_identity_frame",
        )
        self.assertLess(
            result.metrics["optimized_rotation_rpe_rmse_rad"],
            result.metrics["baseline_rotation_rpe_rmse_rad"],
        )
        self.assertEqual(result.metrics["graph_component_count"], 1)
        self.assertEqual(result.metrics["false_loop_rejected"], 1)
        self.assertGreater(result.metrics["selected_correction_interval_samples"], 0)
        self.assertGreater(result.metrics["selected_correction_payload_bytes_total"], 0)
        self.assertGreater(len(result.details["cadence_sweep"]), 3)
        self.assertGreater(len(result.details["correction_noise_sweep"]), 3)
        self.assertGreater(
            len(result.details["correction_rotation_noise_sweep"]),
            3,
        )
        self.assertGreaterEqual(
            len(result.details["cross_agent_cadence_sweep"]),
            3,
        )
        self.assertGreaterEqual(
            len(result.details["cross_agent_translation_noise_sweep"]),
            3,
        )
        self.assertEqual(
            result.metrics["dense_cross_agent_only_global_target_met"],
            0,
        )
        self.assertGreater(
            result.metrics["dense_cross_agent_only_global_ate_m"],
            result.metrics["target_global_ate_m"],
        )
        self.assertLess(
            result.metrics["dense_cross_agent_relative_translation_rmse_m"],
            result.metrics["baseline_cross_agent_relative_translation_rmse_m"],
        )
        self.assertGreater(
            result.metrics["dense_cross_agent_relative_improvement_percent"],
            50.0,
        )
        self.assertGreater(
            result.metrics["dense_cross_agent_factor_rate_per_minute"],
            0.0,
        )
        self.assertGreater(
            result.metrics["cross_agent_relative_rmse_at_0_2m_translation_noise_m"],
            result.metrics["cross_agent_relative_rmse_at_0_05m_translation_noise_m"],
        )
        self.assertFalse(
            result.details["vision_correction_limits"][
                "cross_agent_factors_add_absolute_global_information"
            ]
        )
        self.assertNotIn("trajectories", result.details)
        self.assertNotIn("trace", result.details["adaptive_scheduler"])
        self.assertEqual(
            result.details["data_load_policy"]["shape"],
            "aggregate_metrics_and_bounded_sweeps",
        )
        adaptive_sweep = result.details["adaptive_scheduler_demand_sweep"]
        self.assertEqual(len(adaptive_sweep), 4)
        if result.metrics["selected_correction_strategy"] == "adaptive_per_wingman":
            selected_demand = result.metrics["selected_scheduler_demand_error_m"]
            selected_row = next(
                row for row in adaptive_sweep if row["demand_target_error_m"] == selected_demand
            )
            self.assertTrue(selected_row["all_agents_target_met"])
            self.assertEqual(
                selected_row["global_correction_count"],
                result.metrics["selected_global_correction_count"],
            )
            self.assertLess(
                result.metrics["selected_global_correction_count"],
                result.metrics["fixed_global_correction_count"],
            )
            self.assertGreater(
                result.metrics["selected_correction_load_reduction_percent"],
                0.0,
            )
        else:
            self.assertEqual(
                result.metrics["selected_global_correction_count"],
                result.metrics["fixed_global_correction_count"],
            )
        self.assertEqual(
            result.details["adaptive_scheduler"]["per_agent_ate_m"].keys(),
            {"Alpha", "Bob", "Carol"},
        )
        self.assertEqual(
            result.details["claim_evidence"]["all_agents_position_target_met"],
            bool(result.metrics["all_agents_global_position_target_met"]),
        )
        self.assertTrue(result.details["adaptive_scheduler"]["target_ate_met"])
        self.assertTrue(result.details["adaptive_scheduler"]["target_pose_met"])
        self.assertEqual(
            result.details["adaptive_scheduler"]["scheduler_metrics"]["cycles"],
            result.metrics["sample_count_per_agent"] - 1,
        )
        self.assertIn("not an end-to-end visual", result.warnings[0])
        self.assertIn("real orientation score", result.warnings[0])

    def test_cli_runs_s3e_global_pose_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._fixture(Path(directory))
            report = Path(directory) / "global-pose.json"
            standard_output = io.StringIO()
            with redirect_stdout(standard_output):
                return_code = main(
                    [
                        "benchmark",
                        "--suite",
                        "s3e-global-pose",
                        "--s3e-root",
                        str(root),
                        "--output",
                        str(report),
                    ]
                )
            payload = json.loads(standard_output.getvalue())
            report_exists = report.is_file()

        self.assertEqual(return_code, 0)
        self.assertEqual(payload["dataset"], "s3e-global-pose")
        self.assertEqual(payload["status"], "passed")
        self.assertTrue(report_exists)


if __name__ == "__main__":
    unittest.main()
