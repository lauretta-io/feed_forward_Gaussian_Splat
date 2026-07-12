import sqlite3
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from ariadne.datasets import evaluate_dataset


class DatasetEvaluationTest(unittest.TestCase):
    def test_miluv_three_agent_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "miluv.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                for agent in ("ifo001", "ifo002", "ifo003"):
                    handle.writestr(
                        f"experiment/{agent}/imu_cam.csv", "timestamp,gx\n0.0,0\n0.005,0\n"
                    )
                    handle.writestr(
                        f"experiment/{agent}/mocap.csv",
                        "timestamp,pose.position.x,pose.position.y,pose.position.z\n"
                        "0.0,0,0,0\n1.0,1,0,0\n",
                    )
                    handle.writestr(f"experiment/{agent}/uwb_range.csv", "timestamp,range\n1,1\n")
                    handle.writestr(f"experiment/{agent}/color/0.001.png", b"png")
            result = evaluate_dataset("miluv", archive)
            self.assertEqual(result.status, "passed")
            self.assertEqual(result.metrics["agent_count"], 3)
            self.assertEqual(result.metrics["image_count"], 3)

    def test_qdrone_is_partial_single_agent_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("indoor.zip", "indoor-gt.zip"):
                with zipfile.ZipFile(root / name, "w") as handle:
                    handle.writestr("flight.csv", "2,1.0,0.0\n0,1.1,100,3.0\n")
            result = evaluate_dataset("qdrone", root)
            self.assertEqual(result.status, "passed")
            self.assertEqual(result.agents, ("qdrone",))
            self.assertIn("no vision", result.warnings[0])

    def test_s3e_ros2_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bag = Path(directory) / "test.db3"
            connection = sqlite3.connect(bag)
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
                    connection.execute(
                        "INSERT INTO messages VALUES (?, ?, ?, ?)",
                        (message_id, topic_id, 1_000_000_000 + topic_id, b"data"),
                    )
                    topic_id += 1
                    message_id += 1
            connection.commit()
            connection.close()
            result = evaluate_dataset("s3e", bag)
            self.assertEqual(result.status, "passed")
            self.assertEqual(result.metrics["agent_count"], 3)
            self.assertEqual(result.metrics["vision_message_count"], 3)

    def test_d2slam_archive_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            (fixture / "drone1/cam0").mkdir(parents=True)
            (fixture / "drone1/cam0/frame.png").write_bytes(b"png")
            (fixture / "drone1/imu.csv").write_text("time,ax\n1,0\n", encoding="utf-8")
            archive = root / "d2.7z"
            subprocess.run(["7z", "a", str(archive), str(fixture)], check=True, capture_output=True)
            result = evaluate_dataset("d2slam", archive)
            self.assertEqual(result.status, "passed")
            self.assertEqual(result.metrics["named_agent_count"], 1)

    def test_simulation_is_deterministic_and_reduces_drift(self) -> None:
        first = evaluate_dataset("simulation", seed=17)
        second = evaluate_dataset("simulation", seed=17)
        self.assertEqual(first.metrics["replay_hash"], second.metrics["replay_hash"])
        self.assertGreater(first.metrics["drift_improvement_percent"], 30.0)
        self.assertGreater(first.metrics["recovery_packets"], 0)


if __name__ == "__main__":
    unittest.main()
