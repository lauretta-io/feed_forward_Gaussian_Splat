from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ariadne.backends import TrajectoryPose, evaluate_trajectory, export_euroc
from ariadne.common import Timestamp
from ariadne.replay import GroundTruthPose, ImageFrame, ImuSample, ReplayBatch


class ExternalVioTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
