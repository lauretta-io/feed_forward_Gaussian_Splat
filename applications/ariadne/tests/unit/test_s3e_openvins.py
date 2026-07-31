from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ariadne.backends import export_euroc_ros1_bag, prepare_openvins_s3e_config


class S3EOpenVinsTest(unittest.TestCase):
    def test_config_uses_raw_cameras_and_rectified_baseline_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "alpha.yaml"
            calibration.write_text(
                """%YAML:1.0
Camera.fx: 100
Camera.bf: 20
Camera.width: 4
Camera.height: 3
IMU.NoiseGyro: 0.01
IMU.NoiseAcc: 0.02
IMU.GyroWalk: 0.001
IMU.AccWalk: 0.002
IMU.Frequency: 100
Tic: !!opencv-matrix
  rows: 4
  cols: 4
  dt: d
  data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
LEFT.D: !!opencv-matrix
  rows: 1
  cols: 5
  dt: d
  data: [0.1, 0.2, 0.3, 0.4, 0]
LEFT.K: !!opencv-matrix
  rows: 3
  cols: 3
  dt: d
  data: [100, 0, 2, 0, 101, 1.5, 0, 0, 1]
LEFT.R: !!opencv-matrix
  rows: 3
  cols: 3
  dt: d
  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]
RIGHT.D: !!opencv-matrix
  rows: 1
  cols: 5
  dt: d
  data: [0.5, 0.6, 0.7, 0.8, 0]
RIGHT.K: !!opencv-matrix
  rows: 3
  cols: 3
  dt: d
  data: [102, 0, 2, 0, 103, 1.5, 0, 0, 1]
RIGHT.R: !!opencv-matrix
  rows: 3
  cols: 3
  dt: d
  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]
""",
                encoding="utf-8",
            )
            template = root / "template"
            template.mkdir()
            (template / "estimator_config.yaml").write_text(
                """calib_cam_extrinsics: true
calib_cam_intrinsics: true
calib_cam_timeoffset: true
gravity_mag: 9.80766
init_dyn_use: false
init_dyn_mle_max_iter: 50
init_dyn_mle_max_time: 0.05
init_imu_thresh: 0.45
track_frequency: 21.0
use_mask: true
""",
                encoding="utf-8",
            )
            estimator = prepare_openvins_s3e_config(
                calibration,
                template,
                root / "output",
            )
            estimator_content = estimator.read_text(encoding="utf-8")
            camera_content = (estimator.parent / "kalibr_imucam_chain.yaml").read_text(
                encoding="utf-8"
            )

        self.assertIn("calib_cam_extrinsics: false", estimator_content)
        self.assertIn("init_dyn_use: false", estimator_content)
        self.assertIn("init_imu_thresh: 1.0", estimator_content)
        self.assertIn("track_frequency: 10.0", estimator_content)
        self.assertIn("use_mask: false", estimator_content)
        self.assertIn("distortion_coeffs: [0.1, 0.2, 0.3, 0.4]", camera_content)
        self.assertIn("intrinsics: [102, 103, 2, 1.5]", camera_content)
        self.assertIn("- [1, 0, 0, -0.2]", camera_content)

    def test_euroc_bridge_writes_synchronized_ros1_topics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "euroc"
            for camera in ("cam0", "cam1"):
                image_dir = sequence / f"mav0/{camera}/data"
                image_dir.mkdir(parents=True)
                (image_dir / "1000000000.png").write_bytes(b"compressed")
            imu_dir = sequence / "mav0/imu0"
            imu_dir.mkdir(parents=True)
            (sequence / "times.txt").write_text("1000000000\n", encoding="utf-8")
            (imu_dir / "data.csv").write_text(
                "#timestamp,gyro,acceleration\n"
                "999000000,1,2,3,4,5,6\n"
                "1000000000,7,8,9,10,11,12\n",
                encoding="utf-8",
            )
            output = export_euroc_ros1_bag(sequence, root / "window.bag")

            from rosbags.highlevel import AnyReader

            with AnyReader([output]) as reader:
                topics = {connection.topic for connection in reader.connections}
                messages = list(reader.messages())

        self.assertEqual(
            topics,
            {"/cam0/image_raw/compressed", "/cam1/image_raw/compressed", "/imu0"},
        )
        self.assertEqual(len(messages), 4)
