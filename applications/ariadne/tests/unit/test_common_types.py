import json
import unittest

import numpy as np

from ariadne.common import (
    CameraCalibration,
    FrameId,
    ImuCalibration,
    ModelVersion,
    PoseCovariance,
    PoseEstimate,
    SensorHealth,
    SensorHealthState,
    Timestamp,
    TransformSE3,
    from_resplat_opencv_c2w,
    to_resplat_opencv_c2w,
)


class CommonTypesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = FrameId("camera_front")
        self.body = FrameId("body")
        self.local = FrameId("local_wingman_01")
        self.camera_to_body = TransformSE3.from_translation_quaternion(
            self.camera, self.body, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 4.0]
        )

    def test_transform_inverse_round_trip_below_tolerance(self) -> None:
        identity = self.camera_to_body.then(self.camera_to_body.inverse())
        np.testing.assert_allclose(identity.matrix, np.eye(4), atol=1e-12)
        self.assertEqual(identity.source, self.camera)
        self.assertEqual(identity.destination, self.camera)

    def test_transform_composition_and_frame_mismatch(self) -> None:
        body_to_local = TransformSE3.from_translation_quaternion(
            self.body, self.local, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        )
        camera_to_local = self.camera_to_body.then(body_to_local)
        np.testing.assert_allclose(camera_to_local.translation_m, [1.0, 3.0, 3.0])
        with self.assertRaisesRegex(ValueError, "frame mismatch"):
            body_to_local.then(self.camera_to_body)

    def test_quaternion_is_normalized(self) -> None:
        np.testing.assert_allclose(self.camera_to_body.quaternion_xyzw(), [0.0, 0.0, 0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "non-zero"):
            TransformSE3.from_translation_quaternion(
                self.camera, self.body, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]
            )

    def test_timestamp_ordering_and_lossless_serialization(self) -> None:
        earlier = Timestamp(10, 1_700_000_000_000_000_010)
        later = Timestamp(11, 1_700_000_000_000_000_011)
        self.assertLess(earlier, later)
        payload = json.loads(json.dumps(later.to_dict()))
        self.assertEqual(Timestamp.from_dict(payload), later)
        self.assertLess(Timestamp(11), Timestamp(11, 100))
        with self.assertRaisesRegex(ValueError, "integer"):
            Timestamp(1.5)  # type: ignore[arg-type]

    def test_covariance_validation(self) -> None:
        covariance = PoseCovariance(np.eye(6))
        with self.assertRaisesRegex(ValueError, "shape"):
            PoseCovariance(np.eye(3))
        with self.assertRaisesRegex(ValueError, "finite"):
            PoseCovariance(np.full((6, 6), np.nan))
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            PoseCovariance(-np.eye(6))
        with self.assertRaises(ValueError):
            covariance.matrix[0, 0] = 3.0

    def test_pose_serialization_round_trip(self) -> None:
        pose = PoseEstimate(Timestamp(123, 456), self.camera_to_body, PoseCovariance(np.eye(6)))
        restored = PoseEstimate.from_dict(json.loads(json.dumps(pose.to_dict())))
        self.assertEqual(restored.timestamp, pose.timestamp)
        self.assertEqual(restored.transform.source, pose.transform.source)
        np.testing.assert_array_equal(restored.transform.matrix, pose.transform.matrix)
        np.testing.assert_array_equal(restored.covariance.matrix, pose.covariance.matrix)

    def test_calibration_health_and_model_serialization(self) -> None:
        camera = CameraCalibration(self.camera, np.eye(3), 640, 480, [0.1, 0.01])
        restored_camera = CameraCalibration.from_dict(camera.to_dict())
        np.testing.assert_array_equal(restored_camera.intrinsic_matrix, camera.intrinsic_matrix)

        imu_to_body = TransformSE3.from_translation_quaternion(
            FrameId("imu"), self.body, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        )
        imu = ImuCalibration(FrameId("imu"), imu_to_body, 0.01, 0.02)
        restored_imu = ImuCalibration.from_dict(imu.to_dict())
        self.assertEqual(restored_imu.imu_frame, imu.imu_frame)
        with self.assertRaisesRegex(ValueError, "finite"):
            ImuCalibration(FrameId("imu"), imu_to_body, float("nan"), 0.02)

        health = SensorHealth("front_camera", SensorHealthState.HEALTHY, Timestamp(99), "ok")
        self.assertEqual(SensorHealth.from_dict(health.to_dict()), health)
        version = ModelVersion("detector", "1.0", "a" * 64)
        self.assertEqual(ModelVersion.from_dict(version.to_dict()), version)

    def test_resplat_opencv_c2w_round_trip(self) -> None:
        matrix = np.eye(4)
        matrix[:3, 3] = [1.0, -2.0, 3.0]
        transform = from_resplat_opencv_c2w(
            matrix, camera_frame=self.camera, destination_frame=self.local
        )
        np.testing.assert_array_equal(to_resplat_opencv_c2w(transform), matrix)

    def test_invalid_frame_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            FrameId("world")
        with self.assertRaisesRegex(ValueError, "UUID"):
            FrameId("object_not-a-uuid")


if __name__ == "__main__":
    unittest.main()
