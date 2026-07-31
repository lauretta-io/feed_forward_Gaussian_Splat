from __future__ import annotations

import unittest

import numpy as np

from ariadne.backends import (
    S3EImuSample,
    S3ETimestampSample,
    evaluate_s3e_sensor_contract,
)


def _sensor_samples(
    *,
    right_offset_ns: int = 2_000_000,
    gyro_scale: float = 1.0,
) -> tuple[
    tuple[S3ETimestampSample, ...],
    tuple[S3ETimestampSample, ...],
    tuple[S3EImuSample, ...],
]:
    left = tuple(
        S3ETimestampSample(timestamp, timestamp)
        for timestamp in range(1_000_000_000, 2_000_000_000, 100_000_000)
    )
    right = tuple(
        S3ETimestampSample(timestamp + right_offset_ns, timestamp + right_offset_ns)
        for timestamp in range(1_000_000_000, 2_000_000_000, 100_000_000)
    )
    angular_rate = 0.2
    imu = []
    for index, timestamp in enumerate(
        range(1_000_000_000, 2_000_000_001, 10_000_000)
    ):
        angle = angular_rate * index * 0.01
        imu.append(
            S3EImuSample(
                timestamp,
                timestamp,
                np.asarray([0.0, 0.0, angular_rate / gyro_scale]),
                np.asarray([0.0, 0.0, 9.81]),
                np.asarray([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]),
            )
        )
    return left, right, tuple(imu)


class S3ESensorDiagnosticsTest(unittest.TestCase):
    def test_consistent_sensor_window_passes_contract(self) -> None:
        metrics = evaluate_s3e_sensor_contract(*_sensor_samples())

        self.assertEqual(metrics["s3e_sensor_contract_healthy"], 1)
        self.assertAlmostEqual(metrics["s3e_sensor_stereo_sync_p95_ms"], 2.0)
        self.assertAlmostEqual(metrics["s3e_sensor_imu_cadence_median_ms"], 10.0)
        self.assertAlmostEqual(metrics["s3e_sensor_acceleration_norm_median_mps2"], 9.81)
        self.assertAlmostEqual(metrics["s3e_sensor_ahrs_gyro_rate_scale"], 1.0)
        self.assertGreater(metrics["s3e_sensor_ahrs_gyro_correlation"], 0.99)
        self.assertGreater(
            metrics["s3e_sensor_gravity_direction_concentration"],
            0.99,
        )

    def test_excessive_stereo_skew_fails_contract(self) -> None:
        metrics = evaluate_s3e_sensor_contract(
            *_sensor_samples(right_offset_ns=25_000_000)
        )

        self.assertEqual(metrics["s3e_sensor_contract_healthy"], 0)
        self.assertGreater(metrics["s3e_sensor_stereo_sync_p95_ms"], 10.0)

    def test_ahrs_gyro_scale_mismatch_fails_contract(self) -> None:
        metrics = evaluate_s3e_sensor_contract(*_sensor_samples(gyro_scale=2.0))

        self.assertEqual(metrics["s3e_sensor_contract_healthy"], 0)
        self.assertGreater(metrics["s3e_sensor_ahrs_gyro_rate_scale"], 1.2)
