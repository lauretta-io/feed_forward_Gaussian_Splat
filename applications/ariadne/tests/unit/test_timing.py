import unittest

import numpy as np

from ariadne.datasets.timing import nearest_timestamp_errors_ms, summarize_errors_ms


class TimingMetricsTest(unittest.TestCase):
    def test_nearest_timestamp_errors(self) -> None:
        imu_ns = [0, 5_000_000, 10_000_000, 15_000_000]
        camera_ns = [1_000_000, 9_000_000, 14_000_000]
        errors = nearest_timestamp_errors_ms(imu_ns, camera_ns)
        np.testing.assert_array_equal(errors, [1.0, 1.0, 1.0])
        summary = summarize_errors_ms(errors)
        self.assertEqual(summary["camera_imu_sync_median_ms"], 1.0)
        self.assertEqual(summary["camera_imu_sync_p95_ms"], 1.0)

    def test_empty_streams_return_nan_metrics(self) -> None:
        summary = summarize_errors_ms(nearest_timestamp_errors_ms([], []))
        self.assertTrue(np.isnan(summary["camera_imu_sync_median_ms"]))


if __name__ == "__main__":
    unittest.main()
