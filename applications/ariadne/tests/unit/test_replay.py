import unittest

import numpy as np

from ariadne.common import Timestamp
from ariadne.replay import ImageFrame, ImuSample, ReplaySynchronizer


class ReplaySynchronizerTest(unittest.TestCase):
    def test_synchronizes_per_agent_windows_and_drops_bad_alignment(self) -> None:
        images = [
            ImageFrame(Timestamp(10_000_000), "one", np.zeros((4, 4)), 0),
            ImageFrame(Timestamp(20_000_000), "one", np.zeros((4, 4)), 1),
            ImageFrame(Timestamp(30_000_000), "two", np.zeros((4, 4)), 0),
        ]
        imu = [
            ImuSample(Timestamp(value), "one", np.zeros(3), np.zeros(3))
            for value in (9_000_000, 15_000_000, 20_000_000)
        ]
        result = ReplaySynchronizer(max_sync_error_ms=2.0).synchronize(images, imu)
        self.assertEqual(len(result.packets), 2)
        self.assertEqual(result.dropped_frames, 1)
        self.assertEqual(len(result.packets[1].imu_window), 2)
        self.assertEqual(result.median_error_ms, 0.5)

    def test_invalid_shapes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "image"):
            ImageFrame(Timestamp(0), "one", np.zeros((0, 4)), 0)
        with self.assertRaisesRegex(ValueError, "three-vector|length 3"):
            ImuSample(Timestamp(0), "one", np.zeros(2), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
