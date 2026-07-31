import unittest

import numpy as np

from ariadne.common import Timestamp
from ariadne.models.features import (
    GradientPatchExtractor,
    SpatialPyramidEmbedder,
    benchmark_feature_pair,
)
from ariadne.models.vio import ImuDeadReckoningVio, VisualInertialComplementaryVio
from ariadne.replay import ImageFrame, ImuSample, SynchronizedPacket


class ReferenceModelTest(unittest.TestCase):
    def test_visual_measurement_limits_inertial_bias(self) -> None:
        imu_backend = ImuDeadReckoningVio()
        fused_backend = VisualInertialComplementaryVio()
        for index in range(6):
            frame = ImageFrame(
                Timestamp(index * 100_000_000),
                "one",
                np.zeros((8, 8)),
                index,
                np.array([0.1, 0.0, 0.0]) if index else np.zeros(3),
            )
            sample = ImuSample(frame.timestamp, "one", np.array([2.0, 0.0, 0.0]), np.zeros(3))
            packet = SynchronizedPacket(frame, (sample,), 0.0)
            imu_estimate = imu_backend.process(packet)
            fused_estimate = fused_backend.process(packet)
        self.assertLess(abs(fused_estimate.position_m[0] - 0.5), 0.1)
        self.assertGreater(abs(imu_estimate.position_m[0] - 0.5), 0.1)

    def test_feature_pair_separates_related_and_unrelated_images(self) -> None:
        reference = np.zeros((24, 24))
        reference[4:12, 7:15] = 1.0
        positive = np.roll(reference, 1, axis=1)
        negative = np.random.default_rng(4).random((24, 24))
        frames = [
            ImageFrame(Timestamp(index), "one", image, index)
            for index, image in enumerate((reference, positive, negative))
        ]
        result = benchmark_feature_pair(GradientPatchExtractor(), SpatialPyramidEmbedder(), *frames)
        self.assertGreater(result["semantic_separation"], 0.0)
        self.assertGreater(result["geometric_match_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
