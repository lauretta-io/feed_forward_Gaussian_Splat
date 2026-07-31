from __future__ import annotations

import unittest

import cv2
import numpy as np

from ariadne.backends import evaluate_stereo_disparity_direction


class StereoDiagnosticsTest(unittest.TestCase):
    def test_signed_disparity_detects_camera_order(self) -> None:
        generator = np.random.default_rng(7)
        left = generator.integers(0, 256, size=(240, 320), dtype=np.uint8)
        left = cv2.GaussianBlur(left, (3, 3), 0)
        transform = np.asarray([[1.0, 0.0, -12.0], [0.0, 1.0, 0.0]])
        right = cv2.warpAffine(left, transform, (320, 240))

        forward = evaluate_stereo_disparity_direction((left,), (right,))
        reversed_order = evaluate_stereo_disparity_direction((right,), (left,))

        self.assertEqual(forward["stereo_disparity_direction_healthy"], 1)
        self.assertGreater(forward["stereo_disparity_median_px"], 10.0)
        self.assertLess(forward["stereo_row_model_residual_abs_p95_px"], 1.0)
        self.assertAlmostEqual(forward["stereo_row_model_intercept_px"], 0.0, delta=0.5)
        self.assertEqual(reversed_order["stereo_disparity_direction_healthy"], 0)
        self.assertLess(reversed_order["stereo_disparity_median_px"], -10.0)
