"""Explicit adapters for parent repository camera conventions."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ariadne.common.types import FrameId, TransformSE3


def from_resplat_opencv_c2w(
    matrix: Any,
    *,
    camera_frame: FrameId,
    destination_frame: FrameId,
) -> TransformSE3:
    """Convert a ReSplat/OpenCV camera-to-world matrix to an explicit transform."""
    if not camera_frame.value.startswith("camera_"):
        raise ValueError("camera_frame must use the camera_<id> convention")
    return TransformSE3(camera_frame, destination_frame, matrix)


def to_resplat_opencv_c2w(transform: TransformSE3) -> npt.NDArray[np.float64]:
    """Return a copy suitable for parent code expecting an OpenCV camera-to-world matrix."""
    if not transform.source.value.startswith("camera_"):
        raise ValueError("transform source must be a camera_<id> frame")
    return np.array(transform.matrix, copy=True)
