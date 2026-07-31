"""Deterministic image preprocessing and quality control."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
import numpy.typing as npt

from ariadne.replay import ImageFrame

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameQuality:
    blur_score: float
    exposure_score: float
    occlusion_fraction: float
    accepted: bool


@dataclass(frozen=True)
class PreprocessedFrame:
    source: ImageFrame
    image: npt.NDArray[np.float32]
    valid_mask: npt.NDArray[np.bool_]
    original_resolution: tuple[int, int]
    processed_resolution: tuple[int, int]
    pixel_transform: npt.NDArray[np.float64]
    quality: FrameQuality
    latency_ms: float

    def __post_init__(self) -> None:
        image = np.asarray(self.image, dtype=np.float32)
        mask = np.asarray(self.valid_mask, dtype=np.bool_)
        transform = np.asarray(self.pixel_transform, dtype=np.float64)
        if image.ndim != 2 or image.shape != mask.shape or transform.shape != (3, 3):
            raise ValueError("preprocessed image, mask, or pixel transform shape is invalid")
        if not np.all(np.isfinite(image)) or not np.all(np.isfinite(transform)):
            raise ValueError("preprocessed frame must contain finite values")
        image = image.copy()
        mask = mask.copy()
        transform = transform.copy()
        image.setflags(write=False)
        mask.setflags(write=False)
        transform.setflags(write=False)
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "pixel_transform", transform)


class ImagePreprocessor:
    """CPU grayscale, resize, normalization, and quality reference path."""

    def __init__(
        self,
        *,
        width: int = 64,
        height: int = 64,
        min_blur_score: float = 0.002,
        min_exposure_score: float = 0.15,
        max_occlusion_fraction: float = 0.55,
    ) -> None:
        if width < 2 or height < 2:
            raise ValueError("processed dimensions must be at least two pixels")
        if min_blur_score < 0 or not 0 <= min_exposure_score <= 1:
            raise ValueError("quality thresholds are invalid")
        if not 0 <= max_occlusion_fraction <= 1:
            raise ValueError("max_occlusion_fraction must be between zero and one")
        self.width = width
        self.height = height
        self.min_blur_score = min_blur_score
        self.min_exposure_score = min_exposure_score
        self.max_occlusion_fraction = max_occlusion_fraction
        self.metrics = {"frames": 0, "rejected": 0, "latency_ms": 0.0}

    def process(self, frame: ImageFrame) -> PreprocessedFrame:
        start_ns = perf_counter_ns()
        image = np.asarray(frame.image, dtype=np.float64)
        if image.ndim == 3:
            image = np.mean(image[..., :3], axis=2)
        if float(np.max(image)) > 1.0:
            image = image / 255.0
        image = np.clip(image, 0.0, 1.0)
        original_height, original_width = image.shape
        y_indices = np.rint(np.linspace(0, original_height - 1, self.height)).astype(int)
        x_indices = np.rint(np.linspace(0, original_width - 1, self.width)).astype(int)
        resized = image[np.ix_(y_indices, x_indices)].astype(np.float32)
        gradient_y, gradient_x = np.gradient(resized.astype(np.float64))
        blur_score = float(np.mean(gradient_x**2 + gradient_y**2))
        mean_intensity = float(np.mean(resized))
        exposure_score = max(0.0, 1.0 - abs(mean_intensity - 0.5) * 2.0)
        occlusion_fraction = float(np.mean((resized <= 0.01) | (resized >= 0.99)))
        accepted = (
            blur_score >= self.min_blur_score
            and exposure_score >= self.min_exposure_score
            and occlusion_fraction <= self.max_occlusion_fraction
        )
        transform = np.diag(
            [
                self.width / original_width,
                self.height / original_height,
                1.0,
            ]
        )
        latency_ms = (perf_counter_ns() - start_ns) / 1e6
        self.metrics["frames"] += 1
        self.metrics["rejected"] += int(not accepted)
        self.metrics["latency_ms"] += latency_ms
        if not accepted:
            LOGGER.warning("frame_rejected agent=%s frame=%d", frame.agent_id, frame.frame_index)
        return PreprocessedFrame(
            frame,
            resized,
            np.ones(resized.shape, dtype=np.bool_),
            (original_width, original_height),
            (self.width, self.height),
            transform,
            FrameQuality(blur_score, exposure_score, occlusion_fraction, accepted),
            latency_ms,
        )
