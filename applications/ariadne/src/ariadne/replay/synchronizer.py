"""Deterministic camera/IMU synchronization for offline replay."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ariadne.common import Timestamp


def _finite_vector(value: Any, length: int, name: str) -> npt.NDArray[np.float64]:
    vector = np.array(value, dtype=np.float64, copy=True)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector of length {length}")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True)
class ImageFrame:
    timestamp: Timestamp
    agent_id: str
    image: npt.NDArray[np.generic]
    frame_index: int
    visual_delta_m: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id must not be empty")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        image = np.array(self.image, copy=True)
        if image.ndim not in (2, 3) or min(image.shape[:2], default=0) == 0:
            raise ValueError("image must have non-empty HxW or HxWxC shape")
        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise ValueError("image channel count must be 1, 3, or 4")
        if not np.issubdtype(image.dtype, np.number) or not np.all(np.isfinite(image)):
            raise ValueError("image must contain finite numeric values")
        image.setflags(write=False)
        object.__setattr__(self, "image", image)
        if self.visual_delta_m is not None:
            object.__setattr__(
                self,
                "visual_delta_m",
                _finite_vector(self.visual_delta_m, 3, "visual_delta_m"),
            )


@dataclass(frozen=True)
class ImuSample:
    timestamp: Timestamp
    agent_id: str
    acceleration_mps2: npt.NDArray[np.float64]
    angular_velocity_rps: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id must not be empty")
        object.__setattr__(
            self,
            "acceleration_mps2",
            _finite_vector(self.acceleration_mps2, 3, "acceleration_mps2"),
        )
        object.__setattr__(
            self,
            "angular_velocity_rps",
            _finite_vector(self.angular_velocity_rps, 3, "angular_velocity_rps"),
        )


@dataclass(frozen=True)
class SynchronizedPacket:
    frame: ImageFrame
    imu_window: tuple[ImuSample, ...]
    sync_error_ms: float

    @property
    def agent_id(self) -> str:
        return self.frame.agent_id


@dataclass(frozen=True)
class SynchronizationResult:
    packets: tuple[SynchronizedPacket, ...]
    dropped_frames: int
    median_error_ms: float
    p95_error_ms: float


class ReplaySynchronizer:
    """Associate each image with the IMU samples since the previous accepted image."""

    def __init__(self, max_sync_error_ms: float = 10.0) -> None:
        if not np.isfinite(max_sync_error_ms) or max_sync_error_ms <= 0:
            raise ValueError("max_sync_error_ms must be finite and positive")
        self.max_sync_error_ms = max_sync_error_ms

    def synchronize(
        self, images: list[ImageFrame], imu_samples: list[ImuSample]
    ) -> SynchronizationResult:
        imu_by_agent: dict[str, list[ImuSample]] = {}
        for sample in imu_samples:
            imu_by_agent.setdefault(sample.agent_id, []).append(sample)
        for samples in imu_by_agent.values():
            samples.sort(key=lambda sample: sample.timestamp.monotonic_ns)

        previous_frame_ns: dict[str, int] = {}
        packets: list[SynchronizedPacket] = []
        errors_ms: list[float] = []
        dropped = 0
        for frame in sorted(images, key=lambda item: (item.timestamp.monotonic_ns, item.agent_id)):
            samples = imu_by_agent.get(frame.agent_id, [])
            timestamps = [sample.timestamp.monotonic_ns for sample in samples]
            if not timestamps:
                dropped += 1
                continue
            position = bisect_left(timestamps, frame.timestamp.monotonic_ns)
            candidates = [index for index in (position - 1, position) if 0 <= index < len(samples)]
            nearest_index = min(
                candidates,
                key=lambda index: abs(timestamps[index] - frame.timestamp.monotonic_ns),
            )
            error_ms = abs(timestamps[nearest_index] - frame.timestamp.monotonic_ns) / 1e6
            if error_ms > self.max_sync_error_ms:
                dropped += 1
                continue
            start_ns = previous_frame_ns.get(frame.agent_id, -1)
            window = tuple(
                sample
                for sample in samples
                if start_ns < sample.timestamp.monotonic_ns <= frame.timestamp.monotonic_ns
            )
            if not window:
                window = (samples[nearest_index],)
            packets.append(SynchronizedPacket(frame, window, error_ms))
            errors_ms.append(error_ms)
            previous_frame_ns[frame.agent_id] = frame.timestamp.monotonic_ns

        errors = np.asarray(errors_ms, dtype=np.float64)
        return SynchronizationResult(
            packets=tuple(packets),
            dropped_frames=dropped,
            median_error_ms=float(np.median(errors)) if errors.size else float("nan"),
            p95_error_ms=float(np.percentile(errors, 95)) if errors.size else float("nan"),
        )
