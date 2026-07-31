"""Immutable time, frame, geometry, calibration, and health types."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from functools import total_ordering
from typing import Any
from uuid import UUID

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
_FRAME_PATTERN = re.compile(
    r"^(?:camera_[A-Za-z0-9_-]+|imu|body|local_[A-Za-z0-9_-]+|global|object_[A-Za-z0-9_-]+)$"
)


def _immutable_array(value: Any, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, order=True)
class FrameId:
    value: str

    def __post_init__(self) -> None:
        if not _FRAME_PATTERN.fullmatch(self.value):
            raise ValueError(f"unsupported coordinate frame: {self.value!r}")
        if self.value.startswith("object_"):
            object_id = self.value.removeprefix("object_")
            try:
                parsed_id = UUID(object_id)
            except ValueError as error:
                raise ValueError(f"object frame must contain a UUID: {self.value!r}") from error
            if str(parsed_id) != object_id.lower():
                raise ValueError(f"object frame UUID must use canonical form: {self.value!r}")

    def __str__(self) -> str:
        return self.value


@total_ordering
@dataclass(frozen=True)
class Timestamp:
    monotonic_ns: int
    utc_ns: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.monotonic_ns, int)
            or isinstance(self.monotonic_ns, bool)
            or self.monotonic_ns < 0
        ):
            raise ValueError("monotonic_ns must be a non-negative integer")
        if self.utc_ns is not None and (
            not isinstance(self.utc_ns, int) or isinstance(self.utc_ns, bool) or self.utc_ns < 0
        ):
            raise ValueError("utc_ns must be a non-negative integer when provided")

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Timestamp):
            return NotImplemented
        self_key = (self.monotonic_ns, self.utc_ns is not None, self.utc_ns or 0)
        other_key = (other.monotonic_ns, other.utc_ns is not None, other.utc_ns or 0)
        return self_key < other_key

    def to_dict(self) -> dict[str, int | None]:
        return {"monotonic_ns": self.monotonic_ns, "utc_ns": self.utc_ns}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Timestamp:
        return cls(monotonic_ns=payload["monotonic_ns"], utc_ns=payload.get("utc_ns"))


@dataclass(frozen=True)
class PoseCovariance:
    matrix: FloatArray

    def __post_init__(self) -> None:
        matrix = _immutable_array(self.matrix, (6, 6), "pose covariance")
        if not np.allclose(matrix, matrix.T, atol=1e-12):
            raise ValueError("pose covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(matrix)) < -1e-12:
            raise ValueError("pose covariance must be positive semidefinite")
        object.__setattr__(self, "matrix", matrix)

    def to_dict(self) -> dict[str, list[list[float]]]:
        return {"matrix": self.matrix.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PoseCovariance:
        return cls(payload["matrix"])


@dataclass(frozen=True)
class TransformSE3:
    """Rigid transform mapping points from ``source`` into ``destination``."""

    source: FrameId
    destination: FrameId
    matrix: FloatArray

    def __post_init__(self) -> None:
        matrix = _immutable_array(self.matrix, (4, 4), "transform")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12):
            raise ValueError("transform must have homogeneous bottom row [0, 0, 0, 1]")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
            raise ValueError("transform rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9):
            raise ValueError("transform rotation determinant must be 1")
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def from_translation_quaternion(
        cls,
        source: FrameId,
        destination: FrameId,
        translation_m: Any,
        quaternion_xyzw: Any,
    ) -> TransformSE3:
        translation = _immutable_array(translation_m, (3,), "translation")
        quaternion = _immutable_array(quaternion_xyzw, (4,), "quaternion")
        norm = float(np.linalg.norm(quaternion))
        if norm <= np.finfo(np.float64).eps:
            raise ValueError("quaternion norm must be non-zero")
        x, y, z, w = quaternion / norm
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        return cls(source, destination, matrix)

    @property
    def translation_m(self) -> FloatArray:
        return self.matrix[:3, 3]

    def quaternion_xyzw(self) -> FloatArray:
        rotation = self.matrix[:3, :3]
        trace = float(np.trace(rotation))
        if trace > 0:
            scale = np.sqrt(trace + 1.0) * 2
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    0.25 * scale,
                ]
            )
        else:
            axis = int(np.argmax(np.diag(rotation)))
            if axis == 0:
                scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
                quaternion = np.array(
                    [
                        0.25 * scale,
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                        (rotation[2, 1] - rotation[1, 2]) / scale,
                    ]
                )
            elif axis == 1:
                scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
                quaternion = np.array(
                    [
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        0.25 * scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                        (rotation[0, 2] - rotation[2, 0]) / scale,
                    ]
                )
            else:
                scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
                quaternion = np.array(
                    [
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                        0.25 * scale,
                        (rotation[1, 0] - rotation[0, 1]) / scale,
                    ]
                )
        quaternion /= np.linalg.norm(quaternion)
        quaternion.setflags(write=False)
        return quaternion

    def inverse(self) -> TransformSE3:
        rotation = self.matrix[:3, :3]
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation.T
        matrix[:3, 3] = -(rotation.T @ self.matrix[:3, 3])
        return TransformSE3(self.destination, self.source, matrix)

    def then(self, following: TransformSE3) -> TransformSE3:
        if self.destination != following.source:
            raise ValueError(
                f"frame mismatch: {self.destination} does not match {following.source}"
            )
        return TransformSE3(self.source, following.destination, following.matrix @ self.matrix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "destination": self.destination.value,
            "matrix": self.matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TransformSE3:
        return cls(FrameId(payload["source"]), FrameId(payload["destination"]), payload["matrix"])


@dataclass(frozen=True)
class PoseEstimate:
    timestamp: Timestamp
    transform: TransformSE3
    covariance: PoseCovariance

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.to_dict(),
            "transform": self.transform.to_dict(),
            "covariance": self.covariance.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PoseEstimate:
        return cls(
            Timestamp.from_dict(payload["timestamp"]),
            TransformSE3.from_dict(payload["transform"]),
            PoseCovariance.from_dict(payload["covariance"]),
        )


@dataclass(frozen=True)
class CameraCalibration:
    camera_frame: FrameId
    intrinsic_matrix: FloatArray
    image_width: int
    image_height: int
    distortion_coefficients: FloatArray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        if not self.camera_frame.value.startswith("camera_"):
            raise ValueError("camera calibration requires a camera_<id> frame")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        intrinsic = _immutable_array(self.intrinsic_matrix, (3, 3), "intrinsic matrix")
        distortion = np.array(self.distortion_coefficients, dtype=np.float64, copy=True)
        if distortion.ndim != 1 or not np.all(np.isfinite(distortion)):
            raise ValueError("distortion coefficients must be a finite vector")
        distortion.setflags(write=False)
        object.__setattr__(self, "intrinsic_matrix", intrinsic)
        object.__setattr__(self, "distortion_coefficients", distortion)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_frame": self.camera_frame.value,
            "intrinsic_matrix": self.intrinsic_matrix.tolist(),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "distortion_coefficients": self.distortion_coefficients.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CameraCalibration:
        return cls(
            FrameId(payload["camera_frame"]),
            payload["intrinsic_matrix"],
            payload["image_width"],
            payload["image_height"],
            payload["distortion_coefficients"],
        )


@dataclass(frozen=True)
class ImuCalibration:
    imu_frame: FrameId
    body_from_imu: TransformSE3
    accelerometer_noise_density: float
    gyroscope_noise_density: float

    def __post_init__(self) -> None:
        if self.imu_frame != FrameId("imu"):
            raise ValueError("IMU calibration frame must be 'imu'")
        if self.body_from_imu.source != self.imu_frame or self.body_from_imu.destination != FrameId(
            "body"
        ):
            raise ValueError("body_from_imu must map imu to body")
        noise = np.array(
            [self.accelerometer_noise_density, self.gyroscope_noise_density], dtype=np.float64
        )
        if not np.all(np.isfinite(noise)) or np.any(noise < 0):
            raise ValueError("IMU noise densities must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "imu_frame": self.imu_frame.value,
            "body_from_imu": self.body_from_imu.to_dict(),
            "accelerometer_noise_density": self.accelerometer_noise_density,
            "gyroscope_noise_density": self.gyroscope_noise_density,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ImuCalibration:
        return cls(
            FrameId(payload["imu_frame"]),
            TransformSE3.from_dict(payload["body_from_imu"]),
            payload["accelerometer_noise_density"],
            payload["gyroscope_noise_density"],
        )


class SensorHealthState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass(frozen=True)
class SensorHealth:
    sensor_id: str
    state: SensorHealthState
    timestamp: Timestamp
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.sensor_id:
            raise ValueError("sensor_id must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "state": self.state.value,
            "timestamp": self.timestamp.to_dict(),
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SensorHealth:
        return cls(
            payload["sensor_id"],
            SensorHealthState(payload["state"]),
            Timestamp.from_dict(payload["timestamp"]),
            payload.get("detail", ""),
        )


@dataclass(frozen=True)
class ModelVersion:
    name: str
    version: str
    checksum_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("model name and version must not be empty")
        if self.checksum_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.checksum_sha256
        ):
            raise ValueError("checksum_sha256 must contain 64 lowercase hexadecimal characters")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "version": self.version,
            "checksum_sha256": self.checksum_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelVersion:
        return cls(payload["name"], payload["version"], payload.get("checksum_sha256"))
