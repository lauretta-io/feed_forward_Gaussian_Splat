"""Typed feed-forward Gaussian reconstruction boundary with a CPU reference."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter_ns
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ariadne.common import ModelVersion, Timestamp
from ariadne.intelligence import RegisteredObservation

LOGGER = logging.getLogger(__name__)


class GaussianBackendError(RuntimeError):
    """Normalized failure raised at the model-backend boundary."""


class GaussianResourceError(GaussianBackendError):
    """A request exceeds an explicit executor resource bound."""


class GaussianOutOfMemoryError(GaussianResourceError):
    """A backend reported host or accelerator memory exhaustion."""


@dataclass(frozen=True)
class ReconstructionLimits:
    max_observations: int = 64
    max_objects: int = 32
    max_estimated_memory_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if (
            self.max_observations <= 0
            or self.max_objects <= 0
            or self.max_estimated_memory_bytes <= 0
        ):
            raise ValueError("reconstruction limits must be positive")


@dataclass(frozen=True)
class ReconstructionDiagnostics:
    backend: str
    model_version: str
    device: str
    observation_count: int
    object_count: int
    estimated_working_set_bytes: int

    def __post_init__(self) -> None:
        if (
            not self.backend
            or not self.model_version
            or not self.device
            or self.observation_count <= 0
            or self.object_count <= 0
            or self.estimated_working_set_bytes <= 0
        ):
            raise ValueError("reconstruction diagnostics are invalid")


def _finite_vector(value: npt.ArrayLike, size: int, name: str) -> npt.NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GaussianPrimitive:
    primitive_id: str
    object_id: str
    mean_m: npt.NDArray[np.float64]
    scale_m: npt.NDArray[np.float64]
    color_rgb: npt.NDArray[np.float64]
    opacity: float
    confidence: float
    source_observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.primitive_id or not self.object_id or not self.source_observation_ids:
            raise ValueError("Gaussian identifiers and provenance must not be empty")
        mean = _finite_vector(self.mean_m, 3, "mean_m")
        scale = _finite_vector(self.scale_m, 3, "scale_m")
        color = _finite_vector(self.color_rgb, 3, "color_rgb")
        if np.any(scale <= 0) or np.any((color < 0) | (color > 1)):
            raise ValueError("Gaussian scale or color is invalid")
        if not 0 <= self.opacity <= 1 or not 0 <= self.confidence <= 1:
            raise ValueError("Gaussian opacity and confidence must be between zero and one")
        object.__setattr__(self, "mean_m", mean)
        object.__setattr__(self, "scale_m", scale)
        object.__setattr__(self, "color_rgb", color)


@dataclass(frozen=True)
class ReconstructionResult:
    timestamp: Timestamp
    model_version: ModelVersion
    primitives: tuple[GaussianPrimitive, ...]
    latency_ms: float
    input_observations: int
    diagnostics: ReconstructionDiagnostics | None = None

    def __post_init__(self) -> None:
        if (
            self.latency_ms < 0
            or self.input_observations <= 0
            or not np.isfinite(self.latency_ms)
        ):
            raise ValueError("reconstruction result metrics are invalid")


@runtime_checkable
class GaussianSplatBackend(Protocol):
    version: ModelVersion

    def reconstruct(
        self,
        observations: tuple[RegisteredObservation, ...],
        *,
        timestamp: Timestamp,
    ) -> ReconstructionResult: ...


BackendFactory = Callable[[], GaussianSplatBackend]


class GaussianBackendRegistry:
    def __init__(self, *, max_backends: int = 16) -> None:
        if max_backends <= 0:
            raise ValueError("max_backends must be positive")
        self.max_backends = max_backends
        self._factories: dict[str, BackendFactory] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def register(self, name: str, factory: BackendFactory) -> None:
        if not name or name in self._factories:
            raise ValueError("Gaussian backend name must be non-empty and unique")
        if len(self._factories) >= self.max_backends:
            raise ValueError("Gaussian backend registry capacity exceeded")
        backend = factory()
        if not isinstance(backend, GaussianSplatBackend):
            raise TypeError("Gaussian backend does not satisfy the required protocol")
        self._factories[name] = factory

    def create(self, name: str) -> GaussianSplatBackend:
        factory = self._factories.get(name)
        if factory is None:
            raise ValueError(f"unknown Gaussian backend: {name}")
        backend = factory()
        if not isinstance(backend, GaussianSplatBackend):
            raise TypeError("Gaussian backend factory returned an incompatible object")
        return backend


class GaussianReconstructionExecutor:
    def __init__(
        self,
        backend: GaussianSplatBackend,
        *,
        limits: ReconstructionLimits | None = None,
        device: str = "cpu",
    ) -> None:
        if not isinstance(backend, GaussianSplatBackend) or not device:
            raise ValueError("executor backend and device must be valid")
        self.backend = backend
        self.limits = ReconstructionLimits() if limits is None else limits
        self.device = device
        self.metrics = {
            "requests": 0,
            "completed": 0,
            "resource_rejected": 0,
            "backend_failures": 0,
            "out_of_memory": 0,
        }

    def reconstruct(
        self,
        observations: tuple[RegisteredObservation, ...],
        *,
        timestamp: Timestamp,
    ) -> ReconstructionResult:
        self.metrics["requests"] += 1
        if not observations:
            self.metrics["resource_rejected"] += 1
            raise GaussianResourceError("reconstruction requires at least one observation")
        object_count = len({item.local_id.split(":")[-1] for item in observations})
        estimated_bytes = self._estimate_working_set(observations, object_count)
        if (
            len(observations) > self.limits.max_observations
            or object_count > self.limits.max_objects
            or estimated_bytes > self.limits.max_estimated_memory_bytes
        ):
            self.metrics["resource_rejected"] += 1
            raise GaussianResourceError("reconstruction request exceeds configured limits")
        try:
            result = self.backend.reconstruct(observations, timestamp=timestamp)
        except Exception as error:
            if isinstance(error, MemoryError) or error.__class__.__name__ in {
                "OutOfMemoryError",
                "CUDAOutOfMemoryError",
            }:
                self.metrics["out_of_memory"] += 1
                raise GaussianOutOfMemoryError(
                    f"Gaussian backend {self.backend.version.name} exhausted memory"
                ) from error
            self.metrics["backend_failures"] += 1
            raise GaussianBackendError(
                f"Gaussian backend {self.backend.version.name} failed"
            ) from error
        self._validate_result(result, observations, timestamp, object_count)
        diagnostics = ReconstructionDiagnostics(
            backend=self.backend.version.name,
            model_version=self.backend.version.version,
            device=self.device,
            observation_count=len(observations),
            object_count=object_count,
            estimated_working_set_bytes=estimated_bytes,
        )
        self.metrics["completed"] += 1
        return replace(result, diagnostics=diagnostics)

    @staticmethod
    def _estimate_working_set(
        observations: tuple[RegisteredObservation, ...], object_count: int
    ) -> int:
        input_bytes = sum(
            item.position_m.nbytes
            + item.embedding.nbytes
            + item.covariance_diagonal.nbytes
            for item in observations
        )
        return max(1, input_bytes * 4 + object_count * 256)

    def _validate_result(
        self,
        result: ReconstructionResult,
        observations: tuple[RegisteredObservation, ...],
        timestamp: Timestamp,
        object_count: int,
    ) -> None:
        source_ids = {item.observation_id for item in observations}
        output_sources = {
            source_id
            for primitive in result.primitives
            for source_id in primitive.source_observation_ids
        }
        if (
            result.timestamp != timestamp
            or result.model_version != self.backend.version
            or result.input_observations != len(observations)
            or not result.primitives
            or len(result.primitives) > object_count
            or not output_sources.issubset(source_ids)
        ):
            self.metrics["backend_failures"] += 1
            raise GaussianBackendError("Gaussian backend returned an invalid result")


class ReferenceGaussianSplatAdapter:
    """Convert associated static observations into provenance-preserving object Gaussians."""

    version = ModelVersion("object-gaussian-reference", "1.0.0")

    def reconstruct(
        self,
        observations: tuple[RegisteredObservation, ...],
        *,
        timestamp: Timestamp,
    ) -> ReconstructionResult:
        if not observations:
            raise ValueError("reconstruction requires at least one static observation")
        start_ns = perf_counter_ns()
        grouped: dict[str, list[RegisteredObservation]] = {}
        for observation in observations:
            grouped.setdefault(observation.local_id.split(":")[-1], []).append(observation)
        primitives: list[GaussianPrimitive] = []
        for object_id, group in sorted(grouped.items()):
            weights = np.asarray([item.confidence for item in group], dtype=np.float64)
            epsilon = float(np.finfo(np.float64).eps)
            weights /= max(float(weights.sum()), epsilon)
            mean = np.sum(np.stack([item.position_m for item in group]) * weights[:, None], axis=0)
            covariance = np.sum(
                np.stack([item.covariance_diagonal for item in group]) * weights[:, None], axis=0
            )
            embedding = np.sum(
                np.stack([item.embedding for item in group]) * weights[:, None], axis=0
            )
            color = np.abs(np.resize(embedding, 3))
            color /= max(float(np.max(color)), epsilon)
            primitives.append(
                GaussianPrimitive(
                    f"gaussian_{object_id}",
                    object_id,
                    mean,
                    np.sqrt(np.maximum(covariance, 1e-4)),
                    color,
                    min(0.98, float(np.mean([item.confidence for item in group]))),
                    float(np.mean([item.confidence for item in group])),
                    tuple(item.observation_id for item in group),
                )
            )
        latency_ms = (perf_counter_ns() - start_ns) / 1e6
        LOGGER.info(
            "gaussian_reconstruction_complete observations=%d primitives=%d",
            len(observations),
            len(primitives),
        )
        return ReconstructionResult(
            timestamp,
            self.version,
            tuple(primitives),
            latency_ms,
            len(observations),
        )
