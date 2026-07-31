from __future__ import annotations

import numpy as np
import pytest

from ariadne.common import ModelVersion, Timestamp
from ariadne.intelligence import RegisteredObservation
from ariadne.splatting import (
    GaussianBackendError,
    GaussianBackendRegistry,
    GaussianOutOfMemoryError,
    GaussianReconstructionExecutor,
    GaussianResourceError,
    ReconstructionLimits,
    ReconstructionResult,
    ReferenceGaussianSplatAdapter,
)


def observation(agent: str, index: int, object_id: str = "tower") -> RegisteredObservation:
    return RegisteredObservation(
        f"{agent}:0:{index}",
        agent,
        f"{agent}:{object_id}",
        Timestamp(100 + index),
        np.array([4.0 + index * 0.1, 2.0, 0.0]),
        np.array([1.0, 0.2, 0.1]),
        np.array([0.04, 0.09, 0.01]),
        0.9,
        ModelVersion("reference", "1"),
    )


def test_executor_attaches_bounded_backend_diagnostics() -> None:
    backend = ReferenceGaussianSplatAdapter()
    executor = GaussianReconstructionExecutor(
        backend,
        limits=ReconstructionLimits(
            max_observations=2,
            max_objects=1,
            max_estimated_memory_bytes=4096,
        ),
        device="cpu",
    )
    result = executor.reconstruct(
        (observation("wingman_01", 0), observation("wingman_02", 1)),
        timestamp=Timestamp(200),
    )

    assert result.diagnostics is not None
    assert result.diagnostics.backend == backend.version.name
    assert result.diagnostics.observation_count == 2
    assert result.diagnostics.object_count == 1
    assert 0 < result.diagnostics.estimated_working_set_bytes <= 4096
    assert executor.metrics["completed"] == 1


def test_executor_rejects_request_limits_before_backend_execution() -> None:
    executor = GaussianReconstructionExecutor(
        ReferenceGaussianSplatAdapter(),
        limits=ReconstructionLimits(max_observations=1),
    )
    with pytest.raises(GaussianResourceError, match="limits"):
        executor.reconstruct(
            (observation("wingman_01", 0), observation("wingman_02", 1)),
            timestamp=Timestamp(200),
        )
    assert executor.metrics["resource_rejected"] == 1


def test_executor_normalizes_backend_oom_and_invalid_output() -> None:
    class OomBackend:
        version = ModelVersion("oom-test", "1")

        def reconstruct(self, observations, *, timestamp):
            raise MemoryError("simulated")

    oom_executor = GaussianReconstructionExecutor(OomBackend())
    with pytest.raises(GaussianOutOfMemoryError, match="exhausted"):
        oom_executor.reconstruct((observation("wingman_01", 0),), timestamp=Timestamp(200))
    assert oom_executor.metrics["out_of_memory"] == 1

    class InvalidBackend:
        version = ModelVersion("invalid-test", "1")

        def reconstruct(self, observations, *, timestamp):
            return ReconstructionResult(timestamp, self.version, (), 0.1, len(observations))

    invalid_executor = GaussianReconstructionExecutor(InvalidBackend())
    with pytest.raises(GaussianBackendError, match="invalid result"):
        invalid_executor.reconstruct(
            (observation("wingman_01", 0),),
            timestamp=Timestamp(200),
        )
    assert invalid_executor.metrics["backend_failures"] == 1


def test_backend_registry_is_bounded_and_rejects_duplicates() -> None:
    registry = GaussianBackendRegistry(max_backends=1)
    registry.register("reference", ReferenceGaussianSplatAdapter)
    assert registry.names == ("reference",)
    assert isinstance(registry.create("reference"), ReferenceGaussianSplatAdapter)
    with pytest.raises(ValueError, match="unique"):
        registry.register("reference", ReferenceGaussianSplatAdapter)
    with pytest.raises(ValueError, match="capacity"):
        registry.register("second", ReferenceGaussianSplatAdapter)
    with pytest.raises(ValueError, match="unknown"):
        registry.create("missing")
