"""Feed-forward Gaussian adapter and versioned global scene map."""

from ariadne.splatting.adapter import (
    GaussianBackendError,
    GaussianBackendRegistry,
    GaussianOutOfMemoryError,
    GaussianPrimitive,
    GaussianReconstructionExecutor,
    GaussianResourceError,
    GaussianSplatBackend,
    ReconstructionDiagnostics,
    ReconstructionLimits,
    ReconstructionResult,
    ReferenceGaussianSplatAdapter,
)
from ariadne.splatting.scene_map import (
    GlobalGaussianMap,
    SceneSnapshot,
    SceneSnapshotStore,
)

__all__ = [
    "GaussianBackendError",
    "GaussianBackendRegistry",
    "GaussianOutOfMemoryError",
    "GaussianPrimitive",
    "GaussianReconstructionExecutor",
    "GaussianResourceError",
    "GaussianSplatBackend",
    "GlobalGaussianMap",
    "ReconstructionDiagnostics",
    "ReconstructionLimits",
    "ReconstructionResult",
    "ReferenceGaussianSplatAdapter",
    "SceneSnapshot",
    "SceneSnapshotStore",
]
