from __future__ import annotations

import importlib
from types import ModuleType


class OpenSplatBackendError(ImportError):
    """Raised when the optional OpenSplat CPU extension is unavailable."""


REQUIRED_FUNCTIONS = (
    "project_gaussians_cpu",
    "rasterize_gaussians_cpu",
)


def load_backend() -> ModuleType:
    try:
        backend = importlib.import_module("opensplat_cpu_ext")
    except ImportError as exc:
        raise OpenSplatBackendError(
            "OpenSplat CPU rendering requires an optional Python extension named "
            "`opensplat_cpu_ext`. Build pierotofy/OpenSplat with GPU_RUNTIME=CPU "
            "and expose wrappers for project_gaussians_cpu and "
            "rasterize_gaussians_cpu. See OPENSPLAT_CPU_INTEGRATION.md."
        ) from exc

    missing = [name for name in REQUIRED_FUNCTIONS if not hasattr(backend, name)]
    if missing:
        raise OpenSplatBackendError(
            "The installed `opensplat_cpu_ext` module is missing required "
            f"function(s): {', '.join(missing)}. See "
            "OPENSPLAT_CPU_INTEGRATION.md for the expected backend contract."
        )

    return backend
