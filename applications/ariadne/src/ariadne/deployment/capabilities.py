"""Explicit hardware capability detection and fail-closed profile checks."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareCapabilities:
    architecture: str
    cpu_cores: int
    memory_bytes: int
    accelerators: tuple[str, ...]
    camera_devices: tuple[str, ...]


@dataclass(frozen=True)
class DeploymentProfile:
    name: str
    minimum_cpu_cores: int
    minimum_memory_bytes: int
    required_accelerators: tuple[str, ...] = ()
    minimum_cameras: int = 0

    def __post_init__(self) -> None:
        if not self.name or self.minimum_cpu_cores <= 0 or self.minimum_memory_bytes <= 0:
            raise ValueError("deployment profile requirements are invalid")
        if self.minimum_cameras < 0:
            raise ValueError("minimum_cameras must be non-negative")


@dataclass(frozen=True)
class ProfileValidation:
    compatible: bool
    missing: tuple[str, ...]


class CapabilityProbe:
    @staticmethod
    def inspect() -> HardwareCapabilities:
        cpu_cores = os.cpu_count() or 1
        memory_bytes = 0
        if hasattr(os, "sysconf"):
            try:
                memory_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
            except (OSError, ValueError):
                memory_bytes = 0
        accelerators: list[str] = []
        if os.path.exists("/dev/nvidia0"):
            accelerators.append("cuda")
        if os.path.exists("/dev/hailo0"):
            accelerators.append("hailo")
        cameras = tuple(path for index in range(8) if os.path.exists(path := f"/dev/video{index}"))
        return HardwareCapabilities(
            platform.machine() or "unknown",
            cpu_cores,
            memory_bytes,
            tuple(accelerators),
            cameras,
        )

    @staticmethod
    def validate(
        capabilities: HardwareCapabilities, profile: DeploymentProfile
    ) -> ProfileValidation:
        missing: list[str] = []
        if capabilities.cpu_cores < profile.minimum_cpu_cores:
            missing.append(f"cpu_cores>={profile.minimum_cpu_cores}")
        if capabilities.memory_bytes < profile.minimum_memory_bytes:
            missing.append(f"memory_bytes>={profile.minimum_memory_bytes}")
        for accelerator in profile.required_accelerators:
            if accelerator not in capabilities.accelerators:
                missing.append(f"accelerator:{accelerator}")
        if len(capabilities.camera_devices) < profile.minimum_cameras:
            missing.append(f"cameras>={profile.minimum_cameras}")
        return ProfileValidation(not missing, tuple(missing))
