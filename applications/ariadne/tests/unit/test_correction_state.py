from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ariadne.common import FrameId, Timestamp, TransformSE3
from ariadne.pose_correction import (
    CorrectionApplier,
    CorrectionDeltaGenerator,
    CorrectionResetRequiredError,
)


def pose(destination: str, x: float, yaw_rad: float = 0.0) -> TransformSE3:
    return TransformSE3.from_translation_quaternion(
        FrameId("body"),
        FrameId(destination),
        (x, 0.0, 0.0),
        (0.0, 0.0, np.sin(yaw_rad / 2.0), np.cos(yaw_rad / 2.0)),
    )


def test_generator_and_applier_preserve_sequences_and_dedupe_after_restart(
    tmp_path: Path,
) -> None:
    local = pose("local_wingman_01", 1.0)
    optimized = pose("global", 1.8)
    generator = CorrectionDeltaGenerator(max_history=2)
    first = generator.generate(
        "wingman_01",
        local,
        optimized,
        issued_at=Timestamp(100),
    )
    applier = CorrectionApplier(max_applied_history=2)
    applier.apply(local, first, now=Timestamp(101))

    restored_generator = CorrectionDeltaGenerator.read_json(
        generator.write_json(tmp_path / "generator.json")
    )
    restored_applier = CorrectionApplier.read_json(
        applier.write_json(tmp_path / "applier.json")
    )
    with pytest.raises(ValueError, match="already"):
        restored_applier.apply(local, first, now=Timestamp(102))
    second = restored_generator.generate(
        "wingman_01",
        local,
        optimized,
        issued_at=Timestamp(103),
    )
    assert second.correction_id == "correction_wingman_01_000001"
    assert restored_generator.metrics["restores"] == 1
    assert restored_applier.metrics["restores"] == 1


def test_generator_rejects_time_travel_and_bounds_history() -> None:
    local = pose("local_wingman_01", 1.0)
    optimized = pose("global", 1.8)
    generator = CorrectionDeltaGenerator(max_history=1)
    generator.generate("wingman_01", local, optimized, issued_at=Timestamp(100))
    generator.generate("wingman_01", local, optimized, issued_at=Timestamp(101))
    assert len(generator.history) == 1
    with pytest.raises(ValueError, match="monotonic"):
        generator.generate("wingman_01", local, optimized, issued_at=Timestamp(99))
    assert generator.metrics["rejected"] == 1


def test_applier_flags_reset_required_for_unsafe_global_jump() -> None:
    local = pose("local_wingman_01", 0.0)
    optimized = pose("global", 20.0)
    correction = CorrectionDeltaGenerator().generate(
        "wingman_01",
        local,
        optimized,
        issued_at=Timestamp(100),
    )
    applier = CorrectionApplier(
        max_translation_step_m=0.5,
        max_total_translation_m=5.0,
    )
    with pytest.raises(CorrectionResetRequiredError, match="continuity"):
        applier.apply(local, correction, now=Timestamp(101))
    assert applier.metrics["reset_required"] == 1


def test_applier_flags_reset_required_for_unsafe_rotation_jump() -> None:
    local = pose("local_wingman_01", 0.0)
    optimized = pose("global", 0.0, yaw_rad=1.0)
    correction = CorrectionDeltaGenerator().generate(
        "wingman_01",
        local,
        optimized,
        issued_at=Timestamp(100),
    )
    applier = CorrectionApplier(
        max_rotation_step_rad=0.1,
        max_total_rotation_rad=0.5,
    )

    with pytest.raises(CorrectionResetRequiredError, match=r"SE\(3\)"):
        applier.apply(local, correction, now=Timestamp(101))

    assert applier.metrics["reset_required"] == 1


def test_applier_smooths_rotation_independently_from_translation() -> None:
    local = pose("local_wingman_01", 0.0)
    optimized = pose("global", 0.0, yaw_rad=0.4)
    correction = CorrectionDeltaGenerator().generate(
        "wingman_01",
        local,
        optimized,
        issued_at=Timestamp(100),
    )
    applier = CorrectionApplier(
        max_rotation_step_rad=0.1,
        max_total_rotation_rad=0.5,
    )

    applied = applier.apply(local, correction, now=Timestamp(101))
    quaternion = applied.corrected_pose.quaternion_xyzw()
    applied_angle = 2.0 * np.arccos(abs(float(quaternion[3])))

    assert applied.applied_fraction == pytest.approx(0.25)
    assert applied_angle <= 0.100000001
    assert applier.metrics["rotation_limited"] == 1
