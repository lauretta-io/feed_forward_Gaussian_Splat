from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ariadne.splatting import DenseGaussianContribution, fuse_static_gaussian_plys


def _write_ascii_gaussians(path: Path, rows: list[tuple[float, ...]]) -> None:
    properties = ("x", "y", "z", "nx", "ny", "nz", "opacity", "rot_0", "rot_1", "rot_2", "rot_3")
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(rows)}",
        *(f"property float {name}" for name in properties),
        "end_header",
        *(" ".join(str(value) for value in row) for row in rows),
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")


def _transform(*, translation: tuple[float, float, float], yaw_degrees: float = 0.0) -> np.ndarray:
    yaw = np.deg2rad(yaw_degrees)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    return np.asarray(
        [
            [cosine, -sine, 0.0, translation[0]],
            [sine, cosine, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def test_dense_fusion_accepts_out_of_order_time_and_preserves_provenance(tmp_path: Path) -> None:
    later = tmp_path / "later.ply"
    earlier = tmp_path / "earlier.ply"
    _write_ascii_gaussians(
        later,
        [
            (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0),
            (float("nan"), 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0),
        ],
    )
    _write_ascii_gaussians(
        earlier,
        [(0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)],
    )
    result = fuse_static_gaussian_plys(
        (
            DenseGaussianContribution(
                "wingman_later", 300, later, _transform(translation=(10.0, 0.0, 0.0)),
                "pose-graph", True,
            ),
            DenseGaussianContribution(
                "wingman_earlier", 100, earlier,
                _transform(translation=(0.0, 0.0, 1.0), yaw_degrees=90.0),
                "pose-graph", True,
            ),
        ),
        tmp_path / "global.ply",
    )
    assert result.input_gaussians == 3
    assert result.output_gaussians == 2
    assert result.filtered_gaussians == 1
    assert result.global_registration_verified
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "static-asynchronous"
    assert payload["temporal_alignment"] == "none"
    assert payload["inputs_arrived_in_timestamp_order"] is False
    assert [source["capture_timestamp_ns"] for source in payload["sources"]] == [300, 100]
    assert payload["bounds_m"]["minimum"] == pytest.approx([-2.0, 0.0, 0.0])
    assert payload["bounds_m"]["maximum"] == pytest.approx([11.0, 0.0, 1.0])


def test_dense_fusion_marks_unverified_registration_ineligible(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    _write_ascii_gaussians(
        source,
        [(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)],
    )
    result = fuse_static_gaussian_plys(
        (
            DenseGaussianContribution(
                "atlas_source", 10, source, np.eye(4), "manual-atlas", False
            ),
        ),
        tmp_path / "global.ply",
    )
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert not result.global_registration_verified
    assert payload["global_metric_claim_eligible"] is False
    assert payload["warnings"]


def test_dense_fusion_rejects_non_rigid_transform(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    _write_ascii_gaussians(
        source,
        [(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0)],
    )
    scaled = np.eye(4)
    scaled[0, 0] = 2.0
    with pytest.raises(ValueError, match="rigid rotation"):
        DenseGaussianContribution("wingman", 0, source, scaled, "invalid")
