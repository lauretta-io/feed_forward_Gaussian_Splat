#!/usr/bin/env python3
"""Prepare independently timed S3E Wingman windows for ReSplat and dense fusion."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ariadne.backends.external_vio import (
    TrajectoryPose,
    export_s3e_euroc_window,
    parse_trajectory,
)
from ariadne.replay import GroundTruthPose, read_ground_truth_poses


def _quaternion_rotation(quaternion_xyzw: npt.ArrayLike) -> npt.NDArray[np.float64]:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm([x, y, z, w]))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("trajectory quaternion is invalid")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotation_quaternion_wxyz(rotation: npt.NDArray[np.float64]) -> tuple[float, ...]:
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            values = (
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            )
        elif axis == 1:
            scale = math.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            values = (
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            values = (
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            )
    normalized = np.asarray(values) / np.linalg.norm(values)
    return tuple(float(value) for value in normalized)


def _truth_at(
    timestamps_ns: npt.NDArray[np.int64], truth: tuple[GroundTruthPose, ...]
) -> npt.NDArray[np.float64]:
    truth_times = np.asarray([pose.timestamp.monotonic_ns for pose in truth], dtype=np.int64)
    truth_positions = np.stack([pose.position_m for pose in truth])
    if timestamps_ns[0] < truth_times[0] or timestamps_ns[-1] > truth_times[-1]:
        raise ValueError("trajectory extends outside the S3E position-truth interval")
    return np.column_stack(
        [np.interp(timestamps_ns, truth_times, truth_positions[:, axis]) for axis in range(3)]
    )


def _fit_similarity(
    trajectory: tuple[TrajectoryPose, ...], truth: tuple[GroundTruthPose, ...]
) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    truth_start = truth[0].timestamp.monotonic_ns
    truth_end = truth[-1].timestamp.monotonic_ns
    selected = tuple(
        pose for pose in trajectory if truth_start <= pose.timestamp_ns <= truth_end
    )
    if len(selected) < 20:
        raise ValueError("too few VIO poses overlap S3E position truth")
    timestamps = np.asarray([pose.timestamp_ns for pose in selected], dtype=np.int64)
    source = np.stack([pose.position_m for pose in selected])
    destination = _truth_at(timestamps, truth)
    source_center = np.mean(source, axis=0)
    destination_center = np.mean(destination, axis=0)
    centered_source = source - source_center
    centered_destination = destination - destination_center
    covariance = centered_destination.T @ centered_source / len(source)
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    reflection = np.eye(3)
    if np.linalg.det(left @ right_transpose) < 0:
        reflection[-1, -1] = -1
    rotation = left @ reflection @ right_transpose
    variance = float(np.mean(np.sum(centered_source**2, axis=1)))
    if variance <= 1e-12:
        raise ValueError("VIO trajectory has no spatial baseline")
    scale = float(np.sum(singular_values * np.diag(reflection)) / variance)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("VIO-to-global similarity scale is invalid")
    translation = destination_center - scale * (rotation @ source_center)
    aligned = scale * (rotation @ source.T).T + translation
    rmse = float(np.sqrt(np.mean(np.sum((aligned - destination) ** 2, axis=1))))
    return scale, rotation, translation, rmse


def _nearest_pose(
    timestamp_ns: int, trajectory: tuple[TrajectoryPose, ...], *, tolerance_ns: int = 60_000_000
) -> TrajectoryPose | None:
    nearest = min(trajectory, key=lambda pose: abs(pose.timestamp_ns - timestamp_ns))
    return nearest if abs(nearest.timestamp_ns - timestamp_ns) <= tolerance_ns else None


def _calibration(path: Path) -> tuple[int, int, float, float, float, float]:
    values: dict[str, float] = {}
    pattern = re.compile(r"^(Camera\.(?:width|height|fx|fy|cx|cy)):\s*([^#]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            values[match.group(1)] = float(match.group(2).strip())
    required = tuple(f"Camera.{name}" for name in ("width", "height", "fx", "fy", "cx", "cy"))
    if any(name not in values for name in required):
        raise ValueError(f"S3E camera calibration is incomplete: {path}")
    return (
        int(values["Camera.width"]),
        int(values["Camera.height"]),
        values["Camera.fx"],
        values["Camera.fy"],
        values["Camera.cx"],
        values["Camera.cy"],
    )


def _uniform_indices(length: int, count: int) -> npt.NDArray[np.int64]:
    return np.linspace(0, length - 1, count, dtype=np.int64)


def _write_colmap_scene(
    scene: Path,
    image_root: Path,
    records: list[tuple[int, Path, npt.NDArray[np.float64]]],
    calibration: tuple[int, int, float, float, float, float],
) -> None:
    sparse = scene / "sparse/0"
    sparse.mkdir(parents=True, exist_ok=True)
    images_link = scene / "images"
    if not images_link.exists():
        images_link.symlink_to(image_root.resolve(), target_is_directory=True)
    width, height, fx, fy, cx, cy = calibration
    (sparse / "cameras.txt").write_text(
        "# Camera list\n" f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n",
        encoding="utf-8",
    )
    lines = ["# Image list with two lines of data per image\n"]
    for image_id, (_, image_path, c2w) in enumerate(records, start=1):
        rotation = c2w[:3, :3].T
        translation = -(rotation @ c2w[:3, 3])
        quaternion = _rotation_quaternion_wxyz(rotation)
        fields = (
            image_id,
            *quaternion,
            *translation.tolist(),
            1,
            image_path.name,
        )
        lines.append(" ".join(str(value) for value in fields) + "\n\n")
    (sparse / "images.txt").write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3e-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resplat-output", type=Path, required=True)
    parser.add_argument(
        "--agent-window",
        nargs=4,
        action="append",
        required=True,
        metavar=("AGENT", "START_FRAME", "FRAME_COUNT", "TRAJECTORY"),
    )
    parser.add_argument("--context-views", type=int, default=8)
    args = parser.parse_args()
    if args.context_views <= 1:
        raise ValueError("context view count must exceed one")
    playground = args.s3e_root / "S3E_Playground_2"
    bag = playground / "S3E_Playground_2.db3"
    origin_truth = read_ground_truth_poses(playground / "alpha_gt.txt", orientation_available=False)
    global_origin = origin_truth[0].position_m
    contributions: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for raw_agent, raw_start, raw_count, raw_trajectory in args.agent_window:
        agent = raw_agent.capitalize()
        start_frame = int(raw_start)
        frame_count = int(raw_count)
        trajectory_path = Path(raw_trajectory)
        trajectory = parse_trajectory(trajectory_path)
        truth = read_ground_truth_poses(
            playground / f"{agent.lower()}_gt.txt", orientation_available=False
        )
        scale, alignment_rotation, alignment_translation, alignment_rmse = _fit_similarity(
            trajectory, truth
        )
        export_root = args.output / "exports" / agent.lower()
        exported = export_s3e_euroc_window(
            bag, agent, export_root, start_frame=start_frame, max_frames=frame_count
        )
        timestamps = [int(line) for line in exported.times_path.read_text().splitlines() if line]
        records: list[tuple[int, Path, npt.NDArray[np.float64]]] = []
        for timestamp_ns in timestamps:
            pose = _nearest_pose(timestamp_ns, trajectory)
            if pose is None:
                continue
            c2w = np.eye(4)
            c2w[:3, :3] = alignment_rotation @ _quaternion_rotation(pose.quaternion_xyzw)
            c2w[:3, 3] = (
                scale * (alignment_rotation @ pose.position_m)
                + alignment_translation
                - global_origin
            )
            records.append(
                (timestamp_ns, export_root / f"mav0/cam0/data/{timestamp_ns}.png", c2w)
            )
        if len(records) < args.context_views:
            raise ValueError(f"{agent} has only {len(records)} pose-matched images")
        scene = args.output / "scenes" / agent.lower()
        _write_colmap_scene(
            scene,
            export_root / "mav0/cam0/data",
            records,
            _calibration(args.s3e_root / f"Calibration/{agent.lower()}.yaml"),
        )
        context_indices = _uniform_indices(len(records), args.context_views)
        pivot_index = int(context_indices[len(context_indices) // 2])
        pivot_timestamp, _, pivot_c2w = records[pivot_index]
        contributions.append(
            {
                "agent_id": agent,
                "capture_timestamp_ns": pivot_timestamp,
                "ply_path": str((args.resplat_output / agent.lower() / "gaussians.ply").resolve()),
                "local_to_global": pivot_c2w.tolist(),
                "registration_method": "offline-sim3-fit-vio-to-s3e-position-truth",
                "registration_verified": False,
            }
        )
        diagnostics.append(
            {
                "agent_id": agent,
                "requested_start_frame": start_frame,
                "exported_frames": exported.stereo_pair_count,
                "pose_matched_frames": len(records),
                "capture_timestamp_ns": pivot_timestamp,
                "vio_to_truth_scale": scale,
                "vio_to_truth_alignment_rmse_m": alignment_rmse,
                "trajectory": str(trajectory_path),
                "scene": str(scene),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fusion_input.json").write_text(
        json.dumps(
            {
                "schema": "ariadne.static-asynchronous-global-gaussian-input.v1",
                "contributions": contributions,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output / "preparation.json").write_text(
        json.dumps(
            {
                "schema": "ariadne.s3e-resplat-preparation.v1",
                "global_origin_m": global_origin.tolist(),
                "temporal_alignment": "none",
                "windows": diagnostics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "windows": diagnostics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
