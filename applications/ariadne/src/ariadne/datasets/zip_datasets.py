"""ZIP archive probes for MILUV and QDrone datasets."""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from ariadne.datasets.base import DatasetEvaluation
from ariadne.datasets.timing import nearest_timestamp_errors_ms, summarize_errors_ms

_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}


def _zip_inventory(archives: list[Path]) -> tuple[list[str], int, int]:
    names: list[str] = []
    compressed_bytes = 0
    uncompressed_bytes = 0
    for archive in archives:
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                if info.is_dir():
                    continue
                names.append(info.filename)
                compressed_bytes += info.compress_size
                uncompressed_bytes += info.file_size
    return names, compressed_bytes, uncompressed_bytes


def evaluate_miluv(path: Path) -> DatasetEvaluation:
    if not path.is_file():
        raise FileNotFoundError(f"MILUV archive does not exist: {path}")
    names, compressed_bytes, uncompressed_bytes = _zip_inventory([path])
    agents = sorted(set(re.findall(r"ifo\d+", "\n".join(names), flags=re.IGNORECASE)))
    lowered = [name.lower() for name in names]
    image_count = sum(Path(name).suffix.lower() in _IMAGE_SUFFIXES for name in names)
    sensor_files: Counter[str] = Counter()
    for name in lowered:
        if "imu" in name:
            sensor_files["imu"] += 1
        if "mocap" in name or "vicon" in name:
            sensor_files["ground_truth"] += 1
        if "uwb" in name:
            sensor_files["uwb"] += 1
    modalities = {key for key, value in sensor_files.items() if value}
    if image_count:
        modalities.add("vision")
    sync_errors: list[float] = []
    agent_details: dict[str, dict[str, int | float]] = {}
    ground_truth_starts: list[float] = []
    ground_truth_ends: list[float] = []
    with zipfile.ZipFile(path) as handle:
        for agent in agents:
            imu_member = next((name for name in names if f"/{agent}/imu_cam.csv" in name), None)
            mocap_member = next((name for name in names if f"/{agent}/mocap.csv" in name), None)
            color_names = [name for name in names if f"/{agent}/color/" in name]
            color_timestamps: list[float] = []
            for name in color_names:
                try:
                    color_timestamps.append(float(Path(name).stem))
                except ValueError:
                    continue
            imu_timestamps: list[float] = []
            if imu_member is not None:
                with (
                    handle.open(imu_member) as raw,
                    io.TextIOWrapper(raw, encoding="utf-8") as text,
                ):
                    imu_timestamps = [float(row["timestamp"]) for row in csv.DictReader(text)]
            if imu_timestamps and color_timestamps:
                sync_errors.extend(
                    nearest_timestamp_errors_ms(
                        np.rint(np.asarray(imu_timestamps) * 1e9).astype(np.int64),
                        np.rint(np.asarray(color_timestamps) * 1e9).astype(np.int64),
                    )
                )
            mocap_timestamps: list[float] = []
            mocap_positions: list[list[float]] = []
            if mocap_member is not None:
                with (
                    handle.open(mocap_member) as raw,
                    io.TextIOWrapper(raw, encoding="utf-8") as text,
                ):
                    for row in csv.DictReader(text):
                        mocap_timestamps.append(float(row["timestamp"]))
                        mocap_positions.append(
                            [
                                float(row["pose.position.x"]),
                                float(row["pose.position.y"]),
                                float(row["pose.position.z"]),
                            ]
                        )
            path_length = 0.0
            if len(mocap_positions) > 1:
                positions = np.asarray(mocap_positions, dtype=np.float64)
                path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
                ground_truth_starts.append(mocap_timestamps[0])
                ground_truth_ends.append(mocap_timestamps[-1])
            agent_details[agent] = {
                "color_frame_count": len(color_timestamps),
                "imu_sample_count": len(imu_timestamps),
                "ground_truth_sample_count": len(mocap_timestamps),
                "ground_truth_path_length_m": path_length,
            }
    sync_metrics = summarize_errors_ms(np.asarray(sync_errors, dtype=np.float64))
    overlap_seconds = (
        max(min(ground_truth_ends) - max(ground_truth_starts), 0.0)
        if ground_truth_starts and ground_truth_ends
        else 0.0
    )
    warnings: list[str] = []
    if len(agents) < 3:
        warnings.append("expected three MILUV agents in selected experiment")
    status = (
        "passed"
        if len(agents) >= 3
        and {"imu", "vision", "ground_truth"} <= modalities
        and sync_metrics["camera_imu_sync_median_ms"] < 2.0
        else "failed"
    )
    return DatasetEvaluation(
        dataset="miluv",
        status=status,
        agents=tuple(agents),
        modalities=tuple(sorted(modalities)),
        metrics={
            "agent_count": len(agents),
            "archive_member_count": len(names),
            "image_count": image_count,
            "compressed_bytes": compressed_bytes,
            "uncompressed_bytes": uncompressed_bytes,
            "imu_file_count": sensor_files["imu"],
            "ground_truth_file_count": sensor_files["ground_truth"],
            "uwb_file_count": sensor_files["uwb"],
            "ground_truth_overlap_seconds": overlap_seconds,
            "total_ground_truth_path_length_m": sum(
                details["ground_truth_path_length_m"] for details in agent_details.values()
            ),
            **sync_metrics,
        },
        warnings=tuple(warnings),
        details={"path": str(path), "agents": agent_details, "sample_members": names[:50]},
    )


def evaluate_qdrone(path: Path) -> DatasetEvaluation:
    archives = sorted(path.glob("*.zip")) if path.is_dir() else [path]
    if not archives or any(not archive.is_file() for archive in archives):
        raise FileNotFoundError(f"QDrone archives do not exist: {path}")
    names, compressed_bytes, uncompressed_bytes = _zip_inventory(archives)
    record_types: set[int] = set()
    for archive in archives:
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                if not info.filename.lower().endswith(".csv"):
                    continue
                with handle.open(info) as raw, io.TextIOWrapper(raw, encoding="utf-8") as text:
                    for index, line in enumerate(text):
                        if index >= 20_000 or {0, 2} <= record_types:
                            break
                        try:
                            record_types.add(int(line.split(",", maxsplit=1)[0]))
                        except ValueError:
                            continue
                if {0, 2} <= record_types:
                    break
    modalities = []
    if 2 in record_types:
        modalities.append("imu")
    if 0 in record_types:
        modalities.append("uwb")
    has_ground_truth = any("gt" in archive.stem.lower() for archive in archives)
    if has_ground_truth:
        modalities.append("ground_truth")
    return DatasetEvaluation(
        dataset="qdrone",
        status="passed" if "imu" in modalities and "uwb" in modalities else "failed",
        agents=("qdrone",),
        modalities=tuple(modalities),
        metrics={
            "agent_count": 1,
            "archive_count": len(archives),
            "archive_member_count": len(names),
            "compressed_bytes": compressed_bytes,
            "uncompressed_bytes": uncompressed_bytes,
            "has_ground_truth": int(has_ground_truth),
        },
        warnings=("QDrone has no vision stream and cannot run end-to-end ARIADNE",),
        details={
            "path": str(path),
            "archives": [archive.name for archive in archives],
            "record_types": sorted(record_types),
        },
    )
