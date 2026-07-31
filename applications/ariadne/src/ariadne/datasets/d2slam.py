"""D2SLAM archive inventory and modality checks."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

from ariadne.datasets.base import DatasetEvaluation
from ariadne.datasets.timing import nearest_timestamp_errors_ms, summarize_errors_ms


def _evaluate_bag_directory(path: Path) -> DatasetEvaluation:
    try:
        from rosbags.rosbag1 import Reader
    except ImportError as error:
        raise RuntimeError(
            "install ARIADNE with the evaluation extra to inspect ROS bags"
        ) from error
    bags = sorted(path.rglob("*.bag"))
    if not bags:
        raise ValueError(f"no ROS1 bags found beneath: {path}")
    starts: list[int] = []
    ends: list[int] = []
    total_messages = 0
    imu_messages = 0
    vision_messages = 0
    sync_errors: list[float] = []
    bag_topics: dict[str, dict[str, int]] = {}
    for bag in bags:
        with Reader(bag) as reader:
            starts.append(reader.start_time)
            ends.append(reader.end_time)
            total_messages += reader.message_count
            topics = {connection.topic: connection.msgcount for connection in reader.connections}
            bag_topics[bag.name] = topics
            imu_messages += sum(
                connection.msgcount
                for connection in reader.connections
                if "imu" in connection.topic.lower() or "imu" in connection.msgtype.lower()
            )
            vision_messages += sum(
                connection.msgcount
                for connection in reader.connections
                if "image" in connection.topic.lower() or "image" in connection.msgtype.lower()
            )
            imu_connections = [
                connection
                for connection in reader.connections
                if "imu" in connection.topic.lower() or "imu" in connection.msgtype.lower()
            ]
            vision_connections = [
                connection
                for connection in reader.connections
                if "image" in connection.topic.lower() or "image" in connection.msgtype.lower()
            ]
            imu_timestamps = [timestamp for _, timestamp, _ in reader.messages(imu_connections)]
            vision_timestamps = [
                timestamp for _, timestamp, _ in reader.messages(vision_connections)
            ]
            sync_errors.extend(nearest_timestamp_errors_ms(imu_timestamps, vision_timestamps))
    ground_truth_files = sorted(path.glob("groundtruth_*.csv"))
    agents = tuple(f"agent_{index}" for index in range(1, len(bags) + 1))
    return DatasetEvaluation(
        dataset="d2slam",
        status="passed" if imu_messages and vision_messages and len(bags) >= 3 else "failed",
        agents=agents,
        modalities=("ground_truth", "imu", "stereo_vision"),
        metrics={
            "agent_count": len(bags),
            "rosbag_count": len(bags),
            "ground_truth_file_count": len(ground_truth_files),
            "message_count": total_messages,
            "imu_message_count": imu_messages,
            "vision_message_count": vision_messages,
            "max_agent_start_skew_ms": (max(starts) - min(starts)) / 1e6,
            "max_duration_seconds": (max(ends) - min(starts)) / 1e9,
            **summarize_errors_ms(np.asarray(sync_errors, dtype=np.float64)),
        },
        warnings=("aligned TUM set emulates multi-robot replay; it is not a simultaneous flight",),
        details={"path": str(path), "bag_topics": bag_topics},
    )


def evaluate_d2slam(path: Path) -> DatasetEvaluation:
    if path.is_dir():
        return _evaluate_bag_directory(path)
    if not path.is_file():
        raise FileNotFoundError(f"D2SLAM archive does not exist: {path}")
    completed = subprocess.run(
        ["7z", "l", "-slt", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"cannot inspect D2SLAM archive: {completed.stderr.strip()}")
    members = [
        match.group(1)
        for line in completed.stdout.splitlines()
        if (match := re.match(r"Path = (.+)", line)) is not None
    ][1:]
    lowered = "\n".join(members).lower()
    bag_count = sum(name.lower().endswith(".bag") for name in members)
    image_count = sum(Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"} for name in members)
    modalities: set[str] = set()
    if bag_count or image_count:
        modalities.add("vision")
    if "imu" in lowered or bag_count:
        modalities.add("imu")
    config_member = next((member for member in members if member.endswith("/d2vins.yaml")), None)
    config_text = ""
    if config_member is not None:
        config = subprocess.run(
            ["7z", "x", "-so", str(path), config_member],
            check=False,
            capture_output=True,
            text=True,
        )
        if config.returncode == 0:
            config_text = config.stdout
    agent_tokens = set(
        re.findall(r"(?:drone|swarm|agent)[_-]?(\d+)", lowered + config_text.lower())
    )
    warnings = []
    if not agent_tokens:
        warnings.append("archive names do not expose explicit physical drone IDs")
    warnings.append("aligned TUM set emulates multi-robot replay; it is not a simultaneous flight")
    return DatasetEvaluation(
        dataset="d2slam",
        status="passed" if {"vision", "imu"} <= modalities else "failed",
        agents=tuple(f"agent_{token}" for token in sorted(agent_tokens)),
        modalities=tuple(sorted(modalities)),
        metrics={
            "archive_bytes": path.stat().st_size,
            "archive_member_count": len(members),
            "rosbag_count": bag_count,
            "image_file_count": image_count,
            "named_agent_count": len(agent_tokens),
        },
        warnings=tuple(warnings),
        details={
            "path": str(path),
            "sample_members": members[:100],
            "bag_files": [member for member in members if member.lower().endswith(".bag")],
        },
    )
