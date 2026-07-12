"""Metadata-level evaluation of S3E ROS2 SQLite bags."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from ariadne.datasets.base import DatasetEvaluation
from ariadne.datasets.timing import nearest_timestamp_errors_ms, summarize_errors_ms


def _modality(topic_name: str, topic_type: str) -> str | None:
    lowered = f"{topic_name} {topic_type}".lower()
    if "image" in lowered or "camera" in lowered:
        return "vision"
    if "imu" in lowered:
        return "imu"
    if "nlink" in lowered or "uwb" in lowered:
        return "uwb"
    if "pointcloud" in lowered or "velodyne" in lowered:
        return "lidar"
    if "navsat" in lowered or topic_name.endswith("/fix"):
        return "gnss"
    return None


def evaluate_s3e(path: Path) -> DatasetEvaluation:
    if not path.is_file():
        raise FileNotFoundError(f"S3E ROS2 bag does not exist: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        topics = connection.execute("SELECT id, name, type FROM topics ORDER BY id").fetchall()
        counts = dict(
            connection.execute(
                "SELECT topic_id, COUNT(*) FROM messages GROUP BY topic_id"
            ).fetchall()
        )
        bounds = connection.execute(
            "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM messages"
        ).fetchone()
        first_by_topic = dict(
            connection.execute(
                "SELECT topic_id, MIN(timestamp) FROM messages GROUP BY topic_id"
            ).fetchall()
        )
    finally:
        connection.close()

    agents: set[str] = set()
    modalities: set[str] = set()
    agent_first_ns: dict[str, int] = {}
    topic_metrics: dict[str, int] = {}
    agent_modalities: dict[str, set[str]] = defaultdict(set)
    topic_ids: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for topic_id, name, topic_type in topics:
        parts = name.strip("/").split("/")
        agent = parts[0] if parts else "unknown"
        agents.add(agent)
        modality = _modality(name, topic_type)
        if modality is not None:
            modalities.add(modality)
            agent_modalities[agent].add(modality)
            topic_ids[agent][modality].append(topic_id)
        topic_metrics[name] = counts.get(topic_id, 0)
        first_ns = first_by_topic.get(topic_id)
        if first_ns is not None:
            agent_first_ns[agent] = min(agent_first_ns.get(agent, first_ns), first_ns)

    starts = list(agent_first_ns.values())
    start_skew_ms = (max(starts) - min(starts)) / 1e6 if starts else 0.0
    message_count, start_ns, end_ns = bounds
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    sync_errors: list[float] = []
    try:
        for agent in sorted(agents):
            imu_ids = topic_ids[agent]["imu"]
            vision_ids = topic_ids[agent]["vision"]
            if not imu_ids or not vision_ids:
                continue
            imu_placeholders = ",".join("?" for _ in imu_ids)
            vision_placeholders = ",".join("?" for _ in vision_ids)
            imu_timestamps = [
                row[0]
                for row in connection.execute(
                    f"SELECT timestamp FROM messages WHERE topic_id IN ({imu_placeholders})",
                    imu_ids,
                )
            ]
            vision_timestamps = [
                row[0]
                for row in connection.execute(
                    f"SELECT timestamp FROM messages WHERE topic_id IN ({vision_placeholders})",
                    vision_ids,
                )
            ]
            sync_errors.extend(nearest_timestamp_errors_ms(imu_timestamps, vision_timestamps))
    finally:
        connection.close()
    sync_metrics = summarize_errors_ms(np.asarray(sync_errors, dtype=np.float64))
    ground_truth_files = sorted(path.parent.glob("*_gt.txt"))
    if ground_truth_files:
        modalities.add("ground_truth")
    warnings: list[str] = []
    if "vision" not in modalities:
        warnings.append("selected S3E bag has no camera topics")
    if "imu" not in modalities:
        warnings.append("selected S3E bag has no IMU topics")
    status = "passed" if len(agents) >= 2 and "imu" in modalities else "failed"
    return DatasetEvaluation(
        dataset="s3e",
        status=status,
        agents=tuple(sorted(agents)),
        modalities=tuple(sorted(modalities)),
        metrics={
            "agent_count": len(agents),
            "topic_count": len(topics),
            "message_count": message_count,
            "duration_seconds": (end_ns - start_ns) / 1e9,
            "max_agent_start_skew_ms": start_skew_ms,
            "vision_message_count": sum(
                count
                for topic_id, name, topic_type in topics
                if _modality(name, topic_type) == "vision"
                for count in (counts.get(topic_id, 0),)
            ),
            "imu_message_count": sum(
                count
                for topic_id, name, topic_type in topics
                if _modality(name, topic_type) == "imu"
                for count in (counts.get(topic_id, 0),)
            ),
            "ground_truth_file_count": len(ground_truth_files),
            **sync_metrics,
        },
        warnings=tuple(warnings),
        details={
            "path": str(path),
            "agent_modalities": {
                agent: sorted(values) for agent, values in sorted(agent_modalities.items())
            },
            "topic_message_counts": topic_metrics,
            "ground_truth_files": [file.name for file in ground_truth_files],
        },
    )
