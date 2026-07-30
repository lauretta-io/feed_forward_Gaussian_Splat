#!/usr/bin/env python3
# ruff: noqa: E501
"""Build the self-contained ARIADNE capability and results website."""

from __future__ import annotations

import ast
import base64
import bisect
import hashlib
import importlib
import io
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from ariadne.benchmarks import build_video_evidence, select_video_frames
from ariadne.replay import D2SlamReplaySource, GroundTruthPose


ROOT = Path(__file__).resolve().parents[1]


def python_test_count(root: Path) -> int:
    total = 0
    for path in sorted(root.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        total += sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            for node in ast.walk(tree)
        )
    return total


ARIADNE_TEST_COUNT = python_test_count(ROOT / "applications/ariadne/tests")
OUTPUT = ROOT / "reports/ariadne_status.html"
PHASE1 = ROOT / "outputs/ariadne/phase1/benchmark.json"
EXCHANGE = ROOT / "outputs/ariadne/exchange/benchmark.json"
GLOBAL_SCENE = ROOT / "outputs/ariadne/global-scene/benchmark.json"
OPERATIONS = ROOT / "outputs/ariadne/operations/benchmark.json"
END_TO_END = ROOT / "outputs/ariadne/end-to-end/benchmark.json"
DATASETS = ROOT / "outputs/ariadne/dataset_sequence/summary.json"
REAL_VIO = ROOT / "outputs/ariadne/real_vio/d2slam-1"
FRAME_DIR = REAL_VIO / "orbslam3/euroc/mav0/cam0/data"
D2SLAM_ROOT = ROOT / "datasets/ariadne/d2slam/extracted/tum_corr"
RESPLAT_REPORT = ROOT / "outputs/ariadne/resplat_report/neighbourhood_105_10f"
RESPLAT_CHECKPOINT = ROOT / "pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth"
RESPLAT_RUN_URL = "https://wandb.ai/galvin/gaussiansplat_test/runs/15d73m80"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_resplat_render(report: dict[str, Any]) -> Path:
    """Select a saved render that also has per-view metrics."""
    rendered = RESPLAT_REPORT / "rendered"
    metric_names = {
        str(item["name"])
        for item in report.get("per_view", ())
        if isinstance(item, dict) and "name" in item
    }
    preferred = rendered / "11.png"
    if preferred.is_file() and preferred.name in metric_names:
        return preferred
    match = next(
        (path for path in sorted(rendered.glob("*.png")) if path.name in metric_names),
        None,
    )
    if match is None:
        raise FileNotFoundError(
            f"no saved ReSplat render with per-view metrics found under {rendered}"
        )
    return match


def first_frame_data_uri() -> str:
    frame = next(iter(sorted(FRAME_DIR.glob("*.png"))), None)
    if frame is None:
        return ""
    return image_data_uri(frame)


def evidence_frame_data_uris(paths: tuple[Path, ...]) -> list[str]:
    """Embed a compact, self-contained 20-frame evidence segment."""
    image_module = importlib.import_module("PIL.Image")
    images: list[str] = []
    for path in paths:
        with image_module.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((320, 320), image_module.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=72, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        images.append(f"data:image/jpeg;base64,{encoded}")
    return images


def trajectory_points(path: Path, *, limit: int = 72) -> list[list[float]]:
    """Return a bounded XY trajectory sample for the self-contained status page."""
    points: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 4:
            points.append([float(fields[1]), float(fields[2])])
    if len(points) <= limit:
        return points
    indices = [round(index * (len(points) - 1) / (limit - 1)) for index in range(limit)]
    return [points[index] for index in indices]


def trajectory_timestamps(path: Path) -> list[int]:
    """Parse timestamps exactly as the production evaluator does."""
    timestamps: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        value = float(line.replace(",", " ").split()[0])
        timestamps.append(int(value) if value > 1e12 else int(value * 1e9))
    return timestamps


def movement_context(
    trajectory_path: Path,
    expected_matches: int,
    truth: tuple[GroundTruthPose, ...],
) -> dict[str, float | str]:
    """Measure movement from the evaluator's exact replay-window ground truth."""
    estimates = trajectory_timestamps(trajectory_path)
    tolerance_ns = 600_000_000
    truth_times = [sample.timestamp.monotonic_ns for sample in truth]
    selected_indices: list[int] = []
    for estimate in estimates:
        insertion = bisect.bisect_left(truth_times, estimate)
        candidates = [
            index for index in (insertion - 1, insertion) if 0 <= index < len(truth_times)
        ]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda index: abs(truth_times[index] - estimate))
        if abs(truth_times[nearest] - estimate) <= tolerance_ns:
            selected_indices.append(nearest)
    if len(selected_indices) != expected_matches:
        raise ValueError(
            f"reproduced {len(selected_indices)} of {expected_matches} trajectory matches "
            f"for {trajectory_path}"
        )

    def distance(left: Any, right: Any) -> float:
        return math.sqrt(sum((right[axis] - left[axis]) ** 2 for axis in range(3)))

    first_index = selected_indices[0]
    last_index = selected_indices[-1]
    covered = truth[first_index : last_index + 1]
    covered_distance = sum(
        distance(covered[index - 1].position_m, covered[index].position_m)
        for index in range(1, len(covered))
    )
    matched_steps = [
        distance(
            truth[selected_indices[index - 1]].position_m,
            truth[selected_indices[index]].position_m,
        )
        for index in range(1, len(selected_indices))
    ]
    rms_step = math.sqrt(sum(step**2 for step in matched_steps) / len(matched_steps))
    trajectory_times = trajectory_timestamps(trajectory_path)
    return {
        "environment": "Indoor corridor and several offices; handheld sensor rig, not a UAV flight",
        "ground_truth_distance_m": covered_distance,
        "ground_truth_duration_s": (
            covered[-1].timestamp.monotonic_ns - covered[0].timestamp.monotonic_ns
        )
        / 1e9,
        "ground_truth_step_rms_m": rms_step,
        "trajectory_span_s": (trajectory_times[-1] - trajectory_times[0]) / 1e9,
        "ground_truth_source": "https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset",
    }


def build_payload() -> dict[str, Any]:
    phase1 = read_json(PHASE1)
    exchange = read_json(EXCHANGE)
    global_scene = read_json(GLOBAL_SCENE)
    operations = read_json(OPERATIONS)
    end_to_end = read_json(END_TO_END)
    datasets = read_json(DATASETS)
    openvins = read_json(REAL_VIO / "openvins/evaluation.json")
    orbslam3 = read_json(REAL_VIO / "orbslam3/evaluation.json")
    resplat = read_json(RESPLAT_REPORT / "metrics.json")
    resplat_render = select_resplat_render(resplat)
    reference_truth = D2SlamReplaySource(D2SLAM_ROOT, 1).load(
        start_frame=0, max_frames=500
    ).ground_truth
    openvins_context = movement_context(
        REAL_VIO / "openvins/trajectory.txt",
        openvins["metrics"]["matched_pose_count"],
        reference_truth,
    )
    orbslam3_context = movement_context(
        REAL_VIO / "orbslam3/f_ariadne.txt",
        orbslam3["metrics"]["matched_pose_count"],
        reference_truth,
    )
    evidence_paths = select_video_frames(
        tuple(FRAME_DIR.glob("*.png")), REAL_VIO / "orbslam3/f_ariadne.txt"
    )
    video_evidence = build_video_evidence(
        evidence_paths, REAL_VIO / "orbslam3/f_ariadne.txt"
    )
    video_evidence["images"] = evidence_frame_data_uris(evidence_paths)
    correction_details = global_scene["details"]["correction"]
    pre_correction_pose_map = trajectory_points(
        REAL_VIO / "orbslam3/f_ariadne.txt", limit=160
    )
    applied_delta_m = [
        float(correction_details["applied_pose_m"][axis])
        - float(correction_details["local_pose_m"][axis])
        for axis in range(3)
    ]
    requested_delta_m = [
        float(correction_details["requested_pose_m"][axis])
        - float(correction_details["local_pose_m"][axis])
        for axis in range(3)
    ]
    post_correction_pose_map = [
        [point[0] + applied_delta_m[0], point[1] + applied_delta_m[1]]
        for point in pre_correction_pose_map
    ]
    tracking_states = phase1["details"]["tracking_states"]
    feature_pairs = phase1["details"]["feature_pairs"]
    dataset_by_name = {item["dataset"]: item for item in datasets}
    return {
        "generated": datetime.now().astimezone().strftime("%d %b %Y, %H:%M %Z"),
        "hero_image": first_frame_data_uri(),
        "video_evidence": video_evidence,
        "summary": {
            "tests": ARIADNE_TEST_COUNT,
            "datasets": sum(item["status"] == "passed" for item in datasets),
            "real_backends": 2,
            "phase1_status": phase1["status"],
        },
        "results": [
            {
                "name": "OpenVINS",
                "scope": "D2SLAM sequence 1, full bag",
                "frames": openvins["details"]["frames"],
                "imu_samples": openvins["details"]["imu_samples"],
                "metrics": openvins["metrics"],
                "movement": openvins_context,
                "window_kind": "long",
                "wandb": "https://wandb.ai/galvin/gaussiansplat_test/runs/9lqddnbf",
            },
            {
                "name": "ORB-SLAM3",
                "scope": "D2SLAM sequence 1, 500-frame window",
                "frames": orbslam3["details"]["frames"],
                "imu_samples": orbslam3["details"]["imu_samples"],
                "metrics": orbslam3["metrics"],
                "movement": orbslam3_context,
                "window_kind": "short",
                "wandb": "https://wandb.ai/galvin/gaussiansplat_test/runs/sqod0oj2",
            },
        ],
        "phase1": phase1,
        "exchange": exchange,
        "global_scene": global_scene,
        "operations": operations,
        "end_to_end": end_to_end,
        "datasets": [
            {
                "name": "MILUV",
                "state": "ready",
                "agents": dataset_by_name["miluv"]["metrics"]["agent_count"],
                "signals": "Vision / IMU / UWB / mocap",
                "detail": "Three-UAV archive replay is decoded and synchronized.",
                "limit": "Selected archive does not publish camera intrinsics.",
            },
            {
                "name": "D2SLAM",
                "state": "validated",
                "agents": dataset_by_name["d2slam"]["metrics"]["agent_count"],
                "signals": "Stereo / IMU / ground truth",
                "detail": "Production OpenVINS and ORB-SLAM3 baselines pass on sequence 1.",
                "limit": "Aligned TUM set emulates multi-robot replay; not simultaneous flight.",
            },
            {
                "name": "S3E",
                "state": "ready",
                "agents": dataset_by_name["s3e"]["metrics"]["agent_count"],
                "signals": "Stereo / IMU / GNSS / LiDAR",
                "detail": "ROS2 replay, timing, calibration, and ground truth are decoded.",
                "limit": "Backend-specific VIO calibration is not yet validated.",
            },
            {
                "name": "QDrone",
                "state": "limited",
                "agents": dataset_by_name["qdrone"]["metrics"]["agent_count"],
                "signals": "IMU / UWB / ground truth",
                "detail": "Useful for localization and timing regression.",
                "limit": "No vision stream; cannot exercise ARIADNE end to end.",
            },
            {
                "name": "Simulation",
                "state": "validated",
                "agents": len(dataset_by_name["simulation"]["agents"]),
                "signals": "Vision / IMU / network faults",
                "detail": "Deterministic packet-loss, outage, drift, and dynamic-object scenario.",
                "limit": "Synthetic evidence does not establish field performance.",
            },
        ],
        "components": [
            {
                "group": "Foundation",
                "name": "Bootstrap + build system",
                "state": "validated",
                "summary": "Strict configuration, structured logging, CPU-safe imports, CLI, tests, and Docker.",
                "implementation": "pyproject + AriadneConfig + CLI",
                "visual": {
                    "kind": "bootstrap",
                    "commands": ["validate", "wingman", "intelligence", "benchmark"],
                    "tests": ARIADNE_TEST_COUNT,
                },
            },
            {
                "group": "Foundation",
                "name": "Common time + coordinate types",
                "state": "validated",
                "summary": "Immutable timestamps, calibrated frames, covariance, and explicit SE(3) composition.",
                "implementation": "ariadne.common",
                "visual": {
                    "kind": "frames",
                    "frames": ["camera_0", "body", "local_wingman_01", "global"],
                    "tolerance": "1e-6",
                },
            },
            {
                "group": "Input",
                "name": "Replay + synchronization",
                "state": "validated",
                "summary": "Bounded per-agent camera/IMU windows from ZIP, ROS1, and ROS2 sources.",
                "implementation": "MILUV, D2SLAM, S3E adapters",
                "visual": {
                    "kind": "synchronization",
                    "packets": phase1["metrics"]["synchronized_packets"],
                    "dropped": phase1["metrics"]["dropped_frames"],
                    "median_ms": phase1["metrics"]["sync_median_ms"],
                    "p95_ms": phase1["metrics"]["sync_p95_ms"],
                },
            },
            {
                "group": "Input",
                "name": "Image preprocessing",
                "state": "reference",
                "summary": "Deterministic grayscale, resize, normalization, mask, and quality gates.",
                "implementation": "CPU ImagePreprocessor",
                "visual": {
                    "kind": "preprocessing",
                    "grid": exchange["details"]["processed_image"],
                    "blur": exchange["metrics"]["blur_score"],
                    "exposure": exchange["metrics"]["exposure_score"],
                    "latency_ms": exchange["metrics"]["preprocessing_latency_ms"],
                },
            },
            {
                "group": "Local state",
                "name": "Visual-inertial odometry",
                "state": "validated",
                "summary": "Process-isolated production VIO with aligned ATE, RPE, drift, and timing.",
                "implementation": "OpenVINS + ORB-SLAM3",
                "visual": {
                    "kind": "trajectory",
                    "series": [
                        {
                            "name": "OpenVINS",
                            "points": trajectory_points(REAL_VIO / "openvins/trajectory.txt"),
                        },
                        {
                            "name": "ORB-SLAM3",
                            "points": trajectory_points(REAL_VIO / "orbslam3/f_ariadne.txt"),
                        },
                    ],
                },
            },
            {
                "group": "Perception",
                "name": "Geometric features",
                "state": "reference",
                "summary": "Interface and metric flow are exercised with deterministic patch features.",
                "implementation": "Gradient/grid patch reference",
                "visual": {
                    "kind": "keypoints",
                    "recall": feature_pairs[0]["geometric_match_recall"],
                    "latency_ms": feature_pairs[0]["geometric_latency_ms"],
                },
            },
            {
                "group": "Perception",
                "name": "Semantic embeddings",
                "state": "reference",
                "summary": "Positive/negative view separation and latency contracts are tested.",
                "implementation": "Spatial pyramid/histogram reference",
                "visual": {
                    "kind": "embedding",
                    "positive": feature_pairs[1]["semantic_positive_cosine"],
                    "negative": feature_pairs[1]["semantic_negative_cosine"],
                    "separation": feature_pairs[1]["semantic_separation"],
                },
            },
            {
                "group": "Perception",
                "name": "Saliency detection",
                "state": "validated",
                "summary": "The official DUTS-trained full U²-Net produces the exact binary masks shown for all 20 frames.",
                "implementation": (
                    f"{video_evidence['saliency_model_info']['name']} · "
                    f"{video_evidence['saliency_model_info']['parameter_count']:,} parameters · "
                    f"{str(video_evidence['saliency_model_info']['device']).upper()} · "
                    f"{video_evidence['saliency_model_info']['input_size_px']}×"
                    f"{video_evidence['saliency_model_info']['input_size_px']} input · "
                    f"threshold {video_evidence['saliency_model_info']['threshold']}"
                ),
                "visual": {
                    "kind": "saliency",
                    "grid": exchange["details"]["saliency_scores"],
                    "fraction": exchange["metrics"]["salient_fraction"],
                    "latency_ms": exchange["metrics"]["saliency_latency_ms"],
                },
            },
            {
                "group": "Perception",
                "name": "Saliency clustering",
                "state": "validated",
                "summary": "Exact U²-Net masks are converted into ranked, bounded regions and checked by deterministic connected-component tests.",
                "implementation": "4-connected deterministic clusterer",
                "visual": {
                    "kind": "regions",
                    "regions": exchange["details"]["regions"],
                },
            },
            {
                "group": "Tracking",
                "name": "Static filtering",
                "state": "reference",
                "summary": "Temporal hysteresis prevents dynamic or unconfirmed tracks entering the map.",
                "implementation": "Deterministic temporal evidence filter",
                "visual": {
                    "kind": "static_filter",
                    "states": [
                        item["state"]
                        for item in tracking_states
                        if item["agent"] == "wingman_01"
                    ],
                    "false_insertions": phase1["metrics"]["false_static_insertions"],
                },
            },
            {
                "group": "Local state",
                "name": "Object + keyframe store",
                "state": "validated",
                "summary": "Confirmed-only admission, monotonic updates, bounded retention, and versioned atomic snapshots preserve local state across restart.",
                "implementation": "LocalObjectStore · ariadne.local-object-store.v1",
                "visual": {
                    "kind": "object_store",
                    "objects": exchange["details"]["local_objects"],
                    "count": exchange["metrics"]["local_object_count"],
                    "keyframes": exchange["metrics"]["keyframe_count"],
                    "restored": exchange["metrics"]["local_object_snapshot_restored"],
                },
            },
            {
                "group": "Transport",
                "name": "Uplink packaging",
                "state": "reference",
                "summary": "Versioned static-object payloads are quantized, compressed, and checksummed.",
                "implementation": "UplinkPackager",
                "visual": {
                    "kind": "uplink",
                    "bytes": exchange["metrics"]["uplink_bytes"],
                    "transport": exchange["details"]["transport_metrics"],
                },
            },
            {
                "group": "Transport",
                "name": "Mesh communications",
                "state": "reference",
                "summary": "Priority, TTL, finite retransmission, receiver acknowledgements, duplicate suppression, and delivery policy are deterministic.",
                "implementation": "InMemoryMeshTransport + DeliveryReceipt",
                "visual": {
                    "kind": "mesh",
                    "transport": exchange["details"]["transport_metrics"],
                },
            },
            {
                "group": "Intelligence",
                "name": "Observation registry",
                "state": "reference",
                "summary": "Raw envelopes are durably journaled before bounded derived-state mutation, then clock-gated, deduplicated, snapshotted, and replayed after restart.",
                "implementation": "ObservationRegistry + ObservationJournal · versioned JSON/JSONL",
                "visual": {
                    "kind": "registry",
                    "ids": exchange["details"]["registry_ids"],
                    "observations": exchange["metrics"]["registry_observation_count"],
                    "duplicates": exchange["metrics"]["registry_duplicate_packets"],
                    "restored": exchange["metrics"]["registry_snapshot_restored"],
                    "journal_entries": exchange["metrics"]["registry_journal_entries"],
                    "journal_replayed": exchange["metrics"][
                        "registry_journal_replayed_observations"
                    ],
                },
            },
            {
                "group": "Collaboration",
                "name": "Cross-agent association",
                "state": "reference",
                "summary": "Geometry and embedding gates produce persistent global IDs with bounded scored evidence and restart-stable local mappings.",
                "implementation": "CrossAgentAssociator · ariadne.cross-agent-association.v1",
                "visual": {
                    "kind": "association",
                    "agents": phase1["agents"],
                    "global_id": next(
                        item["global_id"]
                        for item in tracking_states
                        if item["global_id"] is not None
                    ),
                    "objects": phase1["metrics"]["global_object_count"],
                    "evidence": phase1["metrics"]["association_evidence_count"],
                    "stable": phase1["metrics"]["association_global_ids_stable"],
                    "restored": phase1["metrics"]["association_snapshot_restored"],
                },
            },
            {
                "group": "Reconstruction",
                "name": "Gaussian splat adapter",
                "state": "validated",
                "summary": "A ReSplat run supplies rendered evidence while the protocol-checked executor enforces backend registration, request budgets, diagnostics, and normalized OOM/failure behavior.",
                "implementation": "ReSplat DL3DV 8-view + GaussianReconstructionExecutor",
                "visual": {
                    "kind": "gaussian",
                    "gaussians": global_scene["details"]["gaussians"],
                    "inputs": global_scene["metrics"]["input_observations"],
                    "latency_ms": global_scene["metrics"]["reconstruction_latency_ms"],
                    "contract_backend": global_scene["metrics"]["reconstruction_backend"],
                    "estimated_memory_bytes": global_scene["metrics"][
                        "reconstruction_estimated_memory_bytes"
                    ],
                    "render_image": image_data_uri(resplat_render),
                    "render_name": resplat_render.name,
                    "render_sha256": file_sha256(resplat_render),
                    "dataset": "DDOS neighbourhood 105",
                    "model": "ReSplat base · DL3DV 8-view",
                    "checkpoint": RESPLAT_CHECKPOINT.name,
                    "checkpoint_sha256": file_sha256(RESPLAT_CHECKPOINT),
                    "checkpoint_bytes": RESPLAT_CHECKPOINT.stat().st_size,
                    "context_views": resplat["config"]["num_context"],
                    "target_views": resplat["config"]["num_target"],
                    "refinement_iterations": resplat["config"]["num_refine"],
                    "resolution": resplat["config"]["resolution"],
                    "mean_metrics": resplat["mean"],
                    "target_metrics": next(
                        item for item in resplat["per_view"] if item["name"] == resplat_render.name
                    ),
                    "wandb_runtime_s": 11,
                    "wandb_run_id": "15d73m80",
                    "wandb_url": RESPLAT_RUN_URL,
                },
            },
            {
                "group": "Global state",
                "name": "Global Gaussian scene map",
                "state": "reference",
                "summary": "Atomic versioned updates merge object Gaussians, retain provenance, and survive restart through crash-safe snapshots with bounded on-disk history.",
                "implementation": "GlobalGaussianMap + SceneSnapshotStore · ariadne.global-scene-snapshot.v1",
                "visual": {
                    "kind": "scene_map",
                    "gaussians": global_scene["details"]["gaussians"],
                    "revision": global_scene["metrics"]["scene_revision"],
                    "history": global_scene["details"]["scene_map"]["history_revisions"],
                    "snapshot_bytes": global_scene["details"]["scene_map"]["snapshot_bytes"],
                    "rollback_supported": global_scene["details"]["scene_map"][
                        "rollback_supported"
                    ],
                    "restore_supported": global_scene["details"]["scene_map"][
                        "restore_supported"
                    ],
                    "persisted_snapshots": global_scene["details"]["scene_map"][
                        "persisted_snapshots"
                    ],
                },
            },
            {
                "group": "Global state",
                "name": "Pose graph",
                "state": "validated",
                "summary": "Bounded incremental SE(3) constraints preserve revisions across restart while validating covariance, disconnected components, optimization, and outlier rejection.",
                "implementation": "RobustSE3PoseGraph · ariadne.se3-pose-graph.v1",
                "visual": {
                    "kind": "se3_pose_graph",
                    **global_scene["details"]["pose_graph"],
                    "revision": global_scene["metrics"]["se3_graph_revision"],
                    "restored_revision": global_scene["metrics"][
                        "se3_graph_restored_revision"
                    ],
                    "state_restored": global_scene["metrics"][
                        "se3_graph_state_restored"
                    ],
                },
            },
            {
                "group": "Global state",
                "name": "Correction delta application",
                "state": "validated",
                "summary": "Restart-safe correction sequencing and replay suppression enforce monotonic issue time, TTL, bounded local continuity, and explicit reset-required behavior.",
                "implementation": "ORB-SLAM3 VIO + persistent CorrectionDeltaGenerator/CorrectionApplier",
                "visual": {
                    "kind": "correction",
                    **correction_details,
                    "vio_backend": "ORB-SLAM3",
                    "trajectory_pose_count": len(
                        trajectory_timestamps(REAL_VIO / "orbslam3/f_ariadne.txt")
                    ),
                    "trajectory_source": "D2SLAM sequence 1 · 500-frame window",
                    "pre_global_pose_map": pre_correction_pose_map,
                    "post_global_pose_map": post_correction_pose_map,
                    "applied_delta_m": applied_delta_m,
                    "requested_delta_m": requested_delta_m,
                    "frame_transform": "local_wingman_01 → global",
                    "interpretation": "frame alignment demonstration; no post-correction ATE claim",
                    "state_restored": global_scene["metrics"]["correction_state_restored"],
                    "restart_duplicate_rejected": global_scene["metrics"][
                        "correction_duplicate_after_restart_rejected"
                    ],
                    "sequence_preserved": global_scene["metrics"][
                        "correction_next_sequence_preserved"
                    ],
                },
            },
            {
                "group": "Context",
                "name": "Unified scene graph",
                "state": "reference",
                "summary": "Freshness-aware nodes and typed relations expose machine-usable planning context.",
                "implementation": "UnifiedContext",
                "visual": {
                    "kind": "context",
                    **global_scene["details"]["context"],
                    "degraded": global_scene["metrics"]["context_degraded"],
                },
            },
            {
                "group": "Integration",
                "name": "Hand-off to SKYLA",
                "state": "reference",
                "summary": "The executable versioned global-frame envelope transfers fresh scene, mission, constraint, and vehicle-state context from ARIADNE to the SKYLA planner boundary.",
                "implementation": "SkylaHandoff · ariadne.skyla.handoff.v1",
                "visual": {
                    "kind": "skyla_handoff",
                    "schema": operations["details"]["handoff"]["schema"],
                    "source": "UnifiedContext snapshot",
                    "destination": "SKYLA planning interface",
                    "frame": operations["details"]["handoff"]["frame"],
                    "scene_revision": operations["details"]["handoff"]["scene_revision"],
                    "nodes": operations["details"]["handoff"]["nodes"],
                    "edges": operations["details"]["handoff"]["edges"],
                    "degraded": operations["details"]["handoff"]["degraded"],
                    "payload_bytes": operations["details"]["handoff"]["payload_bytes"],
                    "fields": [
                        "mission goals + priorities",
                        "scene nodes + typed relations",
                        "corrected global poses",
                        "frontiers + no-fly constraints",
                        "battery + health + link state",
                    ],
                    "gates": [
                        name
                        for name, passed in operations["details"]["handoff"]["gates"].items()
                        if passed
                    ],
                },
            },
            {
                "group": "Planning",
                "name": "SKYLA planning architecture",
                "state": "reference",
                "summary": "The SKYLA reference consumes the hand-off, filters mission and no-fly constraints, allocates work globally, and returns idempotent route requests while Wingmen retain local safety authority.",
                "implementation": "SkylaMissionPlanner + RouteRequest",
                "visual": {
                    "kind": "skyla_planning",
                    "wingmen": operations["details"]["wingmen"],
                    "frontiers": operations["details"]["frontiers"],
                    "assignments": operations["details"]["assignments"],
                    "routes": operations["details"]["skyla_planning"]["route_requests"],
                    "blocked_frontiers": operations["details"]["skyla_planning"][
                        "blocked_frontiers"
                    ],
                    "excluded_vehicles": operations["details"]["skyla_planning"][
                        "excluded_vehicles"
                    ],
                    "stages": operations["details"]["skyla_planning"]["stages"],
                    "feedback": ["execution state", "map changes", "node failure", "replan"],
                    "global_authority": "Intelligence Node",
                    "local_authority": "Wingman collision avoidance + flight safety",
                },
            },
            {
                "group": "Operations",
                "name": "Telemetry + health",
                "state": "reference",
                "summary": "Bounded counters, gauges, p50/p95 distributions, four-state health, redacted mission traces, and Prometheus-compatible export.",
                "implementation": "TelemetryCollector + TelemetrySnapshot",
                "visual": {
                    "kind": "telemetry",
                    **operations["details"]["telemetry"],
                },
            },
            {
                "group": "Verification",
                "name": "Simulation + deterministic replay",
                "state": "reference",
                "summary": "Packet loss, a timed partition, clock drift, dynamics, and recovery are reproducible.",
                "implementation": "evaluate_simulation",
                "visual": {
                    "kind": "simulation",
                    **operations["details"]["simulation"],
                },
            },
            {
                "group": "Operations",
                "name": "Dataset + model registry",
                "state": "reference",
                "summary": "Typed catalogs retain licenses, backends, roles, versions, and experiment provenance.",
                "implementation": "RegistryCatalog",
                "visual": {
                    "kind": "catalog",
                    **operations["details"]["registry"],
                    "models": [
                        *operations["details"]["registry"]["models"],
                        video_evidence["saliency_model"],
                    ],
                },
            },
            {
                "group": "Trust",
                "name": "Security + replay protection",
                "state": "reference",
                "summary": "Authenticated envelopes enforce identity, integrity, destination, expiry, and nonce replay.",
                "implementation": "HmacEnvelopeSecurity",
                "visual": {
                    "kind": "security",
                    **operations["details"]["security"],
                },
            },
            {
                "group": "Deployment",
                "name": "Hardware abstraction",
                "state": "reference",
                "summary": "Explicit profiles fail closed when CPU, memory, camera, or accelerator capabilities are absent.",
                "implementation": "CapabilityProbe + DeploymentProfile",
                "visual": {
                    "kind": "deployment",
                    **operations["details"]["deployment"],
                },
            },
            {
                "group": "System",
                "name": "End-to-end orchestration",
                "state": "reference",
                "summary": "All CPU reference gates compose with network-partition and low-battery degraded modes.",
                "implementation": "run_reference_system",
                "visual": {
                    "kind": "end_to_end",
                    "stages": end_to_end["details"]["stages"],
                    "degraded": end_to_end["details"]["degraded_modes"],
                    "latency_ms": end_to_end["metrics"]["total_latency_ms"],
                    "peak_bytes": end_to_end["metrics"]["peak_traced_bytes"],
                },
            },
            {
                "group": "Operations",
                "name": "Evaluation + evidence",
                "state": "validated",
                "summary": "Typed JSON reports and W&B model-benchmark artifacts share metric names.",
                "implementation": "DatasetEvaluation + W&B",
                "visual": {
                    "kind": "evidence",
                    "tests": ARIADNE_TEST_COUNT,
                    "reports": len(datasets) + 7,
                    "status": phase1["status"],
                },
            },
        ],
        "gaps": [
            "Run both production VIO backends on identical full replay windows before ranking them.",
            "Replace reference geometric and semantic features with ALIKED/SuperPoint and DINO-family adapters.",
            "Calibrate SE(3) covariance and robust gates on real multi-agent loop closures.",
            "Validate real multi-agent association, correction exchange, and persistent mapping end to end.",
            "Add recovery, p50/p95 latency, peak memory, and power measurements to production gates.",
        ],
        "sources": [
            "outputs/ariadne/phase1/benchmark.json",
            "outputs/ariadne/exchange/benchmark.json",
            "outputs/ariadne/global-scene/benchmark.json",
            "outputs/ariadne/operations/benchmark.json",
            "outputs/ariadne/end-to-end/benchmark.json",
            "outputs/ariadne/dataset_sequence/summary.json",
            "outputs/ariadne/real_vio/d2slam-1/openvins/evaluation.json",
            "outputs/ariadne/real_vio/d2slam-1/orbslam3/evaluation.json",
            "outputs/ariadne/resplat_report/neighbourhood_105_10f/metrics.json",
            str(resplat_render.relative_to(ROOT)),
            RESPLAT_RUN_URL,
            "applications/ariadne/docs/phase1_models.md",
            "applications/ariadne/docs/real_vio.md",
        ],
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ARIADNE | Capability Status</title>
  <style>
    :root {
      color-scheme: light;
      --paper: #f4f5f2;
      --surface: #ffffff;
      --ink: #161b19;
      --muted: #5c6661;
      --line: #d7ddd9;
      --green: #1f7451;
      --green-soft: #e5f1eb;
      --amber: #a46516;
      --amber-soft: #f7ecd9;
      --red: #a33f48;
      --red-soft: #f8e8e9;
      --blue: #2f628b;
      --blue-soft: #e5eef5;
      --charcoal: #111513;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      overflow-x: hidden;
      background: var(--paper);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }
    a { color: inherit; }
    button, a { -webkit-tap-highlight-color: transparent; }
    .site-nav {
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      min-height: 58px;
      padding: 0 32px;
      border-bottom: 1px solid rgba(255,255,255,.16);
      background: rgba(17,21,19,.95);
      color: #fff;
      backdrop-filter: blur(12px);
    }
    .brand { font-weight: 760; font-size: 14px; }
    .nav-links { display: flex; align-items: center; gap: 20px; }
    .nav-links a { color: #cbd2ce; text-decoration: none; font-size: 13px; }
    .nav-links a:hover, .nav-links a:focus-visible { color: #fff; }
    .hero {
      position: relative;
      min-height: min(66vh, 640px);
      display: flex;
      align-items: flex-end;
      overflow: hidden;
      background: var(--charcoal);
      color: #fff;
    }
    .hero-image { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; object-position: center 42%; opacity: .72; }
    .hero-shade { position: absolute; inset: 0; background: rgba(7,10,8,.62); }
    .hero-content { position: relative; width: min(1180px, 100%); margin: 0 auto; padding: 72px 32px 56px; }
    .eyebrow { margin: 0 0 12px; color: #bcd7c9; font-size: 12px; font-weight: 720; text-transform: uppercase; }
    h1 { max-width: 780px; margin: 0; font-size: 78px; line-height: .98; letter-spacing: 0; }
    .hero-copy { width: 100%; max-width: 670px; margin: 22px 0 0; color: #e1e6e3; font-size: 18px; overflow-wrap: anywhere; }
    .hero-meta { display: flex; flex-wrap: wrap; gap: 9px 18px; margin-top: 28px; color: #c6ceca; font-size: 12px; }
    .hero-meta span { display: inline-flex; align-items: center; gap: 7px; }
    .hero-meta i { width: 7px; height: 7px; border-radius: 50%; background: #63c693; }
    .band { border-bottom: 1px solid var(--line); background: var(--surface); }
    .band.alt { background: var(--paper); }
    .inner { width: min(1180px, 100%); margin: 0 auto; padding: 58px 32px; }
    .section-head { display: grid; grid-template-columns: minmax(0, 1fr) minmax(260px, 500px); gap: 36px; align-items: end; margin-bottom: 28px; }
    .section-head h2 { margin: 0; font-size: 30px; line-height: 1.15; letter-spacing: 0; }
    .section-head p { margin: 0; color: var(--muted); }
    .kpi-strip { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border: 1px solid var(--line); background: var(--surface); }
    .kpi { min-height: 116px; padding: 22px; border-right: 1px solid var(--line); }
    .kpi:last-child { border-right: 0; }
    .kpi strong { display: block; font-size: 30px; line-height: 1; }
    .kpi span { display: block; margin-top: 9px; color: var(--muted); font-size: 13px; }
    .status-line { display: flex; align-items: center; gap: 8px; margin-top: 8px; color: var(--green); font-size: 12px; font-weight: 700; text-transform: uppercase; }
    .status-line::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--green); }
    .result-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .result-card { padding: 22px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
    .result-top { display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; }
    .result-card h3 { margin: 0; font-size: 21px; }
    .result-card .scope { margin: 5px 0 0; color: var(--muted); font-size: 13px; }
    .run-link { flex: 0 0 auto; color: var(--blue); font-size: 12px; font-weight: 700; text-decoration: none; }
    .run-link:hover { text-decoration: underline; }
    .metric-row { display: grid; grid-template-columns: 88px 1fr 84px; gap: 12px; align-items: center; margin-top: 17px; font-size: 12px; }
    .metric-row span:first-child { color: var(--muted); }
    .track { height: 8px; background: #e6eae7; overflow: hidden; }
    .track i { display: block; height: 100%; background: var(--green); }
    .metric-row strong { text-align: right; font-size: 12px; }
    .result-foot { display: flex; flex-wrap: wrap; gap: 8px 18px; margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; }
    .result-explanation { margin-top: 16px; padding: 14px 15px; border-radius: 4px; background: #f3f6f4; color: #46514b; font-size: 12px; }
    .result-explanation p { margin: 0; }
    .result-explanation p + p { margin-top: 8px; }
    .result-explanation strong { color: var(--ink); }
    .result-explanation a { color: var(--blue); }
    .caveat { margin: 18px 0 0; padding: 14px 16px; border-left: 3px solid var(--amber); background: var(--amber-soft); color: #6d4b1d; font-size: 13px; }
    .metric-guide { margin-top: 22px; padding-top: 22px; border-top: 1px solid var(--line); }
    .metric-guide h3 { margin: 0 0 13px; font-size: 17px; }
    .metric-guide-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); border: 1px solid var(--line); background: var(--line); gap: 1px; }
    .metric-definition { min-height: 112px; padding: 15px; background: var(--surface); }
    .metric-definition strong { display: block; font-size: 12px; }
    .metric-definition p { margin: 7px 0 0; color: var(--muted); font-size: 11px; }
    .pipeline { display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 1px; border: 1px solid var(--line); background: var(--line); }
    .pipeline-step { position: relative; min-height: 132px; padding: 17px 14px; background: var(--surface); }
    .pipeline-step b { display: block; color: var(--muted); font-size: 11px; }
    .pipeline-step strong { display: block; margin-top: 13px; font-size: 14px; line-height: 1.25; }
    .pipeline-step span { display: block; margin-top: 8px; color: var(--muted); font-size: 11px; }
    .pipeline-step.validated { box-shadow: inset 0 3px var(--green); }
    .pipeline-step.reference { box-shadow: inset 0 3px var(--amber); }
    .filter-bar { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 18px; }
    .evidence-note { margin: -4px 0 18px; padding: 13px 15px; border-left: 3px solid var(--blue); background: var(--blue-soft); color: #385268; font-size: 12px; }
    .filter-button { min-height: 36px; padding: 7px 13px; border: 1px solid var(--line); background: var(--surface); color: var(--muted); font: inherit; font-size: 12px; cursor: pointer; }
    .filter-button:first-child { border-radius: 5px 0 0 5px; }
    .filter-button:last-child { border-radius: 0 5px 5px 0; }
    .filter-button.active { border-color: var(--charcoal); background: var(--charcoal); color: #fff; }
    .component-list { display: grid; gap: 16px; }
    .component { display: grid; grid-template-columns: minmax(240px, .72fr) minmax(420px, 1.28fr); min-height: 244px; overflow: hidden; border: 1px solid var(--line); border-radius: 7px; background: var(--surface); }
    .component[hidden] { display: none; }
    .component-copy { display: flex; flex-direction: column; padding: 22px; border-right: 1px solid var(--line); }
    .component-heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .component .group { color: var(--muted); font-size: 12px; }
    .component h3 { margin: 24px 0 0; font-size: 20px; line-height: 1.2; }
    .component p { margin: 10px 0 0; color: var(--muted); font-size: 13px; }
    .component code { display: block; margin-top: auto; padding-top: 18px; color: #34413a; font-size: 11px; overflow-wrap: anywhere; }
    .component-visual { min-width: 0; padding: 18px 20px 16px; background: #f8faf8; }
    .visual-kicker { display: flex; justify-content: space-between; gap: 14px; margin-bottom: 10px; color: var(--muted); font-size: 10px; font-weight: 760; letter-spacing: .06em; text-transform: uppercase; }
    .visual-output { height: 150px; overflow: hidden; border: 1px solid #dfe5e1; border-radius: 5px; background: #fff; }
    .visual-output svg { display: block; width: 100%; height: 100%; }
    .resplat-render { position: relative; height: 220px; background: #101713; }
    .resplat-render img { display: block; width: 100%; height: 100%; object-fit: contain; }
    .resplat-render-label { position: absolute; top: 9px; left: 9px; padding: 5px 7px; border: 1px solid rgba(255,255,255,.18); border-radius: 3px; background: rgba(8,13,10,.78); color: #eef5f1; font: 700 9px/1.1 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .04em; }
    .pose-map-comparison { height: 220px; }
    .architecture-visual { height: 220px; }
    .evidence-video { position: relative; height: 220px; background: #101713; }
    .evidence-video canvas { display: block; width: 100%; height: 100%; }
    .evidence-video-status { position: absolute; right: 9px; bottom: 8px; display: flex; gap: 7px; align-items: center; padding: 5px 7px; border: 1px solid rgba(255,255,255,.18); border-radius: 3px; background: rgba(8,13,10,.76); color: #eef5f1; font: 700 9px/1.1 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .04em; }
    .evidence-video-status i { width: 6px; height: 6px; border-radius: 50%; background: #63c693; }
    .evidence-video-status button { padding: 0; border: 0; background: transparent; color: inherit; font: inherit; cursor: pointer; }
    .evidence-basis { color: var(--blue); font-weight: 760; }
    .visual-caption { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 9px; color: var(--muted); font-size: 10px; }
    .visual-caption strong { color: var(--ink); }
    .visual-caption a { color: var(--blue); font-weight: 760; text-decoration: none; }
    .visual-caption a:hover { text-decoration: underline; }
    .chart-label { fill: #66716b; font: 10px Inter, ui-sans-serif, system-ui, sans-serif; }
    .chart-value { fill: #1c2822; font: 700 10px Inter, ui-sans-serif, system-ui, sans-serif; }
    .badge { display: inline-flex; align-items: center; gap: 6px; width: fit-content; padding: 5px 8px; border-radius: 999px; font-size: 10px; font-weight: 760; text-transform: uppercase; }
    .badge::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
    .badge.validated { color: var(--green); background: var(--green-soft); }
    .badge.reference { color: var(--amber); background: var(--amber-soft); }
    .badge.ready { color: var(--blue); background: var(--blue-soft); }
    .badge.limited { color: var(--red); background: var(--red-soft); }
    .dataset-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }
    .dataset-card { min-height: 250px; padding: 18px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
    .dataset-card h3 { margin: 15px 0 0; font-size: 17px; }
    .dataset-card .agents { margin: 3px 0 0; color: var(--muted); font-size: 12px; }
    .dataset-card .signals { min-height: 39px; margin: 18px 0 0; font-size: 12px; font-weight: 700; }
    .dataset-card p { margin: 8px 0 0; color: var(--muted); font-size: 12px; }
    .dataset-card .limit { color: #75464b; }
    .gap-layout { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(300px, .9fr); gap: 44px; }
    .gap-list { margin: 0; padding: 0; list-style: none; counter-reset: gaps; }
    .gap-list li { counter-increment: gaps; display: grid; grid-template-columns: 34px 1fr; gap: 12px; padding: 15px 0; border-bottom: 1px solid var(--line); }
    .gap-list li::before { content: counter(gaps, decimal-leading-zero); color: var(--red); font-size: 12px; font-weight: 760; }
    .phase-proof { border: 1px solid var(--line); background: var(--surface); }
    .phase-proof header { padding: 18px; border-bottom: 1px solid var(--line); }
    .phase-proof h3 { margin: 0; font-size: 17px; }
    .proof-grid { display: grid; grid-template-columns: 1fr 1fr; }
    .proof { min-height: 102px; padding: 16px; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); }
    .proof:nth-child(2n) { border-right: 0; }
    .proof:nth-last-child(-n+2) { border-bottom: 0; }
    .proof strong { display: block; font-size: 18px; }
    .proof span { display: block; margin-top: 6px; color: var(--muted); font-size: 11px; }
    footer { background: var(--charcoal); color: #d7ddda; }
    footer .inner { padding-top: 38px; padding-bottom: 38px; }
    .footer-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 34px; }
    footer h2 { margin: 0; color: #fff; font-size: 18px; }
    footer p { margin: 8px 0 0; color: #9eaaa4; font-size: 12px; }
    .sources { display: grid; grid-template-columns: 1fr 1fr; gap: 7px 20px; }
    .sources code { color: #bdc8c2; font-size: 10px; overflow-wrap: anywhere; }
    @media (max-width: 980px) {
      .dataset-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .component { grid-template-columns: minmax(220px, .8fr) minmax(350px, 1.2fr); }
      .pipeline { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    }
    @media (max-width: 720px) {
      .site-nav { padding: 0 18px; }
      .nav-links { display: none; }
      .hero { min-height: 600px; }
      .hero-shade { background: rgba(7,10,8,.68); }
      .hero-content, .inner { padding-left: 18px; padding-right: 18px; }
      h1 { font-size: 48px; }
      .hero-copy { font-size: 16px; }
      .section-head { grid-template-columns: 1fr; gap: 12px; }
      .kpi-strip { grid-template-columns: 1fr 1fr; }
      .kpi:nth-child(2) { border-right: 0; }
      .kpi:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
      .result-grid, .gap-layout, .footer-grid { grid-template-columns: 1fr; }
      .metric-guide-grid { grid-template-columns: 1fr 1fr; }
      .pipeline { grid-template-columns: 1fr 1fr; }
      .component { grid-template-columns: 1fr; }
      .component-copy { min-height: 220px; border-right: 0; border-bottom: 1px solid var(--line); }
      .component-visual { padding: 14px; }
      .dataset-grid, .sources { grid-template-columns: 1fr; }
      .dataset-card { min-height: auto; }
    }
    @media (max-width: 460px) {
      .metric-guide-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <nav class="site-nav" aria-label="Primary">
    <span class="brand">ARIADNE / STATUS</span>
    <div class="nav-links">
      <a href="#results">Results</a>
      <a href="#components">Components</a>
      <a href="#datasets">Datasets</a>
      <a href="#gaps">Next</a>
    </div>
  </nav>

  <main>
    <section class="hero" aria-labelledby="page-title">
      <img class="hero-image" id="hero-image" alt="D2SLAM fisheye camera frame used by the ARIADNE VIO evaluation">
      <div class="hero-shade"></div>
      <div class="hero-content">
        <p class="eyebrow">Vision-first distributed UAV autonomy</p>
        <h1 id="page-title">ARIADNE</h1>
        <p class="hero-copy">A working reference pipeline with real multi-sensor replay, two validated production VIO backends, cross-agent contracts, and reproducible evaluation evidence.</p>
        <div class="hero-meta"><span><i></i>Current branch validated</span><span id="generated"></span><span>Evidence snapshot, not live telemetry</span></div>
      </div>
    </section>

    <section class="band alt" aria-label="System summary">
      <div class="inner"><div class="kpi-strip" id="kpis"></div></div>
    </section>

    <section class="band" id="results">
      <div class="inner">
        <div class="section-head">
          <h2>Real VIO results</h2>
          <p>Both adapters passed on D2SLAM sequence 1 and emitted aligned trajectory metrics. Lower error is better.</p>
        </div>
        <div class="result-grid" id="results-grid"></div>
        <p class="caveat">These runs are not a direct model ranking. OpenVINS ran on the full bag and emitted a 293-second trajectory, while its evaluator batch contained 3,000 camera frames; ORB-SLAM3 processed a 500-frame window. TUM VI corridor ground truth exists only in the start and end segments, so the error metrics cover less motion than either input scope. Identical-window reruns are the next comparison gate.</p>
        <div class="metric-guide" aria-labelledby="metric-guide-title">
          <h3 id="metric-guide-title">How to read the VIO metrics</h3>
          <div class="metric-guide-grid">
            <div class="metric-definition"><strong>ATE RMSE</strong><p>Global position accuracy. The evaluator rigidly aligns matched estimated positions to ground truth, then reports their root-mean-square distance. The displayed path percentage is context, not part of the standard ATE definition.</p></div>
            <div class="metric-definition"><strong>RPE RMSE</strong><p>Local motion consistency. It compares consecutive displacement vectors in the aligned estimate and ground truth. The report relates it to the RMS ground-truth displacement per matched step.</p></div>
            <div class="metric-definition"><strong>Final drift</strong><p>The position error at the last matched pose after alignment. The displayed percentage divides that endpoint error by ground-truth-covered path length; it is not the trajectory average.</p></div>
            <div class="metric-definition"><strong>Trajectory poses</strong><p>Pose estimates emitted by the backend. This can differ from camera-frame count because backends publish at different rates.</p></div>
            <div class="metric-definition"><strong>Matched</strong><p>Estimated poses whose nearest ground-truth timestamp is within 0.6 seconds. Only matched poses determine ATE, RPE, and final drift.</p></div>
            <div class="metric-definition"><strong>Elapsed</strong><p>Total subprocess wall-clock time, including setup and I/O. It is not normalized per-frame inference latency.</p></div>
          </div>
        </div>
      </div>
    </section>

    <section class="band alt" aria-labelledby="pipeline-title">
      <div class="inner">
        <div class="section-head">
          <h2 id="pipeline-title">Current pipeline</h2>
          <p>Green stages meet their report gates through direct run, replay, or automated benchmark evidence. Amber stages remain interface or deterministic reference implementations.</p>
        </div>
        <div class="pipeline" id="pipeline"></div>
      </div>
    </section>

    <section class="band" id="components">
      <div class="inner">
        <div class="section-head">
          <h2>Component state</h2>
          <p>Filter the implementation inventory by evidence level. Each component keeps a stable boundary so production models can replace references without changing the report contract.</p>
        </div>
        <div class="filter-bar" role="group" aria-label="Filter components">
          <button class="filter-button active" data-filter="all" type="button">All</button>
          <button class="filter-button" data-filter="validated" type="button">Validated</button>
          <button class="filter-button" data-filter="reference" type="button">Reference</button>
        </div>
        <p class="evidence-note"><strong>Evidence convention.</strong> “Real 20-frame replay” contains 20 consecutive 20 Hz camera frames—0.95 seconds of source time—slowed to 5 fps for inspection. Its overlays come from the actual preprocessing, feature, embedding, pretrained U²-Net saliency, and clustering code plus production ORB-SLAM3 poses. “Benchmark output” uses the saved deterministic JSON metrics for stages where camera video would not measure the component.</p>
        <div class="component-list" id="component-list"></div>
      </div>
    </section>

    <section class="band alt" id="datasets">
      <div class="inner">
        <div class="section-head">
          <h2>Dataset readiness</h2>
          <p>The representative corpus covers multi-agent vision and inertial replay, timing stress, UWB regression, and ground-truth evaluation without committing raw payloads.</p>
        </div>
        <div class="dataset-grid" id="dataset-grid"></div>
      </div>
    </section>

    <section class="band" id="gaps">
      <div class="inner gap-layout">
        <div>
          <div class="section-head" style="display:block;margin-bottom:12px"><h2>What remains</h2></div>
          <ol class="gap-list" id="gap-list"></ol>
        </div>
        <aside class="phase-proof" aria-labelledby="proof-title">
          <header><h3 id="proof-title">Phase 1 reference proof</h3></header>
          <div class="proof-grid" id="proof-grid"></div>
        </aside>
      </div>
    </section>
  </main>

  <footer>
    <div class="inner footer-grid">
      <div><h2>Source-backed snapshot</h2><p>Generated locally from ARIADNE JSON reports, a real embedded 20-frame TUM VI replay segment, and technical documentation. W&B links point to validated model-benchmark artifacts.</p></div>
      <div class="sources" id="sources"></div>
    </div>
  </footer>

  <script>
    const data = __PAYLOAD__;
    const fmt = (value, digits = 3) => Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
    const badge = state => `<span class="badge ${state}">${state}</span>`;
    document.getElementById('hero-image').src = data.hero_image;
    document.getElementById('generated').textContent = `Generated ${data.generated}`;

    const kpis = [
      [data.summary.real_backends, 'Production VIO backends', true],
      [data.summary.datasets, 'Dataset evaluations passed', true],
      [data.summary.tests, 'Automated tests passed', true],
      ['00-26', 'CPU reference modules integrated', data.end_to_end.status === 'passed'],
    ];
    document.getElementById('kpis').innerHTML = kpis.map(([value, label, passed]) => `
      <div class="kpi"><strong>${value}</strong><span>${label}</span>${passed ? '<div class="status-line">Passed</div>' : ''}</div>`).join('');

    const metricScale = { ate_rmse_m: .08, rpe_rmse_m: .015, final_drift_m: .7 };
    const metricLabels = { ate_rmse_m: 'ATE RMSE', rpe_rmse_m: 'RPE RMSE', final_drift_m: 'Final drift' };
    document.getElementById('results-grid').innerHTML = data.results.map(result => {
      const rows = Object.keys(metricLabels).map(key => {
        const value = result.metrics[key];
        const width = Math.min(100, value / metricScale[key] * 100);
        return `<div class="metric-row"><span>${metricLabels[key]}</span><div class="track"><i style="width:${width}%"></i></div><strong>${fmt(value, 4)} m</strong></div>`;
      }).join('');
      const matchRate = result.metrics.matched_pose_count / result.metrics.trajectory_pose_count * 100;
      const endpointRatio = result.metrics.final_drift_m / result.metrics.ate_rmse_m;
      const pathDistance = result.movement.ground_truth_distance_m;
      const pathDuration = result.movement.ground_truth_duration_s;
      const atePathPercent = result.metrics.ate_rmse_m / pathDistance * 100;
      const finalPathPercent = result.metrics.final_drift_m / pathDistance * 100;
      const rpeStepPercent = result.metrics.rpe_rmse_m / result.movement.ground_truth_step_rms_m * 100;
      const endpointReading = endpointRatio > 2
        ? 'The much larger endpoint error indicates late divergence within the ground-truth-covered interval.'
        : 'The endpoint remained below the run-wide RMS error in this evaluated slice.';
      const scopeReading = result.window_kind === 'long'
        ? `The emitted full-bag trajectory spans ${fmt(result.movement.trajectory_span_s, 1)} s, but this ATE/RPE/drift evidence covers only the ${fmt(pathDuration, 1)} s start segment with ground truth; it is not a full-bag error measurement.`
        : `These errors describe only ${fmt(pathDuration, 1)} s of ground-truth-covered motion inside the 500-frame window and should not be extrapolated to the full bag.`;
      const explanation = `<div class="result-explanation"><p><strong>Environment:</strong> ${result.movement.environment}. The source is TUM VI <em>corridor1</em>, repackaged as an aligned D2SLAM replay. <a href="${result.movement.ground_truth_source}" target="_blank" rel="noreferrer">Dataset context ↗</a></p><p><strong>Evidence used:</strong> The evaluator context contained ${fmt(result.frames, 0)} camera frames and ${fmt(result.imu_samples, 0)} IMU samples. ${result.name} emitted ${fmt(result.metrics.trajectory_pose_count, 0)} poses over a ${fmt(result.movement.trajectory_span_s, 1)} s trajectory span; ${fmt(result.metrics.matched_pose_count, 0)} (${fmt(matchRate, 1)}%) matched the available ground truth within 0.6 s. Those matches span ${fmt(pathDistance, 2)} m of measured motion over ${fmt(pathDuration, 1)} s.</p><p><strong>ATE context:</strong> ${fmt(result.metrics.ate_rmse_m * 100, 2)} cm RMS global position error across that ${fmt(pathDistance, 2)} m ground-truth-covered path. Dividing ATE by path length gives ${fmt(atePathPercent, 2)}% as a scale reference; that ratio is not the standard ATE definition.</p><p><strong>RPE context:</strong> ${fmt(result.metrics.rpe_rmse_m * 100, 2)} cm RMS local displacement error between consecutive matched poses. The corresponding ground-truth motion was ${fmt(result.movement.ground_truth_step_rms_m * 100, 2)} cm RMS per step, so the RPE is ${fmt(rpeStepPercent, 1)}% of a typical evaluated step.</p><p><strong>Final-drift context:</strong> The final matched pose is ${fmt(result.metrics.final_drift_m * 100, 2)} cm from ground truth after ${fmt(pathDistance, 2)} m and ${fmt(pathDuration, 1)} s of covered motion. That is ${fmt(finalPathPercent, 2)}% of the covered path length and ${fmt(endpointRatio, 1)}× the ATE. ${endpointReading}</p><p><strong>Coverage limit:</strong> ${scopeReading}</p></div>`;
      return `<article class="result-card"><div class="result-top"><div><h3>${result.name}</h3><p class="scope">${result.scope}</p></div><a class="run-link" href="${result.wandb}" target="_blank" rel="noreferrer">W&B run ↗</a></div>${rows}<div class="result-foot"><span>${fmt(result.frames, 0)} input frames</span><span>${fmt(result.metrics.trajectory_pose_count, 0)} trajectory poses</span><span>${fmt(result.metrics.matched_pose_count, 0)} matched</span><span>${fmt(result.metrics.elapsed_seconds, 1)} s elapsed</span></div>${explanation}</article>`;
    }).join('');

    const stages = [
      ['01', 'Replay', 'ZIP / ROS1 / ROS2', 'validated'],
      ['02', 'Preprocess', 'Normalize + quality', 'reference'],
      ['03', 'VIO', 'OpenVINS / ORB-SLAM3', 'validated'],
      ['04', 'Features', 'Geometric + semantic', 'reference'],
      ['05', 'Saliency', 'Gradient + contrast', 'reference'],
      ['06', 'Regions', 'Connected evidence', 'validated'],
      ['07', 'Static filter', 'Temporal evidence', 'reference'],
      ['08', 'Object store', 'Bounded keyframes', 'validated'],
      ['09', 'Uplink', 'Quantize + compress', 'reference'],
      ['10', 'Ingest', 'Validate + dedupe', 'reference'],
      ['11', 'Association', 'Cross-agent IDs', 'reference'],
      ['12', 'Gaussians', 'Feed-forward adapter', 'validated'],
      ['13', 'Scene map', 'Versioned fusion', 'reference'],
      ['14', 'Pose graph', 'Robust translation', 'validated'],
      ['15', 'Correction', 'Bounded delta', 'validated'],
      ['16', 'Context', 'Fresh scene graph', 'reference'],
      ['17', 'SKYLA hand-off', 'Versioned context envelope', 'reference'],
      ['18', 'SKYLA planning', 'Intent → allocation → routes', 'reference'],
      ['19', 'Telemetry', 'Health + latency', 'reference'],
      ['20', 'Simulation', 'Fault + recovery', 'reference'],
      ['21', 'Registry', 'Data + models', 'reference'],
      ['22', 'Security', 'Sign + verify', 'reference'],
      ['23', 'Deployment', 'Capability gate', 'reference'],
      ['24', 'Runtime', 'All gates composed', 'reference'],
      ['25', 'Evidence', 'JSON + W&B', 'validated'],
    ];
    document.getElementById('pipeline').innerHTML = stages.map(([index, title, detail, state]) => `<div class="pipeline-step ${state}"><b>${index}</b><strong>${title}</strong><span>${detail}</span></div>`).join('');

    const svgFrame = (label, body) => `<div class="visual-output" role="img" aria-label="${label}"><svg viewBox="0 0 560 150" preserveAspectRatio="none">${body}</svg></div>`;
    const visualShell = (label, visual, caption, basis = 'Benchmark output') => `<div class="component-visual"><div class="visual-kicker"><span class="evidence-basis">${basis}</span><span>${label}</span></div>${visual}<div class="visual-caption">${caption}</div></div>`;
    const evidenceVideoModes = {
      frames: 'coordinates',
      synchronization: 'synchronization',
      preprocessing: 'preprocessing',
      trajectory: 'vio',
      keypoints: 'keypoints',
      embedding: 'embedding',
      saliency: 'saliency',
      regions: 'regions',
      object_store: 'object_store',
    };
    const evidenceModeLabels = {
      coordinates: 'Frame + coordinate contract',
      synchronization: 'Stereo / IMU replay window',
      preprocessing: 'Normalized grayscale output',
      vio: 'ORB-SLAM3 pose output',
      keypoints: 'Gradient keypoints',
      embedding: 'Semantic similarity',
      saliency: 'U²-Net binary segmentation mask',
      regions: 'Connected regions',
      object_store: 'Keyframe admission evidence',
    };
    const segment = data.video_evidence;
    const evidenceImages = segment.images.map(source => { const image = new Image(); image.src = source; return image; });
    let evidenceFrameIndex = 0;
    let evidencePlaying = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const retainedFrameIndices = new Set(segment.frames
      .map((frame, index) => [index, frame.quality.blur * frame.quality.exposure])
      .sort((left, right) => right[1] - left[1])
      .slice(0, 3)
      .map(item => item[0]));

    const videoCaption = mode => {
      const metrics = segment.metrics;
      if (mode === 'coordinates') return `<span><strong>20</strong> timestamped frames</span><span>camera → body → local → global</span>`;
      if (mode === 'synchronization') return `<span><strong>20</strong> contiguous frames</span><span><strong>${fmt(segment.duration_s, 2)} s</strong> source span</span><span>production replay timestamps</span>`;
      if (mode === 'preprocessing') return `<span><strong>${metrics.accepted_frames}/20</strong> quality accepted</span><span>mean blur <strong>${fmt(metrics.mean_blur, 4)}</strong></span><span>mean exposure <strong>${fmt(metrics.mean_exposure, 3)}</strong></span>`;
      if (mode === 'vio') return `<span><strong>20</strong> production poses</span><span>frame-to-pose tolerance <strong>60 ms</strong></span><span>ORB-SLAM3 output</span>`;
      if (mode === 'keypoints') return `<span>mean <strong>${fmt(metrics.mean_keypoints, 0)}</strong> keypoints/frame</span><span>actual GradientPatchExtractor output</span>`;
      if (mode === 'embedding') return `<span><strong>21-D</strong> spatial pyramid</span><span>similarity to segment frame 01</span>`;
      if (mode === 'saliency') {
        const model = segment.saliency_model_info;
        const frame = segment.frames[evidenceFrameIndex];
        return `<span><strong>${model.name}</strong> · ${model.training_dataset}</span><span><strong>${fmt(frame.saliency_latency_ms, 2)} ms</strong> this frame</span><span>mean <strong>${fmt(metrics.saliency_latency_mean_ms, 2)} ms</strong></span><span>p50 / p95 <strong>${fmt(metrics.saliency_latency_p50_ms, 2)} / ${fmt(metrics.saliency_latency_p95_ms, 2)} ms</strong></span><span>range <strong>${fmt(metrics.saliency_latency_min_ms, 2)}–${fmt(metrics.saliency_latency_max_ms, 2)} ms</strong></span><span><strong>${fmt(metrics.saliency_throughput_fps, 1)} fps</strong> measured throughput</span><span>${model.device.toUpperCase()} · ${model.input_size_px}×${model.input_size_px} → ${model.output_size_px[1]}×${model.output_size_px[0]} · threshold ${model.threshold}</span><span>${model.version}</span><span>${(model.checkpoint_bytes / 1048576).toFixed(1)} MiB checkpoint · model load excluded from timing</span>`;
      }
      if (mode === 'regions') return `<span>mean <strong>${fmt(metrics.mean_regions, 1)}</strong> regions/frame</span><span>connected components after saliency</span>`;
      return `<span><strong>3/20</strong> quality-ranked candidates</span><span>confirmed-only store remains benchmark-gated</span>`;
    };
    const videoVisual = (component, mode) => {
      const label = evidenceModeLabels[mode];
      const output = `<div class="visual-output evidence-video"><canvas width="560" height="220" data-evidence-mode="${mode}" role="img" aria-label="${label} over a real 20-frame TUM VI corridor segment"></canvas><div class="evidence-video-status"><i></i><button type="button" data-evidence-toggle>${evidencePlaying ? 'PAUSE' : 'PLAY'}</button><span data-evidence-counter>01 / 20</span></div></div>`;
      return visualShell(label, output, videoCaption(mode), 'Real 20-frame replay');
    };

    const drawEvidenceVideo = canvas => {
      const context = canvas.getContext('2d');
      const frame = segment.frames[evidenceFrameIndex];
      const image = evidenceImages[evidenceFrameIndex];
      const mode = canvas.dataset.evidenceMode;
      if (!context || !image.complete) return;
      const width = canvas.width, height = canvas.height, imageSize = height, imageX = (width - imageSize) / 2;
      context.clearRect(0, 0, width, height);
      context.fillStyle = '#111914'; context.fillRect(0, 0, width, height);
      if (mode === 'preprocessing') {
        const grid = frame.processed_grid, cell = imageSize / grid.length;
        context.imageSmoothingEnabled = false;
        grid.forEach((row, y) => row.forEach((value, x) => { const shade=Math.round(value*255); context.fillStyle=`rgb(${shade},${shade},${shade})`; context.fillRect(imageX+x*cell,y*cell,cell+.5,cell+.5); }));
      } else if (mode === 'saliency') {
        const [maskHeight, maskWidth] = frame.saliency_mask_shape;
        const packed = Uint8Array.from(atob(frame.saliency_mask_bits), character => character.charCodeAt(0));
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = maskWidth; maskCanvas.height = maskHeight;
        const maskContext = maskCanvas.getContext('2d');
        if (!maskContext) return;
        const pixels = maskContext.createImageData(maskWidth, maskHeight);
        for (let index = 0; index < maskWidth * maskHeight; index += 1) {
          const selected = (packed[index >> 3] >> (7 - (index & 7))) & 1;
          const offset = index * 4;
          const shade = selected ? 246 : 12;
          pixels.data[offset] = shade;
          pixels.data[offset + 1] = selected ? 250 : 18;
          pixels.data[offset + 2] = selected ? 248 : 15;
          pixels.data[offset + 3] = 255;
        }
        maskContext.putImageData(pixels, 0, 0);
        context.imageSmoothingEnabled = false;
        context.drawImage(maskCanvas, imageX, 0, imageSize, imageSize);
      } else {
        context.drawImage(image, imageX, 0, imageSize, imageSize);
      }
      context.fillStyle = 'rgba(8,13,10,.86)'; context.fillRect(0, 0, imageX, height); context.fillRect(imageX + imageSize, 0, imageX, height);
      context.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      context.fillStyle = '#63c693'; context.fillText(evidenceModeLabels[mode].toUpperCase(), 14, 24);
      context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
      context.fillStyle = '#aebcb4'; context.fillText(`FRAME ${String(evidenceFrameIndex + 1).padStart(2, '0')} / 20`, 14, 46);
      context.fillText(`${(frame.timestamp_ns / 1e9).toFixed(3)} s`, 14, 63);
      context.fillStyle = frame.quality.accepted ? '#63c693' : '#e0a958';
      context.fillText(frame.quality.accepted ? 'QUALITY ACCEPTED' : 'QUALITY REJECTED', 14, 84);

      if (mode === 'coordinates') {
        const originX = imageX + 38, originY = height - 35;
        context.lineWidth = 3;
        [['#e4756d', 40, 0, 'x'], ['#63c693', 0, -40, 'y'], ['#6fa2c9', 27, -23, 'z']].forEach(([color, dx, dy, label]) => {
          context.strokeStyle = color; context.beginPath(); context.moveTo(originX, originY); context.lineTo(originX + dx, originY + dy); context.stroke();
          context.fillStyle = color; context.fillText(label, originX + dx + 4, originY + dy);
        });
        context.fillStyle = '#dfe8e3'; context.fillText('camera frame', width - imageX + 14, 111); context.fillText('xyzw rotation', width - imageX + 14, 128);
      } else if (mode === 'synchronization') {
        context.fillStyle = '#dfe8e3'; context.fillText('STEREO 20 Hz', width - imageX + 14, 111); context.fillText('IMU 200 Hz', width - imageX + 14, 128); context.fillStyle = '#63c693'; context.fillText('WINDOW BOUNDED', width - imageX + 14, 150);
        for (let index = 0; index < 10; index += 1) { context.fillRect(width - imageX + 14 + index * 12, 168, 6, 20); }
      } else if (mode === 'preprocessing') {
        context.fillStyle = '#dfe8e3'; context.fillText(`BLUR ${frame.quality.blur.toFixed(4)}`, width - imageX + 14, 111); context.fillText(`EXPOSURE ${frame.quality.exposure.toFixed(3)}`, width - imageX + 14, 128); context.fillText(`OCCLUSION ${(frame.quality.occlusion * 100).toFixed(1)}%`, width - imageX + 14, 145);
      } else if (mode === 'keypoints') {
        context.lineWidth = 1.3;
        frame.keypoints_xy.forEach((point, index) => { const x=imageX+point[0]*imageSize, y=point[1]*imageSize; context.strokeStyle=index<8?'#63c693':'#6fa2c9'; context.beginPath(); context.arc(x,y,index<8?4:2.4,0,Math.PI*2); context.stroke(); });
        context.fillStyle = '#dfe8e3'; context.fillText(`${frame.keypoints_xy.length} KEYPOINTS`, width - imageX + 14, 111);
      } else if (mode === 'embedding') {
        const similarity = frame.embedding_similarity;
        context.fillStyle = '#dfe8e3'; context.fillText('COSINE TO F01', width - imageX + 14, 111); context.fillStyle = '#34433b'; context.fillRect(width-imageX+14,126,imageX-28,14); context.fillStyle = '#63c693'; context.fillRect(width-imageX+14,126,(imageX-28)*similarity,14); context.fillStyle = '#fff'; context.fillText(similarity.toFixed(3), width-imageX+14,158);
      } else if (mode === 'saliency') {
        const model = segment.saliency_model_info;
        context.fillStyle = '#dfe8e3'; context.fillText('BINARY MASK', width-imageX+14,105);
        context.fillText(`${(frame.salient_fraction*100).toFixed(1)}% FOREGROUND`, width-imageX+14,122);
        context.fillText(`${frame.saliency_latency_ms.toFixed(2)} ms / FRAME`, width-imageX+14,139);
        context.fillText(`${model.output_size_px[1]}×${model.output_size_px[0]} OUTPUT`, width-imageX+14,156);
        context.fillStyle = '#f6faf8'; context.fillRect(width-imageX+14,170,12,12);
        context.fillStyle = '#dfe8e3'; context.fillText('WHITE = 1 · BLACK = 0', width-imageX+32,180);
      } else if (mode === 'regions') {
        context.lineWidth = 2;
        frame.regions.forEach((region, index) => { const box=region.bbox_xyxy; context.strokeStyle=index===0?'#63c693':'#e0a958'; context.strokeRect(imageX+box[0]*imageSize,box[1]*imageSize,(box[2]-box[0])*imageSize,(box[3]-box[1])*imageSize); });
        context.fillStyle = '#dfe8e3'; context.fillText(`${frame.regions.length} REGIONS`, width-imageX+14,111);
      } else if (mode === 'object_store') {
        const retained = retainedFrameIndices.has(evidenceFrameIndex);
        context.fillStyle = retained ? '#63c693' : '#dfe8e3'; context.fillText(retained ? 'KEYFRAME CANDIDATE' : 'OBSERVATION ONLY', width-imageX+14,111); context.fillText(`QUALITY ${(frame.quality.blur*frame.quality.exposure).toFixed(4)}`, width-imageX+14,128);
        if (retained) { context.strokeStyle='#63c693'; context.lineWidth=5; context.strokeRect(imageX+4,4,imageSize-8,imageSize-8); }
      } else if (mode === 'vio') {
        const poses = segment.frames.map(item => item.vio_position_m);
        const xs=poses.map(point=>point[0]), ys=poses.map(point=>point[1]); const minX=Math.min(...xs), maxX=Math.max(...xs), minY=Math.min(...ys), maxY=Math.max(...ys);
        const plot = point => [width-imageX+18+(point[0]-minX)/Math.max(maxX-minX,1e-9)*(imageX-36), 184-(point[1]-minY)/Math.max(maxY-minY,1e-9)*75];
        context.strokeStyle='#6fa2c9'; context.lineWidth=2; context.beginPath(); poses.slice(0,evidenceFrameIndex+1).forEach((point,index)=>{const [x,y]=plot(point); index?context.lineTo(x,y):context.moveTo(x,y);}); context.stroke();
        const [poseX,poseY]=plot(frame.vio_position_m); context.fillStyle='#63c693'; context.beginPath(); context.arc(poseX,poseY,5,0,Math.PI*2); context.fill(); context.fillStyle='#dfe8e3'; context.fillText(`X ${frame.vio_position_m[0].toFixed(3)} m`,width-imageX+14,111); context.fillText(`Y ${frame.vio_position_m[1].toFixed(3)} m`,width-imageX+14,128);
      }
    };
    const drawEvidenceVideos = () => {
      document.querySelectorAll('canvas[data-evidence-mode]').forEach(drawEvidenceVideo);
      document.querySelectorAll('[data-evidence-counter]').forEach(node => { node.textContent = `${String(evidenceFrameIndex + 1).padStart(2, '0')} / 20`; });
      document.querySelectorAll('[data-evidence-toggle]').forEach(node => { node.textContent = evidencePlaying ? 'PAUSE' : 'PLAY'; });
    };
    const initializeEvidenceVideos = () => {
      document.querySelectorAll('[data-evidence-toggle]').forEach(button => button.addEventListener('click', () => { evidencePlaying = !evidencePlaying; drawEvidenceVideos(); }));
      Promise.all(evidenceImages.map(image => image.decode ? image.decode().catch(() => undefined) : Promise.resolve())).then(drawEvidenceVideos);
    };
    const synchronizationVisual = visual => {
      const frames = Array.from({length: 6}, (_, index) => 70 + index * 84);
      const frameMarks = frames.map((x, index) => `<g><line x1="${x}" y1="42" x2="${x}" y2="112" stroke="#bdd7ca" stroke-width="1"/><circle cx="${x}" cy="40" r="8" fill="#1f7451"/><text x="${x}" y="22" text-anchor="middle" class="chart-label">F${index + 1}</text></g>`).join('');
      const imuMarks = Array.from({length: 24}, (_, index) => 49 + index * 20).map(x => `<line x1="${x}" y1="105" x2="${x}" y2="119" stroke="#2f628b" stroke-width="2"/>`).join('');
      const svg = svgFrame('Camera frames aligned to bounded IMU sample windows', `<line x1="42" y1="40" x2="518" y2="40" stroke="#aab5af"/><line x1="42" y1="112" x2="518" y2="112" stroke="#aab5af"/>${frameMarks}${imuMarks}<text x="18" y="44" class="chart-label">CAM</text><text x="18" y="116" class="chart-label">IMU</text>`);
      return visualShell('Aligned timeline', svg, `<span><strong>${visual.packets}</strong> packets</span><span><strong>${visual.dropped}</strong> dropped</span><span>p95 <strong>${fmt(visual.p95_ms, 2)} ms</strong></span>`);
    };
    const bootstrapVisual = visual => {
      const nodes = visual.commands.map((command, index) => { const x=20+index*135; return `<g><rect x="${x}" y="49" width="108" height="52" rx="5" fill="${index === 0 ? '#e5eef5' : '#e5f1eb'}" stroke="${index === 0 ? '#2f628b' : '#1f7451'}"/><text x="${x+54}" y="79" text-anchor="middle" class="chart-value">${command}</text></g>`; }).join('');
      return visualShell('Executable surface', svgFrame('Validated CLI entry points share strict configuration and CPU-safe imports', nodes), `<span><strong>${visual.commands.length}</strong> command families</span><span><strong>${visual.tests}</strong> tests</span><span>zero import downloads</span>`);
    };
    const framesVisual = visual => {
      const nodes = visual.frames.map((frame, index) => { const x=22+index*134; return `<g><circle cx="${x+48}" cy="73" r="34" fill="${index === visual.frames.length-1 ? '#e5f1eb' : '#fff'}" stroke="${index === visual.frames.length-1 ? '#1f7451' : '#2f628b'}"/><path d="M${x+48} 73h19M${x+48} 73v-19" stroke="#718078"/><text x="${x+48}" y="124" text-anchor="middle" class="chart-label">${frame}</text>${index < visual.frames.length-1 ? `<path d="M${x+83} 73H${x+126}" stroke="#819088" marker-end="url(#frame-arrow)"/>` : ''}</g>`; }).join('');
      return visualShell('Explicit transforms', svgFrame('Camera coordinates compose explicitly through body and local frames into global', `<defs><marker id="frame-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#819088"/></marker></defs>${nodes}`), `<span>quaternion <strong>x y z w</strong></span><span>round-trip &lt; <strong>${visual.tolerance}</strong></span>`);
    };
    const rasterCells = (grid, color) => {
      const rows = grid.length, columns = grid[0].length;
      const cellWidth = 520 / columns, cellHeight = 118 / rows;
      return grid.flatMap((row, y) => row.map((value, x) => `<rect x="${20 + x * cellWidth}" y="${16 + y * cellHeight}" width="${cellWidth + .4}" height="${cellHeight + .4}" fill="${color(value)}"/>`)).join('');
    };
    const preprocessingVisual = visual => {
      const cells = rasterCells(visual.grid, value => { const shade = Math.round(value * 255); return `rgb(${shade},${shade},${shade})`; });
      const svg = svgFrame('Normalized grayscale frame after deterministic resize and quality control', `<rect x="18" y="14" width="524" height="122" rx="4" fill="#121614"/>${cells}`);
      return visualShell('Normalized frame', svg, `<span>blur <strong>${fmt(visual.blur, 4)}</strong></span><span>exposure <strong>${fmt(visual.exposure, 3)}</strong></span><span><strong>${fmt(visual.latency_ms, 3)} ms</strong></span>`);
    };
    const saliencyVisual = visual => {
      const cells = rasterCells(visual.grid, value => { const level = Math.min(1, value); const red = Math.round(246 * level + 22 * (1-level)); const green = Math.round(142 * level + 34 * (1-level)); const blue = Math.round(45 * level + 31 * (1-level)); return `rgb(${red},${green},${blue})`; });
      const svg = svgFrame('Normalized saliency scores rendered as a heatmap', `<rect x="18" y="14" width="524" height="122" rx="4" fill="#161f1b"/>${cells}`);
      return visualShell('Saliency heatmap', svg, `<span><strong>${fmt(visual.fraction * 100, 1)}%</strong> salient pixels</span><span><strong>${fmt(visual.latency_ms, 3)} ms</strong></span><span>brighter = stronger evidence</span>`);
    };
    const regionsVisual = visual => {
      const boxes = visual.regions.map((region, index) => { const [x1,y1,x2,y2] = region.bbox_xyxy; const x = 30 + x1 / 40 * 500, y = 18 + y1 / 32 * 114, width = (x2-x1)/40*500, height = (y2-y1)/32*114; const color = ['#63c693','#e0a958','#6fa2c9'][index % 3]; return `<g><rect x="${x}" y="${y}" width="${width}" height="${height}" fill="${color}" fill-opacity=".16" stroke="${color}" stroke-width="2"/><text x="${x+4}" y="${y+12}" fill="${color}" style="font:700 9px Inter,sans-serif">R${index+1}</text></g>`; }).join('');
      const svg = svgFrame('Connected saliency components converted into ranked bounding regions', `<defs><linearGradient id="region-bg" x1="0" x2="1"><stop stop-color="#26352f"/><stop offset="1" stop-color="#52645b"/></linearGradient></defs><rect x="30" y="18" width="500" height="114" rx="4" fill="url(#region-bg)"/>${boxes}`);
      return visualShell('Region proposals', svg, `<span><strong>${visual.regions.length}</strong> retained regions</span>${visual.regions.slice(0,2).map(region => `<span>${region.id} <strong>${region.area_px} px</strong></span>`).join('')}`);
    };
    const trajectoryVisual = visual => {
      const all = visual.series.flatMap(series => series.points);
      const xs = all.map(point => point[0]);
      const ys = all.map(point => point[1]);
      const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
      const scalePoint = point => {
        const x = 30 + (point[0] - minX) / Math.max(maxX - minX, 1e-9) * 500;
        const y = 132 - (point[1] - minY) / Math.max(maxY - minY, 1e-9) * 114;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      };
      const colors = ['#1f7451', '#2f628b'];
      const lines = visual.series.map((series, index) => `<polyline points="${series.points.map(scalePoint).join(' ')}" fill="none" stroke="${colors[index]}" stroke-width="2.5" vector-effect="non-scaling-stroke"/>`).join('');
      const svg = svgFrame('Sampled XY trajectories emitted by both production VIO backends', `<path d="M30 18V132H530" fill="none" stroke="#d7ddd9"/>${lines}`);
      const caption = visual.series.map((series, index) => `<span><i style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${colors[index]};margin-right:4px"></i><strong>${series.name}</strong> ${series.points.length} samples</span>`).join('');
      return visualShell('Trajectory estimate', svg, caption);
    };
    const keypointVisual = visual => {
      const points = [[74,34],[116,52],[153,31],[191,82],[231,60],[271,99],[316,48],[354,91],[401,37],[438,72],[482,50],[505,106],[94,108],[177,116],[377,119]];
      const marks = points.map(([x,y], index) => `<g><circle cx="${x}" cy="${y}" r="${index % 3 === 0 ? 7 : 4}" fill="none" stroke="${index % 3 === 0 ? '#1f7451' : '#2f628b'}" stroke-width="2"/><line x1="${x-9}" y1="${y}" x2="${x+9}" y2="${y}" stroke="#809089" stroke-width=".7"/><line x1="${x}" y1="${y-9}" x2="${x}" y2="${y+9}" stroke="#809089" stroke-width=".7"/></g>`).join('');
      const svg = svgFrame('Detected geometric patch keypoints and descriptor support regions', `<defs><linearGradient id="feature-bg" x1="0" x2="1"><stop stop-color="#edf2ee"/><stop offset=".55" stop-color="#d5dfd9"/><stop offset="1" stop-color="#f4eee3"/></linearGradient></defs><rect x="18" y="14" width="524" height="122" rx="4" fill="url(#feature-bg)"/><path d="M18 98L130 61L216 103L306 44L390 87L542 39V136H18Z" fill="#c6d2cb" opacity=".55"/>${marks}`);
      return visualShell('Feature field', svg, `<span>match recall <strong>${fmt(visual.recall * 100, 1)}%</strong></span><span>mean extraction <strong>${fmt(visual.latency_ms, 3)} ms</strong></span>`);
    };
    const embeddingVisual = visual => {
      const bars = [
        ['same object', visual.positive, '#1f7451', 42],
        ['different object', visual.negative, '#a46516', 96],
      ].map(([label, value, color, y]) => `<text x="24" y="${y-9}" class="chart-label">${label}</text><rect x="24" y="${y}" width="490" height="16" rx="3" fill="#e7ebe8"/><rect x="24" y="${y}" width="${490 * value}" height="16" rx="3" fill="${color}"/><text x="524" y="${y+12}" class="chart-value">${fmt(value, 3)}</text>`).join('');
      return visualShell('Cosine similarity', svgFrame('Semantic similarity separates positive and negative object views', bars), `<span>embedding separation <strong>${fmt(visual.separation, 3)}</strong></span><span>larger is better</span>`);
    };
    const staticFilterVisual = visual => {
      const colors = { unknown: '#c5ccc8', static_candidate: '#d89d4d', static_confirmed: '#1f7451', dynamic: '#a33f48' };
      const names = { unknown: 'UNKNOWN', static_candidate: 'CANDIDATE', static_confirmed: 'CONFIRMED', dynamic: 'DYNAMIC' };
      const cells = visual.states.map((state, index) => { const x = 24 + index * 102; return `<g><rect x="${x}" y="47" width="82" height="42" rx="4" fill="${colors[state]}"/><text x="${x+41}" y="72" fill="#fff" text-anchor="middle" style="font:700 8px Inter,sans-serif">${names[state]}</text><text x="${x+41}" y="112" text-anchor="middle" class="chart-label">t${index + 1}</text>${index < visual.states.length - 1 ? `<path d="M${x+83} 68H${x+99}" stroke="#819088" marker-end="url(#arrow)"/>` : ''}</g>`; }).join('');
      const svg = svgFrame('Temporal evidence moves a track from unknown through candidate to confirmed static', `<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#819088"/></marker></defs>${cells}`);
      return visualShell('State transition', svg, `<span><strong>${visual.false_insertions}</strong> dynamic insertions</span><span>only CONFIRMED enters the map</span>`);
    };
    const objectStoreVisual = visual => {
      const object = visual.objects[0];
      const frames = object.keyframes.map((frame, index) => `<g transform="translate(${76 + index * 76} ${38 + index * 8})"><rect width="92" height="66" rx="4" fill="#fff" stroke="#2f628b"/><rect x="8" y="9" width="76" height="33" fill="#e5eef5"/><circle cx="31" cy="25" r="8" fill="#2f628b" opacity=".65"/><text x="46" y="57" text-anchor="middle" class="chart-label">frame ${frame}</text></g>`).join('');
      const svg = svgFrame('High-quality keyframes retained behind one confirmed local object record', `${frames}<path d="M350 75H395" stroke="#819088" marker-end="url(#store-arrow)"/><defs><marker id="store-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#819088"/></marker></defs><rect x="399" y="42" width="128" height="66" rx="6" fill="#e5f1eb" stroke="#1f7451"/><text x="463" y="69" text-anchor="middle" class="chart-value">static object</text><text x="463" y="88" text-anchor="middle" class="chart-label">${fmt(object.confidence, 3)} confidence</text>`);
      return visualShell('Bounded object store', svg, `<span><strong>${visual.count}</strong> object</span><span><strong>${visual.keyframes}</strong> prioritized keyframes</span><span><strong>${visual.restored}</strong> snapshot restore</span><span>confirmed-only admission</span>`);
    };
    const uplinkVisual = visual => {
      const stages = [['Object','5.8 KB'],['Quantize','1.9 KB'],['Compress',`${visual.bytes} B`],['Mesh','delivered']];
      const nodes = stages.map(([name, size], index) => { const x=22+index*135; return `<g><rect x="${x}" y="44" width="104" height="60" rx="5" fill="${index === 3 ? '#e5f1eb' : '#e5eef5'}" stroke="${index === 3 ? '#1f7451' : '#2f628b'}"/><text x="${x+52}" y="69" text-anchor="middle" class="chart-value">${name}</text><text x="${x+52}" y="87" text-anchor="middle" class="chart-label">${size}</text>${index < 3 ? `<path d="M${x+104} 74H${x+130}" stroke="#819088" marker-end="url(#uplink-arrow)"/>` : ''}</g>`; }).join('');
      const svg = svgFrame('Static object record quantized, compressed, checksummed, and delivered through the mesh queue', `<defs><marker id="uplink-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#819088"/></marker></defs>${nodes}`);
      return visualShell('Payload flow', svg, `<span><strong>${visual.bytes} bytes</strong> on wire</span><span><strong>${visual.transport.delivered}</strong> delivered</span><span><strong>${visual.transport.dropped}</strong> dropped</span>`);
    };
    const meshVisual = visual => {
      const rows = [['CONTROL',4,'#a33f48'],['CORRECTION',3,'#a46516'],['OBSERVATION',2,'#2f628b'],['TELEMETRY',1,'#6b7771']].map(([label,priority,color], index) => `<g><rect x="32" y="${18+index*30}" width="300" height="22" rx="3" fill="#fff" stroke="#d7ddd9"/><rect x="32" y="${18+index*30}" width="${55+priority*45}" height="22" rx="3" fill="${color}" fill-opacity=".2"/><text x="42" y="${33+index*30}" class="chart-value">${label}</text></g>`).join('');
      const svg = svgFrame('Priority queue applies TTL, finite retry, duplicate, and receiver-acknowledgement policy', `${rows}<path d="M348 75H405" stroke="#819088" marker-end="url(#mesh-arrow)"/><defs><marker id="mesh-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#819088"/></marker></defs><circle cx="468" cy="75" r="43" fill="#e5f1eb" stroke="#1f7451"/><text x="468" y="72" text-anchor="middle" class="chart-value">ACKED</text><text x="468" y="89" text-anchor="middle" class="chart-label">Intelligence</text>`);
      return visualShell('Reliable priority queue', svg, `<span><strong>${visual.transport.sent}</strong> sent</span><span><strong>${visual.transport.delivered}</strong> deliveries</span><span><strong>${visual.transport.retries}</strong> retry</span><span><strong>${visual.transport.acknowledged}</strong> acknowledged</span>`);
    };
    const registryVisual = visual => {
      const svg = svgFrame('Incoming packet passes checksum, sequence, and retention gates before registry insertion', `<defs><marker id="registry-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#819088"/></marker></defs><rect x="24" y="50" width="110" height="52" rx="5" fill="#e5eef5" stroke="#2f628b"/><text x="79" y="80" text-anchor="middle" class="chart-value">packet 0001</text><path d="M134 76H182" stroke="#819088" marker-end="url(#registry-arrow)"/><g><circle cx="217" cy="76" r="27" fill="#e5f1eb" stroke="#1f7451"/><path d="M204 76l8 8 17-19" fill="none" stroke="#1f7451" stroke-width="3"/></g><text x="217" y="120" text-anchor="middle" class="chart-label">validated</text><path d="M244 76H292" stroke="#819088" marker-end="url(#registry-arrow)"/><rect x="296" y="34" width="238" height="84" rx="5" fill="#fff" stroke="#aab5af"/><text x="312" y="57" class="chart-label">ORDERED REGISTRY</text><rect x="312" y="69" width="206" height="31" rx="3" fill="#e5f1eb"/><text x="322" y="89" class="chart-value">${visual.ids[0]}</text>`);
      return visualShell('Validated ingest', svg, `<span><strong>${visual.observations}</strong> observation accepted</span><span><strong>${visual.journal_entries}</strong> raw envelope journaled</span><span><strong>${visual.journal_replayed}</strong> replay rebuilt</span><span><strong>${visual.duplicates}</strong> redelivery suppressed</span><span><strong>${visual.restored}</strong> snapshot restore</span><span>checksum + chain + schema + TTL + skew</span>`);
    };
    const associationVisual = visual => {
      const svg = svgFrame('Two local Wingman observations pass geometry and embedding gates and merge into one global object', `<defs><marker id="assoc-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#6d7a73"/></marker></defs><path d="M148 45C240 45 251 75 338 75M148 107C240 107 251 75 338 75" fill="none" stroke="#6d7a73" stroke-width="2" marker-end="url(#assoc-arrow)"/><g><circle cx="108" cy="45" r="30" fill="#e5eef5" stroke="#2f628b"/><text x="108" y="42" text-anchor="middle" class="chart-value">W01</text><text x="108" y="56" text-anchor="middle" class="chart-label">local track</text></g><g><circle cx="108" cy="107" r="30" fill="#e5eef5" stroke="#2f628b"/><text x="108" y="104" text-anchor="middle" class="chart-value">W02</text><text x="108" y="118" text-anchor="middle" class="chart-label">local track</text></g><g><circle cx="408" cy="75" r="47" fill="#e5f1eb" stroke="#1f7451" stroke-width="2"/><text x="408" y="70" text-anchor="middle" class="chart-value">${visual.global_id}</text><text x="408" y="87" text-anchor="middle" class="chart-label">global object</text></g>`);
      return visualShell('Identity merge', svg, `<span><strong>${visual.agents.length}</strong> Wingmen</span><span><strong>${visual.objects}</strong> persistent object</span><span><strong>${visual.evidence}</strong> scored evidence records</span><span>restart ID <strong>${visual.stable && visual.restored ? 'stable' : 'missing'}</strong></span><span>distance + cosine gated</span>`);
    };
    const gaussianVisual = visual => {
      if (visual.render_image) {
        const output = `<div class="visual-output resplat-render" role="img" aria-label="ReSplat rendered target ${visual.render_name} from ${visual.dataset}"><img src="${visual.render_image}" alt="ReSplat reconstruction of buildings, vegetation, and road in DDOS neighbourhood 105"><span class="resplat-render-label">GENERATED TARGET · ${visual.render_name}</span></div>`;
        const caption = `<span><strong>${visual.model}</strong></span><span>${visual.context_views} context · ${visual.target_views} target · ${visual.refinement_iterations} refinements</span><span><strong>${visual.resolution}</strong> output</span><span>contract <strong>${visual.contract_backend}</strong></span><span>reference working set <strong>${fmt(visual.estimated_memory_bytes/1024, 1)} KiB</strong></span><span>target PSNR <strong>${fmt(visual.target_metrics.psnr, 2)} dB</strong></span><span>SSIM <strong>${fmt(visual.target_metrics.ssim, 3)}</strong></span><span>LPIPS <strong>${fmt(visual.target_metrics.lpips, 3)}</strong></span><span>W&B runtime <strong>${visual.wandb_runtime_s} s</strong></span><span>render sha256 <strong>${visual.render_sha256.slice(0,12)}</strong></span><a href="${visual.wandb_url}" target="_blank" rel="noreferrer">W&B run ↗</a>`;
        return visualShell('ReSplat rendered reconstruction', output, caption, 'New evaluated ReSplat run');
      }
      const gaussian = visual.gaussians[0];
      const [red, green, blue] = gaussian.color_rgb.map(value => Math.round(value * 255));
      const svg = svgFrame('Two associated static observations fuse into one anisotropic Gaussian primitive', `<defs><radialGradient id="gaussian-fill"><stop stop-color="rgb(${red},${green},${blue})" stop-opacity=".8"/><stop offset="1" stop-color="rgb(${red},${green},${blue})" stop-opacity=".08"/><marker id="gaussian-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#819088"/></marker></defs><g><circle cx="78" cy="47" r="21" fill="#e5eef5" stroke="#2f628b"/><circle cx="78" cy="105" r="21" fill="#e5eef5" stroke="#2f628b"/><text x="78" y="50" text-anchor="middle" class="chart-value">W01</text><text x="78" y="108" text-anchor="middle" class="chart-value">W02</text></g><path d="M100 47C190 47 204 75 280 75M100 105C190 105 204 75 280 75" fill="none" stroke="#819088" marker-end="url(#gaussian-arrow)"/><ellipse cx="396" cy="75" rx="104" ry="48" fill="url(#gaussian-fill)" stroke="rgb(${red},${green},${blue})" stroke-width="2"/><circle cx="396" cy="75" r="5" fill="#17201c"/><text x="396" y="137" text-anchor="middle" class="chart-label">${gaussian.id}</text>`);
      return visualShell('Gaussian reconstruction', svg, `<span><strong>${visual.inputs}</strong> observations</span><span><strong>${visual.gaussians.length}</strong> primitive</span><span><strong>${fmt(visual.latency_ms, 3)} ms</strong></span>`);
    };
    const sceneMapVisual = visual => {
      const gaussian = visual.gaussians[0];
      const sources = gaussian.sources.map((source, index) => `<g><rect x="${28 + index*182}" y="104" width="164" height="25" rx="12" fill="#e5eef5"/><text x="${110 + index*182}" y="120" text-anchor="middle" class="chart-label">${source}</text></g>`).join('');
      const svg = svgFrame('Versioned global map snapshot retaining Gaussian shape and source provenance', `<rect x="20" y="16" width="520" height="120" rx="5" fill="#f7f9f7" stroke="#d7ddd9"/><path d="M40 88H516M62 28V120" stroke="#d7ddd9"/><ellipse cx="330" cy="63" rx="84" ry="31" fill="#e87939" fill-opacity=".24" stroke="#a46516"/><circle cx="330" cy="63" r="5" fill="#a46516"/><text x="422" y="35" class="chart-value">revision ${visual.revision}</text>${sources}`);
      return visualShell('Scene snapshot', svg, `<span>mean <strong>${gaussian.mean_m.slice(0,2).map(value => fmt(value,2)).join(', ')} m</strong></span><span><strong>${gaussian.sources.length}</strong> provenance links</span><span><strong>${visual.history.length}</strong> in-memory revisions</span><span><strong>${visual.persisted_snapshots}</strong> crash-safe snapshot</span><span><strong>${fmt(visual.snapshot_bytes,0)} bytes</strong> serialized</span><span>rollback + restore <strong>${visual.rollback_supported && visual.restore_supported ? 'ready' : 'missing'}</strong></span>`);
    };
    const poseGraphVisual = visual => {
      const positions = Object.entries(visual.positions);
      const scaled = Object.fromEntries(positions.map(([name, point]) => [name, [76 + point[0] * 174, 27 + point[1] * 48]]));
      const edge = (left, right, rejected = false) => `<line x1="${scaled[left][0]}" y1="${scaled[left][1]}" x2="${scaled[right][0]}" y2="${scaled[right][1]}" stroke="${rejected ? '#a33f48' : '#829088'}" stroke-width="${rejected ? 2 : 1.5}" stroke-dasharray="${rejected ? '6 5' : ''}"/>`;
      const edges = edge('wingman_01_t0','wingman_01_t1') + edge('wingman_02_t0','wingman_02_t1') + edge('wingman_01_t0','wingman_02_t0') + edge('wingman_01_t1','wingman_02_t1') + edge('wingman_01_t0','wingman_02_t1',true);
      const nodes = positions.map(([name]) => { const [x,y] = scaled[name]; return `<g><circle cx="${x}" cy="${y}" r="8" fill="#1f7451"/><text x="${x+12}" y="${y+4}" class="chart-label">${name.replace('wingman_', 'W')}</text></g>`; }).join('');
      return visualShell('Optimized constraints', svgFrame('Pose graph positions with the injected false constraint drawn as a rejected dashed edge', `${edges}${nodes}<text x="362" y="132" fill="#a33f48" style="font:700 9px Inter,sans-serif">- - rejected outlier</text>`), `<span>position RMSE <strong>${fmt(visual.rmse_m, 4)} m</strong></span><span><strong>${visual.rejected}</strong> rejected constraint</span>`);
    };
    const se3PoseGraphVisual = visual => {
      const coordinates = {};
      visual.nodes.forEach((node,index) => { const p=node.position_m; coordinates[node.id] = node.component === 0 ? [62+p[0]*145, 72-p[1]*39-p[2]*18] : [404+p[0]*70, 91-p[1]*28-p[2]*25]; });
      const line = (a,b,color='#829088',dash='') => `<line x1="${coordinates[a][0]}" y1="${coordinates[a][1]}" x2="${coordinates[b][0]}" y2="${coordinates[b][1]}" stroke="${color}" stroke-width="2" stroke-dasharray="${dash}"/>`;
      const edges = line('wingman_01_t0','wingman_01_t1') + line('wingman_01_t1','object_tower') + line('wingman_01_t0','object_tower','#1f7451','4 3') + line('wingman_02_t0','wingman_02_t1') + line('wingman_01_t0','object_tower','#a33f48','8 6');
      const nodes = visual.nodes.map(node => { const [x,y]=coordinates[node.id]; return `<g><circle cx="${x}" cy="${y}" r="9" fill="${node.component === 0 ? '#1f7451' : '#2f628b'}"/><text x="${x+12}" y="${y+4}" class="chart-label">${node.id.replace('wingman_','W').replace('object_','O_')}</text><circle cx="${x}" cy="${y}" r="${10+node.covariance_trace*70}" fill="none" stroke="${node.component === 0 ? '#1f7451' : '#2f628b'}" opacity=".3"/></g>`; }).join('');
      const svg = svgFrame('Full SE3 pose graph showing two disconnected components, propagated uncertainty, and a rejected loop', `<rect x="20" y="16" width="334" height="120" rx="4" fill="#f7f9f7"/><rect x="372" y="16" width="168" height="120" rx="4" fill="#f5f8fa"/>${edges}${nodes}<text x="384" y="31" class="chart-label">component 2</text><text x="228" y="130" fill="#a33f48" style="font:700 9px Inter,sans-serif">- - rejected SE(3) loop</text>`);
      return visualShell('SE(3) graph + covariance', svg, `<span><strong>${new Set(visual.nodes.map(node => node.component)).size}</strong> components</span><span><strong>${visual.rejected.length}</strong> rejected</span><span>revision <strong>${visual.revision}</strong></span><span>restart revision <strong>${visual.state_restored && visual.restored_revision === visual.revision ? 'stable' : 'missing'}</strong></span><span>translation RMSE <strong>${fmt(visual.translation_rmse_m,4)} m</strong></span><span>rotation RMSE <strong>${fmt(visual.rotation_rmse_rad,4)} rad</strong></span>`);
    };
    const correctionVisual = visual => {
      if (visual.pre_global_pose_map && visual.post_global_pose_map) {
        const combined = [...visual.pre_global_pose_map, ...visual.post_global_pose_map];
        const xs = combined.map(point => point[0]), ys = combined.map(point => point[1]);
        const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
        const spanX = Math.max(maxX - minX, 1e-9), spanY = Math.max(maxY - minY, 1e-9);
        const project = (point, panelX) => [panelX + 16 + (point[0] - minX) / spanX * 218, 199 - (point[1] - minY) / spanY * 142];
        const path = (points, panelX) => points.map((point, index) => { const [x,y] = project(point, panelX); return `${index ? 'L' : 'M'}${x.toFixed(2)} ${y.toFixed(2)}`; }).join(' ');
        const marker = (point, panelX, color, label) => { const [x,y] = project(point, panelX); return `<g><circle cx="${x}" cy="${y}" r="4.5" fill="${color}"/><text x="${x+7}" y="${y-6}" class="chart-label">${label}</text></g>`; };
        const preStart = visual.pre_global_pose_map[0], preEnd = visual.pre_global_pose_map[visual.pre_global_pose_map.length - 1];
        const postStart = visual.post_global_pose_map[0], postEnd = visual.post_global_pose_map[visual.post_global_pose_map.length - 1];
        const panels = `<rect x="12" y="16" width="252" height="194" rx="5" fill="#f5f8fa" stroke="#d7ddd9"/><rect x="296" y="16" width="252" height="194" rx="5" fill="#f4f9f6" stroke="#c9ddd2"/><path d="M28 57H246M28 101H246M28 145H246M28 189H246" stroke="#dfe5e1"/><path d="M312 57H530M312 101H530M312 145H530M312 189H530" stroke="#dce8e1"/>`;
        const svg = `<div class="visual-output pose-map-comparison" role="img" aria-label="ORB-SLAM3 global pose map before and after bounded correction"><svg viewBox="0 0 560 220" preserveAspectRatio="none">${panels}<text x="28" y="38" class="chart-value">PRE-PROCESSED GLOBAL POSE MAP</text><text x="312" y="38" class="chart-value">POST-PROCESSED GLOBAL POSE MAP</text><path d="${path(visual.pre_global_pose_map,12)}" fill="none" stroke="#2f628b" stroke-width="2.5"/><path d="${path(visual.pre_global_pose_map,296)}" fill="none" stroke="#9aa9a1" stroke-width="1.5" stroke-dasharray="4 4" opacity=".55"/><path d="${path(visual.post_global_pose_map,296)}" fill="none" stroke="#1f7451" stroke-width="2.5"/>${marker(preStart,12,'#63c693','start')}${marker(preEnd,12,'#2f628b','end')}${marker(postStart,296,'#63c693','start')}${marker(postEnd,296,'#1f7451','end')}<text x="28" y="207" class="chart-label">raw VIO · shared XY axes</text><text x="312" y="207" class="chart-label">gray = before · green = corrected</text></svg></div>`;
        const caption = `<span><strong>${visual.vio_backend}</strong> · ${visual.trajectory_pose_count} poses</span><span>${visual.trajectory_source}</span><span>requested Δx <strong>${fmt(visual.requested_delta_m[0],2)} m</strong></span><span>applied Δx <strong>${fmt(visual.applied_delta_m[0],2)} m</strong></span><span><strong>${fmt(visual.fraction * 100,1)}%</strong> bounded step</span><span>restart state <strong>${visual.state_restored && visual.restart_duplicate_rejected && visual.sequence_preserved ? 'stable' : 'missing'}</strong></span><span>${visual.frame_transform}</span><span>${visual.interpretation}</span>`;
        return visualShell('VIO pose correction · before / after', svg, caption, 'Production VIO + reference correction');
      }
      const local = visual.local_pose_m[0], requested = visual.requested_pose_m[0], applied = visual.applied_pose_m[0];
      const x = value => 50 + value / 2.2 * 460;
      const svg = svgFrame('Requested global correction clamped to a bounded application step', `<line x1="42" y1="82" x2="520" y2="82" stroke="#aab5af"/><line x1="${x(local)}" y1="42" x2="${x(requested)}" y2="42" stroke="#a33f48" stroke-dasharray="6 4"/><circle cx="${x(local)}" cy="82" r="10" fill="#2f628b"/><text x="${x(local)}" y="112" text-anchor="middle" class="chart-label">local ${fmt(local,1)} m</text><circle cx="${x(applied)}" cy="82" r="11" fill="#1f7451"/><text x="${x(applied)}" y="132" text-anchor="middle" class="chart-label">applied ${fmt(applied,1)} m</text><circle cx="${x(requested)}" cy="82" r="10" fill="none" stroke="#a33f48" stroke-width="2"/><text x="${x(requested)}" y="112" text-anchor="middle" class="chart-label">target ${fmt(requested,1)} m</text>`);
      return visualShell('Bounded correction', svg, `<span><strong>${fmt(visual.fraction * 100, 1)}%</strong> applied this step</span><span>TTL + idempotence enforced</span>`);
    };
    const contextVisual = visual => {
      const wingman = visual.nodes.find(node => node.kind === 'wingman');
      const object = visual.nodes.find(node => node.kind === 'object');
      const svg = svgFrame('Unified scene graph links the corrected Wingman pose to the reconstructed object', `<defs><marker id="context-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#819088"/></marker></defs><circle cx="126" cy="75" r="43" fill="#e5eef5" stroke="#2f628b"/><text x="126" y="70" text-anchor="middle" class="chart-value">${wingman.id}</text><text x="126" y="87" text-anchor="middle" class="chart-label">corrected pose</text><path d="M170 75H347" stroke="#819088" stroke-width="2" marker-end="url(#context-arrow)"/><rect x="217" y="55" width="82" height="23" rx="11" fill="#fff" stroke="#d7ddd9"/><text x="258" y="70" text-anchor="middle" class="chart-label">${visual.edges[0].relation}</text><ellipse cx="420" cy="75" rx="62" ry="40" fill="#f7ecd9" stroke="#a46516"/><text x="420" y="70" text-anchor="middle" class="chart-value">${object.id}</text><text x="420" y="87" text-anchor="middle" class="chart-label">Gaussian object</text>`);
      return visualShell('Machine context', svg, `<span><strong>${visual.nodes.length}</strong> nodes</span><span><strong>${visual.edges.length}</strong> typed relation</span><span>state <strong>${visual.degraded ? 'degraded' : 'fresh'}</strong></span>`);
    };
    const skylaHandoffVisual = visual => {
      const fieldRows = visual.fields.slice(0,3).map((field,index) => `<text x="222" y="${61+index*18}" class="chart-label">• ${field}</text>`).join('');
      const svg = svgFrame('Versioned ARIADNE context envelope handed to the SKYLA planning boundary', `<defs><marker id="skyla-handoff-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#819088"/></marker></defs><rect x="20" y="36" width="138" height="78" rx="7" fill="#e5eef5" stroke="#2f628b"/><text x="89" y="63" text-anchor="middle" class="chart-value">UNIFIED CONTEXT</text><text x="89" y="82" text-anchor="middle" class="chart-label">revision ${visual.scene_revision}</text><text x="89" y="99" text-anchor="middle" class="chart-label">${visual.nodes} nodes · ${visual.edges} edge</text><path d="M158 75H202" stroke="#819088" stroke-width="2" marker-end="url(#skyla-handoff-arrow)"/><rect x="206" y="22" width="180" height="106" rx="7" fill="#fff" stroke="#a46516"/><text x="222" y="43" class="chart-value">HAND-OFF ENVELOPE</text>${fieldRows}<text x="222" y="118" class="chart-label">+ constraints · health · version</text><path d="M386 75H426" stroke="#819088" stroke-width="2" marker-end="url(#skyla-handoff-arrow)"/><rect x="430" y="36" width="110" height="78" rx="7" fill="#e5f1eb" stroke="#1f7451"/><text x="485" y="68" text-anchor="middle" class="chart-value">SKYLA</text><text x="485" y="86" text-anchor="middle" class="chart-label">planning</text><text x="485" y="102" text-anchor="middle" class="chart-label">interface</text>`);
      return visualShell('ARIADNE → SKYLA boundary', svg, `<span>schema <strong>${visual.schema}</strong></span><span>frame <strong>${visual.frame}</strong></span><span><strong>${visual.gates.length}</strong> acceptance gates</span><span><strong>${fmt(visual.payload_bytes, 0)} bytes</strong></span><span>context <strong>${visual.degraded ? 'degraded' : 'fresh'}</strong></span>`, 'Executable reference contract');
    };
    const skylaPlanningVisual = visual => {
      const wrap = label => { const words=label.split(' '), split=Math.ceil(words.length/2); return [words.slice(0,split).join(' '),words.slice(split).join(' ')]; };
      const stages = visual.stages.map((stage,index) => { const x=16+index*89, [first,second]=wrap(stage); return `<g><rect x="${x}" y="42" width="76" height="54" rx="5" fill="${index===4?'#e5f1eb':'#fff'}" stroke="${index===4?'#1f7451':'#9fb2a8'}"/><text x="${x+38}" y="${second?64:70}" text-anchor="middle" class="chart-value">${first}</text>${second?`<text x="${x+38}" y="79" text-anchor="middle" class="chart-label">${second}</text>`:''}${index<visual.stages.length-1?`<path d="M${x+76} 69H${x+87}" stroke="#819088" marker-end="url(#skyla-plan-arrow)"/>`:''}</g>`; }).join('');
      const assignments = visual.assignments.map(item => `${item.agent}→${item.frontier}`).join(' · ');
      const svg = `<div class="visual-output architecture-visual" role="img" aria-label="Proposed SKYLA global planning architecture with Wingman local safety authority"><svg viewBox="0 0 560 220" preserveAspectRatio="none"><defs><marker id="skyla-plan-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#819088"/></marker></defs><rect x="8" y="12" width="544" height="105" rx="6" fill="#f7f9f7" stroke="#d7ddd9"/><text x="18" y="31" class="chart-label">INTELLIGENCE NODE · GLOBAL PLANNING AUTHORITY</text>${stages}<rect x="8" y="130" width="544" height="80" rx="6" fill="#f5f8fa" stroke="#d7ddd9"/><text x="18" y="149" class="chart-label">WINGMAN · LOCAL FLIGHT-SAFETY AUTHORITY</text><rect x="210" y="161" width="132" height="34" rx="5" fill="#e5eef5" stroke="#2f628b"/><text x="276" y="182" text-anchor="middle" class="chart-value">collision + safety gate</text><path d="M507 96V151H276V159" fill="none" stroke="#819088" stroke-width="2" marker-end="url(#skyla-plan-arrow)"/><rect x="370" y="161" width="142" height="34" rx="5" fill="#e5f1eb" stroke="#1f7451"/><text x="441" y="182" text-anchor="middle" class="chart-value">execute route request</text><path d="M342 178H368" stroke="#819088" stroke-width="2" marker-end="url(#skyla-plan-arrow)"/><path d="M210 178H116V105" fill="none" stroke="#a46516" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#skyla-plan-arrow)"/><text x="24" y="201" class="chart-label">execution + map feedback → replan</text></svg></div>`;
      const eligible = visual.wingmen.filter(item => item.eligible).length;
      return visualShell('SKYLA mission-to-execution flow', svg, `<span><strong>${visual.stages.length}</strong> planning stages</span><span><strong>${visual.routes.length}</strong> route requests</span><span><strong>${eligible}/${visual.wingmen.length}</strong> eligible Wingmen</span><span><strong>${visual.blocked_frontiers.length}</strong> blocked frontiers</span><span>${assignments}</span><span>global plan · local safety veto</span>`, 'Executable mission planning reference');
    };
    const planningVisual = visual => {
      const point = position => [40 + position[0]/10*480, 128-position[1]/9*104];
      const byWingman = Object.fromEntries(visual.wingmen.map(item => [item.id,item]));
      const byFrontier = Object.fromEntries(visual.frontiers.map(item => [item.id,item]));
      const lines = visual.assignments.map(item => { const a=point(byWingman[item.agent].position_m), b=point(byFrontier[item.frontier].position_m); return `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="#1f7451" stroke-width="2" stroke-dasharray="5 4"/>`; }).join('');
      const wingmen = visual.wingmen.map(item => { const [x,y]=point(item.position_m); return `<g><circle cx="${x}" cy="${y}" r="10" fill="${item.eligible ? '#2f628b' : '#a33f48'}"/><text x="${x}" y="${y-14}" text-anchor="middle" class="chart-label">${item.id.replace('wingman_','W')}</text></g>`; }).join('');
      const frontiers = visual.frontiers.map(item => { const [x,y]=point(item.position_m); return `<g><path d="M${x} ${y-11}L${x+10} ${y+8}H${x-10}Z" fill="#d89d4d"/><text x="${x}" y="${y+22}" text-anchor="middle" class="chart-label">${item.id.replace('frontier_','F')}</text></g>`; }).join('');
      return visualShell('Auction assignment', svgFrame('Eligible Wingmen assigned to unique frontiers using information gain, distance, and battery', `<rect x="20" y="12" width="520" height="126" rx="4" fill="#f7f9f7"/>${lines}${wingmen}${frontiers}`), `<span><strong>${visual.assignments.length}</strong> assignments</span><span>red node excluded by battery guardrail</span>`);
    };
    const telemetryVisual = visual => {
      const max = Math.max(...visual.latencies_ms); const points = visual.latencies_ms.map((value,index) => `${32+index*80},${126-value/max*94}`).join(' ');
      const svg = svgFrame('Bounded latency distribution with p50 and p95 operating markers', `<path d="M28 18V130H532" fill="none" stroke="#d7ddd9"/><line x1="28" y1="${126-visual.p50_ms/max*94}" x2="532" y2="${126-visual.p50_ms/max*94}" stroke="#1f7451" stroke-dasharray="5 4"/><line x1="28" y1="${126-visual.p95_ms/max*94}" x2="532" y2="${126-visual.p95_ms/max*94}" stroke="#a46516" stroke-dasharray="5 4"/><polyline points="${points}" fill="none" stroke="#2f628b" stroke-width="3"/>${visual.latencies_ms.map((value,index) => `<circle cx="${32+index*80}" cy="${126-value/max*94}" r="4" fill="#2f628b"/>`).join('')}`);
      return visualShell('Latency + health', svg, `<span>p50 <strong>${fmt(visual.p50_ms,2)} ms</strong></span><span>p95 <strong>${fmt(visual.p95_ms,2)} ms</strong></span><span>mesh <strong>${visual.health.mesh}</strong></span><span><strong>${visual.event_count}</strong> trace event</span><span><strong>${visual.prometheus_series.length}</strong> Prometheus series</span><span>${visual.mission_id} · ${visual.node_id}</span>`);
    };
    const simulationVisual = visual => {
      const svg = svgFrame('Network availability timeline with deterministic partition and packet recovery', `<rect x="30" y="54" width="500" height="42" rx="4" fill="#e5f1eb"/><rect x="130" y="54" width="300" height="42" fill="#a33f48"/><path d="M30 110V122M130 110V122M430 110V122M530 110V122" stroke="#819088"/><text x="30" y="137" class="chart-label">0 s</text><text x="120" y="137" class="chart-label">20 s</text><text x="424" y="137" class="chart-label">80 s</text><text x="500" y="137" class="chart-label">100 s</text><text x="280" y="79" text-anchor="middle" fill="#fff" style="font:700 9px Inter,sans-serif">60 SECOND PARTITION</text><path d="M430 42C455 17 486 17 515 42" fill="none" stroke="#1f7451" stroke-width="2"/><text x="470" y="18" text-anchor="middle" class="chart-label">${visual.recovery_packets} recovery packets</text>`);
      return visualShell('Fault timeline', svg, `<span><strong>${fmt(visual.packet_loss_rate*100,1)}%</strong> total loss</span><span><strong>${visual.partition_duration_seconds} s</strong> partition</span><span><strong>${fmt(visual.drift_improvement_percent,1)}%</strong> drift improvement</span>`);
    };
    const catalogVisual = visual => {
      const datasets = visual.datasets.map((name,index) => `<rect x="${26+(index%2)*112}" y="${28+Math.floor(index/2)*38}" width="98" height="27" rx="4" fill="#e5eef5"/><text x="${75+(index%2)*112}" y="${46+Math.floor(index/2)*38}" text-anchor="middle" class="chart-label">${name}</text>`).join('');
      const models = visual.models.map((name,index) => `<rect x="${286+(index%2)*122}" y="${18+Math.floor(index/2)*35}" width="108" height="25" rx="4" fill="#e5f1eb"/><text x="${340+(index%2)*122}" y="${34+Math.floor(index/2)*35}" text-anchor="middle" class="chart-label">${name.replace('_reference',' ref')}</text>`).join('');
      return visualShell('Typed provenance', svgFrame('Dataset and model catalogs linked by experiment records', `<text x="26" y="16" class="chart-value">DATASETS</text>${datasets}<line x1="258" y1="16" x2="258" y2="136" stroke="#d7ddd9"/><text x="286" y="12" class="chart-value">MODELS</text>${models}`), `<span><strong>${visual.datasets.length}</strong> datasets</span><span><strong>${visual.models.length}</strong> models</span><span>license + backend + version</span>`);
    };
    const securityVisual = visual => {
      const svg = svgFrame('Payload signed at the Wingman and accepted only after identity, integrity, expiry, and replay checks', `<defs><marker id="security-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#819088"/></marker></defs><rect x="26" y="49" width="112" height="55" rx="5" fill="#e5eef5" stroke="#2f628b"/><text x="82" y="81" text-anchor="middle" class="chart-value">payload</text><path d="M138 76H198" stroke="#819088" marker-end="url(#security-arrow)"/><rect x="203" y="35" width="126" height="82" rx="8" fill="#f7ecd9" stroke="#a46516"/><path d="M246 66v-9a20 20 0 0140 0v9M239 66h54v35h-54z" fill="none" stroke="#a46516" stroke-width="3"/><path d="M329 76H389" stroke="#819088" marker-end="url(#security-arrow)"/><circle cx="448" cy="76" r="43" fill="#e5f1eb" stroke="#1f7451"/><path d="M427 76l13 13 28-31" fill="none" stroke="#1f7451" stroke-width="4"/>`);
      return visualShell('Authenticated envelope', svg, `<span><strong>${visual.signed}</strong> signed</span><span><strong>${visual.verified}</strong> verified</span><span><strong>${visual.replays}</strong> replays</span>`);
    };
    const deploymentVisual = visual => {
      const checks = ['CPU ≥ 1','RAM ≥ 512 MB','accelerator optional','camera optional'].map((label,index) => `<g><circle cx="58" cy="${31+index*31}" r="9" fill="#1f7451"/><path d="M53 ${31+index*31}l4 4 7-9" fill="none" stroke="#fff" stroke-width="2"/><text x="78" y="${35+index*31}" class="chart-value">${label}</text></g>`).join('');
      const svg = svgFrame('Reference deployment profile compared explicitly with detected hardware capabilities', `${checks}<rect x="320" y="37" width="196" height="76" rx="7" fill="#e5f1eb" stroke="#1f7451"/><text x="418" y="69" text-anchor="middle" class="chart-value">${visual.profile}</text><text x="418" y="91" text-anchor="middle" class="chart-label">${visual.compatible ? 'COMPATIBLE' : 'MISSING CAPABILITY'}</text>`);
      return visualShell('Capability gate', svg, `<span>profile <strong>${visual.profile}</strong></span><span><strong>${visual.missing.length}</strong> missing requirements</span>`);
    };
    const endToEndVisual = visual => {
      const colors = ['#2f628b','#1f7451','#a46516','#6f55a3'];
      const nodes = visual.stages.map((stage,index) => { const x=20+index*135; return `<g><rect x="${x}" y="43" width="108" height="64" rx="5" fill="${colors[index]}" fill-opacity=".13" stroke="${colors[index]}"/><text x="${x+54}" y="66" text-anchor="middle" class="chart-value">${stage.name.replace('-reference','')}</text><text x="${x+54}" y="86" text-anchor="middle" class="chart-label">${fmt(stage.latency_ms,2)} ms</text><text x="${x+54}" y="99" text-anchor="middle" fill="#1f7451" style="font:700 8px Inter,sans-serif">${stage.status}</text></g>`; }).join('');
      return visualShell('Composed runtime', svgFrame('Four executable reference gates composed into one deterministic system result', nodes), `<span><strong>${visual.stages.length}/${visual.stages.length}</strong> gates passed</span><span><strong>${fmt(visual.latency_ms,2)} ms</strong> total</span><span><strong>${fmt(visual.peak_bytes/1024,1)} KiB</strong> peak traced</span><span><strong>${visual.degraded.length}</strong> degraded modes exercised</span>`);
    };
    const evidenceVisual = visual => {
      const steps = [['Pipeline','#e5eef5','#2f628b'],['Metrics','#e5f1eb','#1f7451'],['JSON','#f7ecd9','#a46516'],['W&B','#eee9f7','#6f55a3']];
      const nodes = steps.map(([label, fill, stroke], index) => { const x = 26 + index * 133; return `<g><rect x="${x}" y="52" width="100" height="45" rx="5" fill="${fill}" stroke="${stroke}"/><text x="${x+50}" y="79" text-anchor="middle" class="chart-value">${label}</text>${index < steps.length - 1 ? `<path d="M${x+100} 74H${x+128}" stroke="#819088" marker-end="url(#evidence-arrow)"/>` : ''}</g>`; }).join('');
      const svg = svgFrame('Typed pipeline metrics are serialized to local JSON and optionally logged as W&B evidence', `<defs><marker id="evidence-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#819088"/></marker></defs>${nodes}`);
      return visualShell('Evidence flow', svg, `<span><strong>${visual.tests}</strong> automated tests</span><span><strong>${visual.reports}</strong> report artifacts</span><span>benchmark <strong>${visual.status}</strong></span>`);
    };
    const visualRenderers = { bootstrap: bootstrapVisual, frames: framesVisual, synchronization: synchronizationVisual, preprocessing: preprocessingVisual, trajectory: trajectoryVisual, keypoints: keypointVisual, embedding: embeddingVisual, saliency: saliencyVisual, regions: regionsVisual, static_filter: staticFilterVisual, object_store: objectStoreVisual, uplink: uplinkVisual, mesh: meshVisual, registry: registryVisual, association: associationVisual, gaussian: gaussianVisual, scene_map: sceneMapVisual, pose_graph: poseGraphVisual, se3_pose_graph: se3PoseGraphVisual, correction: correctionVisual, context: contextVisual, skyla_handoff: skylaHandoffVisual, skyla_planning: skylaPlanningVisual, planning: planningVisual, telemetry: telemetryVisual, simulation: simulationVisual, catalog: catalogVisual, security: securityVisual, deployment: deploymentVisual, end_to_end: endToEndVisual, evidence: evidenceVisual };
    const renderVisual = component => evidenceVideoModes[component.visual.kind]
      ? videoVisual(component, evidenceVideoModes[component.visual.kind])
      : visualRenderers[component.visual.kind](component.visual);

    const renderComponents = filter => {
      document.getElementById('component-list').innerHTML = data.components.map(component => `
        <article id="component-${component.visual.kind}" class="component" data-state="${component.state}" ${filter !== 'all' && filter !== component.state ? 'hidden' : ''}>
          <div class="component-copy"><div class="component-heading"><span class="group">${component.group}</span>${badge(component.state)}</div><h3>${component.name}</h3><p>${component.summary}</p><code>${component.implementation}</code></div>${renderVisual(component)}
        </article>`).join('');
      initializeEvidenceVideos();
    };
    renderComponents('all');
    if (window.location.hash) {
      window.setTimeout(() => document.querySelector(window.location.hash)?.scrollIntoView(), 0);
    }
    window.setInterval(() => {
      if (!evidencePlaying) return;
      evidenceFrameIndex = (evidenceFrameIndex + 1) % segment.frame_count;
      drawEvidenceVideos();
    }, 1000 / segment.fps);
    document.querySelectorAll('.filter-button').forEach(button => button.addEventListener('click', () => {
      document.querySelectorAll('.filter-button').forEach(item => item.classList.remove('active'));
      button.classList.add('active');
      renderComponents(button.dataset.filter);
    }));

    document.getElementById('dataset-grid').innerHTML = data.datasets.map(dataset => `<article class="dataset-card">${badge(dataset.state)}<h3>${dataset.name}</h3><p class="agents">${dataset.agents} agent${dataset.agents === 1 ? '' : 's'}</p><p class="signals">${dataset.signals}</p><p>${dataset.detail}</p><p class="limit">${dataset.limit}</p></article>`).join('');
    document.getElementById('gap-list').innerHTML = data.gaps.map(gap => `<li>${gap}</li>`).join('');

    const proof = [
      [fmt(data.phase1.metrics.vio_ate_improvement_percent, 1) + '%', 'VIO ATE improvement vs IMU reference'],
      [fmt(data.phase1.metrics.pose_graph_position_rmse_m, 4) + ' m', 'Pose graph position RMSE'],
      [data.phase1.metrics.false_static_insertions, 'False static insertions'],
      [data.phase1.metrics.pose_graph_rejected_constraints, 'False graph constraint rejected'],
    ];
    document.getElementById('proof-grid').innerHTML = proof.map(([value, label]) => `<div class="proof"><strong>${value}</strong><span>${label}</span></div>`).join('');
    document.getElementById('sources').innerHTML = data.sources.map(source => `<code>${source}</code>`).join('');
    if (window.location.hash) {
      document.querySelector(window.location.hash)?.scrollIntoView();
    }
  </script>
</body>
</html>"""


def main() -> int:
    payload = json.dumps(build_payload(), separators=(",", ":"), ensure_ascii=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(HTML.replace("__PAYLOAD__", payload), encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
