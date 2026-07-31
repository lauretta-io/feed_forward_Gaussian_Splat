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
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ariadne.benchmarks import build_video_evidence, select_video_frames
from ariadne.config import load_config
from ariadne.pose_correction import CorrectionLoadProfile, assess_correction_capacity
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
MILUV_GLOBAL_POSE = ROOT / "outputs/ariadne/miluv-global-pose/benchmark.json"
S3E_GLOBAL_POSE = ROOT / "outputs/ariadne/s3e-global-pose/benchmark.json"
OPERATIONS = ROOT / "outputs/ariadne/operations/benchmark.json"
END_TO_END = ROOT / "outputs/ariadne/end-to-end/benchmark.json"
DATASETS = ROOT / "outputs/ariadne/dataset_sequence/summary.json"
REAL_VIO = ROOT / "outputs/ariadne/real_vio/d2slam-1"
S3E_REAL_VIO = ROOT / "outputs/ariadne/real_vio"
FRAME_DIR = REAL_VIO / "orbslam3/euroc/mav0/cam0/data"
D2SLAM_ROOT = ROOT / "datasets/ariadne/d2slam/extracted/tum_corr"
RESPLAT_REPORT = ROOT / "outputs/ariadne/resplat_report/neighbourhood_105_10f"
RESPLAT_CHECKPOINT = ROOT / "pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth"
RESPLAT_RUN_URL = "https://wandb.ai/galvin/gaussiansplat_test/runs/15d73m80"
STATIC_GLOBAL_GAUSSIANS = ROOT / "outputs/ariadne/s3e-global-gaussian-static/manifest.json"
STATIC_GLOBAL_GAUSSIANS_PREPARATION = (
    ROOT / "outputs/ariadne/s3e-global-gaussian-static/preparation.json"
)

DOCUMENTATION_EXTRAS = (
    Path("applications/ariadne/docs/static_asynchronous_global_gaussians.md"),
    Path("applications/ariadne/docs/vio_global_pose_experiment_log.md"),
)
DOCUMENTATION_EXCLUDED_NAMES = {
    "LOCAL_SECRETS.md",
}
DOCUMENTATION_EXCLUDED_PARTS = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "node_modules",
    "outputs",
    "wandb",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def documentation_group(relative_path: Path) -> str:
    parts = relative_path.parts
    if parts[:2] == ("applications", "ariadne"):
        return "ARIADNE"
    if parts and parts[0] == "documentation":
        return "Specifications"
    if parts and parts[0] == "datasets":
        return "Datasets"
    if parts and parts[0] == "third_party":
        return "Third party"
    if parts and parts[0] == "src":
        return "Integrations"
    return "Repository"


def documentation_title(markdown: str, relative_path: Path) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return relative_path.stem.replace("_", " ").replace("-", " ").title()


def documentation_paths() -> list[Path]:
    """Return safe project documentation without exposing clone-local material."""
    command = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "--", "*.md"],
        check=False,
        capture_output=True,
    )
    relative_paths = {
        Path(item.decode("utf-8"))
        for item in command.stdout.split(b"\0")
        if item
    }
    relative_paths.update(path for path in DOCUMENTATION_EXTRAS if (ROOT / path).is_file())
    return sorted(
        path
        for path in relative_paths
        if path.name not in DOCUMENTATION_EXCLUDED_NAMES
        and not DOCUMENTATION_EXCLUDED_PARTS.intersection(path.parts)
        and (ROOT / path).is_file()
    )


def documentation_payload() -> list[dict[str, Any]]:
    documents = []
    for relative_path in documentation_paths():
        markdown = (ROOT / relative_path).read_text(encoding="utf-8")
        documents.append(
            {
                "group": documentation_group(relative_path),
                "markdown": markdown,
                "path": relative_path.as_posix(),
                "title": documentation_title(markdown, relative_path),
                "word_count": len(markdown.split()),
            }
        )
    return documents


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
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < len(truth_times)
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
    documentation = documentation_payload()
    phase1 = read_json(PHASE1)
    exchange = read_json(EXCHANGE)
    global_scene = read_json(GLOBAL_SCENE)
    miluv_global_pose = read_json(MILUV_GLOBAL_POSE)
    s3e_global_pose = read_json(S3E_GLOBAL_POSE)
    static_global_gaussians = {
        **read_json(STATIC_GLOBAL_GAUSSIANS),
        "preparation": read_json(STATIC_GLOBAL_GAUSSIANS_PREPARATION),
    }
    operations = read_json(OPERATIONS)
    end_to_end = read_json(END_TO_END)
    datasets = read_json(DATASETS)
    openvins = read_json(REAL_VIO / "openvins/evaluation.json")
    orbslam3 = read_json(REAL_VIO / "orbslam3/evaluation.json")
    s3e_vio = {
        agent: read_json(S3E_REAL_VIO / f"s3e-{agent}/orbslam3/evaluation.json")
        for agent in ("alpha", "bob", "carol")
    }
    s3e_openvins = read_json(S3E_REAL_VIO / "s3e-alpha/openvins/evaluation.json")
    s3e_carol_geometry_corrected = read_json(
        S3E_REAL_VIO / "s3e-carol/orbslam3-auto-geometry/evaluation.json"
    )
    s3e_carol_reproducibility = read_json(
        S3E_REAL_VIO / "s3e-carol/orbslam3-auto-geometry-reproducibility.json"
    )
    s3e_alpha_reproducibility = read_json(
        S3E_REAL_VIO / "s3e-alpha/orbslam3-bf-1.2-fast-init-reproducibility.json"
    )
    s3e_alpha_deterministic_reproducibility = read_json(
        S3E_REAL_VIO
        / "s3e-alpha/orbslam3-bf-1.2-fast-init-deterministic-reproducibility.json"
    )
    s3e_alpha_mapping_sync_reproducibility = read_json(
        S3E_REAL_VIO
        / "s3e-alpha/orbslam3-bf-1.2-fast-init-mapping-sync-reproducibility.json"
    )
    s3e_alpha_modes = {
        "stereo": read_json(S3E_REAL_VIO / "s3e-alpha/orbslam3-stereo/evaluation.json"),
        "calibrated_500": read_json(
            S3E_REAL_VIO / "s3e-alpha/orbslam3-bf-1.15/evaluation.json"
        ),
        "calibrated_1000": read_json(
            S3E_REAL_VIO / "s3e-alpha/orbslam3-bf-1.15-1000/evaluation.json"
        ),
        "fast_init_1000": read_json(
            S3E_REAL_VIO / "s3e-alpha/orbslam3-bf-1.15-fast-init/evaluation.json"
        ),
        "fast_init_scaled_1000": read_json(
            S3E_REAL_VIO / "s3e-alpha/orbslam3-bf-1.2-fast-init/evaluation.json"
        ),
        "high_recall_1000": read_json(
            S3E_REAL_VIO
            / "s3e-alpha/orbslam3-bf-1.2-fast-init-high-recall/evaluation.json"
        ),
        "late_default_1000": read_json(
            S3E_REAL_VIO / "s3e-alpha/orbslam3-fast-init-start-1000/evaluation.json"
        ),
        "late_scaled_1000": read_json(
            S3E_REAL_VIO
            / "s3e-alpha/orbslam3-bf-1.2-fast-init-start-1000/evaluation.json"
        ),
    }
    intelligence_config = load_config(
        ROOT / "applications/ariadne/configs/intelligence/default.yaml"
    )
    if intelligence_config.intelligence is None:
        raise ValueError("status build requires an Intelligence configuration")
    correction_config = intelligence_config.intelligence.correction
    scheduling_config = intelligence_config.intelligence.correction_scheduling
    alpha_repeat_metrics = s3e_alpha_reproducibility["metrics"]
    alpha_deterministic_metrics = s3e_alpha_deterministic_reproducibility["metrics"]
    alpha_mapping_sync_metrics = s3e_alpha_mapping_sync_reproducibility["metrics"]
    alpha_tracking_healthy = (
        int(alpha_repeat_metrics["tracking_healthy_count"])
        == int(alpha_repeat_metrics["replicate_count"])
    )
    alpha_live_correction_eligible = (
        alpha_tracking_healthy
        and int(alpha_repeat_metrics["causal_native_rtk_target_pass_count"])
        == int(alpha_repeat_metrics["replicate_count"])
        and bool(alpha_repeat_metrics["trajectory_reproducible"])
    )
    correction_profiles = (
        CorrectionLoadProfile(
            "Alpha",
            float(
                alpha_repeat_metrics[
                    "causal_native_rtk_correction_messages_per_minute_max"
                ]
            ),
            float(
                alpha_repeat_metrics[
                    "causal_native_rtk_correction_interval_min_seconds_min"
                ]
            ),
            int(
                alpha_repeat_metrics[
                    "causal_native_rtk_correction_burst_per_second_max"
                ]
            ),
            alpha_tracking_healthy,
            alpha_live_correction_eligible,
        ),
        *(
            CorrectionLoadProfile(
                agent_id,
                float(
                    report["metrics"][
                        "causal_native_rtk_sim3_correction_messages_per_minute"
                    ]
                ),
                float(
                    report["metrics"][
                        "causal_native_rtk_sim3_correction_interval_min_seconds"
                    ]
                ),
                int(
                    report["metrics"][
                        "causal_native_rtk_sim3_correction_burst_per_second_max"
                    ]
                ),
                bool(report["metrics"]["tracking_healthy"]),
                bool(
                    report["metrics"]["tracking_healthy"]
                    and report["metrics"]["causal_native_rtk_sim3_target_met"]
                ),
            )
            for agent_id, report in (
                ("Bob", s3e_vio["bob"]),
                ("Carol", s3e_carol_geometry_corrected),
            )
        ),
    )
    correction_capacity = assess_correction_capacity(
        correction_profiles,
        evaluation_period_s=scheduling_config.evaluation_period_seconds,
        max_corrections_per_cycle=scheduling_config.max_corrections_per_cycle,
    )
    correction_profile_details = [
        {
            **asdict(profile),
            "action": (
                "schedule_corrections"
                if profile.correction_eligible
                else (
                    "relocalize_live_pose_failure"
                    if profile.tracking_healthy
                    else "relocalize_tracking_failure"
                )
            ),
        }
        for profile in correction_profiles
    ]
    local_alignment_profiles = [
        {
            "agent_id": agent_id,
            "tracking_healthy": bool(report["metrics"]["tracking_healthy"]),
            "rigid_interval_s": float(
                report["metrics"]["local_rigid_maximum_passing_interval_seconds"]
            ),
            "rigid_messages_per_minute": float(
                report["metrics"]["local_rigid_optimistic_anchor_messages_per_minute"]
            ),
            "sim3_interval_s": float(
                report["metrics"]["local_sim3_maximum_passing_interval_seconds"]
            ),
            "sim3_messages_per_minute": float(
                report["metrics"]["local_sim3_optimistic_anchor_messages_per_minute"]
            ),
            "causal_se3_cadence_s": float(
                report["metrics"]["causal_se3_maximum_passing_cadence_seconds"]
            ),
            "causal_se3_messages_per_minute": float(
                report["metrics"]["causal_se3_anchor_messages_per_minute"]
            ),
            "causal_sim3_cadence_s": float(
                report["metrics"]["causal_sim3_maximum_passing_cadence_seconds"]
            ),
            "causal_sim3_messages_per_minute": float(
                report["metrics"]["causal_sim3_anchor_messages_per_minute"]
            ),
            "action": (
                "retain_event_triggered_translation"
                if bool(report["metrics"]["tracking_healthy"])
                else "relocalize"
            ),
        }
        for agent_id, report in (
            ("Alpha", s3e_alpha_modes["fast_init_scaled_1000"]),
            ("Bob", s3e_vio["bob"]),
            ("Carol", s3e_vio["carol"]),
        )
    ]
    resplat = read_json(RESPLAT_REPORT / "metrics.json")
    resplat_render = select_resplat_render(resplat)
    reference_truth = (
        D2SlamReplaySource(D2SLAM_ROOT, 1)
        .load(start_frame=0, max_frames=500)
        .ground_truth
    )
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
            "documents": len(documentation),
            "real_backends": 2,
            "phase1_status": phase1["status"],
        },
        "documentation": documentation,
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
        "evaluation_progression": {
            "target_ate_m": 0.1,
            "stages": [
                {
                    "label": "Original S3E Alpha",
                    "scope": "500 frames · default baseline",
                    "status": "baseline",
                    "ate_m": s3e_vio["alpha"]["metrics"]["ate_rmse_m"],
                    "sim3_ate_m": s3e_vio["alpha"]["metrics"]["sim3_ate_rmse_m"],
                    "rpe_m": s3e_vio["alpha"]["metrics"]["rpe_rmse_m"],
                    "poses": s3e_vio["alpha"]["metrics"]["trajectory_pose_count"],
                    "lost": s3e_vio["alpha"]["metrics"]["lost_frame_count"],
                    "resets": s3e_vio["alpha"]["metrics"]["map_reset_count"],
                    "delta": "comparison baseline",
                    "change": "Initial production ORB-SLAM3 replay.",
                },
                {
                    "label": "Metric-scale sensitivity",
                    "scope": "500 frames · 1.15x stereo baseline",
                    "status": "diagnostic",
                    "ate_m": s3e_alpha_modes["calibrated_500"]["metrics"]["ate_rmse_m"],
                    "sim3_ate_m": s3e_alpha_modes["calibrated_500"]["metrics"]["sim3_ate_rmse_m"],
                    "rpe_m": s3e_alpha_modes["calibrated_500"]["metrics"]["rpe_rmse_m"],
                    "poses": s3e_alpha_modes["calibrated_500"]["metrics"]["trajectory_pose_count"],
                    "lost": s3e_alpha_modes["calibrated_500"]["metrics"]["lost_frame_count"],
                    "resets": s3e_alpha_modes["calibrated_500"]["metrics"]["map_reset_count"],
                    "delta": "74.3% lower ATE vs 500-frame baseline",
                    "change": "Scaled the stereo baseline; tracking stayed healthy, but the setting was not transferable.",
                },
                {
                    "label": "Original long window",
                    "scope": "1,000 frames · 1.15x baseline",
                    "status": "baseline",
                    "ate_m": s3e_alpha_modes["calibrated_1000"]["metrics"]["ate_rmse_m"],
                    "sim3_ate_m": s3e_alpha_modes["calibrated_1000"]["metrics"]["sim3_ate_rmse_m"],
                    "rpe_m": s3e_alpha_modes["calibrated_1000"]["metrics"]["rpe_rmse_m"],
                    "poses": s3e_alpha_modes["calibrated_1000"]["metrics"]["trajectory_pose_count"],
                    "lost": s3e_alpha_modes["calibrated_1000"]["metrics"]["lost_frame_count"],
                    "resets": s3e_alpha_modes["calibrated_1000"]["metrics"]["map_reset_count"],
                    "delta": "new long-window baseline",
                    "change": "Doubled the evaluation window; exposed resets and lost-frame continuity failure.",
                },
                {
                    "label": "Fast IMU initialization",
                    "scope": "1,000 frames · 1.15x baseline",
                    "status": "retained",
                    "ate_m": s3e_alpha_modes["fast_init_1000"]["metrics"]["ate_rmse_m"],
                    "sim3_ate_m": s3e_alpha_modes["fast_init_1000"]["metrics"]["sim3_ate_rmse_m"],
                    "rpe_m": s3e_alpha_modes["fast_init_1000"]["metrics"]["rpe_rmse_m"],
                    "poses": s3e_alpha_modes["fast_init_1000"]["metrics"]["trajectory_pose_count"],
                    "lost": s3e_alpha_modes["fast_init_1000"]["metrics"]["lost_frame_count"],
                    "resets": s3e_alpha_modes["fast_init_1000"]["metrics"]["map_reset_count"],
                    "delta": "37.8% lower ATE vs long baseline",
                    "change": "Bypassed the static-motion initialization wait; recovered continuity without reaching the accuracy target.",
                },
                {
                    "label": "Selected balanced profile",
                    "scope": "1,000 frames · fast init · 1.20x baseline",
                    "status": "retained",
                    "ate_m": s3e_alpha_modes["fast_init_scaled_1000"]["metrics"]["ate_rmse_m"],
                    "sim3_ate_m": s3e_alpha_modes["fast_init_scaled_1000"]["metrics"]["sim3_ate_rmse_m"],
                    "rpe_m": s3e_alpha_modes["fast_init_scaled_1000"]["metrics"]["rpe_rmse_m"],
                    "poses": s3e_alpha_modes["fast_init_scaled_1000"]["metrics"]["trajectory_pose_count"],
                    "lost": s3e_alpha_modes["fast_init_scaled_1000"]["metrics"]["lost_frame_count"],
                    "resets": s3e_alpha_modes["fast_init_scaled_1000"]["metrics"]["map_reset_count"],
                    "delta": "52.7% lower ATE vs long baseline",
                    "change": "Raised the baseline scale to 1.20x; best healthy single long-window configuration.",
                },
                {
                    "label": "High-recall features",
                    "scope": "1,000 frames · 2,400 ORB features",
                    "status": "rejected",
                    "ate_m": s3e_alpha_modes["high_recall_1000"]["metrics"]["ate_rmse_m"],
                    "sim3_ate_m": s3e_alpha_modes["high_recall_1000"]["metrics"]["sim3_ate_rmse_m"],
                    "rpe_m": s3e_alpha_modes["high_recall_1000"]["metrics"]["rpe_rmse_m"],
                    "poses": s3e_alpha_modes["high_recall_1000"]["metrics"]["trajectory_pose_count"],
                    "lost": s3e_alpha_modes["high_recall_1000"]["metrics"]["lost_frame_count"],
                    "resets": s3e_alpha_modes["high_recall_1000"]["metrics"]["map_reset_count"],
                    "delta": "52.0% higher ATE; 16.0% lower RPE",
                    "change": "Increased feature count; local motion improved while global consistency regressed.",
                },
                {
                    "label": "Normal real-time repeats",
                    "scope": "1,000 frames · three identical runs",
                    "status": "diagnostic",
                    "ate_m": alpha_repeat_metrics["ate_rmse_m_median"],
                    "ate_min_m": alpha_repeat_metrics["ate_rmse_m_min"],
                    "ate_max_m": alpha_repeat_metrics["ate_rmse_m_max"],
                    "sim3_ate_m": alpha_repeat_metrics["sim3_ate_rmse_m_median"],
                    "rpe_m": alpha_repeat_metrics["rpe_rmse_m_median"],
                    "poses": alpha_repeat_metrics["trajectory_pose_count_min"],
                    "lost": alpha_repeat_metrics["lost_frame_count_max"],
                    "resets": alpha_repeat_metrics["map_reset_count_max"],
                    "delta": "1.338–1.635 m ATE range",
                    "change": "Added a three-run gate; identical tracking counts exposed trajectory-shape nondeterminism.",
                },
                {
                    "label": "Single-CPU controlled runtime",
                    "scope": "1,000 frames · one CPU · three runs",
                    "status": "diagnostic",
                    "ate_m": alpha_deterministic_metrics["ate_rmse_m_median"],
                    "ate_min_m": alpha_deterministic_metrics["ate_rmse_m_min"],
                    "ate_max_m": alpha_deterministic_metrics["ate_rmse_m_max"],
                    "sim3_ate_m": alpha_deterministic_metrics["sim3_ate_rmse_m_median"],
                    "rpe_m": alpha_deterministic_metrics["rpe_rmse_m_median"],
                    "poses": alpha_deterministic_metrics["trajectory_pose_count_min"],
                    "lost": alpha_deterministic_metrics["lost_frame_count_max"],
                    "resets": alpha_deterministic_metrics["map_reset_count_max"],
                    "delta": "3.6% lower median; 86.5% wider ATE range",
                    "change": "Pinned ORB-SLAM3 and numeric libraries to one CPU; accuracy varied less in the median but reproducibility worsened.",
                },
                {
                    "label": "Current mapping-synchronized runtime",
                    "scope": "1,000 frames · offline mapper barrier · three runs",
                    "status": "current",
                    "ate_m": alpha_mapping_sync_metrics["ate_rmse_m_median"],
                    "ate_min_m": alpha_mapping_sync_metrics["ate_rmse_m_min"],
                    "ate_max_m": alpha_mapping_sync_metrics["ate_rmse_m_max"],
                    "sim3_ate_m": alpha_mapping_sync_metrics[
                        "sim3_ate_rmse_m_median"
                    ],
                    "rpe_m": alpha_mapping_sync_metrics["rpe_rmse_m_median"],
                    "poses": alpha_mapping_sync_metrics["trajectory_pose_count_min"],
                    "lost": alpha_mapping_sync_metrics["lost_frame_count_max"],
                    "resets": alpha_mapping_sync_metrics["map_reset_count_max"],
                    "delta": "18.5% narrower ATE range; 49.7% narrower Sim(3) range",
                    "change": "Waited for local mapping after every frame and removed real-time pacing; spread improved but the median regressed and reproducibility still failed.",
                },
            ],
            "current_layers": [
                {
                    "label": "Raw rigid VIO",
                    "scope": "three-run median · production backend",
                    "status": "diagnostic",
                    "ate_m": alpha_mapping_sync_metrics["ate_rmse_m_median"],
                    "ate_min_m": alpha_mapping_sync_metrics["ate_rmse_m_min"],
                    "ate_max_m": alpha_mapping_sync_metrics["ate_rmse_m_max"],
                    "detail": "Current three-run median; production tracking, failed reproducibility.",
                },
                {
                    "label": "Global Sim(3) alignment",
                    "scope": "offline whole-trajectory alignment",
                    "status": "diagnostic",
                    "ate_m": alpha_mapping_sync_metrics["sim3_ate_rmse_m_median"],
                    "ate_min_m": alpha_mapping_sync_metrics["sim3_ate_rmse_m_min"],
                    "ate_max_m": alpha_mapping_sync_metrics["sim3_ate_rmse_m_max"],
                    "detail": "Offline whole-trajectory scale alignment; does not repair path shape.",
                },
                {
                    "label": "Native 1 Hz RTK online",
                    "scope": "live causal correction · 58.85/min",
                    "status": "rejected",
                    "ate_m": alpha_mapping_sync_metrics[
                        "causal_native_rtk_sim3_ate_m_median"
                    ],
                    "ate_min_m": alpha_mapping_sync_metrics[
                        "causal_native_rtk_sim3_ate_m_min"
                    ],
                    "ate_max_m": alpha_mapping_sync_metrics[
                        "causal_native_rtk_sim3_ate_m_max"
                    ],
                    "detail": "Live and causal at 58.85 corrections/min; zero of three target passes.",
                },
                {
                    "label": "Adaptive fixed-lag map",
                    "scope": "delayed map history · 38.22/min",
                    "status": "controlled",
                    "ate_m": alpha_mapping_sync_metrics[
                        "adaptive_fixed_lag_native_rtk_sim3_ate_m_median"
                    ],
                    "ate_min_m": alpha_mapping_sync_metrics[
                        "adaptive_fixed_lag_native_rtk_sim3_ate_m_min"
                    ],
                    "ate_max_m": alpha_mapping_sync_metrics[
                        "adaptive_fixed_lag_native_rtk_sim3_ate_m_max"
                    ],
                    "detail": "Delayed map history at 38.22 finalizations/min and 1.890 s p95 delay; three of three controlled passes.",
                },
            ],
            "normal_ate_range_m": alpha_repeat_metrics["ate_rmse_m_range"],
            "current_ate_range_m": alpha_mapping_sync_metrics["ate_rmse_m_range"],
            "normal_sim3_range_m": alpha_repeat_metrics["sim3_ate_rmse_m_range"],
            "current_sim3_range_m": alpha_mapping_sync_metrics[
                "sim3_ate_rmse_m_range"
            ],
            "source": "S3E Playground 2 Alpha ORB-SLAM3 evaluation artifacts",
        },
        "phase1": phase1,
        "exchange": exchange,
        "global_scene": global_scene,
        "static_global_gaussians": static_global_gaussians,
        "miluv_global_pose": miluv_global_pose,
        "s3e_global_pose": s3e_global_pose,
        "s3e_real_vio": s3e_vio,
        "s3e_carol_geometry_corrected": s3e_carol_geometry_corrected,
        "s3e_carol_reproducibility": s3e_carol_reproducibility,
        "s3e_alpha_reproducibility": s3e_alpha_reproducibility,
        "s3e_alpha_deterministic_reproducibility": (
            s3e_alpha_deterministic_reproducibility
        ),
        "s3e_alpha_mapping_sync_reproducibility": (
            s3e_alpha_mapping_sync_reproducibility
        ),
        "s3e_alpha_modes": s3e_alpha_modes,
        "s3e_correction_capacity": asdict(correction_capacity),
        "operations": operations,
        "end_to_end": end_to_end,
        "datasets": [
            {
                "name": "MILUV",
                "state": "validated",
                "agents": dataset_by_name["miluv"]["metrics"]["agent_count"],
                "signals": "Vision / IMU / UWB / mocap",
                "detail": (
                    "Real full-SE(3) mocap truth validates the shared rationalizer at "
                    f"{miluv_global_pose['metrics']['optimized_global_ate_m']:.4f} m ATE "
                    "with five corrections per UAV."
                ),
                "limit": (
                    "Local odometry and cross-agent factors remain controlled; raw UWB "
                    f"is {miluv_global_pose['metrics']['uwb_range_rmse_m']:.3f} m RMSE "
                    "and cannot directly anchor the 0.1 m target."
                ),
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
                "state": "validated",
                "agents": dataset_by_name["s3e"]["metrics"]["agent_count"],
                "signals": "Stereo / IMU / GNSS / LiDAR",
                "detail": (
                    "The controlled rationalization proxy reaches "
                    f"{s3e_global_pose['metrics']['optimized_global_ate_m']:.3f} m ATE, but "
                    "none of three production ORB-SLAM3 Wingman runs meets 0.1 m, "
                    "and matched Alpha OpenVINS diverges."
                ),
                "limit": (
                    "Alpha fast initialization removes resets over 1,000 frames, "
                    "but three identical runs span "
                    f"{s3e_alpha_reproducibility['metrics']['ate_rmse_m_min']:.2f}–"
                    f"{s3e_alpha_reproducibility['metrics']['ate_rmse_m_max']:.2f} m ATE "
                    "and fail the reproducibility gate; "
                    "OpenVINS reaches "
                    f"{s3e_openvins['metrics']['ate_rmse_m']:.1f} m with a "
                    f"{s3e_openvins['metrics']['metric_scale_correction_to_truth']:.3f}× "
                    "scale correction and must relocalize."
                ),
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
                            "points": trajectory_points(
                                REAL_VIO / "openvins/trajectory.txt"
                            ),
                        },
                        {
                            "name": "ORB-SLAM3",
                            "points": trajectory_points(
                                REAL_VIO / "orbslam3/f_ariadne.txt"
                            ),
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
                    "contract_backend": global_scene["metrics"][
                        "reconstruction_backend"
                    ],
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
                        item
                        for item in resplat["per_view"]
                        if item["name"] == resplat_render.name
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
                    "history": global_scene["details"]["scene_map"][
                        "history_revisions"
                    ],
                    "snapshot_bytes": global_scene["details"]["scene_map"][
                        "snapshot_bytes"
                    ],
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
                "summary": (
                    "MILUV real 6-DoF mocap and S3E RTK geometry validate shared multi-agent "
                    "rationalization, per-Wingman load, and false-loop rejection; bounded SE(3) "
                    "state also preserves revisions across restart."
                ),
                "implementation": "RobustSE3PoseGraph + MILUV + S3E · ariadne.se3-pose-graph.v1",
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
                    "miluv_baseline_ate_m": miluv_global_pose["metrics"][
                        "baseline_global_ate_m"
                    ],
                    "miluv_optimized_ate_m": miluv_global_pose["metrics"][
                        "optimized_global_ate_m"
                    ],
                    "miluv_optimized_orientation_rmse_rad": miluv_global_pose[
                        "metrics"
                    ]["optimized_global_orientation_rmse_rad"],
                    "miluv_target_ate_m": miluv_global_pose["metrics"][
                        "target_global_ate_m"
                    ],
                    "miluv_correction_interval_s": miluv_global_pose["metrics"][
                        "selected_correction_interval_seconds"
                    ],
                    "miluv_messages_per_minute_max": miluv_global_pose["metrics"][
                        "selected_correction_messages_per_minute_max"
                    ],
                    "miluv_corrections_by_agent": miluv_global_pose["details"][
                        "intelligence_load"
                    ]["correction_count_by_agent"],
                    "miluv_cross_agent_only_ate_m": miluv_global_pose["details"][
                        "vision_correction_limits"
                    ]["cross_agent_only_global_ate_m"],
                    "miluv_uwb_rmse_m": miluv_global_pose["metrics"][
                        "uwb_range_rmse_m"
                    ],
                    "miluv_uwb_p95_m": miluv_global_pose["metrics"][
                        "uwb_range_absolute_error_p95_m"
                    ],
                    "miluv_uwb_causal_ate_m": miluv_global_pose["metrics"][
                        "uwb_graph_best_tested_global_ate_m"
                    ],
                    "miluv_uwb_causal_messages_per_minute_max": miluv_global_pose[
                        "metrics"
                    ]["uwb_graph_best_tested_messages_per_minute_max"],
                    "miluv_uwb_fixed_lag_position_ate_m": miluv_global_pose["metrics"][
                        "uwb_fixed_lag_global_position_ate_m"
                    ],
                    "miluv_uwb_fixed_lag_max_agent_ate_m": miluv_global_pose["metrics"][
                        "uwb_fixed_lag_max_agent_position_ate_m"
                    ],
                    "miluv_uwb_fixed_lag_orientation_rmse_rad": miluv_global_pose[
                        "metrics"
                    ]["uwb_fixed_lag_global_orientation_rmse_rad"],
                    "miluv_uwb_fixed_lag_duration_s": miluv_global_pose["metrics"][
                        "uwb_fixed_lag_duration_seconds"
                    ],
                    "miluv_uwb_fixed_lag_solve_interval_s": miluv_global_pose[
                        "metrics"
                    ]["uwb_fixed_lag_solve_interval_seconds"],
                    "miluv_uwb_fixed_lag_solve_p95_ms": miluv_global_pose["metrics"][
                        "uwb_fixed_lag_optimization_latency_ms_p95"
                    ],
                    "miluv_uwb_fixed_lag_messages_per_minute_max": (
                        miluv_global_pose["metrics"][
                            "uwb_fixed_lag_correction_messages_per_minute_max"
                        ]
                    ),
                    "miluv_uwb_fixed_lag_all_agents_target_met": (
                        miluv_global_pose["metrics"][
                            "uwb_fixed_lag_all_agents_position_target_met"
                        ]
                    ),
                    "miluv_uwb_fixed_lag_position_claim_eligible": (
                        miluv_global_pose["metrics"][
                            "uwb_fixed_lag_position_claim_eligible"
                        ]
                    ),
                    "miluv_uwb_fixed_lag_full_pose_claim_eligible": (
                        miluv_global_pose["metrics"][
                            "uwb_fixed_lag_full_pose_claim_eligible"
                        ]
                    ),
                    "miluv_uwb_batch_position_ate_m": miluv_global_pose["metrics"][
                        "uwb_batch_global_position_ate_m"
                    ],
                    "miluv_uwb_batch_orientation_rmse_rad": miluv_global_pose[
                        "metrics"
                    ]["uwb_batch_global_orientation_rmse_rad"],
                    "miluv_uwb_batch_messages_per_minute_max": miluv_global_pose[
                        "metrics"
                    ]["uwb_batch_correction_messages_per_minute_max"],
                    "miluv_uwb_batch_position_target_met": miluv_global_pose["metrics"][
                        "uwb_batch_position_target_met"
                    ],
                    "miluv_uwb_batch_full_pose_target_met": miluv_global_pose[
                        "metrics"
                    ]["uwb_batch_full_pose_target_met"],
                    "miluv_loaded_archive_fraction_percent": miluv_global_pose[
                        "metrics"
                    ]["loaded_archive_fraction_percent"],
                    "miluv_adaptive_correction_count": miluv_global_pose["details"][
                        "adaptive_scheduler"
                    ]["global_correction_count"],
                    "miluv_fixed_correction_count": miluv_global_pose["details"][
                        "fixed_cadence_reference"
                    ]["global_correction_count"],
                    "s3e_baseline_ate_m": s3e_global_pose["metrics"][
                        "baseline_global_ate_m"
                    ],
                    "s3e_optimized_ate_m": s3e_global_pose["metrics"][
                        "optimized_global_ate_m"
                    ],
                    "s3e_maximum_agent_ate_m": s3e_global_pose["metrics"][
                        "maximum_agent_global_ate_m"
                    ],
                    "s3e_per_agent_ate_m": s3e_global_pose["details"][
                        "adaptive_scheduler"
                    ]["per_agent_ate_m"],
                    "s3e_baseline_orientation_rmse_rad": s3e_global_pose["metrics"][
                        "baseline_global_orientation_rmse_rad"
                    ],
                    "s3e_optimized_orientation_rmse_rad": s3e_global_pose["metrics"][
                        "optimized_global_orientation_rmse_rad"
                    ],
                    "s3e_target_orientation_rmse_rad": s3e_global_pose["metrics"][
                        "target_global_orientation_rmse_rad"
                    ],
                    "s3e_improvement_percent": s3e_global_pose["metrics"][
                        "global_ate_improvement_percent"
                    ],
                    "s3e_cross_agent_baseline_relative_rmse_m": (
                        s3e_global_pose["metrics"][
                            "baseline_cross_agent_relative_translation_rmse_m"
                        ]
                    ),
                    "s3e_cross_agent_dense_relative_rmse_m": (
                        s3e_global_pose["metrics"][
                            "dense_cross_agent_relative_translation_rmse_m"
                        ]
                    ),
                    "s3e_cross_agent_dense_relative_improvement_percent": (
                        s3e_global_pose["metrics"][
                            "dense_cross_agent_relative_improvement_percent"
                        ]
                    ),
                    "s3e_cross_agent_only_global_ate_m": (
                        s3e_global_pose["metrics"][
                            "dense_cross_agent_only_global_ate_m"
                        ]
                    ),
                    "s3e_cross_agent_factor_rate_per_minute": (
                        s3e_global_pose["metrics"][
                            "dense_cross_agent_factor_rate_per_minute"
                        ]
                    ),
                    "s3e_cross_agent_relative_rmse_at_0_05m_noise": (
                        s3e_global_pose["metrics"][
                            "cross_agent_relative_rmse_at_0_05m_translation_noise_m"
                        ]
                    ),
                    "s3e_cross_agent_relative_rmse_at_0_2m_noise": (
                        s3e_global_pose["metrics"][
                            "cross_agent_relative_rmse_at_0_2m_translation_noise_m"
                        ]
                    ),
                    "s3e_target_ate_m": s3e_global_pose["metrics"][
                        "target_global_ate_m"
                    ],
                    "s3e_correction_interval_s": s3e_global_pose["metrics"][
                        "selected_correction_interval_seconds"
                    ],
                    "s3e_messages_per_minute": s3e_global_pose["metrics"][
                        "selected_correction_messages_per_minute_per_agent"
                    ],
                    "s3e_selected_correction_count": s3e_global_pose["metrics"][
                        "selected_global_correction_count"
                    ],
                    "s3e_fixed_correction_count": s3e_global_pose["metrics"][
                        "fixed_global_correction_count"
                    ],
                    "s3e_correction_load_reduction_percent": s3e_global_pose["metrics"][
                        "selected_correction_load_reduction_percent"
                    ],
                    "s3e_scheduler_demand_error_m": s3e_global_pose["metrics"][
                        "selected_scheduler_demand_error_m"
                    ],
                    "s3e_capacity_override_cycles": s3e_global_pose["metrics"][
                        "selected_capacity_override_cycle_count"
                    ],
                    "s3e_messages_per_minute_by_agent": s3e_global_pose["details"][
                        "intelligence_load"
                    ]["messages_per_minute_by_agent"],
                    "s3e_corrections_by_agent": s3e_global_pose["details"][
                        "intelligence_load"
                    ]["correction_count_by_agent"],
                    "s3e_payload_bytes": s3e_global_pose["metrics"][
                        "selected_correction_payload_bytes_total"
                    ],
                    "s3e_report_payload_bytes": S3E_GLOBAL_POSE.stat().st_size,
                    "s3e_optimization_latency_ms": s3e_global_pose["metrics"][
                        "optimization_latency_ms"
                    ],
                    "s3e_rotation_rationalization_constraints": s3e_global_pose[
                        "metrics"
                    ]["rotation_rationalization_constraint_count"],
                    "s3e_rotation_rationalization_iterations": s3e_global_pose[
                        "metrics"
                    ]["rotation_rationalization_iterations"],
                    "s3e_constraint_rotation_rmse_rad": s3e_global_pose["metrics"][
                        "optimized_constraint_rotation_rmse_rad"
                    ],
                    "s3e_maximum_translation_noise_m": s3e_global_pose["metrics"][
                        "maximum_tested_correction_noise_m_for_target"
                    ],
                    "s3e_maximum_rotation_noise_rad": s3e_global_pose["metrics"][
                        "maximum_tested_correction_rotation_noise_rad_for_target"
                    ],
                    "s3e_strategy": s3e_global_pose["metrics"][
                        "selected_correction_strategy"
                    ],
                    "s3e_position_claim_eligible": s3e_global_pose["metrics"][
                        "position_claim_eligible"
                    ],
                    "s3e_full_pose_claim_eligible": s3e_global_pose["metrics"][
                        "full_pose_claim_eligible"
                    ],
                    "s3e_real_vio_target_met": sum(
                        report["metrics"]["target_ate_met"]
                        for report in s3e_vio.values()
                    ),
                    "s3e_real_vio_agent_count": len(s3e_vio),
                    "s3e_real_vio_worst_ate_m": max(
                        report["metrics"]["ate_rmse_m"] for report in s3e_vio.values()
                    ),
                    "s3e_real_vio_max_required_messages_per_minute": max(
                        report["metrics"]["correction_messages_per_minute_for_0_1m"]
                        for report in s3e_vio.values()
                    ),
                    "s3e_real_vio_unrecoverable_agents": sum(
                        not report["metrics"].get(
                            "correction_target_reachable_with_tested_intervals",
                            report["metrics"][
                                "maximum_correction_interval_seconds_for_0_1m"
                            ]
                            > 0,
                        )
                        for report in s3e_vio.values()
                    ),
                    "s3e_sensor_contract_passes": sum(
                        int(report["metrics"].get("s3e_sensor_contract_healthy", 0))
                        for report in s3e_vio.values()
                    ),
                    "s3e_carol_corrected_ate_m": s3e_carol_geometry_corrected[
                        "metrics"
                    ]["ate_rmse_m"],
                    "s3e_carol_corrected_pose_count": s3e_carol_geometry_corrected[
                        "metrics"
                    ]["trajectory_pose_count"],
                    "s3e_carol_corrected_lost_frames": s3e_carol_geometry_corrected[
                        "metrics"
                    ]["lost_frame_count"],
                    "s3e_carol_corrected_resets": s3e_carol_geometry_corrected[
                        "metrics"
                    ]["map_reset_count"],
                    "s3e_carol_replicate_ate_median": (
                        s3e_carol_reproducibility["metrics"]["ate_rmse_m_median"]
                    ),
                    "s3e_carol_replicate_ate_min": (
                        s3e_carol_reproducibility["metrics"]["ate_rmse_m_min"]
                    ),
                    "s3e_carol_replicate_ate_max": (
                        s3e_carol_reproducibility["metrics"]["ate_rmse_m_max"]
                    ),
                    "s3e_carol_reproducible": s3e_carol_reproducibility["metrics"][
                        "trajectory_reproducible"
                    ],
                    "s3e_alpha_replicate_ate_median": (
                        s3e_alpha_reproducibility["metrics"]["ate_rmse_m_median"]
                    ),
                    "s3e_alpha_replicate_ate_min": (
                        s3e_alpha_reproducibility["metrics"]["ate_rmse_m_min"]
                    ),
                    "s3e_alpha_replicate_ate_max": (
                        s3e_alpha_reproducibility["metrics"]["ate_rmse_m_max"]
                    ),
                    "s3e_alpha_deterministic_ate_min": (
                        s3e_alpha_deterministic_reproducibility["metrics"][
                            "ate_rmse_m_min"
                        ]
                    ),
                    "s3e_alpha_deterministic_ate_max": (
                        s3e_alpha_deterministic_reproducibility["metrics"][
                            "ate_rmse_m_max"
                        ]
                    ),
                    "s3e_alpha_deterministic_sim3_min": (
                        s3e_alpha_deterministic_reproducibility["metrics"][
                            "sim3_ate_rmse_m_min"
                        ]
                    ),
                    "s3e_alpha_deterministic_sim3_max": (
                        s3e_alpha_deterministic_reproducibility["metrics"][
                            "sim3_ate_rmse_m_max"
                        ]
                    ),
                    "s3e_alpha_deterministic_reproducible": (
                        s3e_alpha_deterministic_reproducibility["metrics"][
                            "trajectory_reproducible"
                        ]
                    ),
                    "s3e_alpha_mapping_sync_ate_min": (
                        s3e_alpha_mapping_sync_reproducibility["metrics"][
                            "ate_rmse_m_min"
                        ]
                    ),
                    "s3e_alpha_mapping_sync_ate_max": (
                        s3e_alpha_mapping_sync_reproducibility["metrics"][
                            "ate_rmse_m_max"
                        ]
                    ),
                    "s3e_alpha_mapping_sync_sim3_min": (
                        s3e_alpha_mapping_sync_reproducibility["metrics"][
                            "sim3_ate_rmse_m_min"
                        ]
                    ),
                    "s3e_alpha_mapping_sync_sim3_max": (
                        s3e_alpha_mapping_sync_reproducibility["metrics"][
                            "sim3_ate_rmse_m_max"
                        ]
                    ),
                    "s3e_alpha_mapping_sync_reproducible": (
                        s3e_alpha_mapping_sync_reproducibility["metrics"][
                            "trajectory_reproducible"
                        ]
                    ),
                    "s3e_alpha_native_rtk_target_pass_count": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_native_rtk_target_pass_count"
                        ]
                    ),
                    "s3e_alpha_native_rtk_sim3_ate_min": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_native_rtk_sim3_ate_m_min"
                        ]
                    ),
                    "s3e_alpha_native_rtk_sim3_ate_max": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_native_rtk_sim3_ate_m_max"
                        ]
                    ),
                    "s3e_alpha_native_rtk_se3_ate_max": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_native_rtk_se3_ate_m_max"
                        ]
                    ),
                    "s3e_alpha_native_rtk_anchor_rate": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_native_rtk_anchor_messages_per_minute_max"
                        ]
                    ),
                    "s3e_alpha_native_rtk_correction_rate": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_native_rtk_correction_messages_per_minute_max"
                        ]
                    ),
                    "s3e_alpha_segment_hold_target_pass_count": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_segment_hold_native_rtk_target_pass_count"
                        ]
                    ),
                    "s3e_alpha_segment_hold_ate_min": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_segment_hold_native_rtk_sim3_ate_m_min"
                        ]
                    ),
                    "s3e_alpha_segment_hold_ate_max": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_segment_hold_native_rtk_sim3_ate_m_max"
                        ]
                    ),
                    "s3e_segment_hold_by_agent": {
                        "Alpha": s3e_alpha_reproducibility["metrics"][
                            "causal_segment_hold_native_rtk_sim3_ate_m_max"
                        ],
                        "Bob": s3e_vio["bob"]["metrics"][
                            "causal_segment_hold_native_rtk_sim3_ate_m"
                        ],
                        "Carol": s3e_carol_geometry_corrected["metrics"][
                            "causal_segment_hold_native_rtk_sim3_ate_m"
                        ],
                    },
                    "s3e_segment_hold_updates_by_agent": {
                        "Alpha": s3e_alpha_reproducibility["metrics"][
                            "causal_segment_hold_native_rtk_updates_per_minute_max"
                        ],
                        "Bob": s3e_vio["bob"]["metrics"][
                            "causal_segment_hold_native_rtk_prediction_updates_per_minute"
                        ],
                        "Carol": s3e_carol_geometry_corrected["metrics"][
                            "causal_segment_hold_native_rtk_prediction_updates_per_minute"
                        ],
                    },
                    "s3e_alpha_segment_hold_horizon_ate": {
                        "0.1": s3e_alpha_reproducibility["metrics"][
                            "causal_segment_hold_native_rtk_horizon_0_1s_ate_m_max"
                        ],
                        "0.2": s3e_alpha_reproducibility["metrics"][
                            "causal_segment_hold_native_rtk_horizon_0_2s_ate_m_max"
                        ],
                        "0.5": s3e_alpha_reproducibility["metrics"][
                            "causal_segment_hold_native_rtk_horizon_0_5s_ate_m_max"
                        ],
                        "1.0": s3e_alpha_reproducibility["metrics"][
                            "causal_segment_hold_native_rtk_horizon_1s_ate_m_max"
                        ],
                    },
                    "s3e_alpha_segment_hold_required_observation_rate": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_segment_hold_native_rtk_minimum_observation_rate_per_minute_max"
                        ]
                    ),
                    "s3e_segment_hold_target_horizon_by_agent": {
                        "Alpha": s3e_alpha_reproducibility["metrics"][
                            "causal_segment_hold_native_rtk_maximum_target_horizon_seconds_min"
                        ],
                        "Bob": s3e_vio["bob"]["metrics"][
                            "causal_segment_hold_native_rtk_maximum_target_horizon_seconds"
                        ],
                        "Carol": s3e_carol_geometry_corrected["metrics"][
                            "causal_segment_hold_native_rtk_maximum_target_horizon_seconds"
                        ],
                    },
                    "s3e_alpha_fixed_lag_target_pass_count": (
                        s3e_alpha_reproducibility["metrics"][
                            "fixed_lag_native_rtk_target_pass_count"
                        ]
                    ),
                    "s3e_alpha_fixed_lag_sim3_ate_min": (
                        s3e_alpha_reproducibility["metrics"][
                            "fixed_lag_native_rtk_sim3_ate_m_min"
                        ]
                    ),
                    "s3e_alpha_fixed_lag_sim3_ate_max": (
                        s3e_alpha_reproducibility["metrics"][
                            "fixed_lag_native_rtk_sim3_ate_m_max"
                        ]
                    ),
                    "s3e_alpha_fixed_lag_se3_ate_max": (
                        s3e_alpha_reproducibility["metrics"][
                            "fixed_lag_native_rtk_se3_ate_m_max"
                        ]
                    ),
                    "s3e_alpha_fixed_lag_coverage_min": (
                        s3e_alpha_reproducibility["metrics"][
                            "fixed_lag_native_rtk_pose_coverage_fraction_min"
                        ]
                    ),
                    "s3e_alpha_fixed_lag_updates_per_minute": (
                        s3e_alpha_reproducibility["metrics"][
                            "fixed_lag_native_rtk_finalization_updates_per_minute_max"
                        ]
                    ),
                    "s3e_alpha_fixed_lag_latency_mean_s": (
                        s3e_alpha_reproducibility["metrics"][
                            "fixed_lag_native_rtk_latency_mean_seconds_max"
                        ]
                    ),
                    "s3e_alpha_fixed_lag_latency_p95_s": (
                        s3e_alpha_reproducibility["metrics"][
                            "fixed_lag_native_rtk_latency_p95_seconds_max"
                        ]
                    ),
                    "s3e_alpha_fixed_lag_latency_max_s": (
                        s3e_alpha_reproducibility["metrics"][
                            "fixed_lag_native_rtk_latency_max_seconds_max"
                        ]
                    ),
                    "s3e_alpha_fixed_lag_scale_p05_min": (
                        s3e_alpha_reproducibility["metrics"][
                            "fixed_lag_native_rtk_scale_p05_min"
                        ]
                    ),
                    "s3e_alpha_fixed_lag_scale_p95_max": (
                        s3e_alpha_reproducibility["metrics"][
                            "fixed_lag_native_rtk_scale_p95_max"
                        ]
                    ),
                    "s3e_alpha_fixed_lag_scale_max": (
                        s3e_alpha_reproducibility["metrics"][
                            "fixed_lag_native_rtk_scale_max"
                        ]
                    ),
                    "s3e_alpha_fixed_lag_scale_plausible_fraction": (
                        s3e_alpha_reproducibility["metrics"][
                            "fixed_lag_native_rtk_scale_plausible_fraction_min"
                        ]
                    ),
                    "s3e_alpha_adaptive_fixed_lag_target_pass_count": (
                        s3e_alpha_reproducibility["metrics"][
                            "adaptive_fixed_lag_native_rtk_target_pass_count"
                        ]
                    ),
                    "s3e_alpha_adaptive_fixed_lag_ate_min": (
                        s3e_alpha_reproducibility["metrics"][
                            "adaptive_fixed_lag_native_rtk_sim3_ate_m_min"
                        ]
                    ),
                    "s3e_alpha_adaptive_fixed_lag_ate_max": (
                        s3e_alpha_reproducibility["metrics"][
                            "adaptive_fixed_lag_native_rtk_sim3_ate_m_max"
                        ]
                    ),
                    "s3e_alpha_adaptive_fixed_lag_updates_min": (
                        s3e_alpha_reproducibility["metrics"][
                            "adaptive_fixed_lag_native_rtk_updates_per_minute_min"
                        ]
                    ),
                    "s3e_alpha_adaptive_fixed_lag_updates_max": (
                        s3e_alpha_reproducibility["metrics"][
                            "adaptive_fixed_lag_native_rtk_updates_per_minute_max"
                        ]
                    ),
                    "s3e_alpha_adaptive_fixed_lag_reduction_min": (
                        s3e_alpha_reproducibility["metrics"][
                            "adaptive_fixed_lag_native_rtk_update_reduction_percent_min"
                        ]
                    ),
                    "s3e_alpha_adaptive_fixed_lag_reduction_max": (
                        s3e_alpha_reproducibility["metrics"][
                            "adaptive_fixed_lag_native_rtk_update_reduction_percent_max"
                        ]
                    ),
                    "s3e_alpha_adaptive_fixed_lag_latency_mean_s": (
                        s3e_alpha_reproducibility["metrics"][
                            "adaptive_fixed_lag_native_rtk_latency_mean_seconds_max"
                        ]
                    ),
                    "s3e_alpha_adaptive_fixed_lag_latency_p95_s": (
                        s3e_alpha_reproducibility["metrics"][
                            "adaptive_fixed_lag_native_rtk_latency_p95_seconds_max"
                        ]
                    ),
                    "s3e_alpha_adaptive_fixed_lag_latency_max_s": (
                        s3e_alpha_reproducibility["metrics"][
                            "adaptive_fixed_lag_native_rtk_latency_max_seconds_max"
                        ]
                    ),
                    "s3e_adaptive_fixed_lag_by_agent": {
                        "Alpha": s3e_alpha_reproducibility["metrics"][
                            "adaptive_fixed_lag_native_rtk_sim3_ate_m_max"
                        ],
                        "Bob": s3e_vio["bob"]["metrics"][
                            "adaptive_fixed_lag_native_rtk_sim3_ate_m"
                        ],
                        "Carol": s3e_carol_geometry_corrected["metrics"][
                            "adaptive_fixed_lag_native_rtk_sim3_ate_m"
                        ],
                    },
                    "s3e_adaptive_fixed_lag_updates_by_agent": {
                        "Alpha": s3e_alpha_reproducibility["metrics"][
                            "adaptive_fixed_lag_native_rtk_updates_per_minute_max"
                        ],
                        "Bob": s3e_vio["bob"]["metrics"][
                            "adaptive_fixed_lag_native_rtk_finalization_updates_per_minute"
                        ],
                        "Carol": s3e_carol_geometry_corrected["metrics"][
                            "adaptive_fixed_lag_native_rtk_finalization_updates_per_minute"
                        ],
                    },
                    "s3e_fixed_lag_by_agent": {
                        "Alpha": s3e_alpha_reproducibility["metrics"][
                            "fixed_lag_native_rtk_sim3_ate_m_max"
                        ],
                        "Bob": s3e_vio["bob"]["metrics"][
                            "fixed_lag_native_rtk_sim3_ate_m"
                        ],
                        "Carol": s3e_carol_geometry_corrected["metrics"][
                            "fixed_lag_native_rtk_sim3_ate_m"
                        ],
                    },
                    "s3e_native_rtk_by_agent": {
                        "Alpha": {
                            "ate_m": s3e_alpha_reproducibility["metrics"][
                                "causal_native_rtk_sim3_ate_m_max"
                            ],
                            "anchor_rate": s3e_alpha_reproducibility["metrics"][
                                "causal_native_rtk_anchor_messages_per_minute_max"
                            ],
                            "correction_rate": s3e_alpha_reproducibility["metrics"][
                                "causal_native_rtk_correction_messages_per_minute_max"
                            ],
                            "target_met": bool(
                                s3e_alpha_reproducibility["metrics"][
                                    "causal_native_rtk_target_pass_count"
                                ]
                                == s3e_alpha_reproducibility["metrics"][
                                    "replicate_count"
                                ]
                            ),
                        },
                        "Bob": {
                            "ate_m": s3e_vio["bob"]["metrics"][
                                "causal_native_rtk_sim3_ate_m"
                            ],
                            "anchor_rate": s3e_vio["bob"]["metrics"][
                                "causal_native_rtk_anchor_messages_per_minute"
                            ],
                            "correction_rate": s3e_vio["bob"]["metrics"][
                                "causal_native_rtk_sim3_correction_messages_per_minute"
                            ],
                            "target_met": bool(
                                s3e_vio["bob"]["metrics"][
                                    "causal_native_rtk_sim3_target_met"
                                ]
                            ),
                        },
                        "Carol": {
                            "ate_m": s3e_carol_geometry_corrected["metrics"][
                                "causal_native_rtk_sim3_ate_m"
                            ],
                            "anchor_rate": s3e_carol_geometry_corrected["metrics"][
                                "causal_native_rtk_anchor_messages_per_minute"
                            ],
                            "correction_rate": s3e_carol_geometry_corrected["metrics"][
                                "causal_native_rtk_sim3_correction_messages_per_minute"
                            ],
                            "target_met": bool(
                                s3e_carol_geometry_corrected["metrics"][
                                    "causal_native_rtk_sim3_target_met"
                                ]
                            ),
                        },
                    },
                    "s3e_alpha_replicate_load_min": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_sim3_low_ingress_correction_messages_per_minute_min"
                        ]
                    ),
                    "s3e_alpha_replicate_load_median": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_sim3_low_ingress_correction_messages_per_minute_median"
                        ]
                    ),
                    "s3e_alpha_replicate_load_max": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_sim3_low_ingress_correction_messages_per_minute_max"
                        ]
                    ),
                    "s3e_alpha_replicate_peak_per_second": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_sim3_low_ingress_correction_burst_per_second_max"
                        ]
                    ),
                    "s3e_alpha_replicate_min_interval_s": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_sim3_low_ingress_correction_interval_min_seconds_min"
                        ]
                    ),
                    "s3e_alpha_replicate_causal_ate_max": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_sim3_low_ingress_ate_m_max"
                        ]
                    ),
                    "s3e_alpha_replicate_anchor_load_max": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_sim3_low_ingress_anchor_messages_per_minute_max"
                        ]
                    ),
                    "s3e_alpha_replicate_threshold_min": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_sim3_low_ingress_correction_threshold_m"
                        ]
                    ),
                    "s3e_alpha_replicate_threshold_max": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_sim3_low_ingress_correction_threshold_m"
                        ]
                    ),
                    "s3e_alpha_replicate_hold_p95_max_s": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_sim3_low_ingress_correction_interval_p95_seconds_max"
                        ]
                    ),
                    "s3e_alpha_replicate_hold_max_s": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_sim3_low_ingress_correction_interval_max_seconds_max"
                        ]
                    ),
                    "s3e_alpha_radio_min_ate_max": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_sim3_reference_ate_m_max"
                        ]
                    ),
                    "s3e_alpha_radio_min_correction_load_max": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_sim3_reference_correction_messages_per_minute_max"
                        ]
                    ),
                    "s3e_alpha_radio_min_anchor_load_max": (
                        s3e_alpha_reproducibility["metrics"][
                            "causal_sim3_reference_anchor_messages_per_minute_max"
                        ]
                    ),
                    "s3e_alpha_reproducible": s3e_alpha_reproducibility["metrics"][
                        "trajectory_reproducible"
                    ],
                    "s3e_openvins_ate_m": s3e_openvins["metrics"]["ate_rmse_m"],
                    "s3e_openvins_sim3_ate_m": s3e_openvins["metrics"][
                        "sim3_ate_rmse_m"
                    ],
                    "s3e_openvins_scale_correction": s3e_openvins["metrics"][
                        "metric_scale_correction_to_truth"
                    ],
                    "s3e_openvins_event_messages_per_minute": s3e_openvins["metrics"][
                        "event_triggered_messages_per_minute_for_0_1m"
                    ],
                    "s3e_openvins_event_peak_per_second": s3e_openvins["metrics"][
                        "event_triggered_peak_corrections_per_second"
                    ],
                    "s3e_openvins_tracking_healthy": s3e_openvins["metrics"][
                        "tracking_healthy"
                    ],
                    "s3e_alpha_stereo_ate_m": s3e_alpha_modes["stereo"]["metrics"][
                        "ate_rmse_m"
                    ],
                    "s3e_alpha_calibrated_ate_m": s3e_alpha_modes["calibrated_500"][
                        "metrics"
                    ]["ate_rmse_m"],
                    "s3e_alpha_calibrated_scale_correction": s3e_alpha_modes[
                        "calibrated_500"
                    ]["metrics"]["metric_scale_correction_to_truth"],
                    "s3e_alpha_long_ate_m": s3e_alpha_modes["calibrated_1000"][
                        "metrics"
                    ]["ate_rmse_m"],
                    "s3e_alpha_long_map_resets": s3e_alpha_modes["calibrated_1000"][
                        "metrics"
                    ]["map_reset_count"],
                    "s3e_alpha_long_lost_frames": s3e_alpha_modes["calibrated_1000"][
                        "metrics"
                    ]["lost_frame_count"],
                    "s3e_alpha_event_messages_per_minute": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["event_triggered_messages_per_minute_for_0_1m"],
                    "s3e_alpha_event_rate_reduction_percent": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["event_triggered_rate_reduction_vs_periodic_percent"],
                    "s3e_alpha_event_min_interval_s": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["event_triggered_min_interval_seconds"],
                    "s3e_alpha_event_peak_per_second": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["event_triggered_peak_corrections_per_second"],
                    "s3e_alpha_orientation_reference_available": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["orientation_reference_available"],
                    "s3e_alpha_orientation_proxy_rmse_rad": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["orientation_proxy_rmse_rad"],
                    "s3e_alpha_orientation_proxy_p95_rad": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["orientation_proxy_p95_rad"],
                    "s3e_alpha_orientation_proxy_rpe_rmse_rad": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["orientation_proxy_rpe_rmse_rad"],
                    "s3e_alpha_orientation_proxy_independent": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["orientation_proxy_independent_of_vio"],
                    "s3e_alpha_stereo_orientation_proxy_rmse_rad": s3e_alpha_modes[
                        "stereo"
                    ]["metrics"]["orientation_proxy_rmse_rad"],
                    "s3e_alpha_stereo_orientation_proxy_independent": s3e_alpha_modes[
                        "stereo"
                    ]["metrics"]["orientation_proxy_independent_of_vio"],
                    "s3e_alpha_first_quarter_ate_m": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["ate_first_quarter_m"],
                    "s3e_alpha_middle_half_ate_m": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["ate_middle_half_m"],
                    "s3e_alpha_last_quarter_ate_m": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["ate_last_quarter_m"],
                    "s3e_alpha_peak_error_fraction": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["peak_error_trajectory_fraction"],
                    "s3e_alpha_timing_best_offset_ms": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["timing_best_ate_offset_ms"],
                    "s3e_alpha_timing_gain_percent": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["timing_ate_improvement_percent"],
                    "s3e_alpha_timing_dominant": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["timing_offset_is_dominant"],
                    "s3e_alpha_high_recall_ate_m": s3e_alpha_modes["high_recall_1000"][
                        "metrics"
                    ]["ate_rmse_m"],
                    "s3e_alpha_late_default_ate_m": s3e_alpha_modes[
                        "late_default_1000"
                    ]["metrics"]["ate_rmse_m"],
                    "s3e_alpha_late_scaled_ate_m": s3e_alpha_modes["late_scaled_1000"][
                        "metrics"
                    ]["ate_rmse_m"],
                    "s3e_alpha_late_scaled_residual_scale": s3e_alpha_modes[
                        "late_scaled_1000"
                    ]["metrics"]["metric_scale_correction_to_truth"],
                    "s3e_alpha_fast_init_ate_m": s3e_alpha_modes["fast_init_1000"][
                        "metrics"
                    ]["ate_rmse_m"],
                    "s3e_alpha_fast_init_p95_m": s3e_alpha_modes["fast_init_1000"][
                        "metrics"
                    ]["error_p95_m"],
                    "s3e_alpha_fast_init_map_resets": s3e_alpha_modes["fast_init_1000"][
                        "metrics"
                    ]["map_reset_count"],
                    "s3e_alpha_fast_init_lost_frames": s3e_alpha_modes[
                        "fast_init_1000"
                    ]["metrics"]["lost_frame_count"],
                    "s3e_alpha_best_ate_m": s3e_alpha_modes["fast_init_scaled_1000"][
                        "metrics"
                    ]["ate_rmse_m"],
                    "s3e_alpha_best_sim3_ate_m": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["sim3_ate_rmse_m"],
                    "s3e_alpha_best_scale_correction": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["metric_scale_correction_to_truth"],
                    "s3e_alpha_best_p95_m": s3e_alpha_modes["fast_init_scaled_1000"][
                        "metrics"
                    ]["error_p95_m"],
                    "s3e_alpha_best_final_drift_m": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["final_drift_m"],
                    "s3e_alpha_lever_arm_maximum_norm_m": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["lever_arm_sensitivity_maximum_norm_m"],
                    "s3e_alpha_lever_arm_fitted_norm_m": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["lever_arm_sensitivity_fitted_norm_m"],
                    "s3e_alpha_lever_arm_unconstrained_norm_m": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["lever_arm_sensitivity_unconstrained_norm_m"],
                    "s3e_alpha_lever_arm_bound_active": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["lever_arm_sensitivity_bound_active"],
                    "s3e_alpha_lever_arm_adjusted_ate_m": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["lever_arm_sensitivity_adjusted_ate_m"],
                    "s3e_alpha_lever_arm_full_improvement_percent": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["lever_arm_sensitivity_full_fit_improvement_percent"],
                    "s3e_alpha_lever_arm_holdout_improvement_percent": (
                        s3e_alpha_modes["fast_init_scaled_1000"]["metrics"][
                            "lever_arm_sensitivity_holdout_improvement_percent"
                        ]
                    ),
                    "s3e_alpha_local_rigid_1s_ate_m": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["local_alignment_1s_rigid_ate_m"],
                    "s3e_alpha_local_sim3_2s_ate_m": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["local_alignment_2s_sim3_ate_m"],
                    "s3e_alpha_local_rigid_interval_s": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["local_rigid_maximum_passing_interval_seconds"],
                    "s3e_alpha_local_rigid_messages_per_minute": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["local_rigid_optimistic_anchor_messages_per_minute"],
                    "s3e_alpha_local_sim3_interval_s": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["local_sim3_maximum_passing_interval_seconds"],
                    "s3e_alpha_local_sim3_messages_per_minute": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["local_sim3_optimistic_anchor_messages_per_minute"],
                    "s3e_alpha_local_5s_scale_p05": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["local_alignment_5s_scale_p05"],
                    "s3e_alpha_local_5s_scale_p95": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["local_alignment_5s_scale_p95"],
                    "s3e_alpha_stereo_local_sim3_0_5s_ate_m": s3e_alpha_modes["stereo"][
                        "metrics"
                    ]["local_alignment_0_5s_sim3_ate_m"],
                    "s3e_bob_local_sim3_0_5s_ate_m": s3e_vio["bob"]["metrics"][
                        "local_alignment_0_5s_sim3_ate_m"
                    ],
                    "s3e_carol_local_sim3_0_5s_ate_m": s3e_vio["carol"]["metrics"][
                        "local_alignment_0_5s_sim3_ate_m"
                    ],
                    "s3e_alpha_causal_se3_cadence_s": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["causal_se3_maximum_passing_cadence_seconds"],
                    "s3e_alpha_causal_se3_ate_m": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["causal_se3_ate_at_selected_cadence_m"],
                    "s3e_alpha_causal_se3_messages_per_minute": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["causal_se3_anchor_messages_per_minute"],
                    "s3e_alpha_causal_se3_jump_p95_m": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["causal_se3_correction_jump_p95_m"],
                    "s3e_alpha_causal_sim3_cadence_s": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["causal_sim3_maximum_passing_cadence_seconds"],
                    "s3e_alpha_causal_sim3_ate_m": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["causal_sim3_ate_at_selected_cadence_m"],
                    "s3e_alpha_causal_sim3_messages_per_minute": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["causal_sim3_anchor_messages_per_minute"],
                    "s3e_alpha_causal_sim3_jump_p95_m": s3e_alpha_modes[
                        "fast_init_scaled_1000"
                    ]["metrics"]["causal_sim3_correction_jump_p95_m"],
                    "s3e_alpha_causal_sim3_scale_update_p95_percent": (
                        100.0
                        * (
                            math.exp(
                                s3e_alpha_modes["fast_init_scaled_1000"]["metrics"][
                                    "causal_sim3_scale_update_p95_log"
                                ]
                            )
                            - 1.0
                        )
                    ),
                    "s3e_local_alignment_profiles": local_alignment_profiles,
                    "s3e_capacity_configured_evaluation_period_s": (
                        scheduling_config.evaluation_period_seconds
                    ),
                    "s3e_capacity_recommended_evaluation_period_s": (
                        correction_capacity.recommended_maximum_evaluation_period_s
                    ),
                    "s3e_capacity_feasible": correction_capacity.feasible,
                    "s3e_capacity_action": correction_capacity.action,
                    "s3e_capacity_configured_messages_per_minute": (
                        correction_capacity.configured_messages_per_minute_capacity
                    ),
                    "s3e_capacity_required_messages_per_minute": (
                        correction_capacity.total_messages_per_minute
                    ),
                    "s3e_capacity_suppressed_messages_per_minute": (
                        correction_capacity.suppressed_messages_per_minute
                    ),
                    "s3e_capacity_configured_peak_per_second": (
                        correction_capacity.configured_peak_corrections_per_second
                    ),
                    "s3e_capacity_required_peak_per_second": (
                        correction_capacity.peak_corrections_per_second
                    ),
                    "s3e_capacity_suppressed_peak_per_second": (
                        correction_capacity.suppressed_peak_corrections_per_second
                    ),
                    "s3e_capacity_schedulable_agents": (
                        correction_capacity.schedulable_agent_ids
                    ),
                    "s3e_capacity_relocalization_agents": (
                        correction_capacity.relocalization_agent_ids
                    ),
                    "s3e_capacity_tracking_failure_agents": (
                        correction_capacity.tracking_failure_agent_ids
                    ),
                    "s3e_capacity_live_pose_failure_agents": (
                        correction_capacity.live_pose_failure_agent_ids
                    ),
                    "s3e_correction_profiles": correction_profile_details,
                    "correction_max_rotation_step_rad": (
                        correction_config.max_rotation_step_rad
                    ),
                    "correction_max_total_rotation_rad": (
                        correction_config.max_total_rotation_rad
                    ),
                },
            },
            {
                "group": "Global state",
                "name": "Correction delta application",
                "state": "validated",
                "summary": "Restart-safe correction sequencing, independent translation/rotation safety bounds, observable per-Wingman load envelopes, capacity assessment, and replay suppression distinguish schedulable corrections from mandatory relocalization.",
                "implementation": "ORB-SLAM3 VIO + CorrectionCadenceScheduler + persistent correction state",
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
                    "state_restored": global_scene["metrics"][
                        "correction_state_restored"
                    ],
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
                    "scene_revision": operations["details"]["handoff"][
                        "scene_revision"
                    ],
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
                        for name, passed in operations["details"]["handoff"][
                            "gates"
                        ].items()
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
                    "feedback": [
                        "execution state",
                        "map changes",
                        "node failure",
                        "replan",
                    ],
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
                    "reports": len(datasets) + 9,
                    "status": phase1["status"],
                },
            },
        ],
        "gaps": [
            "Static-asynchronous S3E fusion produced one 184,320-Gaussian PLY from independently timed Alpha, Bob, and Carol ReSplat windows. Offline truth-fitted registration is 1.28/13.31/9.12 m RMSE and Bob/Carol scales are implausible; directional SH was not rebased after rotation, so metric and appearance claims remain closed.",
            "MILUV real full-SE(3) truth passes at 0.0195 m ATE and 0.0069 rad orientation RMSE, but local odometry and cross-agent relative-pose observations remain controlled; production VIO and visual association are not validated by this result.",
            "MILUV delayed position-only UWB corrections stop at 0.1430 m ATE even at 58.2 messages/min, so independently solved scalar-range corrections do not meet the 0.1 m target.",
            "MILUV causal fixed-lag UWB rationalization reaches 0.0930 m fleet ATE and all three Wingmen pass on seed 7 at 17.2-23.1 messages/min; its factor inventory has no production local pose or attitude stream, so the deployment claim gate remains closed.",
            "MILUV full-batch UWB reaches a 0.0783 m non-causal upper bound, while fixed-lag orientation remains controlled at 0.1069 rad; replace controlled odometry/orientation with production VIO and sparse marginalization before claiming full pose.",
            "On the controlled MILUV trace, adaptive scheduling improves ATE to 0.0109 m but increases corrections from 15 to 18, so the 16.08-second fixed cadence remains the lower-load selection.",
            "S3E production ORB-SLAM3 still misses 0.1 m locally. Exact native 1 Hz RTK positions stop at 0.200-0.244 m online causal Sim(3) across three Alpha runs, 1.089 m for Bob, and 1.798 m for automatic-geometry Carol; all three live Wingman poses remain relocalization cases.",
            "Resetting Alpha to each exact RTK position and holding a ten-segment exponentially weighted past-only transform improves live-position ATE by 23.9-28.3% to 0.152-0.175 m, but yields zero target passes at 59.45 updates/min. Bob and Carol worsen to 1.364 m and 2.309 m with implausible scales, so this is an observed live limit rather than a correction candidate.",
            "Alpha past-only live error is 0.007 m through 0.1 s and 0.072 m through 0.2 s, then misses at 0.123 m through 0.5 s and 0.175 m through 1.0 s. The fixed cross-replicate target horizon is therefore 0.2 s, implying at least 300 global observations/min versus the native 60.06/min; Bob and Carol have no passing tested horizon.",
            "Intelligence fixed-lag rationalization uses consecutive native RTK endpoints and relative VIO motion to finalize Alpha history at 0.077-0.085 m Sim(3), with 0.539 s mean and 0.997 s maximum delay. Bob and Carol still miss at 0.420 m and 0.447 m, and the result cannot be used as a live correction claim.",
            "Adaptive fixed-lag coalescing preserves all three Alpha finalized-map passes at 0.091-0.097 m while reducing Intelligence finalizations from 59.45 to 37.0-38.8/min (34.7-37.8%). Its p95 delay is 1.890 s and maximum delay is 1.998 s; native RTK ingress remains about 60/min and live pose remains closed.",
            "Alpha fixed-lag segment scales remain inside the 0.5-2.0x plausibility gate but vary from 0.704x p05 to 1.311x p95, exposing time-varying VIO deformation. Bob and Carol have only 6.1% and 9.7% plausible segments, confirming tracking failure rather than a schedulable correction problem.",
            "All three S3E Wingmen pass the independent timestamp/IMU contract. Automatic stereo geometry removes Carol's lost frames, but three identical replicates span 29.00-69.21 m ATE, 1-4 resets, and 464-491 corrections/min; reproducibility and global-pose claim gates fail.",
            "Applying Carol's fitted 0.42x baseline lowers affine-corrected ATE to 4.97 m but reintroduces 294 lost frames and leaves a 4.88 m Sim(3) floor; static depth scaling is not a transferable calibration.",
            "Matched 500-frame Alpha OpenVINS initializes only after raising the static excitation threshold to 1.0, then diverges to 563.2 m rigid ATE, 8.72 m Sim(3) ATE, and a 0.0225x fitted scale; its 539.1 corrections/min and 10/s peak are unschedulable, so it is rejected rather than promoted as an S3E fallback.",
            "The per-Wingman S3E gate tolerates 0.025 m correction translation noise and 0.005 rad rotation noise at the selected load point; fleet averages can remain below 0.1 m after Carol has failed, so correction limits must be enforced per node.",
            "Controlled dense S3E cross-Wingman factors reduce relative translation RMSE from 14.261 m to 0.133 m at 12.84 factors/min, but absolute global ATE remains 6.028 m. Raising controlled association noise from 0.05 m to 0.20 m degrades relative RMSE from 0.149 m to 0.268 m without repairing the global gauge; a measured global landmark or external observation is still required.",
            "S3E Playground 2 publishes RTK position but identity quaternion placeholders and no RTK antenna lever arm, so real orientation accuracy and full SE(3) correction load are not observable from this reference.",
            "S3E IMU/AHRS provides an orientation-consistency proxy: Alpha stereo-inertial is 0.032 rad RMSE with 0.0022 rad rotational RPE, but it shares the estimator IMU; stereo-only is an independent 0.299 rad check. Neither is orientation ground truth.",
            "A fitted RTK lever-arm sensitivity cannot explain Alpha's long-window error: the fit saturates the conservative 1 m bound, leaves 1.27 m ATE, and worsens held-out ATE by 1.1%; the unconstrained fit requests 6.22 m.",
            "Non-causal local fits put Alpha below target at 1 s with SE(3) (0.073 m, 60 anchors/min) and 2 s with Sim(3) (0.051 m, 30 anchors/min); this motivates a causal fixed-lag Intelligence-node rationalizer but does not replace the measured 0.1 s reaction requirement.",
            "Alpha threshold-held transmission separates fit cadence from radio cadence: the balanced sensitivity reaches 0.0906-0.0937 m across three runs at 75.2-78.9 corrections/min, but it consumes 294.8 RTK-interpolated scoring anchors/min rather than native observations.",
            "Native Alpha RTK provides 60.06 anchors/min and would transmit 58.85 corrections/min with a one/s peak, but its 0.200-0.244 m online Sim(3) ATE fails accuracy. Capacity therefore routes Alpha, Bob, and Carol to relocalization while the fixed-lag Alpha pass is reserved for delayed map finalization.",
            "The live-capacity gate keeps tracking health separate from correction eligibility: Alpha is healthy-but-inaccurate and non-reproducible, while Bob and Carol fail tracking. Fail-closed routing suppresses 170.03 candidate corrections/min and a combined three/s peak without treating avoided traffic as recovered accuracy.",
            "Pinning Alpha ORB-SLAM3 and its numeric libraries to one CPU does not close reproducibility: three runs span 1.005-1.559 m ATE and 0.980-1.502 m Sim(3), widening the Sim(3) spread to 0.521 m. This motivated the explicit offline local-mapping ablation.",
            "Blocking every Alpha frame until local mapping is idle narrows the three-run ATE spread from 0.297 m to 0.242 m and the Sim(3) spread from 0.347 m to 0.175 m, but median ATE regresses from 1.341 m to 1.608 m and the reproducibility gate still fails. Local-mapper overlap is a contributor, not the root cause; next isolate remaining map-state and loop-closing divergence.",
            "A 2,400-feature high-recall ORB profile improves local RPE but regresses Alpha ATE from 1.34 m to 2.03 m; the balanced 1,600-feature profile remains selected.",
            "On non-overlapping Alpha frames 1000-1999, the untouched and 1.20x baselines reach 8.07 m and 4.96 m ATE; scaling helps but leaves a 1.221x residual, so one static multiplier does not generalize.",
            "Resolve S3E metric calibration after the sensor and Carol stereo-observability boundaries before adding another VIO backend; neither matched OpenVINS nor the current ORB-SLAM3 path supports the global-pose claim.",
            "Replace reference geometric and semantic features with ALIKED/SuperPoint and DINO-family adapters.",
            "Calibrate SE(3) covariance and robust gates on real multi-agent loop closures.",
            "Collect production Wingman telemetry to calibrate correction uncertainty, queue pressure, and relocalization recovery; artifact-derived candidate rates remain diagnostic rather than a deployable schedule.",
            "Validate real multi-agent association, correction exchange, and persistent mapping end to end.",
            "Add recovery, p50/p95 latency, peak memory, and power measurements to production gates.",
        ],
        "sources": [
            "outputs/ariadne/phase1/benchmark.json",
            "outputs/ariadne/exchange/benchmark.json",
            "outputs/ariadne/global-scene/benchmark.json",
            "outputs/ariadne/s3e-global-gaussian-static/preparation.json",
            "outputs/ariadne/s3e-global-gaussian-static/manifest.json",
            "outputs/ariadne/s3e-global-gaussian-static/unified_s3e_global_gaussians.ply",
            "outputs/ariadne/miluv-global-pose/benchmark.json",
            "outputs/ariadne/s3e-global-pose/benchmark.json",
            "outputs/ariadne/operations/benchmark.json",
            "outputs/ariadne/end-to-end/benchmark.json",
            "outputs/ariadne/dataset_sequence/summary.json",
            "outputs/ariadne/real_vio/d2slam-1/openvins/evaluation.json",
            "outputs/ariadne/real_vio/d2slam-1/orbslam3/evaluation.json",
            "outputs/ariadne/real_vio/s3e-alpha/orbslam3/evaluation.json",
            "outputs/ariadne/real_vio/s3e-alpha/openvins/evaluation.json",
            "outputs/ariadne/real_vio/s3e-bob/orbslam3/evaluation.json",
            "outputs/ariadne/real_vio/s3e-carol/orbslam3/evaluation.json",
            "outputs/ariadne/real_vio/s3e-carol/orbslam3-auto-geometry-reproducibility.json",
            "outputs/ariadne/real_vio/s3e-alpha/orbslam3-bf-1.2-fast-init-reproducibility.json",
            "outputs/ariadne/real_vio/s3e-alpha/orbslam3-bf-1.2-fast-init-deterministic-reproducibility.json",
            "outputs/ariadne/real_vio/s3e-alpha/orbslam3-bf-1.2-fast-init-mapping-sync-reproducibility.json",
            "outputs/ariadne/real_vio/s3e-alpha/orbslam3-stereo/evaluation.json",
            "outputs/ariadne/real_vio/s3e-alpha/orbslam3-bf-1.15/evaluation.json",
            "outputs/ariadne/real_vio/s3e-alpha/orbslam3-bf-1.15-1000/evaluation.json",
            "outputs/ariadne/real_vio/s3e-alpha/orbslam3-bf-1.15-fast-init/evaluation.json",
            "outputs/ariadne/real_vio/s3e-alpha/orbslam3-bf-1.2-fast-init/evaluation.json",
            "outputs/ariadne/real_vio/s3e-alpha/orbslam3-bf-1.2-fast-init-high-recall/evaluation.json",
            "outputs/ariadne/real_vio/s3e-alpha/orbslam3-fast-init-start-1000/evaluation.json",
            "outputs/ariadne/real_vio/s3e-alpha/orbslam3-bf-1.2-fast-init-start-1000/evaluation.json",
            "outputs/ariadne/resplat_report/neighbourhood_105_10f/metrics.json",
            str(resplat_render.relative_to(ROOT)),
            RESPLAT_RUN_URL,
            "applications/ariadne/docs/phase1_models.md",
            "applications/ariadne/docs/real_vio.md",
            "applications/ariadne/docs/static_asynchronous_global_gaussians.md",
            "applications/ariadne/docs/vio_global_pose_experiment_log.md",
            "applications/ariadne/configs/intelligence/default.yaml",
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
    .progression-summary { margin: 0 0 22px; padding: 16px 18px; border-left: 3px solid var(--blue); background: var(--blue-soft); color: #385268; font-size: 13px; }
    .progression-summary strong { color: var(--ink); }
    .progression-card { overflow: hidden; margin-top: 16px; border: 1px solid var(--line); border-radius: 7px; background: var(--surface); }
    .progression-card header { padding: 19px 22px 15px; border-bottom: 1px solid var(--line); }
    .progression-card h3 { margin: 0; font-size: 18px; }
    .progression-card header p { margin: 6px 0 0; color: var(--muted); font-size: 12px; }
    .progression-chart { overflow-x: auto; padding: 12px 16px 6px; }
    .progression-chart svg { display: block; width: 100%; min-width: 920px; height: auto; }
    .progression-reading { margin: 0; padding: 14px 22px 18px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; }
    .progression-reading strong { color: var(--ink); }
    .progression-table-wrap { overflow-x: auto; }
    .progression-table { width: 100%; min-width: 960px; border-collapse: collapse; font-size: 11px; }
    .progression-table th, .progression-table td { padding: 11px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    .progression-table th { background: #f0f3f1; color: #4e5a54; font-size: 10px; letter-spacing: .04em; text-transform: uppercase; }
    .progression-table tbody tr:last-child td { border-bottom: 0; }
    .progression-table td:nth-child(3), .progression-table td:nth-child(4), .progression-table td:nth-child(5) { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space: nowrap; }
    .progression-table .metric-improved { color: var(--green); font-weight: 760; }
    .progression-table .metric-regressed { color: var(--amber); font-weight: 760; }
    .progression-caveat { margin: 18px 0 0; color: var(--muted); font-size: 12px; }
    .fusion-shell { overflow: hidden; border: 1px solid var(--line); border-radius: 7px; background: var(--surface); }
    .fusion-status { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border-bottom: 1px solid var(--line); }
    .fusion-metric { min-height: 112px; padding: 19px; border-right: 1px solid var(--line); }
    .fusion-metric:last-child { border-right: 0; }
    .fusion-metric strong { display: block; font-size: 23px; }
    .fusion-metric span { display: block; margin-top: 5px; color: var(--muted); font-size: 11px; }
    .fusion-detail { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(280px, .85fr); gap: 28px; padding: 23px; }
    .fusion-detail h3 { margin: 0 0 8px; font-size: 17px; }
    .fusion-detail p { margin: 0 0 12px; color: var(--muted); font-size: 13px; }
    .fusion-detail code { display: block; margin-top: 7px; color: #405048; font-size: 10px; overflow-wrap: anywhere; }
    .fusion-warning { padding: 14px 16px; border-left: 3px solid var(--amber); background: var(--amber-soft); color: #664318; font-size: 12px; }
    .fusion-links { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 15px; }
    .fusion-links a { padding: 8px 11px; border: 1px solid #bfc8c3; border-radius: 4px; color: var(--blue); font-size: 11px; font-weight: 720; text-decoration: none; }
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
    .docs-shell { display: grid; grid-template-columns: minmax(260px, .68fr) minmax(0, 1.7fr); gap: 24px; align-items: start; }
    .docs-sidebar { position: sticky; top: 78px; overflow: hidden; border: 1px solid var(--line); border-radius: 7px; background: var(--surface); }
    .docs-search-wrap { display: block; padding: 16px; border-bottom: 1px solid var(--line); }
    .docs-search-wrap span { display: block; margin-bottom: 7px; color: var(--muted); font-size: 11px; font-weight: 760; letter-spacing: .04em; text-transform: uppercase; }
    .docs-search { width: 100%; min-height: 40px; padding: 8px 11px; border: 1px solid #bfc8c3; border-radius: 4px; background: #fff; color: var(--ink); font: inherit; font-size: 13px; }
    .docs-search:focus-visible { outline: 3px solid rgba(47,98,139,.2); border-color: var(--blue); }
    .docs-count { margin: 0; padding: 10px 16px; border-bottom: 1px solid var(--line); color: var(--muted); font-size: 11px; }
    .docs-nav { max-height: min(66vh, 720px); overflow-y: auto; padding: 8px; }
    .docs-group + .docs-group { margin-top: 12px; }
    .docs-group h3 { margin: 0; padding: 5px 8px; color: var(--muted); font-size: 10px; letter-spacing: .07em; text-transform: uppercase; }
    .doc-link { display: block; width: 100%; padding: 9px 10px; border: 0; border-radius: 4px; background: transparent; color: var(--ink); text-align: left; cursor: pointer; }
    .doc-link:hover, .doc-link:focus-visible { background: #eef2ef; outline: none; }
    .doc-link.active { background: var(--green-soft); color: #15583c; }
    .doc-link strong { display: block; font-size: 12px; line-height: 1.3; }
    .doc-link span { display: block; margin-top: 3px; color: var(--muted); font: 9px/1.35 ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap: anywhere; }
    .docs-empty { padding: 20px 10px; color: var(--muted); font-size: 12px; text-align: center; }
    .docs-reader { min-width: 0; overflow: hidden; border: 1px solid var(--line); border-radius: 7px; background: var(--surface); }
    .docs-reader-head { padding: 21px 24px; border-bottom: 1px solid var(--line); background: #f8faf8; }
    .docs-reader-head h3 { margin: 0; font-size: 22px; line-height: 1.25; }
    .docs-reader-meta { display: flex; flex-wrap: wrap; gap: 6px 16px; margin-top: 8px; color: var(--muted); font: 10px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; }
    .markdown-body { padding: 28px 30px 42px; color: #28312c; font-size: 14px; overflow-wrap: anywhere; }
    .markdown-body > :first-child { margin-top: 0; }
    .markdown-body > :last-child { margin-bottom: 0; }
    .markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4, .markdown-body h5, .markdown-body h6 { margin: 1.6em 0 .55em; color: var(--ink); line-height: 1.25; }
    .markdown-body h1 { padding-bottom: .35em; border-bottom: 1px solid var(--line); font-size: 28px; }
    .markdown-body h2 { padding-bottom: .3em; border-bottom: 1px solid var(--line); font-size: 22px; }
    .markdown-body h3 { font-size: 18px; }
    .markdown-body h4 { font-size: 15px; }
    .markdown-body p { margin: .8em 0; }
    .markdown-body a { color: var(--blue); }
    .markdown-body ul, .markdown-body ol { padding-left: 1.7em; }
    .markdown-body li + li { margin-top: .28em; }
    .markdown-body blockquote { margin: 1em 0; padding: 3px 16px; border-left: 3px solid var(--blue); background: var(--blue-soft); color: #385268; }
    .markdown-body code { padding: 2px 5px; border-radius: 3px; background: #edf0ee; font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
    .markdown-body pre { overflow-x: auto; margin: 1em 0; padding: 16px; border-radius: 5px; background: #151a17; color: #e2e9e5; }
    .markdown-body pre code { padding: 0; background: transparent; color: inherit; }
    .markdown-body table { display: block; width: max-content; max-width: 100%; overflow-x: auto; border-collapse: collapse; }
    .markdown-body th, .markdown-body td { min-width: 110px; padding: 8px 10px; border: 1px solid var(--line); text-align: left; vertical-align: top; }
    .markdown-body th { background: #f0f3f1; }
    .markdown-body hr { margin: 2em 0; border: 0; border-top: 1px solid var(--line); }
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
      .section-head { grid-template-columns: 1fr; gap: 12px; }
      .nav-links { gap: 12px; }
      .nav-links a { font-size: 11px; }
      .dataset-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .component { grid-template-columns: minmax(220px, .8fr) minmax(350px, 1.2fr); }
      .pipeline { grid-template-columns: repeat(4, minmax(0, 1fr)); }
      .fusion-detail { grid-template-columns: 1fr; }
      .docs-shell { grid-template-columns: minmax(230px, .62fr) minmax(0, 1.38fr); }
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
      .fusion-status { grid-template-columns: 1fr 1fr; }
      .fusion-metric:nth-child(2) { border-right: 0; }
      .fusion-metric:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
      .component { grid-template-columns: 1fr; }
      .component-copy { min-height: 220px; border-right: 0; border-bottom: 1px solid var(--line); }
      .component-visual { padding: 14px; }
      .dataset-grid, .sources { grid-template-columns: 1fr; }
      .dataset-card { min-height: auto; }
      .docs-shell { grid-template-columns: 1fr; }
      .docs-sidebar { position: static; }
      .docs-nav { max-height: 330px; }
      .markdown-body { padding: 22px 18px 32px; }
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
      <a href="#progression">Progress</a>
      <a href="#global-gaussians">Global splat</a>
      <a href="#components">Components</a>
      <a href="#datasets">Datasets</a>
      <a href="#documentation">Documentation</a>
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

    <section class="band alt" id="progression" aria-labelledby="progression-title">
      <div class="inner">
        <div class="section-head">
          <h2 id="progression-title">ATE and metric progression</h2>
          <p>Ordered evidence from the original S3E Alpha ORB-SLAM3 baseline to the current three-run controlled-runtime evaluation. Lower ATE and RPE are better.</p>
        </div>
        <p class="progression-summary"><strong>The long-window trajectory improved, but the production gate remains closed.</strong> Fast initialization and the selected baseline scale reduced long-window ATE and restored frame continuity. The current median remains 12.9× above target, and CPU/thread pinning widened the run-to-run ATE range instead of making the trajectory reproducible.</p>
        <article class="progression-card" aria-labelledby="vio-progression-title">
          <header><h3 id="vio-progression-title">S3E Alpha VIO configuration progression</h3><p>Ordered experimental stages; logarithmic ATE axis. The 500-frame and 1,000-frame cohorts are separated and should not be compared as one continuous time series.</p></header>
          <div class="progression-chart" id="vio-progression-chart"></div>
          <p class="progression-reading"><strong>Readout.</strong> The 1.20× balanced profile is the strongest healthy single long-window configuration. High-recall features improve local RPE while degrading global ATE. Three-run intervals then show that neither normal pacing nor the current CPU-controlled runtime is reproducible.</p>
        </article>
        <article class="progression-card" aria-labelledby="layer-progression-title">
          <header><h3 id="layer-progression-title">Current evaluation layers</h3><p>Current three-run medians and ranges on the same logarithmic ATE scale; status labels distinguish live, offline, and delayed-map evidence.</p></header>
          <div class="progression-chart" id="layer-progression-chart"></div>
          <p class="progression-reading"><strong>Readout.</strong> Native 1 Hz RTK is live and causal but stops at 0.218 m median ATE. Only adaptive fixed-lag map finalization crosses 0.1 m, and that result is delayed controlled history rather than a live-pose pass.</p>
        </article>
        <article class="progression-card" aria-labelledby="progression-table-title">
          <header><h3 id="progression-table-title">Key changes at each VIO step</h3><p>Exact checked-out metrics, tracking continuity, and the comparison basis used for every annotated change.</p></header>
          <div class="progression-table-wrap"><table class="progression-table"><thead><tr><th>Stage</th><th>Scope</th><th>ATE</th><th>Sim(3)</th><th>RPE</th><th>Tracking</th><th>Measured change</th><th>Key change</th></tr></thead><tbody id="progression-table-body"></tbody></table></div>
        </article>
        <p class="progression-caveat"><strong>Measurement boundary.</strong> ATE is rigidly aligned position RMSE unless explicitly labeled Sim(3). Ranges are minima and maxima from three runs, not confidence intervals. Percentage changes are descriptive artifact comparisons; fitted truth, offline alignment, or delayed history never opens a production global-pose claim.</p>
      </div>
    </section>

    <section class="band" id="global-gaussians" aria-labelledby="global-gaussians-title">
      <div class="inner">
        <div class="section-head">
          <h2 id="global-gaussians-title">S3E static asynchronous global Gaussian attempt</h2>
          <p>Dense model outputs are fused spatially without sorting, resampling, or synchronizing capture time. Registration evidence remains a separate fail-closed gate.</p>
        </div>
        <article class="fusion-shell">
          <div class="fusion-status" id="fusion-status"></div>
          <div class="fusion-detail">
            <div><h3>What ran</h3><p id="fusion-summary"></p><div id="fusion-sources"></div></div>
            <div><h3>Claim boundary and next step</h3><div class="fusion-warning" id="fusion-warning"></div><p id="fusion-next"></p><div class="fusion-links"><a href="s3e-global-gaussian-static/manifest.json" target="_blank">Open manifest ↗</a><a href="s3e-global-gaussian-static/preparation.json" target="_blank">Open pose diagnostics ↗</a><a href="s3e-global-gaussian-static/unified_s3e_global_gaussians.ply" download>Download 44 MB PLY</a><a href="?doc=applications%2Fariadne%2Fdocs%2Fstatic_asynchronous_global_gaussians.md#documentation">Read method and results</a></div></div>
          </div>
        </article>
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

    <section class="band" id="documentation" aria-labelledby="documentation-title">
      <div class="inner">
        <div class="section-head">
          <h2 id="documentation-title">Documentation library</h2>
          <p>Search and read every published project document without leaving the reporting interface. Repository-relative links open the corresponding document here.</p>
        </div>
        <div class="docs-shell">
          <aside class="docs-sidebar" aria-label="Documentation browser">
            <label class="docs-search-wrap" for="docs-search"><span>Search documentation</span><input class="docs-search" id="docs-search" type="search" placeholder="Title, path, or content" autocomplete="off"></label>
            <p class="docs-count" id="docs-count"></p>
            <nav class="docs-nav" id="docs-nav" aria-label="Available documents"></nav>
          </aside>
          <article class="docs-reader" aria-live="polite" aria-labelledby="doc-reader-title">
            <header class="docs-reader-head"><h3 id="doc-reader-title"></h3><div class="docs-reader-meta" id="doc-reader-meta"></div></header>
            <div class="markdown-body" id="doc-content"></div>
          </article>
        </div>
      </div>
    </section>

    <section class="band alt" id="gaps">
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
    const escapeHtml = value => String(value).replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
    const documentsByPath = new Map(data.documentation.map(document => [document.path, document]));
    let selectedDocumentPath = '';
    let documentationQuery = '';

    const normalizeDocumentPath = (currentPath, target) => {
      const pathOnly = target.split('#', 1)[0].split('?', 1)[0];
      let decoded;
      try { decoded = decodeURIComponent(pathOnly); } catch { decoded = pathOnly; }
      const segments = decoded.startsWith('/') ? [] : currentPath.split('/').slice(0, -1);
      decoded.replace(/^\.\//, '').split('/').forEach(segment => {
        if (!segment || segment === '.') return;
        if (segment === '..') segments.pop();
        else segments.push(segment);
      });
      return segments.join('/');
    };

    const renderInlineMarkdown = (source, currentPath) => {
      const renderPlain = value => escapeHtml(value)
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/~~([^~]+)~~/g, '<del>$1</del>');
      const pattern = /(`[^`\n]+`|\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\))/g;
      let result = '';
      let cursor = 0;
      for (const match of source.matchAll(pattern)) {
        result += renderPlain(source.slice(cursor, match.index));
        if (match[0].startsWith('`')) {
          result += `<code>${escapeHtml(match[0].slice(1, -1))}</code>`;
        } else {
          const label = escapeHtml(match[2]);
          const target = match[3];
          const normalized = normalizeDocumentPath(currentPath, target);
          if (documentsByPath.has(normalized)) {
            result += `<a href="?doc=${encodeURIComponent(normalized)}#documentation" data-doc-path="${escapeHtml(normalized)}">${label}</a>`;
          } else if (/^(https?:|mailto:)/i.test(target)) {
            result += `<a href="${escapeHtml(target)}" target="_blank" rel="noopener noreferrer">${label}</a>`;
          } else {
            result += `<span title="Unavailable from the published documentation set: ${escapeHtml(target)}">${label}</span>`;
          }
        }
        cursor = match.index + match[0].length;
      }
      return result + renderPlain(source.slice(cursor));
    };

    const tableCells = line => line.trim().replace(/^\||\|$/g, '').split('|').map(cell => cell.trim());
    const tableDivider = line => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
    const markdownBlockStart = (lines, index) => {
      const line = lines[index] || '';
      return /^\s*$/.test(line) || /^```/.test(line) || /^#{1,6}\s+/.test(line) || /^\s*([-*+]\s+|\d+\.\s+|>\s?)/.test(line) || /^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line) || (index + 1 < lines.length && line.includes('|') && tableDivider(lines[index + 1]));
    };

    const renderMarkdown = document => {
      const lines = document.markdown.replace(/\r\n?/g, '\n').split('\n');
      const output = [];
      for (let index = 0; index < lines.length;) {
        const line = lines[index];
        if (!line.trim()) { index += 1; continue; }
        const fence = line.match(/^```\s*([^\s`]*)/);
        if (fence) {
          const code = [];
          index += 1;
          while (index < lines.length && !/^```/.test(lines[index])) code.push(lines[index++]);
          if (index < lines.length) index += 1;
          const language = fence[1] ? ` data-language="${escapeHtml(fence[1])}"` : '';
          output.push(`<pre><code${language}>${escapeHtml(code.join('\n'))}</code></pre>`);
          continue;
        }
        const heading = line.match(/^(#{1,6})\s+(.+?)\s*#*$/);
        if (heading) {
          const level = heading[1].length;
          output.push(`<h${level}>${renderInlineMarkdown(heading[2], document.path)}</h${level}>`);
          index += 1;
          continue;
        }
        if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
          output.push('<hr>'); index += 1; continue;
        }
        if (index + 1 < lines.length && line.includes('|') && tableDivider(lines[index + 1])) {
          const headers = tableCells(line);
          index += 2;
          const rows = [];
          while (index < lines.length && lines[index].includes('|') && lines[index].trim()) rows.push(tableCells(lines[index++]));
          output.push(`<table><thead><tr>${headers.map(cell => `<th>${renderInlineMarkdown(cell, document.path)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${headers.map((_, cellIndex) => `<td>${renderInlineMarkdown(row[cellIndex] || '', document.path)}</td>`).join('')}</tr>`).join('')}</tbody></table>`);
          continue;
        }
        if (/^\s*>\s?/.test(line)) {
          const quote = [];
          while (index < lines.length && /^\s*>\s?/.test(lines[index])) quote.push(lines[index++].replace(/^\s*>\s?/, ''));
          output.push(`<blockquote><p>${quote.map(item => renderInlineMarkdown(item, document.path)).join('<br>')}</p></blockquote>`);
          continue;
        }
        const list = line.match(/^\s*([-*+]|\d+\.)\s+(.+)$/);
        if (list) {
          const ordered = /\d+\./.test(list[1]);
          const items = [];
          while (index < lines.length) {
            const item = lines[index].match(/^\s*([-*+]|\d+\.)\s+(.+)$/);
            if (!item || /\d+\./.test(item[1]) !== ordered) break;
            const task = item[2].match(/^\[([ xX])\]\s+(.+)$/);
            items.push(task ? `<li><input type="checkbox" disabled ${task[1].toLowerCase() === 'x' ? 'checked' : ''}> ${renderInlineMarkdown(task[2], document.path)}</li>` : `<li>${renderInlineMarkdown(item[2], document.path)}</li>`);
            index += 1;
          }
          const tag = ordered ? 'ol' : 'ul';
          output.push(`<${tag}>${items.join('')}</${tag}>`);
          continue;
        }
        const paragraph = [line.trim()];
        index += 1;
        while (index < lines.length && !markdownBlockStart(lines, index)) paragraph.push(lines[index++].trim());
        output.push(`<p>${renderInlineMarkdown(paragraph.join(' '), document.path)}</p>`);
      }
      return output.join('');
    };

    const filteredDocuments = () => {
      const query = documentationQuery.trim().toLowerCase();
      if (!query) return data.documentation;
      return data.documentation.filter(document => `${document.title}\n${document.path}\n${document.markdown}`.toLowerCase().includes(query));
    };

    const renderDocumentationNavigation = () => {
      const documents = filteredDocuments();
      document.getElementById('docs-count').textContent = `${documents.length} of ${data.documentation.length} documents`;
      const groups = new Map();
      documents.forEach(document => {
        if (!groups.has(document.group)) groups.set(document.group, []);
        groups.get(document.group).push(document);
      });
      const navigation = document.getElementById('docs-nav');
      navigation.innerHTML = documents.length ? Array.from(groups, ([group, items]) => `<section class="docs-group"><h3>${escapeHtml(group)}</h3>${items.map(document => `<button class="doc-link ${document.path === selectedDocumentPath ? 'active' : ''}" type="button" data-doc-path="${escapeHtml(document.path)}" ${document.path === selectedDocumentPath ? 'aria-current="page"' : ''}><strong>${escapeHtml(document.title)}</strong><span>${escapeHtml(document.path)}</span></button>`).join('')}</section>`).join('') : '<p class="docs-empty">No documents match this search.</p>';
      navigation.querySelectorAll('[data-doc-path]').forEach(button => button.addEventListener('click', () => selectDocument(button.dataset.docPath, true)));
    };

    const selectDocument = (path, updateLocation = false) => {
      const selected = documentsByPath.get(path) || data.documentation[0];
      if (!selected) return;
      selectedDocumentPath = selected.path;
      document.getElementById('doc-reader-title').textContent = selected.title;
      document.getElementById('doc-reader-meta').innerHTML = `<span>${escapeHtml(selected.path)}</span><span>${selected.word_count.toLocaleString()} words</span><span>${escapeHtml(selected.group)}</span>`;
      const content = document.getElementById('doc-content');
      content.innerHTML = renderMarkdown(selected);
      content.querySelectorAll('[data-doc-path]').forEach(link => link.addEventListener('click', event => {
        event.preventDefault();
        selectDocument(link.dataset.docPath, true);
        document.getElementById('documentation').scrollIntoView();
      }));
      renderDocumentationNavigation();
      if (updateLocation) history.replaceState(null, '', `${location.pathname}?doc=${encodeURIComponent(selected.path)}#documentation`);
    };

    document.getElementById('docs-search').addEventListener('input', event => {
      documentationQuery = event.target.value;
      renderDocumentationNavigation();
    });
    const requestedDocument = new URLSearchParams(location.search).get('doc');
    selectDocument(documentsByPath.has(requestedDocument) ? requestedDocument : (documentsByPath.has('applications/ariadne/README.md') ? 'applications/ariadne/README.md' : data.documentation[0]?.path));
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

    const progression = data.evaluation_progression;
    const progressionColors = { baseline: '#66716b', diagnostic: '#a46516', retained: '#2f628b', rejected: '#a33f48', current: '#2f628b', controlled: '#1f7451' };
    const progressionMarker = (status, x, y, color) => {
      if (status === 'diagnostic') return `<path d="M${x} ${y-7}L${x+7} ${y}L${x} ${y+7}L${x-7} ${y}Z" fill="#fff" stroke="${color}" stroke-width="3"/>`;
      if (status === 'rejected') return `<path d="M${x-6} ${y-6}L${x+6} ${y+6}M${x+6} ${y-6}L${x-6} ${y+6}" stroke="${color}" stroke-width="3"/>`;
      if (status === 'retained') return `<rect x="${x-6}" y="${y-6}" width="12" height="12" fill="${color}" stroke="#fff" stroke-width="2"/>`;
      if (status === 'current' || status === 'controlled') return `<circle cx="${x}" cy="${y}" r="7" fill="#fff" stroke="${color}" stroke-width="4"/>`;
      return `<circle cx="${x}" cy="${y}" r="6" fill="${color}" stroke="#fff" stroke-width="2"/>`;
    };
    const renderProgressionChart = (items, chartLabel) => {
      const width = 1100, left = 350, right = 1035, top = 72, rowGap = 58;
      const height = top + items.length * rowGap + 48;
      const domainMin = Math.log10(.08), domainMax = Math.log10(6);
      const x = value => left + (Math.log10(value) - domainMin) / (domainMax - domainMin) * (right - left);
      const ticks = [.1, .2, .5, 1, 2, 5];
      const guides = ticks.map(value => `<g><line x1="${x(value)}" y1="46" x2="${x(value)}" y2="${height-35}" stroke="${value === progression.target_ate_m ? '#1f7451' : '#d7ddd9'}" stroke-width="${value === progression.target_ate_m ? 2 : 1}" ${value === progression.target_ate_m ? '' : 'stroke-dasharray="3 5"'}/><text x="${x(value)}" y="29" text-anchor="middle" class="chart-label">${value} m</text></g>`).join('');
      const rows = items.map((item, index) => {
        const y = top + index * rowGap;
        const color = progressionColors[item.status] || progressionColors.baseline;
        const low = item.ate_min_m ?? item.ate_m;
        const high = item.ate_max_m ?? item.ate_m;
        const interval = high > low ? `<line x1="${x(low)}" y1="${y}" x2="${x(high)}" y2="${y}" stroke="${color}" stroke-opacity=".28" stroke-width="10"/><path d="M${x(low)} ${y-7}V${y+7}M${x(high)} ${y-7}V${y+7}" stroke="${color}" stroke-width="2"/>` : '';
        const valueLabel = high > low ? `${fmt(item.ate_m, 3)} m median · ${fmt(low, 3)}–${fmt(high, 3)}` : `${fmt(item.ate_m, 3)} m`;
        return `<g><line x1="18" y1="${y+29}" x2="${right}" y2="${y+29}" stroke="#edf0ee"/><text x="18" y="${y-5}" class="chart-value">${escapeHtml(item.label)}</text><text x="18" y="${y+12}" class="chart-label">${escapeHtml(item.scope || item.detail)}</text><text x="330" y="${y+4}" text-anchor="end" fill="${color}" style="font:700 9px Inter,sans-serif;text-transform:uppercase">${escapeHtml(item.status)}</text>${interval}${progressionMarker(item.status, x(item.ate_m), y, color)}<text x="${Math.min(x(item.ate_m)+12, right-120)}" y="${y-10}" class="chart-value">${valueLabel}</text></g>`;
      }).join('');
      return `<div role="img" aria-label="${escapeHtml(chartLabel)}"><svg viewBox="0 0 ${width} ${height}" aria-hidden="true"><text x="${left}" y="14" class="chart-label">ATE RMSE · LOGARITHMIC SCALE · LOWER IS BETTER</text>${guides}<text x="${x(progression.target_ate_m)+6}" y="44" fill="#1f7451" style="font:700 9px Inter,sans-serif">0.1 m TARGET</text>${rows}</svg></div>`;
    };
    document.getElementById('vio-progression-chart').innerHTML = renderProgressionChart(progression.stages, 'Ordered S3E Alpha VIO ATE progression from original baseline to current controlled-runtime repeats');
    document.getElementById('layer-progression-chart').innerHTML = renderProgressionChart(progression.current_layers, 'Current raw, aligned, live-corrected, and delayed-map Alpha ATE evaluation layers');
    const formatProgressionRange = stage => stage.ate_min_m == null ? `${fmt(stage.ate_m, 3)} m` : `${fmt(stage.ate_min_m, 3)}–${fmt(stage.ate_max_m, 3)} m (med ${fmt(stage.ate_m, 3)})`;
    document.getElementById('progression-table-body').innerHTML = progression.stages.map(stage => {
      const deltaClass = stage.status === 'retained' ? 'metric-improved' : stage.status === 'rejected' ? 'metric-regressed' : '';
      return `<tr><td><strong>${escapeHtml(stage.label)}</strong></td><td>${escapeHtml(stage.scope)}</td><td>${formatProgressionRange(stage)}</td><td>${fmt(stage.sim3_ate_m, 3)} m</td><td>${fmt(stage.rpe_m, 4)} m</td><td>${fmt(stage.poses, 0)} poses<br>${fmt(stage.lost, 0)} lost · ${fmt(stage.resets, 0)} resets</td><td class="${deltaClass}">${escapeHtml(stage.delta)}</td><td>${escapeHtml(stage.change)}</td></tr>`;
    }).join('');

    const fusion = data.static_global_gaussians;
    const fusionMetrics = [
      [fusion.input_gaussians.toLocaleString(), 'Input Gaussians'],
      [fusion.output_gaussians.toLocaleString(), 'Finite unified Gaussians'],
      [fusion.filtered_gaussians.toLocaleString(), 'Corrupt primitives filtered'],
      [fusion.global_registration_verified ? 'Verified' : 'Unverified', 'Global registration'],
    ];
    document.getElementById('fusion-status').innerHTML = fusionMetrics.map(([value, label]) => `<div class="fusion-metric"><strong>${value}</strong><span>${label}</span></div>`).join('');
    document.getElementById('fusion-summary').textContent = `Three real 62-property ReSplat PLYs were generated from independently timed S3E Alpha, Bob, and Carol windows, then transformed and concatenated in ${fusion.mode} mode. Temporal alignment is ${fusion.temporal_alignment}; input timestamps are deliberately out of order and temporal overlap is not required. The output bounds are ${fusion.bounds_m.minimum.map(value => fmt(value, 1)).join(', ')} m to ${fusion.bounds_m.maximum.map(value => fmt(value, 1)).join(', ')} m.`;
    const fusionPreparation = new Map(fusion.preparation.windows.map(window => [window.agent_id, window]));
    document.getElementById('fusion-sources').innerHTML = fusion.sources.map(source => { const prepared = fusionPreparation.get(source.agent_id); return `<code>${escapeHtml(source.agent_id)} · ${source.input_gaussians.toLocaleString()} Gaussians · ${prepared.pose_matched_frames} pose-matched frames · ${fmt(prepared.vio_to_truth_alignment_rmse_m, 2)} m offline alignment RMSE · ${fmt(prepared.vio_to_truth_scale, 3)}× scale · capture ${source.capture_timestamp_ns}</code>`; }).join('');
    document.getElementById('fusion-warning').textContent = fusion.warnings.join(' ');
    document.getElementById('fusion-next').textContent = 'Next: replace offline truth-fitted transforms with claim-eligible finalized ARIADNE poses, recover Bob/Carol tracking, rotate directional spherical harmonics into the global basis, then evaluate overlap and duplicate geometry. Cross-Wingman clocks remain unsynchronized.';

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
      const badges = [
        `<span><strong>${new Set(visual.nodes.map(node => node.component)).size}</strong> components</span>`,
        `<span><strong>${visual.rejected.length}</strong> rejected constraints</span>`,
        `<span>restart revision <strong>${visual.state_restored && visual.restored_revision === visual.revision ? 'stable' : 'missing'}</strong></span>`,
        `<span>MILUV real 6-DoF ATE <strong>${fmt(visual.miluv_baseline_ate_m,3)} → ${fmt(visual.miluv_optimized_ate_m,4)} m</strong> / ${fmt(visual.miluv_target_ate_m,2)} m target</span>`,
        `<span>MILUV real orientation RMSE <strong>${fmt(visual.miluv_optimized_orientation_rmse_rad,4)} rad</strong></span>`,
        `<span>MILUV fixed cadence <strong>${fmt(visual.miluv_correction_interval_s,2)} s / ${fmt(visual.miluv_messages_per_minute_max,2)} msg min⁻¹ max</strong></span>`,
        `<span>MILUV corrections per UAV <strong>${Object.entries(visual.miluv_corrections_by_agent).map(([agent,count]) => `${agent}: ${count}`).join(' · ')}</strong></span>`,
        `<span>MILUV cross-agent-only ATE <strong>${fmt(visual.miluv_cross_agent_only_ate_m,3)} m</strong> · global anchors required</span>`,
        `<span>MILUV UWB RMSE / p95 <strong>${fmt(visual.miluv_uwb_rmse_m,3)} / ${fmt(visual.miluv_uwb_p95_m,3)} m</strong></span>`,
        `<span>MILUV causal UWB graph <strong>${fmt(visual.miluv_uwb_causal_ate_m,3)} m / ${fmt(visual.miluv_uwb_causal_messages_per_minute_max,1)} msg min⁻¹ max</strong> · target missed</span>`,
        `<span>MILUV causal fixed lag <strong>${fmt(visual.miluv_uwb_fixed_lag_position_ate_m,4)} m fleet / ${fmt(visual.miluv_uwb_fixed_lag_max_agent_ate_m,4)} m max Wingman</strong> · ${visual.miluv_uwb_fixed_lag_all_agents_target_met ? 'controlled pass' : 'per-node miss'} · claim ${visual.miluv_uwb_fixed_lag_position_claim_eligible ? 'eligible' : 'closed'}</span>`,
        `<span>MILUV fixed-lag timing <strong>${fmt(visual.miluv_uwb_fixed_lag_duration_s,2)} s window / ${fmt(visual.miluv_uwb_fixed_lag_solve_interval_s,2)} s updates / ${fmt(visual.miluv_uwb_fixed_lag_solve_p95_ms,1)} ms p95</strong></span>`,
        `<span>MILUV fixed-lag correction load <strong>${fmt(visual.miluv_uwb_fixed_lag_messages_per_minute_max,1)} msg min⁻¹ max</strong> · orientation ${fmt(visual.miluv_uwb_fixed_lag_orientation_rmse_rad,4)} rad remains controlled</span>`,
        `<span>MILUV non-causal UWB batch <strong>${fmt(visual.miluv_uwb_batch_position_ate_m,4)} m position / ${fmt(visual.miluv_uwb_batch_orientation_rmse_rad,4)} rad orientation</strong> · position ${visual.miluv_uwb_batch_position_target_met ? 'passes' : 'fails'}, full pose ${visual.miluv_uwb_batch_full_pose_target_met ? 'passes' : 'fails'}</span>`,
        `<span>MILUV post-batch lower-bound load <strong>${fmt(visual.miluv_uwb_batch_messages_per_minute_max,1)} msg min⁻¹ max</strong></span>`,
        `<span>MILUV adaptive / fixed load <strong>${visual.miluv_adaptive_correction_count} / ${visual.miluv_fixed_correction_count} corrections</strong> · fixed retained</span>`,
        `<span>MILUV archive read <strong>${fmt(visual.miluv_loaded_archive_fraction_percent,3)}%</strong></span>`,
        `<span>S3E proxy ATE <strong>${fmt(visual.s3e_optimized_ate_m,3)} m</strong> / ${fmt(visual.s3e_target_ate_m,2)} m target</span>`,
        `<span>S3E controlled per-Wingman ATE <strong>${Object.entries(visual.s3e_per_agent_ate_m).map(([agent,ate]) => `${agent}: ${fmt(ate,3)} m`).join(' · ')}</strong> · worst ${fmt(visual.s3e_maximum_agent_ate_m,3)} m</span>`,
        `<span>S3E adaptive / fixed load <strong>${visual.s3e_selected_correction_count} / ${visual.s3e_fixed_correction_count} corrections</strong> · ${fmt(visual.s3e_correction_load_reduction_percent,1)}% reduction</span>`,
        `<span>S3E scheduler demand envelope <strong>${fmt(visual.s3e_scheduler_demand_error_m,2)} m</strong> · ${visual.s3e_capacity_override_cycles} capacity-override cycles</span>`,
        `<span>S3E controlled cross-Wingman relative RMSE <strong>${fmt(visual.s3e_cross_agent_baseline_relative_rmse_m,3)} → ${fmt(visual.s3e_cross_agent_dense_relative_rmse_m,3)} m</strong> · ${fmt(visual.s3e_cross_agent_dense_relative_improvement_percent,1)}% improvement</span>`,
        `<span>cross-Wingman-only absolute ATE <strong>${fmt(visual.s3e_cross_agent_only_global_ate_m,3)} m</strong> @ ${fmt(visual.s3e_cross_agent_factor_rate_per_minute,2)} factors/min · global anchor still required</span>`,
        `<span>cross-Wingman RMSE @ 0.05 / 0.20 m association noise <strong>${fmt(visual.s3e_cross_agent_relative_rmse_at_0_05m_noise,3)} / ${fmt(visual.s3e_cross_agent_relative_rmse_at_0_2m_noise,3)} m</strong></span>`,
        `<span>S3E controlled evidence payload <strong>${fmt(visual.s3e_report_payload_bytes/1024,1)} KiB</strong> · aggregate metrics + bounded sweeps</span>`,
        `<span>S3E deployment claim <strong>${visual.s3e_full_pose_claim_eligible ? 'eligible' : 'closed'}</strong> · ground-truth-derived position / controlled orientation</span>`,
        `<span>S3E controlled orientation <strong>${fmt(visual.s3e_baseline_orientation_rmse_rad,3)} → ${fmt(visual.s3e_optimized_orientation_rmse_rad,4)} rad</strong></span>`,
        `<span>passing correction noise <strong>${fmt(visual.s3e_maximum_translation_noise_m,3)} m / ${fmt(visual.s3e_maximum_rotation_noise_rad,3)} rad</strong></span>`,
        `<span>S3E rotation factors / iterations <strong>${visual.s3e_rotation_rationalization_constraints} / ${visual.s3e_rotation_rationalization_iterations}</strong></span>`,
        `<span>S3E constraint rotation RMSE <strong>${fmt(visual.s3e_constraint_rotation_rmse_rad,4)} rad</strong></span>`,
        `<span>real S3E VIO <strong>${visual.s3e_real_vio_target_met}/${visual.s3e_real_vio_agent_count}</strong> target passes</span>`,
        `<span>S3E sensor preflight <strong>${visual.s3e_sensor_contract_passes}/${visual.s3e_real_vio_agent_count}</strong> Wingmen pass</span>`,
        `<span>Carol auto geometry <strong>${fmt(visual.s3e_carol_replicate_ate_median,2)} m median ATE</strong> · ${fmt(visual.s3e_carol_replicate_ate_min,2)}–${fmt(visual.s3e_carol_replicate_ate_max,2)} m · ${visual.s3e_carol_reproducible ? 'reproducible' : 'unstable'}</span>`,
        `<span>Alpha three-run ATE <strong>${fmt(visual.s3e_alpha_replicate_ate_median,2)} m median</strong> · ${fmt(visual.s3e_alpha_replicate_ate_min,2)}–${fmt(visual.s3e_alpha_replicate_ate_max,2)} m · ${visual.s3e_alpha_reproducible ? 'reproducible' : 'unstable'}</span>`,
        `<span>Alpha single-CPU ATE / Sim(3) <strong>${fmt(visual.s3e_alpha_deterministic_ate_min,3)}–${fmt(visual.s3e_alpha_deterministic_ate_max,3)} / ${fmt(visual.s3e_alpha_deterministic_sim3_min,3)}–${fmt(visual.s3e_alpha_deterministic_sim3_max,3)} m</strong> · ${visual.s3e_alpha_deterministic_reproducible ? 'reproducible' : 'still unstable'}</span>`,
        `<span>Alpha mapping-sync ATE / Sim(3) <strong>${fmt(visual.s3e_alpha_mapping_sync_ate_min,3)}–${fmt(visual.s3e_alpha_mapping_sync_ate_max,3)} / ${fmt(visual.s3e_alpha_mapping_sync_sim3_min,3)}–${fmt(visual.s3e_alpha_mapping_sync_sim3_max,3)} m</strong> · ${visual.s3e_alpha_mapping_sync_reproducible ? 'reproducible' : 'still unstable'}</span>`,
        `<span>best Alpha ATE / Sim(3) floor <strong>${fmt(visual.s3e_alpha_best_ate_m,2)} / ${fmt(visual.s3e_alpha_best_sim3_ate_m,2)} m</strong></span>`,
        `<span>matched Alpha OpenVINS ATE / Sim(3) <strong>${fmt(visual.s3e_openvins_ate_m,1)} / ${fmt(visual.s3e_openvins_sim3_ate_m,2)} m</strong> (${visual.s3e_openvins_tracking_healthy ? 'correction eligible' : 'relocalize'})</span>`,
        `<span>OpenVINS fitted scale / correction load <strong>${fmt(visual.s3e_openvins_scale_correction,4)}× / ${fmt(visual.s3e_openvins_event_messages_per_minute,1)} min⁻¹</strong> · peak ${visual.s3e_openvins_event_peak_per_second} s⁻¹</span>`,
        `<span>S3E RTK orientation reference <strong>${visual.s3e_alpha_orientation_reference_available ? 'available' : 'unavailable'}</strong></span>`,
        `<span>Alpha shared-IMU orientation proxy / RPE <strong>${fmt(visual.s3e_alpha_orientation_proxy_rmse_rad,3)} / ${fmt(visual.s3e_alpha_orientation_proxy_rpe_rmse_rad,4)} rad</strong> (${visual.s3e_alpha_orientation_proxy_independent ? 'independent' : 'non-independent'})</span>`,
        `<span>Alpha stereo-only AHRS proxy <strong>${fmt(visual.s3e_alpha_stereo_orientation_proxy_rmse_rad,3)} rad</strong> (${visual.s3e_alpha_stereo_orientation_proxy_independent ? 'independent' : 'non-independent'})</span>`,
        `<span>bounded RTK lever-arm sensitivity <strong>${fmt(visual.s3e_alpha_lever_arm_fitted_norm_m,2)} / ${fmt(visual.s3e_alpha_lever_arm_maximum_norm_m,2)} m</strong> (${visual.s3e_alpha_lever_arm_bound_active ? 'bound saturated' : 'inside bound'})</span>`,
        `<span>lever-arm optimistic ATE / full-fit gain <strong>${fmt(visual.s3e_alpha_lever_arm_adjusted_ate_m,2)} m / ${fmt(visual.s3e_alpha_lever_arm_full_improvement_percent,1)}%</strong></span>`,
        `<span>lever-arm held-out gain <strong>${fmt(visual.s3e_alpha_lever_arm_holdout_improvement_percent,1)}%</strong> · unconstrained norm ${fmt(visual.s3e_alpha_lever_arm_unconstrained_norm_m,2)} m</span>`,
        `<span>Alpha offline local SE(3) <strong>${fmt(visual.s3e_alpha_local_rigid_1s_ate_m,3)} m @ ${fmt(visual.s3e_alpha_local_rigid_interval_s,1)} s</strong> · ${fmt(visual.s3e_alpha_local_rigid_messages_per_minute,0)} anchors/min</span>`,
        `<span>Alpha offline local Sim(3) <strong>${fmt(visual.s3e_alpha_local_sim3_2s_ate_m,3)} m @ ${fmt(visual.s3e_alpha_local_sim3_interval_s,1)} s</strong> · ${fmt(visual.s3e_alpha_local_sim3_messages_per_minute,0)} anchors/min</span>`,
        `<span>Alpha 5 s local scale p05–p95 <strong>${fmt(visual.s3e_alpha_local_5s_scale_p05,3)}–${fmt(visual.s3e_alpha_local_5s_scale_p95,3)}×</strong></span>`,
        `<span>0.5 s Sim(3) Alpha stereo / Bob / Carol <strong>${fmt(visual.s3e_alpha_stereo_local_sim3_0_5s_ate_m,3)} / ${fmt(visual.s3e_bob_local_sim3_0_5s_ate_m,3)} / ${fmt(visual.s3e_carol_local_sim3_0_5s_ate_m,3)} m</strong></span>`,
        `<span>local-fit actions <strong>${visual.s3e_local_alignment_profiles.map(profile => `${profile.agent_id}: ${profile.action.replaceAll('_',' ')}`).join(' · ')}</strong></span>`,
        `<span>local fits <strong>offline and non-causal</strong> · scored ATE unchanged</span>`,
        `<span>Alpha causal trailing SE(3) <strong>${fmt(visual.s3e_alpha_causal_se3_ate_m,3)} m @ ${fmt(visual.s3e_alpha_causal_se3_cadence_s,1)} s</strong> · ${fmt(visual.s3e_alpha_causal_se3_messages_per_minute,1)} anchors/min</span>`,
        `<span>Alpha causal trailing Sim(3) <strong>${fmt(visual.s3e_alpha_causal_sim3_ate_m,3)} m @ ${fmt(visual.s3e_alpha_causal_sim3_cadence_s,1)} s</strong> · ${fmt(visual.s3e_alpha_causal_sim3_messages_per_minute,1)} anchors/min</span>`,
        `<span>causal p95 jumps SE(3) / Sim(3) <strong>${fmt(visual.s3e_alpha_causal_se3_jump_p95_m,3)} / ${fmt(visual.s3e_alpha_causal_sim3_jump_p95_m,3)} m</strong></span>`,
        `<span>causal Sim(3) p95 scale update <strong>${fmt(visual.s3e_alpha_causal_sim3_scale_update_p95_percent,1)}%</strong> · ideal RTK anchors, zero latency</span>`,
        `<span>Alpha phase ATE first / middle / last <strong>${fmt(visual.s3e_alpha_first_quarter_ate_m,2)} / ${fmt(visual.s3e_alpha_middle_half_ate_m,2)} / ${fmt(visual.s3e_alpha_last_quarter_ate_m,2)} m</strong></span>`,
        `<span>Alpha peak-error trajectory point <strong>${fmt(visual.s3e_alpha_peak_error_fraction*100,1)}%</strong></span>`,
        `<span>timing best shift / ATE gain <strong>${visual.s3e_alpha_timing_best_offset_ms} ms / ${fmt(visual.s3e_alpha_timing_gain_percent,2)}%</strong> (${visual.s3e_alpha_timing_dominant ? 'dominant' : 'not dominant'})</span>`,
        `<span>high-recall ORB ATE <strong>${fmt(visual.s3e_alpha_high_recall_ate_m,2)} m</strong> (rejected)</span>`,
        `<span>later-window baseline / 1.20× ATE <strong>${fmt(visual.s3e_alpha_late_default_ate_m,2)} / ${fmt(visual.s3e_alpha_late_scaled_ate_m,2)} m</strong></span>`,
        `<span>later-window residual scale <strong>${fmt(visual.s3e_alpha_late_scaled_residual_scale,3)}×</strong></span>`,
        `<span>Alpha replicated causal Sim(3) <strong>${fmt(visual.s3e_alpha_replicate_causal_ate_max,4)} m worst / ${fmt(visual.s3e_alpha_replicate_load_min,1)}–${fmt(visual.s3e_alpha_replicate_load_max,1)} corrections/min</strong></span>`,
        `<span>Alpha ideal-interpolated ingress / hold threshold <strong>${fmt(visual.s3e_alpha_replicate_anchor_load_max,1)} anchors/min / ${fmt(visual.s3e_alpha_replicate_threshold_min,2)}–${fmt(visual.s3e_alpha_replicate_threshold_max,2)} m</strong></span>`,
        `<span>causal hold p95 / max <strong>${fmt(visual.s3e_alpha_replicate_hold_p95_max_s,2)} / ${fmt(visual.s3e_alpha_replicate_hold_max_s,2)} s</strong> · reaction ${fmt(visual.s3e_alpha_replicate_min_interval_s,3)} s · peak ${visual.s3e_alpha_replicate_peak_per_second} s⁻¹</span>`,
        `<span>Alpha radio-min comparison <strong>${fmt(visual.s3e_alpha_radio_min_ate_max,4)} m / ${fmt(visual.s3e_alpha_radio_min_correction_load_max,1)} corrections/min</strong> · ${fmt(visual.s3e_alpha_radio_min_anchor_load_max,1)} anchors/min</span>`,
        `<span>Alpha native 1 Hz RTK Sim(3) <strong>${fmt(visual.s3e_alpha_native_rtk_sim3_ate_min,3)}–${fmt(visual.s3e_alpha_native_rtk_sim3_ate_max,3)} m</strong> · ${visual.s3e_alpha_native_rtk_target_pass_count}/3 target passes</span>`,
        `<span>Alpha native RTK ingress / corrections <strong>${fmt(visual.s3e_alpha_native_rtk_anchor_rate,1)} / ${fmt(visual.s3e_alpha_native_rtk_correction_rate,1)} min⁻¹</strong> · worst SE(3) ${fmt(visual.s3e_alpha_native_rtk_se3_ate_max,3)} m</span>`,
        `<span>native RTK Sim(3) Alpha / Bob / Carol <strong>${fmt(visual.s3e_native_rtk_by_agent.Alpha.ate_m,3)} / ${fmt(visual.s3e_native_rtk_by_agent.Bob.ate_m,3)} / ${fmt(visual.s3e_native_rtk_by_agent.Carol.ate_m,3)} m</strong></span>`,
        `<span>Alpha past-segment live hold <strong>${fmt(visual.s3e_alpha_segment_hold_ate_min,3)}–${fmt(visual.s3e_alpha_segment_hold_ate_max,3)} m</strong> · ${visual.s3e_alpha_segment_hold_target_pass_count}/3 passes</span>`,
        `<span>past-segment live hold Alpha / Bob / Carol <strong>${fmt(visual.s3e_segment_hold_by_agent.Alpha,3)} / ${fmt(visual.s3e_segment_hold_by_agent.Bob,3)} / ${fmt(visual.s3e_segment_hold_by_agent.Carol,3)} m</strong></span>`,
        `<span>past-segment hold updates Alpha / Bob / Carol <strong>${fmt(visual.s3e_segment_hold_updates_by_agent.Alpha,1)} / ${fmt(visual.s3e_segment_hold_updates_by_agent.Bob,1)} / ${fmt(visual.s3e_segment_hold_updates_by_agent.Carol,1)} min⁻¹</strong> · current pose, position-only, claim closed</span>`,
        `<span>Alpha live hold ATE @ 0.1 / 0.2 / 0.5 / 1.0 s <strong>${fmt(visual.s3e_alpha_segment_hold_horizon_ate['0.1'],3)} / ${fmt(visual.s3e_alpha_segment_hold_horizon_ate['0.2'],3)} / ${fmt(visual.s3e_alpha_segment_hold_horizon_ate['0.5'],3)} / ${fmt(visual.s3e_alpha_segment_hold_horizon_ate['1.0'],3)} m</strong></span>`,
        `<span>live target horizon Alpha / Bob / Carol <strong>${fmt(visual.s3e_segment_hold_target_horizon_by_agent.Alpha,1)} / ${fmt(visual.s3e_segment_hold_target_horizon_by_agent.Bob,1)} / ${fmt(visual.s3e_segment_hold_target_horizon_by_agent.Carol,1)} s</strong> · Alpha requires ≥${fmt(visual.s3e_alpha_segment_hold_required_observation_rate,0)} observations/min</span>`,
        `<span>Alpha fixed-lag native Sim(3) <strong>${fmt(visual.s3e_alpha_fixed_lag_sim3_ate_min,3)}–${fmt(visual.s3e_alpha_fixed_lag_sim3_ate_max,3)} m</strong> · ${visual.s3e_alpha_fixed_lag_target_pass_count}/3 finalized-map passes</span>`,
        `<span>fixed-lag native Sim(3) Alpha / Bob / Carol <strong>${fmt(visual.s3e_fixed_lag_by_agent.Alpha,3)} / ${fmt(visual.s3e_fixed_lag_by_agent.Bob,3)} / ${fmt(visual.s3e_fixed_lag_by_agent.Carol,3)} m</strong></span>`,
        `<span>fixed-lag mean / p95 / max delay <strong>${fmt(visual.s3e_alpha_fixed_lag_latency_mean_s,3)} / ${fmt(visual.s3e_alpha_fixed_lag_latency_p95_s,3)} / ${fmt(visual.s3e_alpha_fixed_lag_latency_max_s,3)} s</strong> · ${fmt(visual.s3e_alpha_fixed_lag_updates_per_minute,1)} map finalizations/min</span>`,
        `<span>Alpha fixed-lag scale p05 / p95 / max <strong>${fmt(visual.s3e_alpha_fixed_lag_scale_p05_min,3)} / ${fmt(visual.s3e_alpha_fixed_lag_scale_p95_max,3)} / ${fmt(visual.s3e_alpha_fixed_lag_scale_max,3)}×</strong> · ${fmt(visual.s3e_alpha_fixed_lag_scale_plausible_fraction*100,0)}% inside 0.5–2.0× gate</span>`,
        `<span>fixed-lag caveat <strong>past trajectory only</strong> · ${fmt(visual.s3e_alpha_fixed_lag_coverage_min*100,1)}% coverage · live Wingman pose still fails</span>`,
        `<span>Alpha adaptive fixed lag <strong>${fmt(visual.s3e_alpha_adaptive_fixed_lag_ate_min,3)}–${fmt(visual.s3e_alpha_adaptive_fixed_lag_ate_max,3)} m</strong> · ${visual.s3e_alpha_adaptive_fixed_lag_target_pass_count}/3 finalized-map passes</span>`,
        `<span>adaptive finalizations <strong>${fmt(visual.s3e_alpha_adaptive_fixed_lag_updates_min,1)}–${fmt(visual.s3e_alpha_adaptive_fixed_lag_updates_max,1)} min⁻¹</strong> · ${fmt(visual.s3e_alpha_adaptive_fixed_lag_reduction_min,1)}–${fmt(visual.s3e_alpha_adaptive_fixed_lag_reduction_max,1)}% below full-rate</span>`,
        `<span>adaptive mean / p95 / max delay <strong>${fmt(visual.s3e_alpha_adaptive_fixed_lag_latency_mean_s,3)} / ${fmt(visual.s3e_alpha_adaptive_fixed_lag_latency_p95_s,3)} / ${fmt(visual.s3e_alpha_adaptive_fixed_lag_latency_max_s,3)} s</strong></span>`,
        `<span>adaptive fixed-lag Alpha / Bob / Carol <strong>${fmt(visual.s3e_adaptive_fixed_lag_by_agent.Alpha,3)} / ${fmt(visual.s3e_adaptive_fixed_lag_by_agent.Bob,3)} / ${fmt(visual.s3e_adaptive_fixed_lag_by_agent.Carol,3)} m</strong></span>`,
        `<span>adaptive finalizations Alpha / Bob / Carol <strong>${fmt(visual.s3e_adaptive_fixed_lag_updates_by_agent.Alpha,1)} / ${fmt(visual.s3e_adaptive_fixed_lag_updates_by_agent.Bob,1)} / ${fmt(visual.s3e_adaptive_fixed_lag_updates_by_agent.Carol,1)} min⁻¹</strong></span>`,
        `<span>adaptive caveat <strong>RTK ingress unchanged</strong> · delayed map only · live pose still closed</span>`,
        `<span>rotation step / reset guard <strong>${fmt(visual.correction_max_rotation_step_rad,2)} / ${fmt(visual.correction_max_total_rotation_rad,2)} rad</strong></span>`,
        `<span>scheduler average capacity / demand <strong>${fmt(visual.s3e_capacity_configured_messages_per_minute,0)} / ${fmt(visual.s3e_capacity_required_messages_per_minute,1)} msg/min</strong></span>`,
        `<span>fail-closed traffic suppression <strong>${fmt(visual.s3e_capacity_suppressed_messages_per_minute,1)} msg/min · peak ${visual.s3e_capacity_suppressed_peak_per_second} s⁻¹</strong></span>`,
        `<span>scheduler peak capacity / demand <strong>${fmt(visual.s3e_capacity_configured_peak_per_second,0)} / ${visual.s3e_capacity_required_peak_per_second} s⁻¹</strong></span>`,
        `<span>configured / required scheduler tick <strong>${fmt(visual.s3e_capacity_configured_evaluation_period_s,1)} / ${fmt(visual.s3e_capacity_recommended_evaluation_period_s,3)} s</strong></span>`,
        `<span>capacity action <strong>${visual.s3e_capacity_action.replaceAll('_',' ')}</strong></span>`,
        `<span>Wingman actions <strong>${visual.s3e_correction_profiles.map(profile => `${profile.agent_id}: ${profile.action.replaceAll('_',' ')}`).join(' · ')}</strong></span>`,
        `<span>tracking / live-pose failures <strong>${visual.s3e_capacity_tracking_failure_agents.join(', ') || 'none'} / ${visual.s3e_capacity_live_pose_failure_agents.join(', ') || 'none'}</strong></span>`,
        `<span>schedulable / relocalize <strong>${visual.s3e_capacity_schedulable_agents.join(', ') || 'none'} / ${visual.s3e_capacity_relocalization_agents.join(', ') || 'none'}</strong></span>`,
        `<span>proxy per Wingman Alpha / Bob / Carol <strong>${fmt(visual.s3e_messages_per_minute_by_agent.Alpha,2)} / ${fmt(visual.s3e_messages_per_minute_by_agent.Bob,2)} / ${fmt(visual.s3e_messages_per_minute_by_agent.Carol,2)} msg/min</strong></span>`,
        `<span>rationalization <strong>${fmt(visual.s3e_optimization_latency_ms,2)} ms</strong></span>`,
      ].join('');
      return visualShell('SE(3) graph + covariance', svg, badges);
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
    const scrollToCurrentSection = () => {
      if (!window.location.hash) return;
      let sectionId;
      try { sectionId = decodeURIComponent(window.location.hash.slice(1)); } catch { sectionId = window.location.hash.slice(1); }
      document.getElementById(sectionId)?.scrollIntoView();
    };
    scrollToCurrentSection();
    window.addEventListener('load', () => window.setTimeout(scrollToCurrentSection, 0), { once: true });
  </script>
</body>
</html>"""


def main() -> int:
    payload = json.dumps(build_payload(), separators=(",", ":"), ensure_ascii=True)
    payload = payload.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    artifact_link = OUTPUT.parent / "s3e-global-gaussian-static"
    artifact_target = STATIC_GLOBAL_GAUSSIANS.parent
    if artifact_link.is_symlink():
        if artifact_link.resolve() != artifact_target.resolve():
            raise RuntimeError(f"report artifact link points to an unexpected target: {artifact_link}")
    elif artifact_link.exists():
        raise RuntimeError(f"report artifact path already exists and is not a symlink: {artifact_link}")
    else:
        artifact_link.symlink_to(
            Path("../outputs/ariadne/s3e-global-gaussian-static"), target_is_directory=True
        )
    OUTPUT.write_text(HTML.replace("__PAYLOAD__", payload), encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
