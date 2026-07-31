"""Real-frame evidence extraction for the ARIADNE status report."""

from __future__ import annotations

import base64
import bisect
import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ariadne.backends.external_vio import parse_trajectory
from ariadne.common import Timestamp
from ariadne.models.features import (
    GradientPatchExtractor,
    SpatialPyramidEmbedder,
    cosine_similarity,
)
from ariadne.perception import (
    ImagePreprocessor,
    PretrainedU2NetSaliencyDetector,
    SaliencyBackend,
    SaliencyClusterer,
)
from ariadne.replay import ImageFrame


def select_video_frames(
    frame_paths: Sequence[Path], trajectory_path: Path, *, frame_count: int = 20
) -> tuple[Path, ...]:
    """Select a contiguous segment beginning when the production trajectory starts."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    paths = tuple(sorted(frame_paths, key=lambda path: int(path.stem)))
    trajectory = parse_trajectory(trajectory_path)
    if not paths or not trajectory:
        raise ValueError("video evidence requires camera frames and a trajectory")
    timestamps = [int(path.stem) for path in paths]
    start = bisect.bisect_left(timestamps, trajectory[0].timestamp_ns)
    selected = paths[start : start + frame_count]
    if len(selected) != frame_count:
        raise ValueError(f"video evidence requires {frame_count} contiguous frames")
    return selected


def _load_rgb(path: Path) -> npt.NDArray[np.uint8]:
    image_module = importlib.import_module("PIL.Image")
    with image_module.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return array


def _sample_grid(values: npt.NDArray[np.float32], size: int = 12) -> list[list[float]]:
    y_indices = np.rint(np.linspace(0, values.shape[0] - 1, size)).astype(int)
    x_indices = np.rint(np.linspace(0, values.shape[1] - 1, size)).astype(int)
    sampled = values[np.ix_(y_indices, x_indices)].round(3)
    return [[float(value) for value in row] for row in sampled]


def _encode_binary_mask(mask: npt.NDArray[np.bool_]) -> str:
    packed = np.packbits(mask.reshape(-1), bitorder="big")
    return base64.b64encode(packed.tobytes()).decode("ascii")


def build_video_evidence(
    selected_frames: Sequence[Path],
    trajectory_path: Path,
    *,
    saliency_detector: SaliencyBackend | None = None,
) -> dict[str, Any]:
    """Run real frames through pretrained saliency and attach production VIO poses."""
    if len(selected_frames) != 20:
        raise ValueError("the report evidence segment must contain exactly 20 frames")
    trajectory = parse_trajectory(trajectory_path)
    trajectory_times = np.asarray([pose.timestamp_ns for pose in trajectory], dtype=np.int64)
    processor = ImagePreprocessor(
        width=96,
        height=96,
        min_blur_score=0.0,
        min_exposure_score=0.0,
        max_occlusion_fraction=1.0,
    )
    feature_extractor = GradientPatchExtractor(max_keypoints=32)
    embedder = SpatialPyramidEmbedder()
    if saliency_detector is None:
        saliency_detector = PretrainedU2NetSaliencyDetector(
            ".cache/ariadne/models/u2net/u2net.pth"
        )
    clusterer = SaliencyClusterer(min_area_px=4, max_regions=8)

    evidence_frames: list[dict[str, Any]] = []
    reference_embedding: npt.NDArray[np.float64] | None = None
    for sequence_index, path in enumerate(selected_frames):
        timestamp_ns = int(path.stem)
        image = _load_rgb(path)
        frame = ImageFrame(Timestamp(timestamp_ns), "d2slam", image, sequence_index)
        processed = processor.process(frame)
        features = feature_extractor.extract(frame)
        embedding = embedder.embed(frame).vector
        if reference_embedding is None:
            reference_embedding = embedding
        saliency = saliency_detector.detect(processed)
        regions = clusterer.cluster(saliency)

        insertion = int(np.searchsorted(trajectory_times, timestamp_ns))
        candidates = [
            index for index in (insertion - 1, insertion) if 0 <= index < len(trajectory)
        ]
        nearest = min(
            candidates,
            key=lambda index: abs(trajectory[index].timestamp_ns - timestamp_ns),
        )
        pose = trajectory[nearest]
        if abs(pose.timestamp_ns - timestamp_ns) > 60_000_000:
            raise ValueError(f"frame {path.name} has no production VIO pose within 60 ms")

        height, width = image.shape[:2]
        evidence_frames.append(
            {
                "sequence_index": sequence_index,
                "source_frame_index": int(path.stem),
                "timestamp_ns": timestamp_ns,
                "quality": {
                    "accepted": processed.quality.accepted,
                    "blur": processed.quality.blur_score,
                    "exposure": processed.quality.exposure_score,
                    "occlusion": processed.quality.occlusion_fraction,
                    "latency_ms": processed.latency_ms,
                },
                "processed_grid": _sample_grid(processed.image, size=24),
                "keypoints_xy": [
                    [float(point[0] / max(width - 1, 1)), float(point[1] / max(height - 1, 1))]
                    for point in features.keypoints_xy
                ],
                "embedding_similarity": cosine_similarity(reference_embedding, embedding),
                "saliency_grid": _sample_grid(saliency.scores),
                "saliency_mask_bits": _encode_binary_mask(saliency.mask),
                "saliency_mask_shape": list(saliency.mask.shape),
                "saliency_backend": saliency.backend,
                "saliency_model": saliency.model_version,
                "salient_fraction": float(np.mean(saliency.mask)),
                "saliency_latency_ms": saliency.latency_ms,
                "regions": [
                    {
                        "id": region.region_id,
                        "bbox_xyxy": [
                            region.bbox_xyxy[0] / processed.processed_resolution[0],
                            region.bbox_xyxy[1] / processed.processed_resolution[1],
                            region.bbox_xyxy[2] / processed.processed_resolution[0],
                            region.bbox_xyxy[3] / processed.processed_resolution[1],
                        ],
                        "area_px": region.area_px,
                        "mean_score": region.mean_score,
                    }
                    for region in regions
                ],
                "vio_position_m": pose.position_m.round(6).tolist(),
            }
        )

    blur_scores = [float(frame["quality"]["blur"]) for frame in evidence_frames]
    exposure_scores = [float(frame["quality"]["exposure"]) for frame in evidence_frames]
    saliency_latencies = np.asarray(
        [float(frame["saliency_latency_ms"]) for frame in evidence_frames], dtype=np.float64
    )
    weights_path = getattr(saliency_detector, "weights_path", None)
    checkpoint_bytes = (
        int(weights_path.stat().st_size)
        if isinstance(weights_path, Path) and weights_path.is_file()
        else None
    )
    return {
        "frame_count": len(evidence_frames),
        "fps": 5,
        "source": "TUM VI corridor1 via the aligned D2SLAM replay",
        "start_timestamp_ns": evidence_frames[0]["timestamp_ns"],
        "end_timestamp_ns": evidence_frames[-1]["timestamp_ns"],
        "duration_s": (
            int(evidence_frames[-1]["timestamp_ns"])
            - int(evidence_frames[0]["timestamp_ns"])
        )
        / 1e9,
        "saliency_backend": saliency_detector.backend,
        "saliency_model": saliency_detector.model_version,
        "saliency_model_info": {
            "name": getattr(saliency_detector, "model_name", type(saliency_detector).__name__),
            "version": saliency_detector.model_version,
            "backend": saliency_detector.backend,
            "training_dataset": getattr(saliency_detector, "training_dataset", None),
            "device": getattr(saliency_detector, "device", "cpu"),
            "input_size_px": getattr(saliency_detector, "input_size", None),
            "output_size_px": list(evidence_frames[0]["saliency_mask_shape"]),
            "threshold": getattr(saliency_detector, "threshold", None),
            "parameter_count": getattr(saliency_detector, "parameter_count", 0),
            "checkpoint_sha256": getattr(saliency_detector, "checkpoint_sha256", None),
            "checkpoint_bytes": checkpoint_bytes,
            "timing_scope": (
                "per-frame detector call including tensor preparation, device transfer, "
                "inference, output resize, and CPU readback; checkpoint load excluded"
            ),
        },
        "frames": evidence_frames,
        "metrics": {
            "accepted_frames": sum(bool(frame["quality"]["accepted"]) for frame in evidence_frames),
            "mean_blur": float(np.mean(blur_scores)),
            "mean_exposure": float(np.mean(exposure_scores)),
            "mean_keypoints": float(
                np.mean([len(frame["keypoints_xy"]) for frame in evidence_frames])
            ),
            "mean_salient_fraction": float(
                np.mean([frame["salient_fraction"] for frame in evidence_frames])
            ),
            "saliency_latency_mean_ms": float(np.mean(saliency_latencies)),
            "saliency_latency_p50_ms": float(np.percentile(saliency_latencies, 50)),
            "saliency_latency_p95_ms": float(np.percentile(saliency_latencies, 95)),
            "saliency_latency_min_ms": float(np.min(saliency_latencies)),
            "saliency_latency_max_ms": float(np.max(saliency_latencies)),
            "saliency_throughput_fps": float(1000.0 / np.mean(saliency_latencies)),
            "mean_regions": float(np.mean([len(frame["regions"]) for frame in evidence_frames])),
        },
    }
