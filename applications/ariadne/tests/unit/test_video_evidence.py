import base64
import importlib
from pathlib import Path

import numpy as np

from ariadne.benchmarks import build_video_evidence, select_video_frames
from ariadne.perception import SaliencyDetector


def _write_frames(root: Path) -> tuple[Path, ...]:
    image_module = importlib.import_module("PIL.Image")
    paths = []
    for index in range(24):
        timestamp_ns = 1_000_000_000 + index * 50_000_000
        image = np.full((32, 32, 3), 30, dtype=np.uint8)
        image[5:15, 4 + index % 4 : 14 + index % 4] = 220
        path = root / f"{timestamp_ns}.png"
        image_module.fromarray(image).save(path)
        paths.append(path)
    return tuple(paths)


def test_video_evidence_runs_twenty_real_frames_through_reference_perception(
    tmp_path: Path,
) -> None:
    frames = _write_frames(tmp_path)
    trajectory = tmp_path / "trajectory.txt"
    trajectory.write_text(
        "\n".join(
            f"{1.1 + index * 0.05:.8f} {index * 0.01:.4f} 0 0 0 0 0 1"
            for index in range(20)
        ),
        encoding="utf-8",
    )

    selected = select_video_frames(frames, trajectory)
    evidence = build_video_evidence(
        selected,
        trajectory,
        saliency_detector=SaliencyDetector(quantile=0.82),
    )

    assert len(selected) == 20
    assert evidence["frame_count"] == 20
    assert evidence["metrics"]["accepted_frames"] == 20
    assert evidence["metrics"]["mean_keypoints"] == 32
    assert evidence["saliency_backend"] == "gradient_contrast"
    assert len(evidence["frames"][0]["processed_grid"]) == 24
    assert len(evidence["frames"][0]["saliency_grid"]) == 12
    mask_shape = tuple(evidence["frames"][0]["saliency_mask_shape"])
    mask = np.unpackbits(
        np.frombuffer(
            base64.b64decode(evidence["frames"][0]["saliency_mask_bits"]),
            dtype=np.uint8,
        ),
        bitorder="big",
    )[: np.prod(mask_shape)].reshape(mask_shape)
    assert mask.shape == (96, 96)
    assert float(mask.mean()) == evidence["frames"][0]["salient_fraction"]
    assert evidence["metrics"]["saliency_latency_mean_ms"] > 0
    assert evidence["saliency_model_info"]["name"] == "Gradient/contrast reference"
    assert evidence["frames"][-1]["vio_position_m"][0] == 0.19
