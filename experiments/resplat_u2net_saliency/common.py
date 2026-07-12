from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class RunSummary:
    run_name: str
    variant: str
    model_state: str
    checkpoint: str
    frame_count: int
    resolution: str
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    fps: float
    gpu_util_mean: float | None
    gpu_mem_peak_mb: float | None
    gpu_power_mean_w: float | None
    gpu_temp_mean_c: float | None
    cpu_util_mean: float | None
    ram_peak_mb: float | None
    process_rss_peak_mb: float | None
    disk_read_mb: float | None
    disk_write_mb: float | None
    saliency_mean: float
    saliency_std: float
    salient_area_ratio: float
    mae: float | None = None
    iou: float | None = None
    precision: float | None = None
    recall: float | None = None
    f_measure: float | None = None

    def as_row(self) -> dict[str, Any]:
        return self.__dict__.copy()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_repo_on_path() -> None:
    import sys

    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def list_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    return sorted(
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_rgb(path: Path, image_size: tuple[int, int] | None = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image_size is not None:
        image = image.resize((image_size[1], image_size[0]), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_mask(path: Path, image_size: tuple[int, int] | None = None) -> torch.Tensor:
    image = Image.open(path).convert("L")
    if image_size is not None:
        image = image.resize((image_size[1], image_size[0]), Image.NEAREST)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def save_saliency(path: Path, saliency: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = saliency.detach().float().cpu()
    if tensor.ndim == 3:
        tensor = tensor[0]
    arr = (tensor.clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_difference(path: Path, left: torch.Tensor, right: torch.Tensor) -> None:
    diff = (right.detach().float().cpu() - left.detach().float().cpu()).abs()
    save_saliency(path, diff)


def saliency_stats(saliency: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    saliency = saliency.detach().float().cpu()
    return {
        "saliency_mean": float(saliency.mean().item()),
        "saliency_std": float(saliency.std(unbiased=False).item()),
        "salient_area_ratio": float((saliency >= threshold).float().mean().item()),
    }


def supervised_metrics(
    saliency: torch.Tensor,
    mask: torch.Tensor | None,
    threshold: float = 0.5,
    beta2: float = 0.3,
) -> dict[str, float | None]:
    if mask is None:
        return {
            "mae": None,
            "iou": None,
            "precision": None,
            "recall": None,
            "f_measure": None,
        }
    pred = saliency.detach().float().cpu().clamp(0, 1)
    gt = mask.detach().float().cpu().clamp(0, 1)
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = F.interpolate(pred.unsqueeze(0), size=gt.shape[-2:], mode="bilinear", align_corners=False)[0]
    pred_bin = pred >= threshold
    gt_bin = gt >= threshold
    tp = float((pred_bin & gt_bin).sum().item())
    fp = float((pred_bin & ~gt_bin).sum().item())
    fn = float((~pred_bin & gt_bin).sum().item())
    union = tp + fp + fn
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f_measure = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
    return {
        "mae": float(torch.mean(torch.abs(pred - gt)).item()),
        "iou": tp / (union + 1e-8),
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure,
    }


def aggregate(values: Iterable[dict[str, float | None]]) -> dict[str, float | None]:
    rows = list(values)
    if not rows:
        return {}
    keys = set().union(*(row.keys() for row in rows))
    out: dict[str, float | None] = {}
    for key in keys:
        nums = [row[key] for row in rows if row.get(key) is not None and not math.isnan(float(row[key]))]
        out[key] = float(np.mean(nums)) if nums else None
    return out


def latency_summary(latencies_s: list[float]) -> dict[str, float]:
    if not latencies_s:
        return {"latency_mean_ms": 0.0, "latency_p50_ms": 0.0, "latency_p95_ms": 0.0, "fps": 0.0}
    arr = np.asarray(latencies_s, dtype=np.float64)
    total = float(arr.sum())
    return {
        "latency_mean_ms": float(arr.mean() * 1000.0),
        "latency_p50_ms": float(np.percentile(arr, 50) * 1000.0),
        "latency_p95_ms": float(np.percentile(arr, 95) * 1000.0),
        "fps": float(len(arr) / total) if total > 0 else 0.0,
    }


def init_wandb(
    project: str,
    name: str,
    mode: str,
    config: dict[str, Any],
    tags: list[str] | None = None,
):
    if mode == "disabled":
        return None
    import wandb

    return wandb.init(project=project, name=name, mode=mode, config=config, tags=tags or [])


def log_wandb(run: Any, data: dict[str, Any]) -> None:
    if run is not None:
        run.log(data)


def finish_wandb(run: Any) -> None:
    if run is not None:
        run.finish()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_comparison_csv(path: Path, summaries: list[RunSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [summary.as_row() for summary in summaries]
    fieldnames = list(RunSummary.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start

