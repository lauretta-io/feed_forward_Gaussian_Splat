from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.resplat_u2net_saliency.common import (
    RunSummary,
    Timer,
    aggregate,
    finish_wandb,
    init_wandb,
    latency_summary,
    log_wandb,
    read_json,
    saliency_stats,
    save_saliency,
    supervised_metrics,
    write_json,
)
from experiments.resplat_u2net_saliency.datasets import EmbeddingSaliencyDataset, saliency_collate
from experiments.resplat_u2net_saliency.models.u2net_resplat_backbone import ResplatBackboneU2Net
from experiments.resplat_u2net_saliency.monitor import HardwareMonitor


def _parse_image_size(values: list[int] | None) -> tuple[int, int] | None:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("--image-size expects HEIGHT WIDTH")
    return values[0], values[1]


def _parse_grid(value: str | None) -> tuple[int, int] | None:
    if value is None or value == "":
        return None
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise ValueError("--token-grid-size must look like HxW")
    return int(parts[0]), int(parts[1])


def _embedding_channels(manifest: dict) -> int | None:
    frames = manifest.get("frames", [])
    if not frames:
        return None
    shape = frames[0].get("tensor_shape", [])
    if len(shape) == 3:
        return int(shape[0])
    if len(shape) == 2:
        return int(shape[-1])
    return None


def _dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.flatten(1)
    target = target.flatten(1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    return 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()


def _preview_payload(output_dir: Path, limit: int = 4) -> dict[str, Any]:
    try:
        import wandb
    except Exception:
        return {}
    paths = sorted((output_dir / "predictions" / "resplat").glob("*.png"))[:limit]
    if not paths:
        return {}
    return {"preview/resplat": [wandb.Image(str(path), caption=path.name) for path in paths]}


def run_resplat(args: argparse.Namespace | SimpleNamespace) -> RunSummary:
    image_size = _parse_image_size(getattr(args, "image_size", None))
    output_dir = Path(args.output_dir)
    prediction_dir = output_dir / "predictions" / "resplat"
    metrics_dir = output_dir / "metrics"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(Path(args.embedding_manifest))
    dataset = EmbeddingSaliencyDataset(
        manifest,
        mask_dir=getattr(args, "mask_dir", None),
        image_size=image_size,
        max_frames=getattr(args, "max_frames", None),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=dataset.has_masks and args.epochs > 0,
        num_workers=args.num_workers,
        collate_fn=saliency_collate,
    )
    device = torch.device(args.device)
    model = ResplatBackboneU2Net(
        embedding_channels=_embedding_channels(manifest),
        adapter_channels=args.adapter_channels,
        decoder_base_channels=args.decoder_base_channels,
        token_grid_size=_parse_grid(getattr(args, "token_grid_size", None)),
    ).to(device)
    model_state = "frozen_resplat_embeddings+untrained_adapter_decoder"

    run = init_wandb(
        project=args.wandb_project,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config=vars(args) if hasattr(args, "__dict__") else {},
        tags=["resplat", "u2net", "saliency"],
    )

    if dataset.has_masks and args.epochs > 0:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(args.epochs):
            losses = []
            for batch in loader:
                embedding = batch["embedding"].to(device)
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                pred = model(embedding, output_size=image.shape[-2:])
                loss = F.binary_cross_entropy(pred, mask) + _dice_loss(pred, mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
            log_wandb(run, {"train/resplat_loss": sum(losses) / max(len(losses), 1), "epoch": epoch})
        model_state = "frozen_resplat_embeddings+trained_adapter_decoder"
    else:
        log_wandb(run, {"train/resplat_skipped_no_masks": not dataset.has_masks})

    monitor = HardwareMonitor(interval_s=args.monitor_interval, gpu_index=args.gpu_index).start()
    model.eval()
    latencies: list[float] = []
    stat_rows: list[dict[str, float | None]] = []
    metric_rows: list[dict[str, float | None]] = []
    resolution = ""
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=saliency_collate):
            embedding = batch["embedding"].to(device)
            image = batch["image"].to(device)
            with Timer() as timer:
                pred = model(embedding, output_size=image.shape[-2:])
            per_frame_latency = timer.elapsed / embedding.shape[0]
            latencies.extend([per_frame_latency] * embedding.shape[0])
            resolution = f"{image.shape[-2]}x{image.shape[-1]}"
            masks = batch["mask"]
            for i, frame_name in enumerate(batch["frame_name"]):
                sal = pred[i].detach().cpu()
                save_saliency(prediction_dir / f"{frame_name}.png", sal)
                stat_rows.append(saliency_stats(sal, args.saliency_threshold))
                mask_i = masks[i].detach().cpu() if masks is not None else None
                metric_rows.append(supervised_metrics(sal, mask_i, args.saliency_threshold))

    hardware = monitor.stop()
    stats = aggregate(stat_rows)
    sup = aggregate(metric_rows)
    perf = latency_summary(latencies)
    summary = RunSummary(
        run_name=args.wandb_name,
        variant="resplat_backbone",
        model_state=model_state,
        checkpoint=str(manifest.get("checkpoint", "")),
        frame_count=len(dataset),
        resolution=resolution,
        **perf,
        **hardware,
        saliency_mean=float(stats.get("saliency_mean") or 0.0),
        saliency_std=float(stats.get("saliency_std") or 0.0),
        salient_area_ratio=float(stats.get("salient_area_ratio") or 0.0),
        mae=sup.get("mae"),
        iou=sup.get("iou"),
        precision=sup.get("precision"),
        recall=sup.get("recall"),
        f_measure=sup.get("f_measure"),
    )
    write_json(metrics_dir / "resplat_summary.json", summary.as_row())
    log_wandb(run, {f"resplat/{k}": v for k, v in summary.as_row().items() if isinstance(v, (int, float)) and v is not None})
    log_wandb(run, monitor.wandb_payload("resplat/hardware"))
    log_wandb(run, _preview_payload(output_dir))
    finish_wandb(run)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReSplat embedding + U2-Net saliency")
    parser.add_argument("--embedding-manifest", default="experiments/resplat_u2net_saliency/outputs/embeddings/manifest.json")
    parser.add_argument("--mask-dir")
    parser.add_argument("--output-dir", default="experiments/resplat_u2net_saliency/outputs")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adapter-channels", type=int, default=128)
    parser.add_argument("--decoder-base-channels", type=int, default=32)
    parser.add_argument("--token-grid-size")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--image-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--saliency-threshold", type=float, default=0.5)
    parser.add_argument("--monitor-interval", type=float, default=0.5)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--wandb-project", default="gaussiansplat_test")
    parser.add_argument("--wandb-name", default="resplat-u2net")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="offline")
    return parser.parse_args()


if __name__ == "__main__":
    summary = run_resplat(parse_args())
    print(summary.as_row())
