from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.resplat_u2net_saliency.common import (
    finish_wandb,
    init_wandb,
    load_mask,
    log_wandb,
    save_difference,
    save_saliency,
    write_comparison_csv,
)
from experiments.resplat_u2net_saliency.train_u2net_baseline import run_baseline
from experiments.resplat_u2net_saliency.train_u2net_resplat_backbone import run_resplat


def _namespace(args: argparse.Namespace, **updates) -> SimpleNamespace:
    data = vars(args).copy()
    data.update(updates)
    return SimpleNamespace(**data)


def _log_visuals(run, output_dir: Path, limit: int = 4) -> None:
    if run is None:
        return
    try:
        import wandb
    except Exception:
        return
    payload = {}
    for name in ["baseline", "resplat", "difference"]:
        paths = sorted((output_dir / "predictions" / name).glob("*.png"))[:limit]
        if paths:
            payload[f"comparison/{name}"] = [wandb.Image(str(path), caption=path.name) for path in paths]
    log_wandb(run, payload)


def _make_difference_maps(output_dir: Path) -> None:
    baseline_dir = output_dir / "predictions" / "baseline"
    resplat_dir = output_dir / "predictions" / "resplat"
    diff_dir = output_dir / "predictions" / "difference"
    diff_dir.mkdir(parents=True, exist_ok=True)
    for base_path in sorted(baseline_dir.glob("*.png")):
        resplat_path = resplat_dir / base_path.name
        if not resplat_path.exists():
            continue
        base = load_mask(base_path)
        resplat = load_mask(resplat_path)
        save_difference(diff_dir / base_path.name, base, resplat)


def run_comparison(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    baseline_summary = run_baseline(
        _namespace(
            args,
            wandb_name=f"{args.wandb_name}-baseline",
            image_dir=args.image_dir,
            checkpoint=args.baseline_checkpoint,
            model_size=args.baseline_model_size,
        )
    )
    resplat_summary = run_resplat(
        _namespace(
            args,
            wandb_name=f"{args.wandb_name}-resplat",
            embedding_manifest=args.embedding_manifest,
            adapter_channels=args.adapter_channels,
            decoder_base_channels=args.decoder_base_channels,
            token_grid_size=args.token_grid_size,
        )
    )
    _make_difference_maps(output_dir)
    comparison_path = output_dir / "metrics" / "comparison.csv"
    write_comparison_csv(comparison_path, [baseline_summary, resplat_summary])

    run = init_wandb(
        project=args.wandb_project,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config=vars(args),
        tags=["comparison", "resplat", "u2net", "saliency"],
    )
    log_wandb(run, {"comparison/frame_count": baseline_summary.frame_count, "comparison/csv": str(comparison_path)})
    _log_visuals(run, output_dir)
    finish_wandb(run)
    return comparison_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare original U2-Net and ReSplat-backed U2-Net saliency")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--embedding-manifest", default="experiments/resplat_u2net_saliency/outputs/embeddings/manifest.json")
    parser.add_argument("--mask-dir")
    parser.add_argument("--output-dir", default="experiments/resplat_u2net_saliency/outputs")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--baseline-model-size", choices=["u2net", "u2netp"], default="u2net")
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
    parser.add_argument("--wandb-name", default="saliency-comparison")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="offline")
    return parser.parse_args()


if __name__ == "__main__":
    path = run_comparison(parse_args())
    print(f"Wrote {path}")
