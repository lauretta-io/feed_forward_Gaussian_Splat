from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.resplat_u2net_saliency.common import (
    Timer,
    ensure_repo_on_path,
    finish_wandb,
    init_wandb,
    latency_summary,
    log_wandb,
    write_json,
)
from experiments.resplat_u2net_saliency.monitor import HardwareMonitor

ensure_repo_on_path()

from src.misc.weave_tools import finish_weave, init_weave  # noqa: E402
from scripts.infer_colmap import (  # noqa: E402
    MODEL_PRESETS,
    build_batch,
    build_model,
    compute_target_shape,
    load_and_preprocess_images,
    load_colmap_scene,
    select_context_views,
    subset_scene_data,
)


def _resolve_scene(args: argparse.Namespace) -> tuple[Path, str]:
    if args.scene_path:
        scene_path = Path(args.scene_path)
        return scene_path, scene_path.name
    if not args.data_dir or not args.scene_name:
        raise ValueError("Specify either --scene-path or --data-dir with --scene-name")
    return Path(args.data_dir) / args.scene_name, args.scene_name


def _camera_normalization(pivotal_pose: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    camera_norm_matrix = torch.inverse(pivotal_pose)
    return torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)


def _target_resolution(args: argparse.Namespace, scene_data: dict) -> tuple[int, int]:
    first_img = Image.open(scene_data["image_paths"][0])
    orig_w, orig_h = first_img.size
    return compute_target_shape(orig_h, orig_w, args.max_resolution, args.image_shape)


def _build_context_batch(
    args: argparse.Namespace,
    scene_data: dict,
    scene_name: str,
    context_indices: np.ndarray,
    resolution: tuple[int, int],
):
    target_h, target_w = resolution
    context_images = load_and_preprocess_images(
        [scene_data["image_paths"][i] for i in context_indices],
        target_h,
        target_w,
    )
    context_c2w = torch.tensor(scene_data["c2w"][context_indices], dtype=torch.float32)
    context_K = torch.tensor(scene_data["intrinsics"][context_indices], dtype=torch.float32)
    aligned = _camera_normalization(context_c2w[len(context_c2w) // 2 : len(context_c2w) // 2 + 1], context_c2w)
    empty_target_images = context_images[:1]
    batch = build_batch(
        context_images,
        empty_target_images,
        context_c2w,
        aligned[:1],
        context_K,
        context_K[:1],
        args.near,
        args.far,
        scene_name,
        args.device,
    )
    image_names = [scene_data["image_names"][i] for i in context_indices]
    image_paths = [scene_data["image_paths"][i] for i in context_indices]
    return batch, image_names, image_paths


def _context_groups(args: argparse.Namespace, scene_data: dict) -> list[np.ndarray]:
    num_total = len(scene_data["image_paths"])
    if args.all_frames:
        window = args.extraction_window_size or args.num_context or num_total
        return [
            np.arange(start, min(start + window, num_total))
            for start in range(0, num_total, window)
        ]
    num_context = min(args.num_context or num_total, num_total)
    return [select_context_views(scene_data["c2w"], num_context, args.context_selection)]


def extract_embeddings(args: argparse.Namespace) -> dict:
    preset = MODEL_PRESETS[args.model_preset]
    if args.checkpoint is None:
        args.checkpoint = preset["checkpoint"]
    if args.num_context is None:
        args.num_context = preset["num_context"]
    args.max_resolution = args.max_resolution or preset["max_resolution"]
    overrides = list(preset.get("overrides", [])) + list(args.overrides or [])
    if str(args.device) == "cpu":
        overrides.extend([
            "model.encoder.use_amp=false",
            "model.encoder.pt_head_amp=false",
            "model.encoder.pt_update_amp=false",
        ])

    output_dir = Path(args.output_dir)
    embedding_dir = output_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    scene_path, scene_name = _resolve_scene(args)
    scene_data = load_colmap_scene(str(scene_path), args.sparse_dir, args.images_dir)
    if args.start_frame is not None:
        scene_data = subset_scene_data(scene_data, args.start_frame, args.frame_distance)
    resolution = _target_resolution(args, scene_data)
    context_groups = _context_groups(args, scene_data)

    encoder, _decoder, data_shim = build_model(
        experiment=args.experiment,
        checkpoint=args.checkpoint,
        num_refine=0,
        image_shape=resolution,
        overrides=overrides,
        device=args.device,
        no_strict_load=True,
    )
    init_weave(args.weave_project, enabled=not args.no_weave)
    monitor = HardwareMonitor(interval_s=args.monitor_interval, gpu_index=args.gpu_index).start()
    start = time.perf_counter()
    latencies: list[float] = []
    frames = []
    all_context_indices: list[int] = []
    try:
        for group_id, context_indices in enumerate(context_groups):
            batch, image_names, image_paths = _build_context_batch(
                args,
                scene_data,
                scene_name,
                context_indices,
                resolution,
            )
            batch = data_shim(batch)
            captured: dict[str, torch.Tensor] = {}

            def hook(_module, inputs):
                captured["embedding"] = inputs[0].detach().float().cpu()

            handle = encoder.gaussian_regressor.register_forward_pre_hook(hook)
            with Timer() as timer:
                with torch.no_grad():
                    _ = encoder(batch["context"], global_step=0, deterministic=True, visualization_dump=None)
            handle.remove()
            latencies.extend([timer.elapsed / max(len(image_names), 1)] * len(image_names))
            if "embedding" not in captured:
                raise RuntimeError("Failed to capture ReSplat embedding from gaussian_regressor input")
            embeddings = captured["embedding"]
            all_context_indices.extend(int(i) for i in context_indices)
            for i, (name, image_path) in enumerate(zip(image_names, image_paths)):
                tensor = embeddings[i].contiguous()
                path = embedding_dir / f"{Path(name).stem}.pt"
                torch.save(
                    {
                        "embedding": tensor,
                        "frame_name": Path(name).stem,
                        "frame_index": int(context_indices[i]),
                        "feature_type": "resplat_fused_gaussian_regressor_input",
                        "group_id": group_id,
                    },
                    path,
                )
                frames.append(
                    {
                        "frame_name": Path(name).stem,
                        "frame_index": int(context_indices[i]),
                        "image_path": str(image_path),
                        "embedding_path": str(path),
                        "feature_type": "resplat_fused_gaussian_regressor_input",
                        "group_id": group_id,
                        "tensor_shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "numel": int(tensor.numel()),
                        "bytes": int(tensor.numel() * tensor.element_size()),
                    }
                )
    finally:
        finish_weave()
    total_runtime = time.perf_counter() - start
    hardware = monitor.stop()

    perf = latency_summary(latencies)
    manifest = {
        "scene_name": scene_name,
        "scene_path": str(scene_path),
        "checkpoint": str(args.checkpoint),
        "model_preset": args.model_preset,
        "experiment": args.experiment,
        "resolution": list(resolution),
        "num_context": len(all_context_indices),
        "context_indices": all_context_indices,
        "all_frames": bool(args.all_frames),
        "extraction_window_size": args.extraction_window_size,
        "weave_project": args.weave_project,
        "total_runtime_s": total_runtime,
        "latency": perf,
        "hardware": hardware,
        "frames": frames,
    }
    manifest_path = embedding_dir / "manifest.json"
    write_json(manifest_path, manifest)

    run = init_wandb(
        project=args.wandb_project,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config=vars(args),
        tags=["resplat", "embedding-extraction", "saliency"],
    )
    log_data = {
        "extract/frame_count": len(frames),
        "extract/total_runtime_s": total_runtime,
        "extract/embedding_bytes_total": int(sum(row["bytes"] for row in frames)),
        "extract/embedding_numel_mean": float(np.mean([row["numel"] for row in frames])),
        **{f"extract/{k}": v for k, v in perf.items()},
        **{f"extract/hardware/{k}": v for k, v in hardware.items() if v is not None},
    }
    log_wandb(run, log_data)
    finish_wandb(run)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract saved ReSplat embeddings for saliency experiments")
    parser.add_argument("--scene-path")
    parser.add_argument("--data-dir")
    parser.add_argument("--scene-name")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--images-dir", default="images_4")
    parser.add_argument("--output-dir", default="experiments/resplat_u2net_saliency/outputs")
    parser.add_argument("--model-preset", choices=list(MODEL_PRESETS.keys()), default="dl3dv_8v_256x448")
    parser.add_argument("--checkpoint")
    parser.add_argument("--experiment", default="dl3dv")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-context", type=int)
    parser.add_argument("--all-frames", action="store_true", help="Extract embeddings for every frame instead of only selected context views")
    parser.add_argument("--extraction-window-size", type=int, help="Frame window size for --all-frames extraction; defaults to --num-context/model preset")
    parser.add_argument("--context-selection", choices=["fps", "uniform"], default="fps")
    parser.add_argument("--max-resolution", type=int)
    parser.add_argument("--image-shape", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--near", type=float, default=0.5)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--frame-distance", type=int, default=10)
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument("--monitor-interval", type=float, default=0.5)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--wandb-project", default="gaussiansplat_test")
    parser.add_argument("--wandb-name", default="extract-resplat-embeddings")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="offline")
    parser.add_argument("--weave-project", default="galvin/gaussiansplat_test")
    parser.add_argument("--no-weave", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    manifest = extract_embeddings(parse_args())
    print(f"Wrote {len(manifest['frames'])} embeddings to {Path(manifest['frames'][0]['embedding_path']).parent}")
