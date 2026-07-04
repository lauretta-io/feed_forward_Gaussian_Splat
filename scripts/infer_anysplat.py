#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ANY_SPLAT_REPO = "https://github.com/QuangTran1608/AnySplat_impact"
ANY_SPLAT_COMMIT = "fe25779ce0ec2747635f6555ecedadcdd565da9e"
DEFAULT_MODEL_ID = "lhjiang/anysplat"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

from src.misc.weave_tools import finish_weave, init_weave

REQUIRED_MODULES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "huggingface_hub": "huggingface_hub",
    "einops": "einops",
    "jaxtyping": "jaxtyping",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "scipy": "scipy",
    "plyfile": "plyfile",
    "gsplat": "gsplat",
    "torch_scatter": "torch_scatter",
}

VIDEO_MODULES = {
    "matplotlib": "matplotlib",
    "imageio": "imageio",
    "skvideo": "sk-video",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AnySplat inference on a folder of images.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input .jpg/.jpeg/.png images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/anysplat/<input-folder-name>.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id or local model directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--save-ply",
        action="store_true",
        help="Write gaussians.ply.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Render and save interpolated RGB/depth videos.",
    )
    parser.add_argument(
        "--wandb-project",
        default="resplat-tests",
        help="W&B project for inference/test result logging.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional W&B entity.",
    )
    parser.add_argument(
        "--wandb-name",
        default=None,
        help="Optional W&B run name. Defaults to an AnySplat input-folder name.",
    )
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B logging mode.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=["anysplat", "inference", "smoke"],
        help="W&B run tags.",
    )
    parser.add_argument(
        "--weave-project",
        default="galvin/gaussiansplat test",
        help="Weave project initialized with weave.init(...).",
    )
    parser.add_argument(
        "--no-weave",
        action="store_true",
        help="Disable Weave initialization for this run.",
    )
    return parser.parse_args()


def ensure_modules(save_video: bool, wandb_mode: str) -> None:
    missing = [
        package
        for module, package in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module) is None
    ]
    if wandb_mode != "disabled" and importlib.util.find_spec("wandb") is None:
        missing.append("wandb")
    if save_video:
        missing.extend(
            package
            for module, package in VIDEO_MODULES.items()
            if importlib.util.find_spec(module) is None
        )
    if missing:
        missing_list = ", ".join(sorted(set(missing)))
        raise SystemExit(
            "AnySplat inference dependencies are not installed: "
            f"{missing_list}. See ANYSPLAT_INTEGRATION.md for the optional "
            "environment notes."
        )


def image_paths(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise SystemExit(f"No supported images found in {input_dir}")
    return paths


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def write_manifest(
    output_dir: Path,
    input_dir: Path,
    images: list[Path],
    model_id: str,
    device: object,
    save_ply: bool,
    save_video: bool,
    artifact_paths: dict[str, str],
    evaluation: dict[str, int | float],
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "integration": "anysplat",
        "source_repo": ANY_SPLAT_REPO,
        "source_commit": ANY_SPLAT_COMMIT,
        "model_id": model_id,
        "device": str(device),
        "input_dir": str(input_dir),
        "source_images": [str(path) for path in images],
        "save_ply": save_ply,
        "save_video": save_video,
        "evaluation": evaluation,
        "artifacts": artifact_paths,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    return config


def wandb_artifact_safe_name(name: str) -> str:
    return "".join(
        char if char.isalnum() or char in "-_." else "-"
        for char in name
    ).strip("-")


def init_wandb(args: argparse.Namespace, input_dir: Path, output_dir: Path):
    if args.wandb_mode == "disabled":
        return None

    import wandb

    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        mode=args.wandb_mode,
        name=args.wandb_name or f"anysplat/{input_dir.name}",
        tags=args.wandb_tags,
        config={
            **serializable_config(args),
            "integration": "anysplat",
            "output_dir": str(output_dir),
            "source_repo": ANY_SPLAT_REPO,
            "source_commit": ANY_SPLAT_COMMIT,
        },
    )


def log_wandb_results(
    run,
    output_dir: Path,
    images: list[Path],
    artifacts: dict[str, str],
    evaluation: dict[str, int | float],
) -> None:
    if run is None:
        return

    import wandb

    run.log(
        {
            "test/frame_count": len(images),
            "test/num_source_images": len(images),
            **{f"test/{key}": value for key, value in evaluation.items()},
            "preview/source_images": [
                wandb.Image(str(path), caption=path.name) for path in images[:8]
            ],
        }
    )

    artifact = wandb.Artifact(
        name=wandb_artifact_safe_name(f"{run.name}-anysplat-results"),
        type="test-results",
        metadata={
            "integration": "anysplat",
            "output_dir": str(output_dir),
        },
    )
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        artifact.add_file(str(manifest_path), name="manifest.json")
    for key, path in artifacts.items():
        artifact_path = Path(path)
        if artifact_path.exists():
            artifact.add_file(str(artifact_path), name=f"{key}/{artifact_path.name}")
    run.log_artifact(artifact)


def main() -> None:
    args = parse_args()
    ensure_modules(args.save_video, args.wandb_mode)

    import torch

    from src.integrations.anysplat.misc.image_io import save_interpolated_video
    from src.integrations.anysplat.model.model.anysplat import AnySplat
    from src.integrations.anysplat.model.ply_export import export_ply
    from src.integrations.anysplat.utils.image import process_image

    input_dir = args.input_dir.resolve()
    images = image_paths(input_dir)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (Path("outputs") / "anysplat" / input_dir.name).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(args, input_dir, output_dir)
    weave_initialized = init_weave(args.weave_project, not args.no_weave)

    device = resolve_device(args.device)
    try:
        model = AnySplat.from_pretrained(args.model_id).to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        image_tensors = [process_image(str(path)) for path in images]
        batch = torch.stack(image_tensors, dim=0).unsqueeze(0).to(device)
        batch_size, _, _, height, width = batch.shape

        with torch.no_grad():
            gaussians, pred_context_pose = model.inference((batch + 1) * 0.5)

        pose_path = output_dir / "predicted_poses.pt"
        torch.save(
            {
                "extrinsic": pred_context_pose["extrinsic"].detach().cpu(),
                "intrinsic": pred_context_pose["intrinsic"].detach().cpu(),
            },
            pose_path,
        )

        artifacts = {"poses": str(pose_path)}
        evaluation = {
            "num_gaussians": int(gaussians.means.shape[1]),
            "opacity_mean": float(gaussians.opacities.mean().item()),
            "opacity_min": float(gaussians.opacities.min().item()),
            "opacity_max": float(gaussians.opacities.max().item()),
            "scale_mean": float(gaussians.scales.mean().item()),
            "scale_min": float(gaussians.scales.min().item()),
            "scale_max": float(gaussians.scales.max().item()),
        }

        if args.save_ply:
            ply_path = output_dir / "gaussians.ply"
            export_ply(
                gaussians.means[0],
                gaussians.scales[0],
                gaussians.rotations[0],
                gaussians.harmonics[0],
                gaussians.opacities[0],
                ply_path,
                save_sh_dc_only=True,
            )
            artifacts["ply"] = str(ply_path)

        if args.save_video:
            rgb_path, depth_path = save_interpolated_video(
                pred_context_pose["extrinsic"],
                pred_context_pose["intrinsic"],
                batch_size,
                height,
                width,
                gaussians,
                output_dir,
                model.decoder,
            )
            artifacts["rgb_video"] = rgb_path
            artifacts["depth_video"] = depth_path

        write_manifest(
            output_dir,
            input_dir,
            images,
            args.model_id,
            device,
            args.save_ply,
            args.save_video,
            artifacts,
            evaluation,
        )
        log_wandb_results(wandb_run, output_dir, images, artifacts, evaluation)
        print(f"AnySplat outputs written to {output_dir}")
    finally:
        if weave_initialized:
            finish_weave()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
