"""Run a production VIO backend against a real ARIADNE dataset sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ariadne.backends import OpenVinsAdapter, OrbSlam3Adapter
from ariadne.datasets import DatasetEvaluation
from ariadne.evaluation import log_evaluation_to_wandb
from ariadne.replay import D2SlamReplaySource

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_D2_ROOT = ROOT / "datasets/ariadne/d2slam/extracted/tum_corr"
DEFAULT_BACKENDS = ROOT / ".cache/ariadne/backends"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("openvins", "orbslam3"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_D2_ROOT)
    parser.add_argument("--backend-root", type=Path, default=DEFAULT_BACKENDS)
    parser.add_argument("--sequence", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"), default="offline"
    )
    parser.add_argument("--wandb-project", default="gaussiansplat_test")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="ariadne-real-vio")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    output_dir = args.output_dir or (
        ROOT / f"outputs/ariadne/real_vio/d2slam-{args.sequence}/{args.backend}"
    )
    source = D2SlamReplaySource(args.dataset_root, args.sequence)
    batch = source.load(start_frame=args.start_frame, max_frames=args.max_frames)
    if args.backend == "openvins":
        config = args.backend_root / "openvins_d2_config/estimator_config.yaml"
        launcher = (str(ROOT / "applications/ariadne/scripts/run_openvins.sh"),)
        result = OpenVinsAdapter().run(
            bag=source.bag,
            config=config,
            truth=batch.ground_truth,
            output_dir=output_dir,
            launcher=launcher,
            launch_target=(str(ROOT / "applications/ariadne/configs/vio/openvins_d2.launch"),),
            timeout_seconds=args.timeout_seconds,
        )
    else:
        orb_root = args.backend_root / "ORB_SLAM3"
        result = OrbSlam3Adapter().run(
            batch=batch,
            executable=ROOT / "applications/ariadne/scripts/run_orbslam3.sh",
            vocabulary=orb_root / "Vocabulary/ORBvoc.txt",
            settings=orb_root / "Examples/Stereo-Inertial/TUM-VI.yaml",
            output_dir=output_dir,
            timeout_seconds=args.timeout_seconds,
        )
    evaluation = DatasetEvaluation(
        dataset=f"d2slam-{args.sequence}",
        status=result.status,
        agents=(batch.agent_id,),
        modalities=("stereo", "imu", "ground-truth"),
        metrics=result.metrics,
        warnings=batch.warnings,
        details={
            "backend": result.backend,
            "command": list(result.command),
            "source": str(batch.source_path),
            "trajectory": str(result.trajectory_path),
            "stdout": str(result.stdout_path),
            "stderr": str(result.stderr_path),
            "detail": result.detail,
            "frames": len(batch.primary_images),
            "imu_samples": len(batch.imu_samples),
        },
    )
    report = output_dir / "evaluation.json"
    evaluation.write_json(report)
    url = log_evaluation_to_wandb(
        evaluation,
        report,
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"ariadne/d2slam-{args.sequence}/{args.backend}",
        group=args.wandb_group,
        tags=["real-vio", args.backend, f"sequence-{args.sequence}"],
        job_type="model-benchmark",
    )
    print(json.dumps({"report": str(report), "status": result.status, "wandb_url": url}))
    return int(result.status != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
