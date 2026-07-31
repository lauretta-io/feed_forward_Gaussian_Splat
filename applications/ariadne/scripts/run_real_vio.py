"""Run a production VIO backend against a real ARIADNE dataset sequence."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import cast

from ariadne.backends import (
    ExternalVioResult,
    OpenVinsAdapter,
    OrbSlam3Adapter,
    apply_euroc_stereo_row_correction,
    diagnose_euroc_stereo_direction,
    diagnose_s3e_sensor_contract,
    evaluate_local_alignment_sensitivity,
    evaluate_orientation_proxy,
    evaluate_rtk_lever_arm_sensitivity,
    export_euroc_ros1_bag,
    export_s3e_euroc_window,
    prepare_openvins_s3e_config,
    prepare_orbslam3_s3e_settings,
    read_s3e_imu_orientation_reference,
    reanalyze_vio_artifacts,
    swap_euroc_stereo_files,
)
from ariadne.datasets import DatasetEvaluation
from ariadne.evaluation import log_evaluation_to_wandb
from ariadne.replay import D2SlamReplaySource, GroundTruthPose, read_ground_truth_poses

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_D2_ROOT = ROOT / "datasets/ariadne/d2slam/extracted/tum_corr"
DEFAULT_S3E_ROOT = ROOT / "datasets/ariadne/s3e/S3Ev1"
DEFAULT_BACKENDS = ROOT / ".cache/ariadne/backends"


def _attach_s3e_diagnostics(
    result: ExternalVioResult,
    *,
    bag: Path,
    agent: str,
    truth: tuple[GroundTruthPose, ...],
    start_timestamp_ns: int,
    end_timestamp_ns: int,
    independent_of_vio: bool,
) -> tuple[ExternalVioResult, int]:
    orientation_reference = read_s3e_imu_orientation_reference(
        bag,
        agent,
        start_timestamp_ns=start_timestamp_ns,
        end_timestamp_ns=end_timestamp_ns,
    )
    metrics = dict(result.metrics)
    metrics.update(
        evaluate_orientation_proxy(
            result.trajectory,
            truth,
            orientation_reference,
            independent_of_vio=independent_of_vio,
        )
    )
    metrics.update(
        evaluate_rtk_lever_arm_sensitivity(
            result.trajectory,
            truth,
            orientation_reference,
            orientation_independent_of_vio=independent_of_vio,
        )
    )
    metrics.update(
        evaluate_local_alignment_sensitivity(
            result.trajectory,
            truth,
        )
    )
    return replace(result, metrics=metrics), len(orientation_reference)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("d2slam", "s3e"), default="d2slam")
    parser.add_argument("--backend", choices=("openvins", "orbslam3"), required=True)
    parser.add_argument(
        "--vio-mode",
        choices=("stereo", "stereo-inertial"),
        default="stereo-inertial",
        help="ORB-SLAM3 sensor mode; OpenVINS remains stereo-inertial",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_D2_ROOT)
    parser.add_argument("--s3e-root", type=Path, default=DEFAULT_S3E_ROOT)
    parser.add_argument("--agent", choices=("Alpha", "Bob", "Carol"), default="Alpha")
    parser.add_argument("--backend-root", type=Path, default=DEFAULT_BACKENDS)
    parser.add_argument("--sequence", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--target-ate-m", type=float, default=0.1)
    parser.add_argument(
        "--stereo-baseline-scale",
        type=float,
        default=1.0,
        help="controlled S3E Camera.bf multiplier for metric-scale calibration",
    )
    parser.add_argument(
        "--imu-fast-init",
        action="store_true",
        help="enable the ORB-SLAM3 S3E fast IMU initialization ablation",
    )
    parser.add_argument(
        "--orb-feature-profile",
        choices=("balanced", "high-recall"),
        default="balanced",
        help="bounded S3E ORB extraction profile",
    )
    parser.add_argument(
        "--orb-deterministic-runtime",
        action="store_true",
        help=(
            "pin ORB-SLAM3 and numeric libraries to one CPU for a controlled "
            "reproducibility ablation"
        ),
    )
    parser.add_argument(
        "--orb-sync-local-mapping",
        action="store_true",
        help=(
            "serialize ORB-SLAM3 tracking against local mapping and disable dataset "
            "real-time pacing for a controlled offline reproducibility ablation"
        ),
    )
    parser.add_argument(
        "--swap-stereo-input",
        action="store_true",
        help="swap S3E left/right image payloads for a reversed-disparity ablation",
    )
    parser.add_argument(
        "--right-image-shift-y-px",
        type=float,
        default=0.0,
        help="bounded S3E ORB vertical correction applied to the right image",
    )
    parser.add_argument(
        "--auto-stereo-geometry",
        "--auto-stereo-row-correction",
        dest="auto_stereo_geometry",
        action="store_true",
        help="repair measured S3E stereo order and row geometry before ORB-SLAM3",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"), default="offline"
    )
    parser.add_argument("--wandb-project", default="gaussiansplat_test")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="ariadne-real-vio")
    parser.add_argument(
        "--keep-export",
        action="store_true",
        help="retain the generated EuRoC image window after an S3E run",
    )
    parser.add_argument(
        "--reanalyze-existing",
        action="store_true",
        help="recompute an existing S3E trajectory report without replaying the backend",
    )
    return parser.parse_args()


def _run_d2slam(
    args: argparse.Namespace, output_dir: Path
) -> tuple[ExternalVioResult, dict[str, object]]:
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
            mode=args.vio_mode,
            deterministic_runtime=args.orb_deterministic_runtime,
            sync_local_mapping=args.orb_sync_local_mapping,
            timeout_seconds=args.timeout_seconds,
        )
    return result, {
        "dataset": f"d2slam-{args.sequence}",
        "agents": (batch.agent_id,),
        "warnings": batch.warnings,
        "source": str(batch.source_path),
        "frames": len(batch.primary_images),
        "imu_samples": len(batch.imu_samples),
        "orb_deterministic_runtime": args.orb_deterministic_runtime,
        "orb_sync_local_mapping": args.orb_sync_local_mapping,
        "temporary_export_removed": False,
    }


def _run_s3e(
    args: argparse.Namespace, output_dir: Path
) -> tuple[ExternalVioResult, dict[str, object]]:
    playground = args.s3e_root / "S3E_Playground_2"
    bag = playground / "S3E_Playground_2.db3"
    calibration = args.s3e_root / "Calibration" / f"{args.agent.lower()}.yaml"
    truth = read_ground_truth_poses(
        playground / f"{args.agent.lower()}_gt.txt",
        orientation_available=False,
    )
    settings: Path
    if args.backend == "orbslam3":
        settings = prepare_orbslam3_s3e_settings(
            calibration,
            output_dir / "orbslam3-settings.yaml",
            stereo_baseline_scale=args.stereo_baseline_scale,
            imu_fast_init=args.imu_fast_init,
            orb_feature_profile=args.orb_feature_profile,
        )
    else:
        settings = prepare_openvins_s3e_config(
            calibration,
            args.backend_root / "openvins_d2_config",
            output_dir / "openvins-config",
        )
    cache_root = ROOT / ".cache/ariadne/tmp"
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_export:
        sequence = output_dir / "euroc"
    else:
        temporary = tempfile.TemporaryDirectory(prefix="s3e-euroc-", dir=cache_root)
        sequence = Path(temporary.name) / "euroc"
    try:
        export = export_s3e_euroc_window(
            bag,
            args.agent,
            sequence,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            swap_stereo=args.swap_stereo_input,
            right_image_vertical_shift_px=args.right_image_shift_y_px,
        )
        selected_truth = tuple(
            pose
            for pose in truth
            if export.start_timestamp_ns - 1_000_000_000
            <= pose.timestamp.monotonic_ns
            <= export.end_timestamp_ns + 1_000_000_000
        )
        sensor_contract = diagnose_s3e_sensor_contract(
            bag,
            args.agent,
            start_timestamp_ns=export.start_timestamp_ns,
            end_timestamp_ns=export.end_timestamp_ns,
        )
        if not int(sensor_contract["s3e_sensor_contract_healthy"]):
            raise RuntimeError(
                "S3E sensor preflight failed; refusing to run an accuracy benchmark"
            )
        stereo_geometry: dict[str, float | int] = {}
        if args.backend == "orbslam3":
            stereo_geometry = diagnose_euroc_stereo_direction(sequence)
            if not int(stereo_geometry["stereo_disparity_direction_healthy"]):
                if not args.auto_stereo_geometry:
                    raise RuntimeError(
                        "S3E stereo input has insufficient positive-disparity support; "
                        "correct camera ordering before running ORB-SLAM3"
                    )
                timestamps = [
                    int(value)
                    for value in export.times_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if value.strip()
                ]
                swap_euroc_stereo_files(
                    sequence / "mav0/cam0/data",
                    sequence / "mav0/cam1/data",
                    timestamps,
                )
                stereo_geometry = {
                    **diagnose_euroc_stereo_direction(sequence),
                    "stereo_auto_order_swapped": 1,
                }
                if not int(stereo_geometry["stereo_disparity_direction_healthy"]):
                    raise RuntimeError(
                        "S3E automatic camera-order repair did not recover "
                        "positive disparity"
                    )
            elif args.auto_stereo_geometry:
                stereo_geometry["stereo_auto_order_swapped"] = 0
            if args.auto_stereo_geometry and abs(
                float(stereo_geometry["stereo_vertical_offset_median_px"])
            ) > 1.0:
                uncorrected_geometry = stereo_geometry
                correction = {
                    "x_slope": float(
                        uncorrected_geometry["stereo_row_model_x_slope"]
                    ),
                    "y_slope": float(
                        uncorrected_geometry["stereo_row_model_y_slope"]
                    ),
                    "intercept_px": float(
                        uncorrected_geometry["stereo_row_model_intercept_px"]
                    ),
                }
                timestamps = [
                    int(value)
                    for value in export.times_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if value.strip()
                ]
                apply_euroc_stereo_row_correction(
                    sequence / "mav0/cam1/data",
                    timestamps,
                    **correction,
                )
                stereo_geometry = {
                    **diagnose_euroc_stereo_direction(sequence),
                    "stereo_uncorrected_vertical_offset_median_px": (
                        uncorrected_geometry["stereo_vertical_offset_median_px"]
                    ),
                    "stereo_uncorrected_row_model_residual_abs_p95_px": (
                        uncorrected_geometry[
                            "stereo_row_model_residual_abs_p95_px"
                        ]
                    ),
                    "stereo_applied_row_model_x_slope": correction["x_slope"],
                    "stereo_applied_row_model_y_slope": correction["y_slope"],
                    "stereo_applied_row_model_intercept_px": correction[
                        "intercept_px"
                    ],
                    "stereo_auto_order_swapped": uncorrected_geometry.get(
                        "stereo_auto_order_swapped",
                        0,
                    ),
                    "stereo_auto_row_correction_applied": 1,
                }
            elif args.auto_stereo_geometry:
                stereo_geometry["stereo_auto_row_correction_applied"] = 0
        if args.backend == "orbslam3":
            orb_root = args.backend_root / "ORB_SLAM3"
            result = OrbSlam3Adapter().run_euroc(
                sequence=sequence,
                times=export.times_path,
                truth=selected_truth,
                executable=ROOT / "applications/ariadne/scripts/run_orbslam3.sh",
                vocabulary=orb_root / "Vocabulary/ORBvoc.txt",
                settings=settings,
                output_dir=output_dir,
                mode=args.vio_mode,
                deterministic_runtime=args.orb_deterministic_runtime,
                sync_local_mapping=args.orb_sync_local_mapping,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            ros1_bag = export_euroc_ros1_bag(
                sequence,
                sequence.parent / "s3e-openvins.bag",
            )
            result = OpenVinsAdapter().run(
                bag=ros1_bag,
                config=settings,
                truth=selected_truth,
                output_dir=output_dir,
                launcher=(str(ROOT / "applications/ariadne/scripts/run_openvins.sh"),),
                launch_target=(
                    str(ROOT / "applications/ariadne/configs/vio/openvins_d2.launch"),
                ),
                timeout_seconds=args.timeout_seconds,
            )
        result, orientation_reference_sample_count = _attach_s3e_diagnostics(
            result,
            bag=bag,
            agent=args.agent,
            truth=selected_truth,
            start_timestamp_ns=export.start_timestamp_ns,
            end_timestamp_ns=export.end_timestamp_ns,
            independent_of_vio=args.backend == "orbslam3" and args.vio_mode == "stereo",
        )
        result = replace(
            result,
            metrics={**result.metrics, **sensor_contract, **stereo_geometry},
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
    return result, {
        "dataset": "s3e-playground-2",
        "agents": (args.agent,),
        "warnings": (),
        "source": str(bag),
        "frames": export.stereo_pair_count,
        "imu_samples": export.imu_sample_count,
        "orientation_reference_samples": orientation_reference_sample_count,
        "orientation_reference_source": "s3e_imu_ahrs_proxy",
        "compressed_image_bytes": export.compressed_image_bytes,
        "window_start_timestamp_ns": export.start_timestamp_ns,
        "window_end_timestamp_ns": export.end_timestamp_ns,
        "start_frame": args.start_frame,
        "requested_max_frames": args.max_frames,
        "temporary_export_removed": not args.keep_export,
        "settings": str(settings),
        "vio_mode": (
            args.vio_mode if args.backend == "orbslam3" else "stereo-inertial"
        ),
        "stereo_baseline_scale": args.stereo_baseline_scale,
        "imu_fast_init": args.imu_fast_init,
        "orb_feature_profile": args.orb_feature_profile,
        "orb_deterministic_runtime": args.orb_deterministic_runtime,
        "orb_sync_local_mapping": args.orb_sync_local_mapping,
        "swap_stereo_input": args.swap_stereo_input,
        "right_image_shift_y_px": args.right_image_shift_y_px,
        "auto_stereo_geometry": args.auto_stereo_geometry,
        **(
            {
                "openvins_initializer": "static",
                "openvins_static_excitation_threshold_mps2": 1.0,
                "openvins_online_camera_calibration": False,
            }
            if args.backend == "openvins"
            else {}
        ),
    }


def _reanalyze_s3e(
    args: argparse.Namespace, output_dir: Path
) -> tuple[ExternalVioResult, dict[str, object]]:
    report_path = output_dir / "evaluation.json"
    if not report_path.is_file():
        raise ValueError(f"existing evaluation report does not exist: {report_path}")
    existing = cast(dict[str, object], json.loads(report_path.read_text(encoding="utf-8")))
    details = cast(dict[str, object], existing["details"])
    metrics = cast(dict[str, object], existing["metrics"])
    stored_mode = str(details.get("vio_mode") or "stereo-inertial")
    if stored_mode != args.vio_mode:
        raise ValueError(
            f"existing report uses {stored_mode}, not requested mode {args.vio_mode}"
        )
    stored_scale = float(
        cast(float | int | str, details.get("stereo_baseline_scale", 1.0))
    )
    if not math.isclose(stored_scale, args.stereo_baseline_scale):
        raise ValueError(
            f"existing report uses baseline scale {stored_scale:g}, "
            f"not requested {args.stereo_baseline_scale:g}"
        )
    stored_fast_init = bool(details.get("imu_fast_init", False))
    if stored_fast_init != args.imu_fast_init:
        raise ValueError(
            f"existing report uses imu_fast_init={stored_fast_init}, "
            f"not requested {args.imu_fast_init}"
        )
    stored_feature_profile = str(
        details.get("orb_feature_profile", "balanced")
    )
    if stored_feature_profile != args.orb_feature_profile:
        raise ValueError(
            f"existing report uses ORB feature profile {stored_feature_profile}, "
            f"not requested {args.orb_feature_profile}"
        )
    stored_deterministic_runtime = bool(
        details.get("orb_deterministic_runtime", False)
    )
    if stored_deterministic_runtime != args.orb_deterministic_runtime:
        raise ValueError(
            "existing report uses orb_deterministic_runtime="
            f"{stored_deterministic_runtime}, not requested "
            f"{args.orb_deterministic_runtime}"
        )
    stored_sync_local_mapping = bool(details.get("orb_sync_local_mapping", False))
    if stored_sync_local_mapping != args.orb_sync_local_mapping:
        raise ValueError(
            "existing report uses orb_sync_local_mapping="
            f"{stored_sync_local_mapping}, not requested "
            f"{args.orb_sync_local_mapping}"
        )
    stored_swap_stereo_input = bool(details.get("swap_stereo_input", False))
    if stored_swap_stereo_input != args.swap_stereo_input:
        raise ValueError(
            f"existing report uses swap_stereo_input={stored_swap_stereo_input}, "
            f"not requested {args.swap_stereo_input}"
        )
    stored_right_image_shift_y_px = float(
        cast(float | int | str, details.get("right_image_shift_y_px", 0.0))
    )
    if not math.isclose(stored_right_image_shift_y_px, args.right_image_shift_y_px):
        raise ValueError(
            f"existing report uses right image shift "
            f"{stored_right_image_shift_y_px:g}px, not requested "
            f"{args.right_image_shift_y_px:g}px"
        )
    stored_auto_stereo_geometry = bool(
        details.get(
            "auto_stereo_geometry",
            details.get("auto_stereo_row_correction", False),
        )
    )
    if stored_auto_stereo_geometry != args.auto_stereo_geometry:
        raise ValueError(
            "existing report uses auto_stereo_geometry="
            f"{stored_auto_stereo_geometry}, not requested "
            f"{args.auto_stereo_geometry}"
        )

    def artifact_path(key: str) -> Path:
        path = Path(str(details[key]))
        return path if path.is_absolute() else ROOT / path

    playground = args.s3e_root / "S3E_Playground_2"
    bag = playground / "S3E_Playground_2.db3"
    truth = read_ground_truth_poses(
        playground / f"{args.agent.lower()}_gt.txt",
        orientation_available=False,
    )
    result = reanalyze_vio_artifacts(
        backend=str(details["backend"]),
        trajectory_path=artifact_path("trajectory"),
        truth=truth,
        stdout_path=artifact_path("stdout"),
        stderr_path=artifact_path("stderr"),
        return_code=int(cast(float | int | str, metrics["return_code"])),
        elapsed_seconds=float(cast(float | int | str, metrics["elapsed_seconds"])),
        command=tuple(str(value) for value in cast(list[object], details["command"])),
    )
    start_timestamp_ns = int(
        cast(float | int | str, details.get("window_start_timestamp_ns", 0))
    )
    end_timestamp_ns = int(
        cast(float | int | str, details.get("window_end_timestamp_ns", 0))
    )
    if (not start_timestamp_ns or not end_timestamp_ns) and result.trajectory:
        start_timestamp_ns = result.trajectory[0].timestamp_ns
        end_timestamp_ns = result.trajectory[-1].timestamp_ns
    result, orientation_reference_sample_count = _attach_s3e_diagnostics(
        result,
        bag=bag,
        agent=args.agent,
        truth=truth,
        start_timestamp_ns=start_timestamp_ns,
        end_timestamp_ns=end_timestamp_ns,
        independent_of_vio=args.vio_mode == "stereo",
    )
    if start_timestamp_ns and end_timestamp_ns:
        result = replace(
            result,
            metrics={
                **result.metrics,
                **diagnose_s3e_sensor_contract(
                    bag,
                    args.agent,
                    start_timestamp_ns=start_timestamp_ns,
                    end_timestamp_ns=end_timestamp_ns,
                ),
            },
        )
    passthrough = {
        key: value
        for key, value in details.items()
        if key
        not in {
            "backend",
            "command",
            "detail",
            "source",
            "stderr",
            "stdout",
            "trajectory",
        }
    }
    passthrough["reanalyzed_existing_artifacts"] = True
    passthrough["vio_mode"] = args.vio_mode
    passthrough["stereo_baseline_scale"] = args.stereo_baseline_scale
    passthrough["imu_fast_init"] = args.imu_fast_init
    passthrough["orb_feature_profile"] = args.orb_feature_profile
    passthrough["orb_deterministic_runtime"] = args.orb_deterministic_runtime
    passthrough["orb_sync_local_mapping"] = args.orb_sync_local_mapping
    passthrough["swap_stereo_input"] = args.swap_stereo_input
    passthrough["right_image_shift_y_px"] = args.right_image_shift_y_px
    passthrough["auto_stereo_geometry"] = args.auto_stereo_geometry
    passthrough["orientation_reference_samples"] = orientation_reference_sample_count
    passthrough["orientation_reference_source"] = "s3e_imu_ahrs_proxy"
    if args.backend == "openvins":
        passthrough["openvins_initializer"] = "static"
        passthrough["openvins_static_excitation_threshold_mps2"] = 1.0
        passthrough["openvins_online_camera_calibration"] = False
    warnings = tuple(
        str(warning)
        for warning in cast(list[object], existing.get("warnings", []))
        if not str(warning).startswith(
            (
                "aligned VIO ATE ",
                "Local alignment sensitivity ",
                "RTK lever-arm sensitivity ",
                "S3E RTK ground truth ",
                "The fitted metric-scale correction ",
                "Threshold-held causal Sim(3) load ",
                "Native-observation causal Sim(3) ",
                "Past-segment causal native-observation Sim(3) ",
                "Fixed-lag native-observation Sim(3) ",
                "Adaptive fixed-lag native-observation Sim(3) ",
            )
        )
    )
    return result, {
        "dataset": str(existing["dataset"]),
        "agents": tuple(str(agent) for agent in cast(list[object], existing["agents"])),
        "warnings": warnings,
        "source": str(details["source"]),
        **passthrough,
    }


def main() -> int:
    args = _arguments()
    if not math.isfinite(args.target_ate_m) or args.target_ate_m <= 0:
        raise ValueError("target ATE must be finite and positive")
    if (
        not math.isfinite(args.stereo_baseline_scale)
        or args.stereo_baseline_scale <= 0
    ):
        raise ValueError("stereo baseline scale must be finite and positive")
    if args.stereo_baseline_scale != 1.0 and (
        args.dataset != "s3e" or args.backend != "orbslam3"
    ):
        raise ValueError("stereo baseline scaling is only supported for S3E ORB-SLAM3")
    if args.imu_fast_init and (
        args.dataset != "s3e"
        or args.backend != "orbslam3"
        or args.vio_mode != "stereo-inertial"
    ):
        raise ValueError("fast IMU initialization requires S3E ORB-SLAM3 stereo-inertial")
    if args.orb_feature_profile != "balanced" and (
        args.dataset != "s3e" or args.backend != "orbslam3"
    ):
        raise ValueError("ORB feature profiles are only supported for S3E ORB-SLAM3")
    if args.orb_deterministic_runtime and args.backend != "orbslam3":
        raise ValueError("deterministic ORB runtime requires the ORB-SLAM3 backend")
    if args.orb_sync_local_mapping and args.backend != "orbslam3":
        raise ValueError("local-mapping synchronization requires the ORB-SLAM3 backend")
    if args.swap_stereo_input and (
        args.dataset != "s3e" or args.backend != "orbslam3"
    ):
        raise ValueError("stereo input swapping is only supported for S3E ORB-SLAM3")
    if (
        not math.isfinite(args.right_image_shift_y_px)
        or abs(args.right_image_shift_y_px) > 100.0
    ):
        raise ValueError("right image shift must be finite and within 100 pixels")
    if args.right_image_shift_y_px != 0.0 and (
        args.dataset != "s3e" or args.backend != "orbslam3"
    ):
        raise ValueError("right image shifting is only supported for S3E ORB-SLAM3")
    if args.auto_stereo_geometry and (
        args.dataset != "s3e" or args.backend != "orbslam3"
    ):
        raise ValueError("automatic stereo geometry is only supported for S3E ORB-SLAM3")
    if args.auto_stereo_geometry and args.right_image_shift_y_px != 0.0:
        raise ValueError("automatic and manual stereo geometry cannot be combined")
    if args.backend == "openvins" and args.vio_mode != "stereo-inertial":
        raise ValueError("OpenVINS does not support the ORB-SLAM3 stereo-only mode")
    if args.reanalyze_existing and args.dataset != "s3e":
        raise ValueError("existing-artifact reanalysis currently supports S3E")
    dataset_name = (
        f"d2slam-{args.sequence}" if args.dataset == "d2slam" else f"s3e-{args.agent.lower()}"
    )
    if args.backend == "orbslam3":
        mode_suffix = "-stereo" if args.vio_mode == "stereo" else ""
        baseline_suffix = (
            f"-bf-{args.stereo_baseline_scale:g}"
            if args.stereo_baseline_scale != 1.0
            else ""
        )
        fast_init_suffix = "-fast-init" if args.imu_fast_init else ""
        feature_suffix = (
            "-high-recall" if args.orb_feature_profile == "high-recall" else ""
        )
        deterministic_suffix = (
            "-deterministic" if args.orb_deterministic_runtime else ""
        )
        mapping_sync_suffix = (
            "-mapping-sync" if args.orb_sync_local_mapping else ""
        )
        stereo_order_suffix = "-stereo-swapped" if args.swap_stereo_input else ""
        vertical_shift_suffix = (
            f"-right-y-{args.right_image_shift_y_px:g}"
            if args.right_image_shift_y_px != 0.0
            else ""
        )
        auto_geometry_suffix = (
            "-auto-geometry" if args.auto_stereo_geometry else ""
        )
        backend_directory = (
            f"orbslam3{mode_suffix}{baseline_suffix}{fast_init_suffix}"
            f"{feature_suffix}{deterministic_suffix}{mapping_sync_suffix}"
            f"{stereo_order_suffix}"
            f"{vertical_shift_suffix}{auto_geometry_suffix}"
        )
    else:
        backend_directory = args.backend
    if args.start_frame:
        backend_directory = f"{backend_directory}-start-{args.start_frame}"
    output_dir = args.output_dir or (
        ROOT / f"outputs/ariadne/real_vio/{dataset_name}/{backend_directory}"
    )
    if args.reanalyze_existing:
        result, context = _reanalyze_s3e(args, output_dir)
    elif args.dataset == "d2slam":
        result, context = _run_d2slam(args, output_dir)
    else:
        result, context = _run_s3e(args, output_dir)
    metrics = dict(result.metrics)
    ate_m = float(metrics["ate_rmse_m"])
    target_met = result.status == "passed" and math.isfinite(ate_m) and ate_m <= args.target_ate_m
    metrics["target_ate_m"] = args.target_ate_m
    metrics["target_ate_met"] = int(target_met)
    metrics["backend_process_status"] = result.status
    warnings = list(cast(tuple[str, ...], context["warnings"]))
    if not int(metrics.get("orientation_reference_available", 1)):
        proxy_relationship = (
            "independent of the stereo-only estimator"
            if int(metrics.get("orientation_proxy_independent_of_vio", 0))
            else "non-independent because the VIO estimator consumes the same IMU"
        )
        warnings.append(
            "S3E RTK ground truth is position-only; real orientation and SE(3) "
            "correction-load metrics are not evaluated. IMU/AHRS orientation is "
            f"reported only as a consistency proxy that is {proxy_relationship}"
        )
    if int(metrics.get("lever_arm_sensitivity_matched_pose_count", 0)) >= 10:
        warnings.append(
            "RTK lever-arm sensitivity fits a bounded offset on evaluation data; "
            "it is an optimistic diagnostic, not physical calibration or a corrected score"
        )
    if int(metrics.get("local_alignment_sensitivity_unique_pose_count", 0)) >= 3:
        warnings.append(
            "Local alignment sensitivity includes future-window offline fits and "
            "zero-latency causal fits to RTK-interpolated scoring anchors; both are "
            "lower bounds, not deployable schedules or corrected scores"
        )
    if int(metrics.get("causal_sim3_target_reachable_with_tested_cadences", 0)):
        warnings.append(
            "Threshold-held causal Sim(3) load separates continuous ideal-anchor "
            "ingress and Intelligence fitting from Wingman correction transmission; "
            "it remains a ground-truth-derived lower bound and is not claim eligible"
        )
    if "causal_native_rtk_sim3_target_met" in metrics:
        warnings.append(
            "Native-observation causal Sim(3) uses only measured S3E RTK timestamps; "
            f"its {float(metrics['causal_native_rtk_sim3_ate_m']):.3f} m ATE "
            f"{'meets' if int(metrics['causal_native_rtk_sim3_target_met']) else 'misses'} "
            "the target and remains position-only, zero-latency sensitivity evidence"
        )
    if "causal_segment_hold_native_rtk_sim3_target_met" in metrics:
        target_outcome = (
            "meets"
            if int(metrics["causal_segment_hold_native_rtk_sim3_target_met"])
            else "misses"
        )
        target_horizon_s = float(
            metrics[
                "causal_segment_hold_native_rtk_maximum_target_horizon_seconds"
            ]
        )
        required_observation_rate = float(
            metrics[
                "causal_segment_hold_native_rtk_minimum_observation_rate_per_minute"
            ]
        )
        timing_boundary = (
            f"the tested target horizon is {target_horizon_s:.1f} s, requiring "
            f"{required_observation_rate:.0f} observations/min"
            if target_horizon_s > 0
            else "no tested horizon passes the accuracy and scale gates"
        )
        warnings.append(
            "Past-segment causal native-observation Sim(3) resets at each measured "
            "RTK position and holds an exponentially weighted transform from prior "
            "segments; "
            f"its {float(metrics['causal_segment_hold_native_rtk_sim3_ate_m']):.3f} m "
            f"ATE {target_outcome} "
            f"the live position target; {timing_boundary}, and the result remains "
            "ineligible for a full-pose claim"
        )
    if "fixed_lag_native_rtk_sim3_target_met" in metrics:
        warnings.append(
            "Fixed-lag native-observation Sim(3) finalizes past trajectory segments "
            "after the next measured RTK endpoint; "
            f"its {float(metrics['fixed_lag_native_rtk_sim3_ate_m']):.3f} m ATE "
            f"{'meets' if int(metrics['fixed_lag_native_rtk_sim3_target_met']) else 'misses'} "
            "the target but is not a live-pose correction or deployment claim"
        )
    if "adaptive_fixed_lag_native_rtk_sim3_target_met" in metrics:
        warnings.append(
            "Adaptive fixed-lag native-observation Sim(3) coalesces at most two "
            "RTK intervals using measured scale/direction changes; "
            f"its {float(metrics['adaptive_fixed_lag_native_rtk_sim3_ate_m']):.3f} m "
            "ATE remains delayed map evidence, not reduced RTK ingress or live pose"
        )
    if int(metrics.get("geometric_divergence_detected", 0)):
        warnings.append(
            "The fitted metric-scale correction is outside the bounded 0.25x-4x "
            "plausibility range; correction scheduling is disabled and relocalization "
            "is required"
        )
    if args.dataset == "s3e" and not int(
        metrics.get("s3e_sensor_contract_healthy", 0)
    ):
        warnings.append(
            "The bounded S3E sensor preflight contract failed; inspect timestamp, "
            "stereo synchronization, IMU cadence, and AHRS/gyro diagnostics before "
            "interpreting backend accuracy"
        )
    if result.status == "passed" and not target_met:
        warnings.append(
            f"aligned VIO ATE {ate_m:.3f} m exceeds the {args.target_ate_m:.3f} m target"
        )
    evaluation = DatasetEvaluation(
        dataset=str(context["dataset"]),
        status="passed" if target_met else "failed",
        agents=cast(tuple[str, ...], context["agents"]),
        modalities=(
            ("stereo", "ground-truth")
            if args.backend == "orbslam3" and args.vio_mode == "stereo"
            else ("stereo", "imu", "ground-truth")
        ),
        metrics=metrics,
        warnings=tuple(warnings),
        details={
            "backend": result.backend,
            "command": list(result.command),
            "source": context["source"],
            "trajectory": str(result.trajectory_path),
            "stdout": str(result.stdout_path),
            "stderr": str(result.stderr_path),
            "detail": result.detail,
            **{
                key: value
                for key, value in context.items()
                if key not in {"dataset", "agents", "warnings", "source"}
            },
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
        name=f"ariadne/{dataset_name}/{args.backend}",
        group=args.wandb_group,
        tags=["real-vio", args.backend, dataset_name],
        job_type="model-benchmark",
    )
    print(json.dumps({"report": str(report), "status": evaluation.status, "wandb_url": url}))
    return int(evaluation.status != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
