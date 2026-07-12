"""Process-isolated OpenVINS and ORB-SLAM3 adapters."""

from __future__ import annotations

import importlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import numpy.typing as npt

from ariadne.replay import GroundTruthPose, ReplayBatch


@dataclass(frozen=True)
class TrajectoryPose:
    timestamp_ns: int
    position_m: npt.NDArray[np.float64]
    quaternion_xyzw: npt.NDArray[np.float64]


@dataclass(frozen=True)
class ExternalVioResult:
    backend: str
    status: str
    return_code: int | None
    elapsed_seconds: float
    trajectory: tuple[TrajectoryPose, ...]
    metrics: dict[str, int | float | str]
    command: tuple[str, ...]
    stdout_path: Path
    stderr_path: Path
    trajectory_path: Path
    detail: str = ""


def parse_trajectory(path: Path) -> tuple[TrajectoryPose, ...]:
    poses: list[TrajectoryPose] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = [float(value) for value in line.replace(",", " ").split()]
        if len(values) < 8:
            continue
        timestamp_ns = int(values[0]) if values[0] > 1e12 else int(values[0] * 1e9)
        poses.append(TrajectoryPose(timestamp_ns, np.asarray(values[1:4]), np.asarray(values[4:8])))
    return tuple(poses)


def _matched_positions(
    estimates: tuple[TrajectoryPose, ...],
    truth: tuple[GroundTruthPose, ...],
    tolerance_seconds: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if not estimates or not truth:
        return np.empty((0, 3)), np.empty((0, 3))
    truth_times = np.asarray([pose.timestamp.monotonic_ns for pose in truth], dtype=np.int64)
    estimated_positions: list[npt.NDArray[np.float64]] = []
    truth_positions: list[npt.NDArray[np.float64]] = []
    tolerance_ns = int(tolerance_seconds * 1e9)
    for estimate in estimates:
        insertion = int(np.searchsorted(truth_times, estimate.timestamp_ns))
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(truth)]
        nearest = min(
            candidates,
            key=lambda index: abs(int(truth_times[index]) - estimate.timestamp_ns),
        )
        if abs(int(truth_times[nearest]) - estimate.timestamp_ns) <= tolerance_ns:
            estimated_positions.append(estimate.position_m)
            truth_positions.append(truth[nearest].position_m)
    return np.asarray(estimated_positions), np.asarray(truth_positions)


def _rigid_align(
    estimated: npt.NDArray[np.float64], truth: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    estimated_center = np.mean(estimated, axis=0)
    truth_center = np.mean(truth, axis=0)
    covariance = (estimated - estimated_center).T @ (truth - truth_center)
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transpose[-1] *= -1
        rotation = right_transpose.T @ left.T
    aligned = (rotation @ (estimated - estimated_center).T).T + truth_center
    return np.asarray(aligned, dtype=np.float64)


def evaluate_trajectory(
    estimates: tuple[TrajectoryPose, ...],
    truth: tuple[GroundTruthPose, ...],
    *,
    tolerance_seconds: float = 0.6,
) -> dict[str, int | float | str]:
    estimated, matched_truth = _matched_positions(estimates, truth, tolerance_seconds)
    if len(estimated) < 3:
        return {
            "trajectory_pose_count": len(estimates),
            "matched_pose_count": len(estimated),
            "ate_rmse_m": float("nan"),
            "rpe_rmse_m": float("nan"),
            "final_drift_m": float("nan"),
        }
    aligned = _rigid_align(estimated, matched_truth)
    errors = np.linalg.norm(aligned - matched_truth, axis=1)
    relative_errors = np.linalg.norm(
        np.diff(aligned, axis=0) - np.diff(matched_truth, axis=0), axis=1
    )
    return {
        "trajectory_pose_count": len(estimates),
        "matched_pose_count": len(estimated),
        "ate_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "rpe_rmse_m": float(np.sqrt(np.mean(relative_errors**2))),
        "final_drift_m": float(errors[-1]),
    }


def _save_png(image: npt.NDArray[np.generic], path: Path) -> None:
    image_module = importlib.import_module("PIL.Image")
    image_module.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def export_euroc(batch: ReplayBatch, output: Path, *, stereo_tolerance_ms: float = 20.0) -> Path:
    left = batch.camera_streams.get("left") or batch.camera_streams.get("color") or ()
    right = batch.camera_streams.get("right") or ()
    if not left:
        raise ValueError("EuRoC export requires a primary camera stream")
    if not right:
        raise ValueError("stereo-inertial EuRoC export requires a right camera stream")
    cam0 = output / "mav0/cam0/data"
    cam1 = output / "mav0/cam1/data"
    imu_root = output / "mav0/imu0"
    cam0.mkdir(parents=True, exist_ok=True)
    cam1.mkdir(parents=True, exist_ok=True)
    imu_root.mkdir(parents=True, exist_ok=True)
    right_times = np.asarray([frame.timestamp.monotonic_ns for frame in right], dtype=np.int64)
    exported_timestamps: list[int] = []
    tolerance_ns = int(stereo_tolerance_ms * 1e6)
    for frame in left:
        insertion = int(np.searchsorted(right_times, frame.timestamp.monotonic_ns))
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(right)]
        nearest = min(
            candidates,
            key=lambda index: abs(int(right_times[index]) - frame.timestamp.monotonic_ns),
        )
        if abs(int(right_times[nearest]) - frame.timestamp.monotonic_ns) > tolerance_ns:
            continue
        filename = f"{frame.timestamp.monotonic_ns}.png"
        _save_png(frame.image, cam0 / filename)
        _save_png(right[nearest].image, cam1 / filename)
        exported_timestamps.append(frame.timestamp.monotonic_ns)
    if not exported_timestamps:
        raise ValueError("no stereo pairs satisfied the synchronization tolerance")
    times_path = output / "times.txt"
    times_path.write_text("".join(f"{value}\n" for value in exported_timestamps), encoding="utf-8")
    imu_lines = [
        "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
        "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
    ]
    for sample in batch.imu_samples:
        gyro = sample.angular_velocity_rps
        acceleration = sample.acceleration_mps2
        imu_lines.append(
            f"{sample.timestamp.monotonic_ns},{gyro[0]},{gyro[1]},{gyro[2]},"
            f"{acceleration[0]},{acceleration[1]},{acceleration[2]}\n"
        )
    (imu_root / "data.csv").write_text("".join(imu_lines), encoding="utf-8")
    return times_path


class _ExternalProcessAdapter:
    backend_name: str

    def _run(
        self,
        command: tuple[str, ...],
        output_dir: Path,
        trajectory_path: Path,
        truth: tuple[GroundTruthPose, ...],
        timeout_seconds: float,
    ) -> ExternalVioResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"
        start = perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=output_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return_code: int | None = completed.returncode
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            detail = ""
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            return_code = None
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(str(error), encoding="utf-8")
            detail = str(error)
        elapsed = perf_counter() - start
        trajectory = parse_trajectory(trajectory_path) if trajectory_path.is_file() else ()
        metrics = evaluate_trajectory(trajectory, truth)
        metrics["elapsed_seconds"] = elapsed
        metrics["return_code"] = return_code if return_code is not None else -1
        status = "passed" if return_code == 0 and len(trajectory) >= 3 else "failed"
        return ExternalVioResult(
            self.backend_name,
            status,
            return_code,
            elapsed,
            trajectory,
            metrics,
            command,
            stdout_path,
            stderr_path,
            trajectory_path,
            detail,
        )


class OpenVinsAdapter(_ExternalProcessAdapter):
    backend_name = "openvins"

    def run(
        self,
        *,
        bag: Path,
        config: Path,
        truth: tuple[GroundTruthPose, ...],
        output_dir: Path,
        launcher: tuple[str, ...] = ("roslaunch",),
        launch_target: tuple[str, ...] = ("ov_msckf", "serial.launch"),
        timeout_seconds: float = 3600.0,
    ) -> ExternalVioResult:
        trajectory = output_dir / "trajectory.txt"
        timing = output_dir / "timing.txt"
        command = (
            *launcher,
            *launch_target,
            f"config_path:={config.resolve()}",
            f"bag:={bag.resolve()}",
            "dosave:=true",
            f"path_est:={trajectory.resolve()}",
            "dotime:=true",
            f"path_time:={timing.resolve()}",
            "dolivetraj:=false",
        )
        return self._run(command, output_dir, trajectory, truth, timeout_seconds)


class OrbSlam3Adapter(_ExternalProcessAdapter):
    backend_name = "orbslam3"

    def run(
        self,
        *,
        batch: ReplayBatch,
        executable: Path,
        vocabulary: Path,
        settings: Path,
        output_dir: Path,
        timeout_seconds: float = 3600.0,
    ) -> ExternalVioResult:
        sequence = output_dir / "euroc"
        times = export_euroc(batch, sequence)
        run_name = "ariadne"
        trajectory = output_dir / f"f_{run_name}.txt"
        command = (
            str(executable.resolve()),
            str(vocabulary.resolve()),
            str(settings.resolve()),
            str(sequence.resolve()),
            str(times.resolve()),
            run_name,
        )
        if shutil.which(command[0]) is None and not executable.is_file():
            output_dir.mkdir(parents=True, exist_ok=True)
        return self._run(command, output_dir, trajectory, batch.ground_truth, timeout_seconds)
