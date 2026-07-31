"""S3E-to-OpenVINS calibration and bounded ROS1 bridge utilities."""

from __future__ import annotations

import csv
import importlib
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def _calibration_scalar(content: str, key: str) -> float:
    match = re.search(
        rf"(?m)^{re.escape(key)}:\s*([-+0-9.eE]+)\s*$",
        content,
    )
    if match is None:
        raise ValueError(f"S3E calibration is missing numeric field {key}")
    value = float(match.group(1))
    if not np.isfinite(value):
        raise ValueError(f"S3E calibration field {key} must be finite")
    return value


def _calibration_matrix(
    content: str,
    key: str,
    shape: tuple[int, int],
) -> npt.NDArray[np.float64]:
    match = re.search(
        rf"(?ms)^{re.escape(key)}:\s*!!opencv-matrix\s*.*?data:\s*\[([^\]]+)\]",
        content,
    )
    if match is None:
        raise ValueError(f"S3E calibration is missing matrix {key}")
    values = np.fromstring(match.group(1).replace("\n", " "), sep=",", dtype=np.float64)
    if values.size != shape[0] * shape[1] or not np.all(np.isfinite(values)):
        raise ValueError(f"S3E calibration matrix {key} must have shape {shape}")
    return np.asarray(values.reshape(shape), dtype=np.float64)


def _yaml_matrix(matrix: npt.NDArray[np.float64], indent: str = "    ") -> str:
    return "\n".join(
        f"{indent}- [{', '.join(f'{value:.16g}' for value in row)}]"
        for row in matrix
    )


def prepare_openvins_s3e_config(
    source: Path,
    template: Path,
    output: Path,
) -> Path:
    """Generate an OpenVINS config from the same raw stereo model used by ORB-SLAM3.

    The S3E file provides each raw camera model, the two rectification rotations,
    a scalar rectified baseline, and the left-camera-to-IMU transform.  Combining
    those fields recovers both raw-camera-to-IMU transforms without treating the
    non-standard fourth column of ``RIGHT.P`` as a physical translation.
    """
    if not source.is_file():
        raise FileNotFoundError(source)
    if not template.is_dir():
        raise FileNotFoundError(template)
    content = source.read_text(encoding="utf-8")
    camera_fx = _calibration_scalar(content, "Camera.fx")
    baseline_m = _calibration_scalar(content, "Camera.bf") / camera_fx
    if baseline_m <= 0:
        raise ValueError("S3E stereo baseline must be positive")

    tic = _calibration_matrix(content, "Tic", (4, 4))
    left_k = _calibration_matrix(content, "LEFT.K", (3, 3))
    right_k = _calibration_matrix(content, "RIGHT.K", (3, 3))
    left_d = _calibration_matrix(content, "LEFT.D", (1, 5))[0, :4]
    right_d = _calibration_matrix(content, "RIGHT.D", (1, 5))[0, :4]
    left_r = _calibration_matrix(content, "LEFT.R", (3, 3))
    right_r = _calibration_matrix(content, "RIGHT.R", (3, 3))
    width = int(_calibration_scalar(content, "Camera.width"))
    height = int(_calibration_scalar(content, "Camera.height"))

    # Tic maps raw left-camera coordinates into the IMU/body frame.  OpenVINS
    # expects the inverse: IMU coordinates into each raw camera frame.
    raw_left_from_imu = np.linalg.inv(tic)
    rect_left_from_imu = np.eye(4, dtype=np.float64)
    rect_left_from_imu[:3, :3] = left_r
    rect_left_from_imu = rect_left_from_imu @ raw_left_from_imu
    rect_right_from_imu = rect_left_from_imu.copy()
    rect_right_from_imu[0, 3] -= baseline_m
    raw_right_from_rectified = np.eye(4, dtype=np.float64)
    raw_right_from_rectified[:3, :3] = right_r.T
    raw_right_from_imu = raw_right_from_rectified @ rect_right_from_imu

    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(template, output)
    for unused_mask in output.glob("mask_tumvi*"):
        unused_mask.unlink()
    estimator = output / "estimator_config.yaml"
    estimator_content = estimator.read_text(encoding="utf-8")
    replacements = {
        "calib_cam_extrinsics": "false",
        "calib_cam_intrinsics": "false",
        "calib_cam_timeoffset": "false",
        "gravity_mag": "9.81",
        "init_dyn_use": "false",
        "init_imu_thresh": "1.0",
        "track_frequency": "10.0",
        "use_mask": "false",
    }
    for key, value in replacements.items():
        estimator_content, count = re.subn(
            rf"(?m)^({re.escape(key)}:\s*)[^#\n]+",
            rf"\g<1>{value} ",
            estimator_content,
            count=1,
        )
        if count != 1:
            raise ValueError(f"OpenVINS template is missing field {key}")
    estimator.write_text(estimator_content, encoding="utf-8")

    imu_config = f"""%YAML:1.0

imu0:
  T_i_b:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
  accelerometer_noise_density: {_calibration_scalar(content, "IMU.NoiseAcc"):.16g}
  accelerometer_random_walk: {_calibration_scalar(content, "IMU.AccWalk"):.16g}
  gyroscope_noise_density: {_calibration_scalar(content, "IMU.NoiseGyro"):.16g}
  gyroscope_random_walk: {_calibration_scalar(content, "IMU.GyroWalk"):.16g}
  rostopic: /imu0
  time_offset: 0.0
  update_rate: {_calibration_scalar(content, "IMU.Frequency"):.16g}
  model: "kalibr"
  Tw:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  R_IMUtoGYRO:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  Ta:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  R_IMUtoACC:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  Tg:
    - [0.0, 0.0, 0.0]
    - [0.0, 0.0, 0.0]
    - [0.0, 0.0, 0.0]
"""
    (output / "kalibr_imu_chain.yaml").write_text(imu_config, encoding="utf-8")

    right_intrinsics = ", ".join(
        f"{value:.16g}"
        for value in (right_k[0, 0], right_k[1, 1], right_k[0, 2], right_k[1, 2])
    )
    camera_config = f"""%YAML:1.0

cam0:
  T_cam_imu:
{_yaml_matrix(raw_left_from_imu)}
  cam_overlaps: [1]
  camera_model: pinhole
  distortion_coeffs: [{', '.join(f'{value:.16g}' for value in left_d)}]
  distortion_model: radtan
  intrinsics: [{left_k[0, 0]:.16g}, {left_k[1, 1]:.16g}, {left_k[0, 2]:.16g}, {left_k[1, 2]:.16g}]
  resolution: [{width}, {height}]
  rostopic: /cam0/image_raw

cam1:
  T_cam_imu:
{_yaml_matrix(raw_right_from_imu)}
  cam_overlaps: [0]
  camera_model: pinhole
  distortion_coeffs: [{', '.join(f'{value:.16g}' for value in right_d)}]
  distortion_model: radtan
  intrinsics: [{right_intrinsics}]
  resolution: [{width}, {height}]
  rostopic: /cam1/image_raw
"""
    (output / "kalibr_imucam_chain.yaml").write_text(
        camera_config,
        encoding="utf-8",
    )
    return estimator


def export_euroc_ros1_bag(sequence: Path, output: Path) -> Path:
    """Write a bounded EuRoC-layout window as an OpenVINS-compatible ROS1 bag."""
    times_path = sequence / "times.txt"
    imu_path = sequence / "mav0/imu0/data.csv"
    if not times_path.is_file() or not imu_path.is_file():
        raise FileNotFoundError("EuRoC window is missing times.txt or IMU data.csv")
    timestamps = [
        int(line)
        for line in times_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records: list[tuple[int, int, str, Any]] = []
    for sequence_number, timestamp_ns in enumerate(timestamps):
        left = sequence / f"mav0/cam0/data/{timestamp_ns}.png"
        right = sequence / f"mav0/cam1/data/{timestamp_ns}.png"
        if not left.is_file() or not right.is_file():
            raise FileNotFoundError(f"EuRoC stereo pair is incomplete at {timestamp_ns}")
        records.append((timestamp_ns, 1, "left", (sequence_number, left)))
        records.append((timestamp_ns, 2, "right", (sequence_number, right)))
    with imu_path.open(encoding="utf-8", newline="") as handle:
        rows = csv.reader(line for line in handle if not line.startswith("#"))
        for sequence_number, row in enumerate(rows):
            if len(row) != 7:
                raise ValueError("EuRoC IMU row must contain timestamp, gyro, and acceleration")
            records.append(
                (
                    int(row[0]),
                    0,
                    "imu",
                    (sequence_number, tuple(float(value) for value in row[1:])),
                )
            )
    records.sort(key=lambda record: (record[0], record[1]))

    rosbag1 = importlib.import_module("rosbags.rosbag1")
    typesys = importlib.import_module("rosbags.typesys")
    typestore = typesys.get_typestore(typesys.Stores.ROS1_NOETIC)
    message_types = typestore.types
    Header = message_types["std_msgs/msg/Header"]
    Time = message_types["builtin_interfaces/msg/Time"]
    CompressedImage = message_types["sensor_msgs/msg/CompressedImage"]
    Imu = message_types["sensor_msgs/msg/Imu"]
    Quaternion = message_types["geometry_msgs/msg/Quaternion"]
    Vector3 = message_types["geometry_msgs/msg/Vector3"]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    with rosbag1.Writer(output) as writer:
        connections = {
            "left": writer.add_connection(
                "/cam0/image_raw/compressed",
                "sensor_msgs/msg/CompressedImage",
                typestore=typestore,
            ),
            "right": writer.add_connection(
                "/cam1/image_raw/compressed",
                "sensor_msgs/msg/CompressedImage",
                typestore=typestore,
            ),
            "imu": writer.add_connection(
                "/imu0",
                "sensor_msgs/msg/Imu",
                typestore=typestore,
            ),
        }
        zero_covariance = np.zeros(9, dtype=np.float64)
        for timestamp_ns, _, kind, payload in records:
            sequence_number, data = payload
            stamp = Time(
                sec=timestamp_ns // 1_000_000_000,
                nanosec=timestamp_ns % 1_000_000_000,
            )
            if kind in {"left", "right"}:
                message = CompressedImage(
                    header=Header(
                        seq=sequence_number,
                        stamp=stamp,
                        frame_id="cam0" if kind == "left" else "cam1",
                    ),
                    format="jpeg",
                    data=np.frombuffer(data.read_bytes(), dtype=np.uint8).copy(),
                )
                message_type = "sensor_msgs/msg/CompressedImage"
            else:
                gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z = data
                message = Imu(
                    header=Header(seq=sequence_number, stamp=stamp, frame_id="imu0"),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                    orientation_covariance=zero_covariance.copy(),
                    angular_velocity=Vector3(x=gyro_x, y=gyro_y, z=gyro_z),
                    angular_velocity_covariance=zero_covariance.copy(),
                    linear_acceleration=Vector3(x=acc_x, y=acc_y, z=acc_z),
                    linear_acceleration_covariance=zero_covariance.copy(),
                )
                message_type = "sensor_msgs/msg/Imu"
            writer.write(
                connections[kind],
                timestamp_ns,
                typestore.serialize_ros1(message, message_type),
            )
    return output
