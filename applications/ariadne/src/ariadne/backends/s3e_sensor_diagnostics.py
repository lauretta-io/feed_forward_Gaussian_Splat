"""Bounded, backend-independent S3E sensor consistency diagnostics."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class S3ETimestampSample:
    """One ROS record timestamp and the timestamp carried by its message header."""

    record_timestamp_ns: int
    header_timestamp_ns: int


@dataclass(frozen=True)
class S3EImuSample:
    """One IMU sample with its optional AHRS orientation estimate."""

    record_timestamp_ns: int
    header_timestamp_ns: int
    angular_velocity_rps: npt.NDArray[np.float64]
    linear_acceleration_mps2: npt.NDArray[np.float64]
    orientation_xyzw: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        for name, value, shape in (
            ("angular_velocity_rps", self.angular_velocity_rps, (3,)),
            ("linear_acceleration_mps2", self.linear_acceleration_mps2, (3,)),
            ("orientation_xyzw", self.orientation_xyzw, (4,)),
        ):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite {shape} vector")
            object.__setattr__(self, name, array)


def _percentile(values: npt.NDArray[np.float64], quantile: float) -> float:
    return float(np.percentile(values, quantile)) if values.size else float("nan")


def _cadence_ms(timestamps_ns: npt.NDArray[np.int64]) -> tuple[float, float, float]:
    if timestamps_ns.size < 2:
        return float("nan"), float("nan"), float("nan")
    cadence = np.diff(timestamps_ns).astype(np.float64) / 1e6
    return _percentile(cadence, 5), _percentile(cadence, 50), _percentile(cadence, 95)


def _nearest_sync_ms(
    left_timestamps_ns: npt.NDArray[np.int64],
    right_timestamps_ns: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    if not left_timestamps_ns.size or not right_timestamps_ns.size:
        return np.empty(0, dtype=np.float64)
    insertion = np.searchsorted(right_timestamps_ns, left_timestamps_ns)
    before = np.clip(insertion - 1, 0, right_timestamps_ns.size - 1)
    after = np.clip(insertion, 0, right_timestamps_ns.size - 1)
    before_delta = np.abs(left_timestamps_ns - right_timestamps_ns[before])
    after_delta = np.abs(left_timestamps_ns - right_timestamps_ns[after])
    return np.minimum(before_delta, after_delta).astype(np.float64) / 1e6


def _quaternion_multiply(
    first: npt.NDArray[np.float64],
    second: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    ax, ay, az, aw = np.moveaxis(first, -1, 0)
    bx, by, bz, bw = np.moveaxis(second, -1, 0)
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


def _orientation_rates(
    timestamps_ns: npt.NDArray[np.int64],
    quaternions_xyzw: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if timestamps_ns.size < 2:
        return np.empty((0, 3), dtype=np.float64)
    conjugates = quaternions_xyzw[:-1].copy()
    conjugates[:, :3] *= -1.0
    relative = _quaternion_multiply(conjugates, quaternions_xyzw[1:])
    relative[relative[:, 3] < 0.0] *= -1.0
    vector_norm = np.linalg.norm(relative[:, :3], axis=1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[:, 3], 0.0, None))
    scale = np.divide(
        angle,
        vector_norm,
        out=np.full_like(angle, 2.0),
        where=vector_norm > 1e-12,
    )
    delta_seconds = np.diff(timestamps_ns).astype(np.float64) / 1e9
    return np.asarray(
        relative[:, :3] * scale[:, None] / delta_seconds[:, None],
        dtype=np.float64,
    )


def _rotate_vectors(
    quaternions_xyzw: npt.NDArray[np.float64],
    vectors: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    vector_quaternions = np.column_stack(
        (vectors, np.zeros(vectors.shape[0], dtype=np.float64))
    )
    conjugates = quaternions_xyzw.copy()
    conjugates[:, :3] *= -1.0
    return _quaternion_multiply(
        _quaternion_multiply(quaternions_xyzw, vector_quaternions),
        conjugates,
    )[:, :3]


def _header_error_ms(
    *sample_groups: tuple[S3ETimestampSample, ...],
    imu_samples: tuple[S3EImuSample, ...],
) -> npt.NDArray[np.float64]:
    deltas = [
        abs(sample.record_timestamp_ns - sample.header_timestamp_ns) / 1e6
        for samples in sample_groups
        for sample in samples
    ]
    deltas.extend(
        abs(sample.record_timestamp_ns - sample.header_timestamp_ns) / 1e6
        for sample in imu_samples
    )
    return np.asarray(deltas, dtype=np.float64)


def evaluate_s3e_sensor_contract(
    left_samples: tuple[S3ETimestampSample, ...],
    right_samples: tuple[S3ETimestampSample, ...],
    imu_samples: tuple[S3EImuSample, ...],
) -> dict[str, float | int]:
    """Evaluate synchronization, rate, units, and AHRS/gyro consistency."""
    left_ns = np.asarray(
        [sample.record_timestamp_ns for sample in left_samples], dtype=np.int64
    )
    right_ns = np.asarray(
        [sample.record_timestamp_ns for sample in right_samples], dtype=np.int64
    )
    imu_ns = np.asarray(
        [sample.record_timestamp_ns for sample in imu_samples], dtype=np.int64
    )
    if any(np.any(np.diff(values) <= 0) for values in (left_ns, right_ns, imu_ns)):
        raise ValueError("S3E sensor timestamps must be strictly increasing")

    header_error_ms = _header_error_ms(
        left_samples,
        right_samples,
        imu_samples=imu_samples,
    )
    stereo_sync_ms = _nearest_sync_ms(left_ns, right_ns)
    left_cadence = _cadence_ms(left_ns)
    right_cadence = _cadence_ms(right_ns)
    imu_cadence = _cadence_ms(imu_ns)

    accelerations = np.asarray(
        [sample.linear_acceleration_mps2 for sample in imu_samples],
        dtype=np.float64,
    )
    gyroscopes = np.asarray(
        [sample.angular_velocity_rps for sample in imu_samples],
        dtype=np.float64,
    )
    quaternions = np.asarray(
        [sample.orientation_xyzw for sample in imu_samples],
        dtype=np.float64,
    )
    quaternion_norms = (
        np.linalg.norm(quaternions, axis=1)
        if quaternions.size
        else np.empty(0, dtype=np.float64)
    )
    quaternion_valid = np.isfinite(quaternion_norms) & (quaternion_norms > 1e-12)
    valid_fraction = (
        float(np.mean(quaternion_valid)) if quaternion_valid.size else 0.0
    )

    gyro_scale = float("nan")
    gyro_correlation = float("nan")
    gyro_rmse = float("nan")
    gravity_concentration = float("nan")
    if np.count_nonzero(quaternion_valid) >= 3:
        valid_indices = np.flatnonzero(quaternion_valid)
        valid_quaternions = quaternions[valid_indices] / quaternion_norms[
            valid_indices, None
        ]
        consecutive = np.diff(valid_indices) == 1
        orientation_rates = _orientation_rates(
            imu_ns[valid_indices],
            valid_quaternions,
        )
        rate_valid = consecutive & np.all(np.isfinite(orientation_rates), axis=1)
        measured_rates = gyroscopes[valid_indices[:-1]][rate_valid]
        orientation_rates = orientation_rates[rate_valid]
        denominator = float(np.sum(measured_rates * measured_rates))
        if measured_rates.size and denominator > 1e-12:
            gyro_scale = float(np.sum(measured_rates * orientation_rates) / denominator)
            residual = orientation_rates - gyro_scale * measured_rates
            gyro_rmse = float(np.sqrt(np.mean(residual * residual)))
            flattened_measured = measured_rates.ravel()
            flattened_orientation = orientation_rates.ravel()
            if (
                np.std(flattened_measured) > 1e-12
                and np.std(flattened_orientation) > 1e-12
            ):
                gyro_correlation = float(
                    np.corrcoef(flattened_measured, flattened_orientation)[0, 1]
                )

        rotated_acceleration = _rotate_vectors(
            valid_quaternions,
            accelerations[valid_indices],
        )
        acceleration_norms = np.linalg.norm(rotated_acceleration, axis=1)
        usable = acceleration_norms > 1e-12
        if np.any(usable):
            directions = rotated_acceleration[usable] / acceleration_norms[usable, None]
            gravity_concentration = float(np.linalg.norm(np.mean(directions, axis=0)))

    acceleration_norms = (
        np.linalg.norm(accelerations, axis=1)
        if accelerations.size
        else np.empty(0, dtype=np.float64)
    )
    header_max = float(np.max(header_error_ms)) if header_error_ms.size else float("nan")
    stereo_p95 = _percentile(stereo_sync_ms, 95)
    accel_p05 = _percentile(acceleration_norms, 5)
    accel_median = _percentile(acceleration_norms, 50)
    accel_p95 = _percentile(acceleration_norms, 95)
    sufficient_samples = (
        len(left_samples) >= 3
        and len(right_samples) >= 3
        and len(imu_samples) >= 20
    )
    healthy = (
        sufficient_samples
        and header_max <= 1.0
        and stereo_p95 <= 10.0
        and 8.0 <= imu_cadence[1] <= 12.0
        and 8.0 <= accel_median <= 11.5
        and valid_fraction >= 0.99
        and 0.8 <= gyro_scale <= 1.2
        and gyro_correlation >= 0.9
        and gyro_rmse <= 0.25
        and gravity_concentration >= 0.9
    )
    return {
        "s3e_sensor_contract_healthy": int(healthy),
        "s3e_sensor_left_frame_count": len(left_samples),
        "s3e_sensor_right_frame_count": len(right_samples),
        "s3e_sensor_imu_sample_count": len(imu_samples),
        "s3e_sensor_header_time_max_error_ms": header_max,
        "s3e_sensor_stereo_sync_median_ms": _percentile(stereo_sync_ms, 50),
        "s3e_sensor_stereo_sync_p95_ms": stereo_p95,
        "s3e_sensor_stereo_sync_max_ms": (
            float(np.max(stereo_sync_ms)) if stereo_sync_ms.size else float("nan")
        ),
        "s3e_sensor_left_cadence_p05_ms": left_cadence[0],
        "s3e_sensor_left_cadence_median_ms": left_cadence[1],
        "s3e_sensor_left_cadence_p95_ms": left_cadence[2],
        "s3e_sensor_right_cadence_p05_ms": right_cadence[0],
        "s3e_sensor_right_cadence_median_ms": right_cadence[1],
        "s3e_sensor_right_cadence_p95_ms": right_cadence[2],
        "s3e_sensor_imu_cadence_p05_ms": imu_cadence[0],
        "s3e_sensor_imu_cadence_median_ms": imu_cadence[1],
        "s3e_sensor_imu_cadence_p95_ms": imu_cadence[2],
        "s3e_sensor_acceleration_norm_p05_mps2": accel_p05,
        "s3e_sensor_acceleration_norm_median_mps2": accel_median,
        "s3e_sensor_acceleration_norm_p95_mps2": accel_p95,
        "s3e_sensor_orientation_valid_fraction": valid_fraction,
        "s3e_sensor_ahrs_gyro_rate_scale": gyro_scale,
        "s3e_sensor_ahrs_gyro_correlation": gyro_correlation,
        "s3e_sensor_ahrs_gyro_rmse_rps": gyro_rmse,
        "s3e_sensor_gravity_direction_concentration": gravity_concentration,
    }


def _header_timestamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def diagnose_s3e_sensor_contract(
    bag: Path,
    agent_id: str,
    *,
    start_timestamp_ns: int,
    end_timestamp_ns: int,
) -> dict[str, float | int]:
    """Stream one bounded S3E window and evaluate its independent sensor contract."""
    if not bag.is_file():
        raise FileNotFoundError(f"S3E bag does not exist: {bag}")
    normalized_agent = agent_id.capitalize()
    if normalized_agent not in {"Alpha", "Bob", "Carol"}:
        raise ValueError("S3E agent must be Alpha, Bob, or Carol")
    if start_timestamp_ns > end_timestamp_ns:
        raise ValueError("S3E sensor diagnostic time window is invalid")
    topics = {
        f"/{normalized_agent}/left_camera/compressed": "left",
        f"/{normalized_agent}/right_camera/compressed": "right",
        f"/{normalized_agent}/imu/data": "imu",
    }
    left: list[S3ETimestampSample] = []
    right: list[S3ETimestampSample] = []
    imu: list[S3EImuSample] = []
    highlevel = importlib.import_module("rosbags.highlevel")
    typesys = importlib.import_module("rosbags.typesys")
    typestore = typesys.get_typestore(typesys.Stores.ROS2_HUMBLE)
    with highlevel.AnyReader([bag], default_typestore=typestore) as reader:
        connections = [
            connection for connection in reader.connections if connection.topic in topics
        ]
        if len(connections) != len(topics):
            missing = sorted(set(topics) - {connection.topic for connection in connections})
            raise ValueError(f"S3E sensor topics are unavailable: {', '.join(missing)}")
        for connection, timestamp_ns, raw in reader.messages(
            connections=connections,
            start=max(0, start_timestamp_ns - 20_000_000),
            stop=end_timestamp_ns + 20_000_001,
        ):
            message = reader.deserialize(raw, connection.msgtype)
            header_timestamp_ns = _header_timestamp_ns(message)
            stream = topics[connection.topic]
            if stream == "left":
                if start_timestamp_ns <= timestamp_ns <= end_timestamp_ns:
                    left.append(S3ETimestampSample(timestamp_ns, header_timestamp_ns))
            elif stream == "right":
                right.append(S3ETimestampSample(timestamp_ns, header_timestamp_ns))
            else:
                if start_timestamp_ns <= timestamp_ns <= end_timestamp_ns:
                    imu.append(
                        S3EImuSample(
                            timestamp_ns,
                            header_timestamp_ns,
                            np.asarray(
                                [
                                    message.angular_velocity.x,
                                    message.angular_velocity.y,
                                    message.angular_velocity.z,
                                ],
                                dtype=np.float64,
                            ),
                            np.asarray(
                                [
                                    message.linear_acceleration.x,
                                    message.linear_acceleration.y,
                                    message.linear_acceleration.z,
                                ],
                                dtype=np.float64,
                            ),
                            np.asarray(
                                [
                                    message.orientation.x,
                                    message.orientation.y,
                                    message.orientation.z,
                                    message.orientation.w,
                                ],
                                dtype=np.float64,
                            ),
                        )
                    )
    return evaluate_s3e_sensor_contract(tuple(left), tuple(right), tuple(imu))
