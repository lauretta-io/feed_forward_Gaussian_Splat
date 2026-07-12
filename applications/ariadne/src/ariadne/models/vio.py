"""Lightweight VIO reference backends and trajectory metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ariadne.common import ModelVersion
from ariadne.models.interfaces import VioEstimate
from ariadne.replay import SynchronizedPacket


@dataclass
class _VioState:
    timestamp_ns: int
    position_m: npt.NDArray[np.float64]
    velocity_mps: npt.NDArray[np.float64]


class ImuDeadReckoningVio:
    """Deterministic IMU integration baseline for adapter and metric validation."""

    version = ModelVersion("imu-dead-reckoning-reference", "1.0.0")

    def __init__(self) -> None:
        self._states: dict[str, _VioState] = {}

    def reset(self) -> None:
        self._states.clear()

    def process(self, packet: SynchronizedPacket) -> VioEstimate:
        timestamp_ns = packet.frame.timestamp.monotonic_ns
        state = self._states.get(packet.agent_id)
        if state is None:
            state = _VioState(timestamp_ns, np.zeros(3), np.zeros(3))
            self._states[packet.agent_id] = state
            return self._estimate(packet, state, 1.0)
        dt = max((timestamp_ns - state.timestamp_ns) / 1e9, 0.0)
        acceleration = np.mean(
            np.stack([sample.acceleration_mps2 for sample in packet.imu_window]), axis=0
        )
        state.position_m = state.position_m + state.velocity_mps * dt + 0.5 * acceleration * dt**2
        state.velocity_mps = state.velocity_mps + acceleration * dt
        state.timestamp_ns = timestamp_ns
        return self._estimate(packet, state, max(0.0, 1.0 - packet.sync_error_ms / 20.0))

    def _estimate(
        self, packet: SynchronizedPacket, state: _VioState, quality: float
    ) -> VioEstimate:
        covariance = np.eye(6, dtype=np.float64) * (1.0 - quality + 1e-3)
        return VioEstimate(
            packet.frame.timestamp,
            packet.agent_id,
            state.position_m.copy(),
            state.velocity_mps.copy(),
            covariance,
            quality,
            "tracking",
        )


class VisualInertialComplementaryVio(ImuDeadReckoningVio):
    """Reference fusion backend blending IMU displacement with visual displacement."""

    version = ModelVersion("visual-inertial-complementary-reference", "1.0.0")

    def __init__(self, visual_weight: float = 0.8) -> None:
        super().__init__()
        if not 0.0 <= visual_weight <= 1.0:
            raise ValueError("visual_weight must be between zero and one")
        self.visual_weight = visual_weight

    def process(self, packet: SynchronizedPacket) -> VioEstimate:
        timestamp_ns = packet.frame.timestamp.monotonic_ns
        state = self._states.get(packet.agent_id)
        if state is None:
            state = _VioState(timestamp_ns, np.zeros(3), np.zeros(3))
            self._states[packet.agent_id] = state
            return self._estimate(packet, state, 1.0)
        dt = max((timestamp_ns - state.timestamp_ns) / 1e9, 0.0)
        acceleration = np.mean(
            np.stack([sample.acceleration_mps2 for sample in packet.imu_window]), axis=0
        )
        inertial_delta = state.velocity_mps * dt + 0.5 * acceleration * dt**2
        delta = inertial_delta
        if packet.frame.visual_delta_m is not None:
            delta = (
                1.0 - self.visual_weight
            ) * inertial_delta + self.visual_weight * packet.frame.visual_delta_m
        state.position_m = state.position_m + delta
        state.velocity_mps = delta / dt if dt > 0 else state.velocity_mps
        state.timestamp_ns = timestamp_ns
        quality = max(0.0, 1.0 - packet.sync_error_ms / 20.0)
        if packet.frame.visual_delta_m is None:
            quality *= 0.5
        return self._estimate(packet, state, quality)


def trajectory_metrics(
    estimates: list[VioEstimate], truth_positions: npt.NDArray[np.float64]
) -> dict[str, float]:
    if len(estimates) != len(truth_positions) or not estimates:
        raise ValueError("estimates and non-empty truth_positions must have equal length")
    predicted = np.stack([estimate.position_m for estimate in estimates])
    truth = np.asarray(truth_positions, dtype=np.float64)
    if truth.shape != predicted.shape:
        raise ValueError("truth_positions must have shape Nx3")
    aligned = predicted - predicted[0] + truth[0]
    errors = np.linalg.norm(aligned - truth, axis=1)
    predicted_steps = np.diff(aligned, axis=0)
    truth_steps = np.diff(truth, axis=0)
    relative_errors = np.linalg.norm(predicted_steps - truth_steps, axis=1)
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "rpe_rmse_m": float(np.sqrt(np.mean(relative_errors**2))),
        "final_drift_m": float(errors[-1]),
        "tracking_uptime": float(np.mean([item.status == "tracking" for item in estimates])),
    }
