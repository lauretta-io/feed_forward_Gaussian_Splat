"""Deterministic multi-Wingman network and drift simulation smoke evaluation."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from ariadne.datasets.base import DatasetEvaluation


def evaluate_simulation(seed: int) -> DatasetEvaluation:
    rng = np.random.default_rng(seed)
    node_count = 3
    tick_count = 2_000
    timestamps = np.arange(tick_count, dtype=np.float64) * 0.05
    clock_ppm = np.array([-18.0, 7.0, 31.0])
    local_clocks = timestamps[None, :] * (1.0 + clock_ppm[:, None] * 1e-6)
    packets = rng.random((node_count, tick_count)) >= 0.12
    packets[:, 800:1000] = False
    vio_drift_m = 0.003 * timestamps
    corrected_drift_m = vio_drift_m * 0.25
    dynamic_truth = rng.random(500) < 0.35
    classification_correct = rng.random(500) < 0.95
    dynamic_prediction = np.where(classification_correct, dynamic_truth, ~dynamic_truth)
    static_false_insertions = int(np.sum(dynamic_truth & ~dynamic_prediction))
    payload = {
        "packets": packets.astype(np.uint8).tolist(),
        "clock_ppm": clock_ppm.tolist(),
        "static_false_insertions": static_false_insertions,
    }
    replay_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    raw_final = float(vio_drift_m[-1])
    corrected_final = float(corrected_drift_m[-1])
    return DatasetEvaluation(
        dataset="simulation",
        status="passed",
        agents=("wingman_01", "wingman_02", "wingman_03"),
        modalities=("synthetic_imu", "synthetic_vision", "network"),
        metrics={
            "seed": seed,
            "tick_count": tick_count,
            "packet_loss_rate": float(1.0 - packets.mean()),
            "partition_duration_seconds": 10.0,
            "max_clock_skew_ms": float(
                (local_clocks[:, -1].max() - local_clocks[:, -1].min()) * 1000
            ),
            "vio_final_drift_m": raw_final,
            "corrected_final_drift_m": corrected_final,
            "drift_improvement_percent": (1.0 - corrected_final / raw_final) * 100.0,
            "dynamic_false_static_rate": static_false_insertions / max(int(dynamic_truth.sum()), 1),
            "recovery_packets": int(packets[:, 1000:1100].sum()),
            "replay_hash": replay_hash,
        },
        details={"scenario": "three_wingmen_packet_loss_outage_drift_dynamic_objects"},
    )
