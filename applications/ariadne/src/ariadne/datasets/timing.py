"""Timestamp alignment metrics shared by dataset probes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def nearest_timestamp_errors_ms(
    reference_ns: Sequence[int] | npt.NDArray[np.int64],
    samples_ns: Sequence[int] | npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    reference = np.sort(np.asarray(reference_ns, dtype=np.int64))
    samples = np.sort(np.asarray(samples_ns, dtype=np.int64))
    if not reference.size or not samples.size:
        return np.empty(0, dtype=np.float64)
    indices = np.searchsorted(reference, samples)
    lower_indices = np.clip(indices - 1, 0, reference.size - 1)
    upper_indices = np.clip(indices, 0, reference.size - 1)
    lower_error = np.abs(samples - reference[lower_indices])
    upper_error = np.abs(reference[upper_indices] - samples)
    return np.minimum(lower_error, upper_error).astype(np.float64) / 1e6


def summarize_errors_ms(errors: npt.NDArray[np.float64]) -> dict[str, float]:
    if not errors.size:
        return {
            "camera_imu_sync_median_ms": float("nan"),
            "camera_imu_sync_p95_ms": float("nan"),
            "camera_imu_sync_max_ms": float("nan"),
        }
    return {
        "camera_imu_sync_median_ms": float(np.median(errors)),
        "camera_imu_sync_p95_ms": float(np.percentile(errors, 95)),
        "camera_imu_sync_max_ms": float(np.max(errors)),
    }
