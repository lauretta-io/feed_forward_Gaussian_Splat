"""Cheap visual observability checks for bounded EuRoC stereo windows."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def _cv2() -> Any:
    return importlib.import_module("cv2")


def _robust_row_model(
    right_points_xy: npt.NDArray[np.float64],
    vertical_offsets_px: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    design = np.column_stack(
        (right_points_xy, np.ones(len(right_points_xy), dtype=np.float64))
    )
    keep = np.ones(len(vertical_offsets_px), dtype=bool)
    for _ in range(5):
        coefficients = np.linalg.lstsq(
            design[keep],
            vertical_offsets_px[keep],
            rcond=None,
        )[0]
        residual = vertical_offsets_px - design @ coefficients
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        threshold = max(1.5, 3.0 * 1.4826 * mad)
        updated = np.abs(residual - median) <= threshold
        if np.array_equal(updated, keep):
            break
        keep = updated
    coefficients = np.asarray(
        np.linalg.lstsq(design[keep], vertical_offsets_px[keep], rcond=None)[0],
        dtype=np.float64,
    )
    return coefficients, vertical_offsets_px - design @ coefficients


def evaluate_stereo_disparity_direction(
    left_images: tuple[npt.NDArray[np.uint8], ...],
    right_images: tuple[npt.NDArray[np.uint8], ...],
) -> dict[str, float | int]:
    """Estimate signed disparity from robust feature correspondences."""
    if not left_images or len(left_images) != len(right_images):
        raise ValueError("stereo diagnostic requires equally sized non-empty streams")
    cv2 = _cv2()
    signed_disparities: list[float] = []
    vertical_offsets: list[float] = []
    matched_right_points: list[tuple[float, float]] = []
    ratio_match_count = 0
    inlier_count = 0
    usable_frames = 0
    for left, right in zip(left_images, right_images, strict=True):
        if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
            raise ValueError("stereo diagnostic images must be same-sized grayscale arrays")
        detector = cv2.SIFT_create(nfeatures=3000)
        left_keypoints, left_descriptors = detector.detectAndCompute(left, None)
        right_keypoints, right_descriptors = detector.detectAndCompute(right, None)
        if left_descriptors is None or right_descriptors is None:
            continue
        nearest = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
            left_descriptors,
            right_descriptors,
            k=2,
        )
        matches = [
            best
            for best, second in nearest
            if best.distance < 0.75 * second.distance
        ]
        ratio_match_count += len(matches)
        if len(matches) < 12:
            continue
        left_points = np.asarray(
            [left_keypoints[match.queryIdx].pt for match in matches],
            dtype=np.float32,
        )
        right_points = np.asarray(
            [right_keypoints[match.trainIdx].pt for match in matches],
            dtype=np.float32,
        )
        _, mask = cv2.findFundamentalMat(
            left_points,
            right_points,
            cv2.FM_RANSAC,
            1.5,
            0.999,
            10_000,
        )
        if mask is None:
            continue
        inliers = np.asarray(mask, dtype=np.uint8).ravel().astype(bool)
        if np.count_nonzero(inliers) < 8:
            continue
        usable_frames += 1
        disparity = left_points[inliers, 0] - right_points[inliers, 0]
        vertical = left_points[inliers, 1] - right_points[inliers, 1]
        signed_disparities.extend(float(value) for value in disparity)
        vertical_offsets.extend(float(value) for value in vertical)
        matched_right_points.extend(
            (float(x), float(y)) for x, y in right_points[inliers]
        )
        inlier_count += int(np.count_nonzero(inliers))

    disparity_array = np.asarray(signed_disparities, dtype=np.float64)
    vertical_array = np.asarray(vertical_offsets, dtype=np.float64)
    right_points_array = np.asarray(matched_right_points, dtype=np.float64)
    positive_fraction = (
        float(np.mean(disparity_array > 0.5)) if disparity_array.size else 0.0
    )
    observable = usable_frames >= 1 and inlier_count >= 40
    direction_healthy = observable and positive_fraction >= 0.75
    row_coefficients = np.full(3, float("nan"), dtype=np.float64)
    row_residual = np.empty(0, dtype=np.float64)
    if observable:
        row_coefficients, row_residual = _robust_row_model(
            right_points_array,
            vertical_array,
        )
    return {
        "stereo_geometry_sampled_frame_count": len(left_images),
        "stereo_geometry_usable_frame_count": usable_frames,
        "stereo_geometry_ratio_match_count": ratio_match_count,
        "stereo_geometry_ransac_inlier_count": inlier_count,
        "stereo_geometry_observable": int(observable),
        "stereo_positive_disparity_fraction": positive_fraction,
        "stereo_disparity_median_px": (
            float(np.median(disparity_array)) if disparity_array.size else float("nan")
        ),
        "stereo_vertical_offset_median_px": (
            float(np.median(vertical_array)) if vertical_array.size else float("nan")
        ),
        "stereo_row_model_x_slope": float(row_coefficients[0]),
        "stereo_row_model_y_slope": float(row_coefficients[1]),
        "stereo_row_model_intercept_px": float(row_coefficients[2]),
        "stereo_row_model_residual_abs_median_px": (
            float(np.median(np.abs(row_residual)))
            if row_residual.size
            else float("nan")
        ),
        "stereo_row_model_residual_abs_p95_px": (
            float(np.percentile(np.abs(row_residual), 95))
            if row_residual.size
            else float("nan")
        ),
        "stereo_disparity_direction_healthy": int(direction_healthy),
    }


def diagnose_euroc_stereo_direction(
    sequence: Path,
    *,
    sample_frames: int = 5,
) -> dict[str, float | int]:
    """Sample an exported window without retaining decoded images."""
    if sample_frames <= 0:
        raise ValueError("stereo diagnostic sample count must be positive")
    times_path = sequence / "times.txt"
    if not times_path.is_file():
        raise FileNotFoundError(times_path)
    timestamps = [
        int(value)
        for value in times_path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    if not timestamps:
        raise ValueError("EuRoC stereo window contains no timestamps")
    indices = np.linspace(
        0,
        len(timestamps) - 1,
        min(sample_frames, len(timestamps)),
        dtype=np.int64,
    )
    cv2 = _cv2()
    left_images: list[npt.NDArray[np.uint8]] = []
    right_images: list[npt.NDArray[np.uint8]] = []
    for index in np.unique(indices):
        timestamp_ns = timestamps[int(index)]
        left = cv2.imread(
            str(sequence / f"mav0/cam0/data/{timestamp_ns}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        right = cv2.imread(
            str(sequence / f"mav0/cam1/data/{timestamp_ns}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if left is None or right is None:
            raise FileNotFoundError(f"stereo images are missing at {timestamp_ns}")
        left_images.append(np.asarray(left, dtype=np.uint8))
        right_images.append(np.asarray(right, dtype=np.uint8))
    return evaluate_stereo_disparity_direction(
        tuple(left_images),
        tuple(right_images),
    )
