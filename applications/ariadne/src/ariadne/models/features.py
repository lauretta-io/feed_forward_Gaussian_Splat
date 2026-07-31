"""CPU reference feature extractors and comparable feature metrics."""

from __future__ import annotations

from time import perf_counter_ns

import numpy as np
import numpy.typing as npt

from ariadne.common import ModelVersion
from ariadne.models.interfaces import FeatureSet, SemanticEmbedding
from ariadne.replay import ImageFrame


def _grayscale(image: npt.NDArray[np.generic]) -> npt.NDArray[np.float64]:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 3:
        array = np.mean(array[..., :3], axis=2)
    scale = float(np.max(array))
    return array / scale if scale > 1.0 else array


def _normalized(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > np.finfo(np.float64).eps else vector


class GradientPatchExtractor:
    version = ModelVersion("gradient-patch-reference", "1.0.0")

    def __init__(self, max_keypoints: int = 64) -> None:
        if max_keypoints <= 0:
            raise ValueError("max_keypoints must be positive")
        self.max_keypoints = max_keypoints

    def extract(self, frame: ImageFrame) -> FeatureSet:
        gray = _grayscale(frame.image)
        gradient_y, gradient_x = np.gradient(gray)
        score_map = np.hypot(gradient_x, gradient_y)
        if min(gray.shape) < 3:
            return FeatureSet(np.empty((0, 2)), np.empty((0, 9)), np.empty(0), self.version)
        interior = score_map[1:-1, 1:-1]
        count = min(self.max_keypoints, interior.size)
        selected = np.argpartition(interior.ravel(), -count)[-count:]
        y_positions, x_positions = np.unravel_index(selected, interior.shape)
        y_positions += 1
        x_positions += 1
        order = np.argsort(score_map[y_positions, x_positions])[::-1]
        y_positions = y_positions[order]
        x_positions = x_positions[order]
        descriptors = np.stack(
            [
                _normalized(gray[y - 1 : y + 2, x - 1 : x + 2].ravel() - gray[y, x])
                for y, x in zip(y_positions, x_positions, strict=True)
            ]
        )
        return FeatureSet(
            np.column_stack((x_positions, y_positions)).astype(np.float64),
            descriptors,
            score_map[y_positions, x_positions],
            self.version,
        )


class GridPatchExtractor:
    version = ModelVersion("grid-patch-reference", "1.0.0")

    def __init__(self, spacing: int = 4) -> None:
        if spacing < 2:
            raise ValueError("spacing must be at least two")
        self.spacing = spacing

    def extract(self, frame: ImageFrame) -> FeatureSet:
        gray = _grayscale(frame.image)
        positions = [
            (y, x)
            for y in range(1, gray.shape[0] - 1, self.spacing)
            for x in range(1, gray.shape[1] - 1, self.spacing)
        ]
        if not positions:
            return FeatureSet(np.empty((0, 2)), np.empty((0, 9)), np.empty(0), self.version)
        descriptors = np.stack(
            [_normalized(gray[y - 1 : y + 2, x - 1 : x + 2].ravel()) for y, x in positions]
        )
        keypoints = np.asarray([(x, y) for y, x in positions], dtype=np.float64)
        return FeatureSet(keypoints, descriptors, np.ones(len(positions)), self.version)


class IntensityHistogramEmbedder:
    version = ModelVersion("intensity-histogram-reference", "1.0.0")

    def __init__(self, bins: int = 16) -> None:
        self.bins = bins

    def embed(self, frame: ImageFrame) -> SemanticEmbedding:
        histogram, _ = np.histogram(_grayscale(frame.image), bins=self.bins, range=(0.0, 1.0))
        return SemanticEmbedding(_normalized(histogram.astype(np.float64)), self.version)


class SpatialPyramidEmbedder:
    version = ModelVersion("spatial-pyramid-reference", "1.0.0")

    def embed(self, frame: ImageFrame) -> SemanticEmbedding:
        gray = _grayscale(frame.image)
        features: list[float] = []
        for cells in (1, 2, 4):
            for y_indices in np.array_split(np.arange(gray.shape[0]), cells):
                for x_indices in np.array_split(np.arange(gray.shape[1]), cells):
                    features.append(float(np.mean(gray[np.ix_(y_indices, x_indices)])))
        return SemanticEmbedding(_normalized(np.asarray(features)), self.version)


def cosine_similarity(left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def descriptor_match_recall(
    reference: FeatureSet, candidate: FeatureSet, *, max_distance: float = 0.25
) -> float:
    if not len(reference.descriptors) or not len(candidate.descriptors):
        return 0.0
    distances = np.linalg.norm(
        reference.descriptors[:, None, :] - candidate.descriptors[None, :, :], axis=2
    )
    return float(np.mean(np.min(distances, axis=1) <= max_distance))


def benchmark_feature_pair(
    geometric: GradientPatchExtractor | GridPatchExtractor,
    semantic: IntensityHistogramEmbedder | SpatialPyramidEmbedder,
    reference: ImageFrame,
    positive: ImageFrame,
    negative: ImageFrame,
) -> dict[str, float | str]:
    start_ns = perf_counter_ns()
    reference_features = geometric.extract(reference)
    positive_features = geometric.extract(positive)
    geometric_latency_ms = (perf_counter_ns() - start_ns) / 2e6
    start_ns = perf_counter_ns()
    reference_embedding = semantic.embed(reference)
    positive_embedding = semantic.embed(positive)
    negative_embedding = semantic.embed(negative)
    semantic_latency_ms = (perf_counter_ns() - start_ns) / 3e6
    positive_similarity = cosine_similarity(reference_embedding.vector, positive_embedding.vector)
    negative_similarity = cosine_similarity(reference_embedding.vector, negative_embedding.vector)
    return {
        "geometric_model": geometric.version.name,
        "semantic_model": semantic.version.name,
        "geometric_match_recall": descriptor_match_recall(reference_features, positive_features),
        "semantic_positive_cosine": positive_similarity,
        "semantic_negative_cosine": negative_similarity,
        "semantic_separation": positive_similarity - negative_similarity,
        "geometric_latency_ms": geometric_latency_ms,
        "semantic_latency_ms": semantic_latency_ms,
    }
