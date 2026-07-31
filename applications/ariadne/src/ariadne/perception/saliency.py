"""Pretrained U²-Net saliency and deterministic connected-region formation."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from ariadne.perception.preprocessing import PreprocessedFrame

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SaliencyMap:
    frame: PreprocessedFrame
    scores: npt.NDArray[np.float32]
    mask: npt.NDArray[np.bool_]
    threshold: float
    latency_ms: float
    backend: str = "gradient_contrast"
    model_version: str = "ariadne-reference-v1"


@dataclass(frozen=True)
class SaliencyRegion:
    region_id: str
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    area_px: int
    mean_score: float


class SaliencyBackend(Protocol):
    """Common detector contract used by runtime and evidence builders."""

    backend: str
    model_version: str

    def detect(self, frame: PreprocessedFrame) -> SaliencyMap: ...


class SaliencyDetector:
    """Deterministic gradient/contrast fallback for tests and offline operation."""

    backend = "gradient_contrast"
    model_name = "Gradient/contrast reference"
    model_version = "ariadne-reference-v1"
    device = "cpu"
    parameter_count = 0

    def __init__(self, *, quantile: float = 0.82, contrast_weight: float = 0.45) -> None:
        if not 0 < quantile < 1 or not 0 <= contrast_weight <= 1:
            raise ValueError("saliency controls are invalid")
        self.quantile = quantile
        self.contrast_weight = contrast_weight
        self.metrics = {"frames": 0, "salient_fraction": 0.0, "latency_ms": 0.0}

    def detect(self, frame: PreprocessedFrame) -> SaliencyMap:
        if not frame.quality.accepted:
            raise ValueError("quality-rejected frames cannot enter saliency detection")
        start_ns = perf_counter_ns()
        image = frame.image.astype(np.float64)
        gradient_y, gradient_x = np.gradient(image)
        gradient = np.hypot(gradient_x, gradient_y)
        contrast = np.abs(image - float(np.mean(image)))
        scores = (1.0 - self.contrast_weight) * gradient + self.contrast_weight * contrast
        maximum = float(np.max(scores))
        if maximum > 0:
            scores /= maximum
        threshold = float(np.quantile(scores[frame.valid_mask], self.quantile))
        mask = (scores >= threshold) & frame.valid_mask
        latency_ms = (perf_counter_ns() - start_ns) / 1e6
        self.metrics["frames"] += 1
        self.metrics["salient_fraction"] += float(np.mean(mask))
        self.metrics["latency_ms"] += latency_ms
        LOGGER.debug(
            "saliency_detected frame=%d threshold=%.4f", frame.source.frame_index, threshold
        )
        return SaliencyMap(
            frame,
            scores.astype(np.float32),
            mask,
            threshold,
            latency_ms,
            self.backend,
            self.model_version,
        )


class PretrainedU2NetSaliencyDetector:
    """Full standard U²-Net detector using the official DUTS-trained checkpoint."""

    backend = "u2net"
    model_name = "U²-Net (full)"
    training_dataset = "DUTS-TR"
    parameter_count = 44_009_869
    checkpoint_sha256 = "10025a17f49cd3208afc342b589890e402ee63123d6f2d289a4a0903695cce58"

    def __init__(
        self,
        weights_path: str | Path,
        *,
        device: str = "auto",
        input_size: int = 320,
        threshold: float = 0.5,
        model: Any | None = None,
        verify_checksum: bool = True,
    ) -> None:
        if input_size < 32 or input_size % 32:
            raise ValueError("U²-Net input_size must be a multiple of 32")
        if not 0 < threshold < 1:
            raise ValueError("U²-Net threshold must be between zero and one")
        self.weights_path = Path(weights_path).expanduser()
        self.input_size = input_size
        self.threshold = threshold
        self._torch = import_module("torch")
        if device == "auto":
            device = "cuda" if self._torch.cuda.is_available() else "cpu"
        self.device = str(self._torch.device(device))

        if model is None:
            if not self.weights_path.is_file():
                raise FileNotFoundError(
                    f"official U²-Net checkpoint does not exist: {self.weights_path}; "
                    "run applications/ariadne/scripts/download_u2net.py"
                )
            digest = self._digest(self.weights_path)
            if verify_checksum and digest != self.checkpoint_sha256:
                raise ValueError(
                    f"U²-Net checkpoint checksum mismatch: expected {self.checkpoint_sha256}, "
                    f"received {digest}"
                )
            model_module = import_module("ariadne.perception.u2net_model")
            model = model_module.U2Net()
            state = self._torch.load(
                self.weights_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(state, Mapping):
                raise ValueError("U²-Net checkpoint must contain a state dictionary")
            model.load_state_dict(state, strict=True)
            self.model_version = f"u2net-official-duts@sha256:{digest[:12]}"
        else:
            self.model_version = "u2net-injected-test-model"
        self._model = model.to(self.device).eval()
        self.metrics = {"frames": 0, "salient_fraction": 0.0, "latency_ms": 0.0}

    @staticmethod
    def _digest(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _input_tensor(self, frame: PreprocessedFrame) -> Any:
        image = np.asarray(frame.source.image, dtype=np.float32)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        image = image[..., :3]
        maximum = float(np.max(image))
        if maximum > 0:
            image = image / maximum
        tensor = self._torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        tensor = tensor.unsqueeze(0).to(self.device, dtype=self._torch.float32)
        tensor = self._torch.nn.functional.interpolate(
            tensor,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = self._torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        std = self._torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def detect(self, frame: PreprocessedFrame) -> SaliencyMap:
        if not frame.quality.accepted:
            raise ValueError("quality-rejected frames cannot enter saliency detection")
        start_ns = perf_counter_ns()
        tensor = self._input_tensor(frame)
        with self._torch.inference_mode():
            outputs = self._model(tensor)
            prediction = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            prediction = self._torch.nn.functional.interpolate(
                prediction,
                size=frame.image.shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            minimum = prediction.amin()
            maximum = prediction.amax()
            prediction = (prediction - minimum) / (maximum - minimum).clamp_min(1e-8)
            scores = prediction.detach().cpu().numpy().astype(np.float32)
        mask = (scores >= self.threshold) & frame.valid_mask
        latency_ms = (perf_counter_ns() - start_ns) / 1e6
        self.metrics["frames"] += 1
        self.metrics["salient_fraction"] += float(np.mean(mask))
        self.metrics["latency_ms"] += latency_ms
        LOGGER.debug(
            "saliency_detected frame=%d backend=%s threshold=%.4f",
            frame.source.frame_index,
            self.backend,
            self.threshold,
        )
        return SaliencyMap(
            frame,
            scores,
            mask,
            self.threshold,
            latency_ms,
            self.backend,
            self.model_version,
        )


class SaliencyClusterer:
    def __init__(self, *, min_area_px: int = 4, max_regions: int = 32) -> None:
        if min_area_px <= 0 or max_regions <= 0:
            raise ValueError("region limits must be positive")
        self.min_area_px = min_area_px
        self.max_regions = max_regions
        self.metrics = {"frames": 0, "regions": 0, "latency_ms": 0.0}

    def cluster(self, saliency: SaliencyMap) -> tuple[SaliencyRegion, ...]:
        start_ns = perf_counter_ns()
        mask = saliency.mask
        visited = np.zeros(mask.shape, dtype=np.bool_)
        regions: list[SaliencyRegion] = []
        height, width = mask.shape
        for seed_y, seed_x in np.argwhere(mask):
            if visited[seed_y, seed_x]:
                continue
            queue = deque([(int(seed_y), int(seed_x))])
            visited[seed_y, seed_x] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                y, x = queue.popleft()
                pixels.append((y, x))
                for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and mask[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        queue.append((next_y, next_x))
            if len(pixels) < self.min_area_px:
                continue
            ys = np.asarray([pixel[0] for pixel in pixels])
            xs = np.asarray([pixel[1] for pixel in pixels])
            regions.append(
                SaliencyRegion(
                    "",
                    (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                    (float(xs.mean()), float(ys.mean())),
                    len(pixels),
                    float(np.mean(saliency.scores[ys, xs])),
                )
            )
        regions.sort(key=lambda region: (-region.mean_score * region.area_px, region.bbox_xyxy))
        selected = tuple(
            SaliencyRegion(
                f"region_{index:04d}",
                region.bbox_xyxy,
                region.centroid_xy,
                region.area_px,
                region.mean_score,
            )
            for index, region in enumerate(regions[: self.max_regions], start=1)
        )
        latency_ms = (perf_counter_ns() - start_ns) / 1e6
        self.metrics["frames"] += 1
        self.metrics["regions"] += len(selected)
        self.metrics["latency_ms"] += latency_ms
        return selected
