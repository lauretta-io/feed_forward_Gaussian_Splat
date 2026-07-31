from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .common import list_images, load_mask, load_rgb


@dataclass(frozen=True)
class FrameRecord:
    image_path: Path
    frame_name: str
    frame_index: int
    mask_path: Path | None = None


class FrameSaliencyDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path | None = None,
        image_size: tuple[int, int] | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.image_size = image_size
        image_paths = list_images(self.image_dir)
        if max_frames is not None:
            image_paths = image_paths[:max_frames]
        if not image_paths:
            raise FileNotFoundError(f"No images found under {self.image_dir}")
        self.records = [
            FrameRecord(
                image_path=path,
                frame_name=path.stem,
                frame_index=i,
                mask_path=self._find_mask(path),
            )
            for i, path in enumerate(image_paths)
        ]

    def _find_mask(self, image_path: Path) -> Path | None:
        if self.mask_dir is None:
            return None
        candidates = [
            self.mask_dir / image_path.name,
            self.mask_dir / f"{image_path.stem}.png",
            self.mask_dir / f"{image_path.stem}.jpg",
            self.mask_dir / f"{image_path.stem}.jpeg",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @property
    def has_masks(self) -> bool:
        return all(record.mask_path is not None for record in self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image = load_rgb(record.image_path, self.image_size)
        mask = load_mask(record.mask_path, self.image_size) if record.mask_path else None
        return {
            "image": image,
            "mask": mask,
            "image_path": str(record.image_path),
            "mask_path": str(record.mask_path) if record.mask_path else "",
            "frame_name": record.frame_name,
            "frame_index": torch.tensor(record.frame_index, dtype=torch.long),
        }


class EmbeddingSaliencyDataset(Dataset):
    def __init__(
        self,
        manifest: dict,
        mask_dir: str | Path | None = None,
        image_size: tuple[int, int] | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.manifest = manifest
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.image_size = image_size
        records = manifest.get("frames", [])
        if max_frames is not None:
            records = records[:max_frames]
        if not records:
            raise ValueError("Embedding manifest contains no frames")
        self.records = records

    @property
    def has_masks(self) -> bool:
        return self.mask_dir is not None and all(self._find_mask(Path(r["image_path"])) for r in self.records)

    def _find_mask(self, image_path: Path) -> Path | None:
        if self.mask_dir is None:
            return None
        for suffix in [image_path.suffix, ".png", ".jpg", ".jpeg"]:
            candidate = self.mask_dir / f"{image_path.stem}{suffix}"
            if candidate.exists():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        embedding = torch.load(record["embedding_path"], map_location="cpu")
        if isinstance(embedding, dict):
            embedding_tensor = embedding["embedding"]
        else:
            embedding_tensor = embedding
        image_path = Path(record["image_path"])
        mask_path = self._find_mask(image_path)
        image = load_rgb(image_path, self.image_size)
        mask = load_mask(mask_path, self.image_size) if mask_path else None
        return {
            "embedding": embedding_tensor.float(),
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
            "mask_path": str(mask_path) if mask_path else "",
            "frame_name": record["frame_name"],
            "frame_index": torch.tensor(record["frame_index"], dtype=torch.long),
        }


def saliency_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [item[key] for item in batch]
        if key in {"image", "embedding", "frame_index"}:
            out[key] = torch.stack(values)  # type: ignore[arg-type]
        elif key == "mask":
            out[key] = None if any(v is None for v in values) else torch.stack(values)  # type: ignore[arg-type]
        else:
            out[key] = values
    return out
