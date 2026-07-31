"""Download resumable ARIADNE dataset subsets with free-space checks."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Download:
    name: str
    url: str
    destination: Path
    size_bytes: int
    checksum: str | None = None
    checksum_kind: str = "sha256"


ROOT = Path(__file__).resolve().parents[3]
DOWNLOADS = (
    Download(
        "miluv-default-3-random-0",
        "https://ndownloader.figshare.com/files/52291625",
        ROOT / "datasets/ariadne/miluv/archives/default_3_random_0.zip",
        3_231_673_980,
        "09236b18470f0dd99c33245ee1c994ee",
        "md5",
    ),
    Download(
        "d2slam-aligned-tum",
        "https://www.dropbox.com/s/ic0yuxr2xym1m0c/tum_corr.7z?dl=1",
        ROOT / "datasets/ariadne/d2slam/archives/tum_corr.7z",
        1_402_927_165,
    ),
    Download(
        "s3e-playground-2",
        "https://huggingface.co/datasets/PengYu-Team/S3E/resolve/main/"
        "S3Ev1/S3E_Playground_2/S3E_Playground_2.db3",
        ROOT / "datasets/ariadne/s3e/S3Ev1/S3E_Playground_2/S3E_Playground_2.db3",
        6_732_963_840,
        "fa2bceb5064fa50318452fba247f49971b4119f3bf6af686629b30a265d1b095",
    ),
    Download(
        "s3e-playground-3-network",
        "https://huggingface.co/datasets/PengYu-Team/S3E/resolve/main/"
        "S3Ev2/S3E_Playground_3/S3E_Playground_3.db3",
        ROOT / "datasets/ariadne/s3e/S3Ev2/S3E_Playground_3/S3E_Playground_3.db3",
        1_837_629_440,
        "c6318632d79845457d64c1b86f5a18ae209efdfc52042b612eaf9336cbf8ac9a",
    ),
    Download(
        "s3e-v1-calibration-alpha",
        "https://huggingface.co/datasets/PengYu-Team/S3E/resolve/main/S3Ev1/Calibration/alpha.yaml",
        ROOT / "datasets/ariadne/s3e/S3Ev1/Calibration/alpha.yaml",
        4_287,
    ),
    Download(
        "s3e-v1-calibration-bob",
        "https://huggingface.co/datasets/PengYu-Team/S3E/resolve/main/S3Ev1/Calibration/bob.yaml",
        ROOT / "datasets/ariadne/s3e/S3Ev1/Calibration/bob.yaml",
        4_328,
    ),
    Download(
        "s3e-v1-calibration-carol",
        "https://huggingface.co/datasets/PengYu-Team/S3E/resolve/main/S3Ev1/Calibration/carol.yaml",
        ROOT / "datasets/ariadne/s3e/S3Ev1/Calibration/carol.yaml",
        4_335,
    ),
    Download(
        "s3e-playground-2-alpha-ground-truth",
        "https://huggingface.co/datasets/PengYu-Team/S3E/resolve/main/"
        "S3Ev1/S3E_Playground_2/alpha_gt.txt",
        ROOT / "datasets/ariadne/s3e/S3Ev1/S3E_Playground_2/alpha_gt.txt",
        19_333,
        "890c8bdceec48f5ba024cfa9fcd7161c58a763990e0ffe358a59aa01f1fb73a3",
    ),
    Download(
        "s3e-playground-2-bob-ground-truth",
        "https://huggingface.co/datasets/PengYu-Team/S3E/resolve/main/"
        "S3Ev1/S3E_Playground_2/bob_gt.txt",
        ROOT / "datasets/ariadne/s3e/S3Ev1/S3E_Playground_2/bob_gt.txt",
        19_202,
        "3913e1dcd160b7b423ef3fb9450d03ebf99154abc9cabb36e922c3cf9265b72a",
    ),
    Download(
        "s3e-playground-2-carol-ground-truth",
        "https://huggingface.co/datasets/PengYu-Team/S3E/resolve/main/"
        "S3Ev1/S3E_Playground_2/carol_gt.txt",
        ROOT / "datasets/ariadne/s3e/S3Ev1/S3E_Playground_2/carol_gt.txt",
        19_394,
        "563c4668fd1fd0dd0f14b76857df2fecfdf4036f40406a69d1e527e8c7a41412",
    ),
    Download(
        "s3e-playground-2-metadata",
        "https://huggingface.co/datasets/PengYu-Team/S3E/resolve/main/"
        "S3Ev1/S3E_Playground_2/metadata.yaml",
        ROOT / "datasets/ariadne/s3e/S3Ev1/S3E_Playground_2/metadata.yaml",
        3_229,
    ),
)


def _verify(item: Download) -> None:
    if item.destination.stat().st_size != item.size_bytes:
        raise RuntimeError(f"size mismatch for {item.name}")
    if item.checksum:
        digest = hashlib.new(item.checksum_kind)
        with item.destination.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != item.checksum:
            raise RuntimeError(f"checksum mismatch for {item.name}")


def _download(item: Download) -> None:
    item.destination.parent.mkdir(parents=True, exist_ok=True)
    existing = item.destination.stat().st_size if item.destination.exists() else 0
    if existing == item.size_bytes:
        _verify(item)
        print(f"verified: {item.name}")
        return
    if existing > item.size_bytes:
        raise RuntimeError(f"existing file is larger than expected for {item.name}")
    request = urllib.request.Request(item.url)
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    with urllib.request.urlopen(request) as response:
        resumes = existing > 0 and response.status == 206
        mode = "ab" if resumes else "wb"
        with item.destination.open(mode) as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
    _verify(item)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=[item.name for item in DOWNLOADS])
    args = parser.parse_args()
    selected = [item for item in DOWNLOADS if item.name in args.names]
    unknown = set(args.names) - {item.name for item in selected}
    if unknown:
        parser.error(f"unknown datasets: {sorted(unknown)}")
    required = sum(
        max(
            item.size_bytes - (item.destination.stat().st_size if item.destination.exists() else 0),
            0,
        )
        for item in selected
    )
    free = shutil.disk_usage(ROOT).free
    if required + 20 * 1024**3 > free:
        print("insufficient disk space while retaining a 20 GiB safety margin", file=sys.stderr)
        return 2
    for item in selected:
        _download(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
