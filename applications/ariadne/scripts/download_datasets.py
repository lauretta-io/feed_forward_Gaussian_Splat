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
)


def _download(item: Download) -> None:
    item.destination.parent.mkdir(parents=True, exist_ok=True)
    existing = item.destination.stat().st_size if item.destination.exists() else 0
    if existing == item.size_bytes:
        print(f"already complete: {item.name}")
        return
    request = urllib.request.Request(item.url)
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    mode = "ab" if existing else "wb"
    with urllib.request.urlopen(request) as response, item.destination.open(mode) as output:
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
    if item.destination.stat().st_size != item.size_bytes:
        raise RuntimeError(f"size mismatch for {item.name}")
    if item.checksum:
        digest = hashlib.new(item.checksum_kind)
        with item.destination.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != item.checksum:
            raise RuntimeError(f"checksum mismatch for {item.name}")


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
