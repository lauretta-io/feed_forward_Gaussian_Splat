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
    Download(
        "qdrone-bridge",
        "https://huggingface.co/datasets/QDrone/UWB_IMU_GT_QDrone_Benchmark_Dataset/"
        "resolve/main/All%20Datasets/bridge.zip",
        ROOT / "datasets/ariadne/qdrone/raw/bridge.zip",
        3_659_534,
        "65b26624977c6c8276a08d48881e4ad20967a0b99220a2b8c9877e607ac73082",
    ),
    Download(
        "qdrone-building",
        "https://huggingface.co/datasets/QDrone/UWB_IMU_GT_QDrone_Benchmark_Dataset/"
        "resolve/main/All%20Datasets/building.zip",
        ROOT / "datasets/ariadne/qdrone/raw/building.zip",
        3_923_719,
        "9e3aef75586c822e249c762105d864a1394e9b1246370512a7964d0342855f16",
    ),
    Download(
        "qdrone-field",
        "https://huggingface.co/datasets/QDrone/UWB_IMU_GT_QDrone_Benchmark_Dataset/"
        "resolve/main/All%20Datasets/field.zip",
        ROOT / "datasets/ariadne/qdrone/raw/field.zip",
        10_808_243,
        "f501cc42d94372861d294474cff1acf5a658fa02c3eca487919bc3e433417dd7",
    ),
    Download(
        "qdrone-indoor",
        "https://huggingface.co/datasets/QDrone/UWB_IMU_GT_QDrone_Benchmark_Dataset/"
        "resolve/main/All%20Datasets/indoor.zip",
        ROOT / "datasets/ariadne/qdrone/raw/indoor.zip",
        9_635_549,
        "0bd32415be7d81250d95eb9f4acae044c797ffe49ac1decd247c4f5b47538ce9",
    ),
    Download(
        "qdrone-tunnel",
        "https://huggingface.co/datasets/QDrone/UWB_IMU_GT_QDrone_Benchmark_Dataset/"
        "resolve/main/All%20Datasets/tunnel.zip",
        ROOT / "datasets/ariadne/qdrone/raw/tunnel.zip",
        5_395_974,
        "8424689a414f1e25b9d17de2133e85bcc3b61e96234fcb35a5b47e746b22f124",
    ),
    Download(
        "qdrone-bridge-ground-truth",
        "https://huggingface.co/datasets/QDrone/UWB_IMU_GT_QDrone_Benchmark_Dataset/"
        "resolve/main/Dataset%20with%20Ground%20Truth/bridge-gt.zip",
        ROOT / "datasets/ariadne/qdrone/raw/bridge-gt.zip",
        1_463_113,
        "9a6434441b5e6370c139ced20ad1234e6b74d525a4a6af825daf7fa5f9529a6c",
    ),
    Download(
        "qdrone-building-ground-truth",
        "https://huggingface.co/datasets/QDrone/UWB_IMU_GT_QDrone_Benchmark_Dataset/"
        "resolve/main/Dataset%20with%20Ground%20Truth/building-gt.zip",
        ROOT / "datasets/ariadne/qdrone/raw/building-gt.zip",
        4_093_145,
        "103a5ca5cbff6bdc8ef0efb3b34f9a9d31aa2311be5c21541fc2e4f6af6df80c",
    ),
    Download(
        "qdrone-field-ground-truth",
        "https://huggingface.co/datasets/QDrone/UWB_IMU_GT_QDrone_Benchmark_Dataset/"
        "resolve/main/Dataset%20with%20Ground%20Truth/field-gt.zip",
        ROOT / "datasets/ariadne/qdrone/raw/field-gt.zip",
        10_130_371,
        "067f5c9b648af21fb8d618f68dcb4967908b967b7cb51b219c88e1a830487017",
    ),
    Download(
        "qdrone-indoor-ground-truth",
        "https://huggingface.co/datasets/QDrone/UWB_IMU_GT_QDrone_Benchmark_Dataset/"
        "resolve/main/Dataset%20with%20Ground%20Truth/indoor-gt.zip",
        ROOT / "datasets/ariadne/qdrone/raw/indoor-gt.zip",
        6_276_458,
        "110dddc745f291fe9693fd42f6540d9116691f922421a22d5511c3e7378c2899",
    ),
    Download(
        "qdrone-tunnel-ground-truth",
        "https://huggingface.co/datasets/QDrone/UWB_IMU_GT_QDrone_Benchmark_Dataset/"
        "resolve/main/Dataset%20with%20Ground%20Truth/tunnel-gt.zip",
        ROOT / "datasets/ariadne/qdrone/raw/tunnel-gt.zip",
        3_719_791,
        "b1d52b2af670e8cb2610b55d624acbc8aa8f7d972abc2bb4cae023ad779ead5a",
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
