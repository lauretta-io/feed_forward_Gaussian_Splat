#!/usr/bin/env python3
"""Download and verify the official full U²-Net checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib
from pathlib import Path

FILE_ID = "1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ"
EXPECTED_SHA256 = "10025a17f49cd3208afc342b589890e402ee63123d6f2d289a4a0903695cce58"
DEFAULT_OUTPUT = Path(".cache/ariadne/models/u2net/u2net.pth")


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    gdown = importlib.import_module("gdown")
    result = gdown.download(id=FILE_ID, output=str(output), quiet=False)
    if result is None or not output.is_file():
        raise RuntimeError("official U²-Net checkpoint download failed")
    actual = digest(output)
    if actual != EXPECTED_SHA256:
        output.unlink()
        raise RuntimeError(
            f"U²-Net checkpoint checksum mismatch: expected {EXPECTED_SHA256}, got {actual}"
        )
    print(f"verified {output} sha256={actual}")


if __name__ == "__main__":
    main()
