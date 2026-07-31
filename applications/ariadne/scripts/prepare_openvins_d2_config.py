"""Create a clone-local OpenVINS TUM-VI config for D2SLAM compressed topics."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openvins-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.openvins_root / "config/tum_vi"
    if not source.is_dir():
        raise FileNotFoundError(source)
    if args.output.exists():
        shutil.rmtree(args.output)
    shutil.copytree(source, args.output)
    print(args.output / "estimator_config.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
