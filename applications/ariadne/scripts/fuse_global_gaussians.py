#!/usr/bin/env python3
"""Fuse dense Gaussian PLY contributions without synchronizing capture time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ariadne.splatting.dense_fusion import contributions_from_manifest, fuse_static_gaussian_plys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="JSON contribution specification")
    parser.add_argument("--output", type=Path, required=True, help="Output unified Gaussian PLY")
    parser.add_argument("--manifest", type=Path, help="Output provenance manifest")
    args = parser.parse_args()
    result = fuse_static_gaussian_plys(
        contributions_from_manifest(args.spec), args.output, manifest_path=args.manifest
    )
    print(
        json.dumps(
            {
                "output_ply": str(result.output_ply),
                "manifest": str(result.manifest_path),
                "input_gaussians": result.input_gaussians,
                "output_gaussians": result.output_gaussians,
                "filtered_gaussians": result.filtered_gaussians,
                "global_registration_verified": result.global_registration_verified,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
