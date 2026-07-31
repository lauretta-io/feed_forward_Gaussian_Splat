"""Summarize identical-configuration VIO evaluation reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

from ariadne.backends import summarize_vio_replicates


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-ate-m", type=float, default=0.1)
    return parser.parse_args()


def _configuration(evaluation: dict[str, object]) -> dict[str, object]:
    details = cast(dict[str, object], evaluation["details"])
    return {
        "dataset": evaluation["dataset"],
        "agents": evaluation["agents"],
        "backend": details["backend"],
        "window_start_timestamp_ns": details.get("window_start_timestamp_ns"),
        "window_end_timestamp_ns": details.get("window_end_timestamp_ns"),
        "start_frame": details.get("start_frame", 0),
        "requested_max_frames": details.get(
            "requested_max_frames",
            details.get("frames"),
        ),
        "vio_mode": details.get("vio_mode"),
        "stereo_baseline_scale": details.get("stereo_baseline_scale", 1.0),
        "imu_fast_init": details.get("imu_fast_init", False),
        "orb_feature_profile": details.get("orb_feature_profile", "balanced"),
        "orb_deterministic_runtime": details.get(
            "orb_deterministic_runtime", False
        ),
        "orb_sync_local_mapping": details.get("orb_sync_local_mapping", False),
        "swap_stereo_input": details.get("swap_stereo_input", False),
        "right_image_shift_y_px": details.get("right_image_shift_y_px", 0.0),
        "auto_stereo_geometry": details.get(
            "auto_stereo_geometry",
            details.get("auto_stereo_row_correction", False),
        ),
    }


def main() -> int:
    args = _arguments()
    evaluations = [
        cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        for path in args.reports
    ]
    configurations = [_configuration(evaluation) for evaluation in evaluations]
    canonical = json.dumps(configurations[0], sort_keys=True, separators=(",", ":"))
    if any(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")) != canonical
        for configuration in configurations[1:]
    ):
        raise ValueError("VIO replicate reports do not share an identical configuration")
    metrics = summarize_vio_replicates(
        tuple(cast(dict[str, object], evaluation["metrics"]) for evaluation in evaluations),
        target_ate_m=args.target_ate_m,
    )
    payload = {
        "dataset": evaluations[0]["dataset"],
        "status": ("passed" if int(metrics["global_pose_claim_eligible"]) else "failed"),
        "agents": evaluations[0]["agents"],
        "metrics": metrics,
        "details": {
            "configuration": configurations[0],
            "configuration_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "reports": [str(path) for path in args.reports],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return int(payload["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
