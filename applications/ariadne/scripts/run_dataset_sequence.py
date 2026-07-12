"""Run the representative ARIADNE dataset evaluation sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ariadne.datasets import evaluate_dataset
from ariadne.evaluation import log_evaluation_to_wandb

ROOT = Path(__file__).resolve().parents[3]
DATASETS = (
    ("miluv", ROOT / "datasets/ariadne/miluv/archives/default_3_random_0.zip"),
    ("d2slam", ROOT / "datasets/ariadne/d2slam/extracted/tum_corr"),
    ("qdrone", ROOT / "datasets/ariadne/qdrone/raw"),
    ("s3e", ROOT / "datasets/ariadne/s3e/S3Ev1/S3E_Playground_2/S3E_Playground_2.db3"),
    ("simulation", None),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/ariadne/dataset_sequence"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"), default="offline"
    )
    parser.add_argument("--wandb-project", default="gaussiansplat_test")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="ariadne-dataset-sequence")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    failed = False
    for index, (name, path) in enumerate(DATASETS, start=1):
        result = evaluate_dataset(name, path, seed=args.seed)
        report_path = output_dir / f"{index:02d}-{name}.json"
        result.write_json(report_path)
        url = log_evaluation_to_wandb(
            result,
            report_path,
            mode=args.wandb_mode,
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"ariadne/sequence-{index:02d}-{name}",
            group=args.wandb_group,
            tags=["representative-corpus", f"sequence-{index}"],
        )
        payload = result.to_dict()
        payload["wandb_url"] = url
        results.append(payload)
        failed = failed or result.status != "passed"
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(summary_path)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
