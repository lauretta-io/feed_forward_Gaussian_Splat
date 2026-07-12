"""ARIADNE command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tracemalloc
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter_ns

from pydantic import ValidationError

from ariadne.common import FrameId, TransformSE3
from ariadne.config import AriadneConfig, load_config
from ariadne.datasets import evaluate_dataset
from ariadne.evaluation import log_evaluation_to_wandb
from ariadne.logging import configure_logging

LOGGER = logging.getLogger("ariadne")


def _config_command(path: str, expected_role: str | None = None) -> AriadneConfig:
    config = load_config(path)
    if expected_role is not None and config.runtime.role != expected_role:
        raise ValueError(f"expected a {expected_role!r} config, got {config.runtime.role!r}")
    configure_logging(config.runtime.log_level, json_output=config.runtime.json_logs)
    return config


def _run_probe(config: AriadneConfig) -> None:
    config.runtime.output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("runtime_ready role=%s node_id=%s", config.runtime.role, config.runtime.node_id)
    print(
        json.dumps(
            {"node_id": config.runtime.node_id, "role": config.runtime.role, "status": "ready"}
        )
    )


def _smoke_benchmark(iterations: int = 1_000) -> dict[str, int | float | str]:
    transform = TransformSE3.from_translation_quaternion(
        FrameId("camera_benchmark"),
        FrameId("body"),
        [0.1, -0.2, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    )
    inverse = transform.inverse()
    tracemalloc.start()
    start_ns = perf_counter_ns()
    for _ in range(iterations):
        transform.then(inverse)
    elapsed_ns = perf_counter_ns() - start_ns
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "suite": "smoke",
        "status": "passed",
        "iterations": iterations,
        "transform_round_trip_ns": elapsed_ns / iterations,
        "peak_traced_bytes": peak_bytes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ariadne")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate a YAML configuration")
    validate.add_argument("--config", required=True)

    for role in ("wingman", "intelligence"):
        role_parser = subparsers.add_parser(role, help=f"{role} runtime commands")
        role_subparsers = role_parser.add_subparsers(dest="action", required=True)
        run = role_subparsers.add_parser("run", help=f"run the {role} bootstrap probe")
        run.add_argument("--config", required=True)

    simulate = subparsers.add_parser("simulate", help="run deterministic simulation probe")
    simulate.add_argument("--scenario", required=True)

    benchmark = subparsers.add_parser("benchmark", help="run a benchmark suite")
    benchmark.add_argument("--suite", choices=("smoke",), required=True)

    evaluate = subparsers.add_parser("evaluate", help="evaluate an ARIADNE dataset replay")
    evaluate.add_argument(
        "--dataset", choices=("miluv", "d2slam", "qdrone", "s3e", "simulation"), required=True
    )
    evaluate.add_argument("--path")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--seed", type=int, default=7)
    evaluate.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    evaluate.add_argument("--wandb-project", default="gaussiansplat_test")
    evaluate.add_argument("--wandb-entity")
    evaluate.add_argument("--wandb-name")
    evaluate.add_argument("--wandb-group")
    evaluate.add_argument("--wandb-tags", nargs="*", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            config = load_config(args.config)
            print(f"valid: {config.runtime.role}:{config.runtime.node_id}")
        elif args.command in ("wingman", "intelligence"):
            _run_probe(_config_command(args.config, args.command))
        elif args.command == "simulate":
            _run_probe(_config_command(args.scenario, "simulation"))
        elif args.command == "benchmark":
            print(json.dumps(_smoke_benchmark()))
        elif args.command == "evaluate":
            result = evaluate_dataset(
                args.dataset,
                Path(args.path) if args.path is not None else None,
                seed=args.seed,
            )
            output_path = Path(args.output)
            result.write_json(output_path)
            run_url = log_evaluation_to_wandb(
                result,
                output_path,
                mode=args.wandb_mode,
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                group=args.wandb_group,
                tags=args.wandb_tags,
            )
            payload = result.to_dict()
            payload["report_path"] = str(output_path)
            payload["wandb_url"] = run_url
            print(json.dumps(payload, sort_keys=True))
            return 0 if result.status == "passed" else 1
        return 0
    except (FileNotFoundError, ValueError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
