"""Optional W&B logging for dataset evaluations."""

from __future__ import annotations

import importlib
from pathlib import Path

from ariadne.datasets.base import DatasetEvaluation


def log_evaluation_to_wandb(
    result: DatasetEvaluation,
    report_path: Path,
    *,
    mode: str,
    project: str,
    entity: str | None,
    name: str | None,
    group: str | None,
    tags: list[str],
) -> str | None:
    if mode == "disabled":
        return None
    try:
        wandb = importlib.import_module("wandb")
    except ImportError as error:
        raise RuntimeError("install ARIADNE with the evaluation extra to use W&B") from error
    run = wandb.init(
        project=project,
        entity=entity,
        name=name or f"ariadne/{result.dataset}",
        group=group,
        tags=["ariadne", "dataset-evaluation", result.dataset, *tags],
        mode=mode,
        job_type="dataset-test",
        config={
            "dataset": result.dataset,
            "agents": list(result.agents),
            "modalities": list(result.modalities),
        },
    )
    numeric_metrics = {
        f"evaluation/{key}": value
        for key, value in result.metrics.items()
        if isinstance(value, int | float)
    }
    numeric_metrics["evaluation/passed"] = int(result.status == "passed")
    run.log(numeric_metrics)
    artifact = wandb.Artifact(
        name=f"ariadne-{result.dataset}-evaluation-{run.id}",
        type="evaluation-report",
        metadata={"status": result.status, "warnings": list(result.warnings)},
    )
    artifact.add_file(str(report_path), name=report_path.name)
    run.log_artifact(artifact)
    run.summary.update(result.metrics)
    run.summary["status"] = result.status
    run_url = getattr(run, "url", None)
    run.finish()
    return run_url
