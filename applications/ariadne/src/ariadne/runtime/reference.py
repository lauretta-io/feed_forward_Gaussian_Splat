"""Deterministic end-to-end orchestration across all CPU reference gates."""

from __future__ import annotations

import tracemalloc
from time import perf_counter_ns

from ariadne.benchmarks.exchange import run_exchange_benchmark
from ariadne.benchmarks.global_scene import run_global_scene_benchmark
from ariadne.benchmarks.operations import run_operations_benchmark
from ariadne.benchmarks.phase1 import run_phase1_benchmark
from ariadne.datasets import DatasetEvaluation


def run_reference_system(seed: int = 7) -> DatasetEvaluation:
    tracemalloc.start()
    start_ns = perf_counter_ns()
    stages = (
        run_phase1_benchmark(seed),
        run_exchange_benchmark(seed),
        run_global_scene_benchmark(seed),
        run_operations_benchmark(seed),
    )
    passed = all(stage.status == "passed" for stage in stages)
    total_latency_ms = (perf_counter_ns() - start_ns) / 1e6
    _, peak_traced_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return DatasetEvaluation(
        dataset="end-to-end-reference",
        status="passed" if passed else "failed",
        agents=("wingman_01", "wingman_02", "wingman_03", "intelligence_01"),
        modalities=(
            "replay",
            "perception",
            "vio",
            "mesh",
            "gaussians",
            "correction",
            "context",
            "planning",
            "route_requests",
            "telemetry",
        ),
        metrics={
            "seed": seed,
            "stage_count": len(stages),
            "passed_stage_count": sum(stage.status == "passed" for stage in stages),
            "total_latency_ms": total_latency_ms,
            "peak_traced_bytes": peak_traced_bytes,
            "network_partition_seconds": stages[3].metrics["partition_duration_seconds"],
            "recovery_packets": stages[3].metrics["recovery_packets"],
            "final_context_nodes": stages[2].metrics["context_node_count"],
            "task_assignments": stages[3].metrics["assignment_count"],
            "route_requests": stages[3].metrics["route_request_count"],
            "handoff_context_revision": stages[3].metrics["handoff_context_revision"],
        },
        details={
            "stages": [
                {
                    "name": stage.dataset,
                    "status": stage.status,
                    "latency_ms": stage.metrics.get("benchmark_latency_ms", 0.0),
                }
                for stage in stages
            ],
            "degraded_modes": [
                {
                    "name": "network_partition",
                    "duration_seconds": stages[3].metrics["partition_duration_seconds"],
                    "recovered": int(stages[3].metrics["recovery_packets"]) > 0,
                },
                {
                    "name": "low_battery_wingman",
                    "excluded_from_planning": True,
                },
                {
                    "name": "no_fly_zone",
                    "blocked_frontiers": stages[3].metrics["blocked_frontier_count"],
                },
            ],
        },
    )
