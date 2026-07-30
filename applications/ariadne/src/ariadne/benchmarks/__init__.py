"""ARIADNE benchmark suites."""

from ariadne.benchmarks.exchange import run_exchange_benchmark
from ariadne.benchmarks.global_scene import run_global_scene_benchmark
from ariadne.benchmarks.operations import run_operations_benchmark
from ariadne.benchmarks.phase1 import run_phase1_benchmark
from ariadne.benchmarks.video_evidence import build_video_evidence, select_video_frames

__all__ = [
    "run_exchange_benchmark",
    "run_global_scene_benchmark",
    "run_operations_benchmark",
    "run_phase1_benchmark",
    "build_video_evidence",
    "select_video_frames",
]
