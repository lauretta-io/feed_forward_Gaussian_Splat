from ariadne.benchmarks import run_operations_benchmark


def test_operations_benchmark_passes_and_covers_degraded_network() -> None:
    result = run_operations_benchmark(seed=19)
    assert result.status == "passed"
    assert result.metrics["assignment_count"] == 2
    assert result.metrics["security_verified"] == 1
    assert result.metrics["deployment_compatible"] == 1
    assert result.metrics["partition_duration_seconds"] == 60.0
    assert result.metrics["recovery_packets"] > 0
    assert result.metrics["telemetry_event_count"] == 1
    assert result.metrics["prometheus_series_count"] >= 7
    assert result.details["telemetry"]["mission_id"] == "mission-reference"
