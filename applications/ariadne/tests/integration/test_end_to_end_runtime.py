from ariadne.runtime import run_reference_system


def test_reference_runtime_composes_every_gate() -> None:
    result = run_reference_system(seed=23)
    assert result.status == "passed"
    assert result.metrics["stage_count"] == 4
    assert result.metrics["passed_stage_count"] == 4
    assert result.metrics["final_context_nodes"] == 2
    assert result.metrics["task_assignments"] == 2
    assert result.metrics["route_requests"] == 2
    assert result.metrics["handoff_context_revision"] == 6
    assert result.metrics["peak_traced_bytes"] > 0
    assert all(stage["status"] == "passed" for stage in result.details["stages"])
    assert result.details["degraded_modes"][0]["recovered"]
    assert result.details["degraded_modes"][2]["blocked_frontiers"] == 1
