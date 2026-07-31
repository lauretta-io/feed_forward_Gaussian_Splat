from ariadne.benchmarks import run_global_scene_benchmark


def test_global_scene_benchmark_passes() -> None:
    result = run_global_scene_benchmark(seed=17)
    assert result.status == "passed"
    assert result.metrics["input_observations"] == 2
    assert result.metrics["gaussian_primitive_count"] == 1
    assert result.metrics["reconstruction_backend"] == "object-gaussian-reference"
    assert result.metrics["reconstruction_estimated_memory_bytes"] > 0
    assert result.metrics["reconstruction_executor_completed"] == 1
    assert result.metrics["scene_revision"] == 1
    assert result.metrics["scene_history_depth"] == 2
    assert result.metrics["scene_snapshot_bytes"] > 0
    assert result.metrics["scene_persisted_snapshot_count"] == 1
    assert result.metrics["scene_restored_revision"] == 1
    assert result.details["scene_map"]["rollback_supported"]
    assert (
        result.details["scene_map"]["persistent_store"]
        == "ariadne.global-scene-snapshot.v1"
    )
    assert result.metrics["se3_graph_component_count"] == 2
    assert result.metrics["se3_graph_rejected_constraints"] == 1
    assert result.metrics["se3_graph_revision"] == 5
    assert result.metrics["se3_graph_restored_revision"] == 5
    assert result.metrics["se3_graph_state_restored"] == 1
    assert result.metrics["correction_state_restored"] == 1
    assert result.metrics["correction_duplicate_after_restart_rejected"] == 1
    assert result.metrics["correction_next_sequence_preserved"] == 1
    assert result.metrics["context_node_count"] == 2
    assert result.metrics["context_degraded"] == 0
