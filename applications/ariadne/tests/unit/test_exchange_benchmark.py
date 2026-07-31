from ariadne.benchmarks import run_exchange_benchmark


def test_exchange_benchmark_passes_and_is_seed_reproducible() -> None:
    first = run_exchange_benchmark(seed=13)
    second = run_exchange_benchmark(seed=13)
    assert first.status == "passed"
    assert first.metrics["saliency_region_count"] == second.metrics["saliency_region_count"]
    assert first.metrics["local_object_count"] == 1
    assert first.metrics["transport_delivered"] == 2
    assert first.metrics["transport_retries"] == 1
    assert first.metrics["transport_acknowledged"] == 1
    assert first.metrics["transport_pending"] == 0
    assert first.metrics["registry_observation_count"] == 1
    assert first.metrics["registry_duplicate_packets"] == 1
    assert first.metrics["local_object_snapshot_restored"] == 1
    assert first.metrics["registry_snapshot_restored"] == 1
    assert first.metrics["registry_journal_entries"] == 1
    assert first.metrics["registry_journal_replayed_observations"] == 1
