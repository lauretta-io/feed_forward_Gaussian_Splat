import unittest

from ariadne.benchmarks import run_phase1_benchmark


class Phase1BenchmarkTest(unittest.TestCase):
    def test_integrated_reference_pipeline_passes_deterministically(self) -> None:
        first = run_phase1_benchmark(seed=11)
        second = run_phase1_benchmark(seed=11)
        self.assertEqual(first.status, "passed")
        self.assertEqual(first.metrics["synchronized_packets"], 50)
        self.assertLess(first.metrics["fused_ate_rmse_m"], first.metrics["imu_ate_rmse_m"])
        self.assertEqual(first.metrics["false_static_insertions"], 0)
        self.assertEqual(first.metrics["global_object_count"], 1)
        self.assertEqual(
            first.metrics["pose_graph_rejected_constraints"],
            second.metrics["pose_graph_rejected_constraints"],
        )


if __name__ == "__main__":
    unittest.main()
