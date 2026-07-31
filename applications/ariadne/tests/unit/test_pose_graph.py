import unittest

import numpy as np

from ariadne.optimization import IncrementalPoseGraph, PoseConstraint


class PoseGraphTest(unittest.TestCase):
    def test_robust_graph_rejects_false_constraint(self) -> None:
        graph = IncrementalPoseGraph("a")
        graph.add_constraint(PoseConstraint("a", "b", np.array([1.0, 0.0, 0.0]), 10.0))
        graph.add_constraint(PoseConstraint("b", "c", np.array([1.0, 0.0, 0.0]), 10.0))
        graph.add_constraint(PoseConstraint("a", "c", np.array([2.0, 0.0, 0.0]), 10.0))
        graph.add_constraint(PoseConstraint("a", "c", np.array([20.0, 0.0, 0.0]), 0.2))
        result = graph.optimize(huber_delta_m=0.25)
        np.testing.assert_allclose(result.positions_m["c"], [2.0, 0.0, 0.0], atol=0.05)
        self.assertEqual(result.rejected_constraints, (3,))


if __name__ == "__main__":
    unittest.main()
