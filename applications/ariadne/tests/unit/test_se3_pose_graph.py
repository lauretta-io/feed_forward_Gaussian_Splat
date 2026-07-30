import numpy as np
import pytest

from ariadne.optimization import RobustSE3PoseGraph, SE3PoseConstraint


def constraint(
    source: str,
    destination: str,
    translation: tuple[float, float, float],
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    information: float = 10.0,
) -> SE3PoseConstraint:
    return SE3PoseConstraint(
        source,
        destination,
        np.asarray(translation),
        np.asarray(quaternion),
        np.eye(6) * 0.01,
        information,
    )


def test_se3_graph_propagates_rotation_covariance_and_rejects_outlier() -> None:
    yaw_90 = (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5))
    graph = RobustSE3PoseGraph("a")
    graph.add_constraint(constraint("a", "b", (1.0, 0.0, 0.0), yaw_90))
    graph.add_constraint(constraint("b", "c", (1.0, 0.0, 0.0)))
    graph.add_constraint(constraint("a", "c", (1.0, 1.0, 0.0), yaw_90, 9.0))
    graph.add_constraint(constraint("a", "c", (8.0, -4.0, 0.0), information=0.1))
    graph.add_constraint(constraint("d", "e", (0.0, 0.0, 2.0)))
    result = graph.optimize()
    np.testing.assert_allclose(result.poses["c"][:3, 3], [1.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(result.covariances["c"], np.eye(6) * 0.02)
    assert result.rejected_constraints == (3,)
    assert result.components["a"] == result.components["c"]
    assert result.components["d"] == result.components["e"]
    assert result.components["a"] != result.components["d"]


def test_se3_graph_validates_covariance_and_controls() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        SE3PoseConstraint(
            "a",
            "b",
            np.zeros(3),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.eye(6) * -1,
        )
    with pytest.raises(ValueError, match="gates"):
        RobustSE3PoseGraph("a").optimize(translation_gate_m=0)


def test_se3_graph_snapshot_preserves_revision_and_is_rerunnable(tmp_path) -> None:
    graph = RobustSE3PoseGraph("a", max_constraints=4, max_results=2)
    first = constraint("a", "b", (1.0, 0.0, 0.0))
    assert graph.add_constraint(first)
    assert not graph.add_constraint(first)
    graph.add_constraint(constraint("b", "c", (1.0, 0.0, 0.0)))
    before = graph.optimize()

    restored = RobustSE3PoseGraph.read_json(graph.write_json(tmp_path / "graph.json"))
    after = restored.optimize()
    assert restored.revision == 2
    assert restored.metrics["restores"] == 1
    assert after.revision == before.revision
    assert after.constraint_count == before.constraint_count
    np.testing.assert_allclose(after.poses["c"], before.poses["c"])


def test_se3_graph_bounds_constraints_and_result_history() -> None:
    graph = RobustSE3PoseGraph("a", max_constraints=1, max_results=1)
    graph.add_constraint(constraint("a", "b", (1.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="capacity"):
        graph.add_constraint(constraint("b", "c", (1.0, 0.0, 0.0)))
    graph.optimize()
    graph.optimize()
    assert len(graph.history) == 1
    assert graph.metrics["optimizations"] == 2
