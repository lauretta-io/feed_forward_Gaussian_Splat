import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ariadne.common import Timestamp
from ariadne.tracking import (
    CrossAgentAssociator,
    StaticTrackState,
    TemporalStaticFilter,
    TrackObservation,
)


class TrackingTest(unittest.TestCase):
    def _observation(self, agent: str, track: str, step: int, motion: float) -> TrackObservation:
        return TrackObservation(
            Timestamp(step),
            agent,
            track,
            np.array([1.0, 2.0, 0.0]),
            np.array([1.0, 0.0]),
            motion,
            0.98 if motion < 0.1 else 0.1,
            0.98 if motion < 0.1 else 0.1,
        )

    def test_confirmation_requires_history_and_dynamic_is_not_associated(self) -> None:
        classifier = TemporalStaticFilter()
        associator = CrossAgentAssociator()
        for step in range(3):
            first = classifier.update(self._observation("one", "landmark", step, 0.01))
            second = classifier.update(self._observation("two", "landmark", step, 0.01))
        self.assertIs(first.state, StaticTrackState.STATIC_CONFIRMED)
        one = associator.associate(first)
        two = associator.associate(second)
        self.assertIsNotNone(one)
        self.assertEqual(one.global_id, two.global_id)
        moving = classifier.update(self._observation("one", "moving", 0, 2.0))
        self.assertIs(moving.state, StaticTrackState.DYNAMIC)
        self.assertIsNone(associator.associate(moving))

    def test_association_evidence_and_global_ids_survive_restart(self) -> None:
        classifier = TemporalStaticFilter()
        states = {}
        for step in range(3):
            for agent in ("one", "two"):
                states[agent] = classifier.update(
                    self._observation(agent, "landmark", step, 0.01)
                )
        associator = CrossAgentAssociator(max_evidence=2)
        first = associator.associate(states["one"])
        second = associator.associate(states["two"])
        associator.associate(states["one"])
        self.assertEqual(first.global_id, second.global_id)
        self.assertEqual(
            [item.decision for item in associator.evidence],
            ["matched", "existing"],
        )

        with TemporaryDirectory() as directory:
            snapshot = associator.write_json(Path(directory) / "association.json")
            restored = CrossAgentAssociator.read_json(snapshot)
        replayed = restored.associate(states["one"])
        self.assertEqual(replayed.global_id, first.global_id)
        self.assertEqual(restored.metrics["restores"], 1)

        for step in range(3, 6):
            far = classifier.update(
                TrackObservation(
                    Timestamp(step),
                    "three",
                    "other",
                    np.array([20.0, 2.0, 0.0]),
                    np.array([0.0, 1.0]),
                    0.01,
                    0.98,
                    0.98,
                )
            )
        created = restored.associate(far)
        self.assertEqual(created.global_id, "global_0002")

    def test_association_object_capacity_fails_closed(self) -> None:
        classifier = TemporalStaticFilter()
        associator = CrossAgentAssociator(max_objects=1)
        states = []
        for agent, position in (("one", 1.0), ("two", 20.0)):
            for step in range(3):
                state = classifier.update(
                    TrackObservation(
                        Timestamp(step),
                        agent,
                        "landmark",
                        np.array([position, 2.0, 0.0]),
                        np.array([1.0, 0.0]),
                        0.01,
                        0.98,
                        0.98,
                    )
                )
            states.append(state)
        self.assertIsNotNone(associator.associate(states[0]))
        self.assertIsNone(associator.associate(states[1]))
        self.assertEqual(associator.metrics["capacity_rejected"], 1)


if __name__ == "__main__":
    unittest.main()
