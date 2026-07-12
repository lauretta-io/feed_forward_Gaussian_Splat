import unittest

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


if __name__ == "__main__":
    unittest.main()
