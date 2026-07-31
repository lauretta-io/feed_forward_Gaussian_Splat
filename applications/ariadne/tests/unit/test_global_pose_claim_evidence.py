import unittest

from ariadne.benchmarks.global_pose_rationalization import GlobalPoseClaimEvidence


class GlobalPoseClaimEvidenceTest(unittest.TestCase):
    def test_independent_measured_causal_factors_are_eligible(self) -> None:
        evidence = GlobalPoseClaimEvidence(
            odometry_source="production",
            position_reference_source="measured",
            orientation_reference_source="measured",
            cross_agent_source="measured",
            uses_ground_truth_in_estimator=False,
            causal=True,
            all_agents_position_target_met=True,
            fleet_position_target_met=True,
            orientation_target_met=True,
        )

        self.assertTrue(evidence.position_claim_eligible)
        self.assertTrue(evidence.full_pose_claim_eligible)
        self.assertEqual(evidence.position_claim_reasons, ())
        self.assertEqual(evidence.full_pose_claim_reasons, ())

    def test_controlled_factors_and_ground_truth_fail_closed(self) -> None:
        evidence = GlobalPoseClaimEvidence(
            odometry_source="controlled",
            position_reference_source="ground_truth_derived",
            orientation_reference_source="controlled",
            cross_agent_source="controlled",
            uses_ground_truth_in_estimator=True,
            causal=True,
            all_agents_position_target_met=True,
            fleet_position_target_met=True,
            orientation_target_met=True,
        )

        self.assertFalse(evidence.position_claim_eligible)
        self.assertFalse(evidence.full_pose_claim_eligible)
        self.assertIn("estimator consumes evaluation ground truth", evidence.position_claim_reasons)
        self.assertIn(
            "orientation factors are not independent measured observations",
            evidence.full_pose_claim_reasons,
        )

    def test_missing_orientation_only_blocks_full_pose_claim(self) -> None:
        evidence = GlobalPoseClaimEvidence(
            odometry_source="production",
            position_reference_source="measured",
            orientation_reference_source="unavailable",
            cross_agent_source="measured",
            uses_ground_truth_in_estimator=False,
            causal=True,
            all_agents_position_target_met=True,
            fleet_position_target_met=True,
            orientation_target_met=False,
        )

        self.assertTrue(evidence.position_claim_eligible)
        self.assertFalse(evidence.full_pose_claim_eligible)


if __name__ == "__main__":
    unittest.main()
