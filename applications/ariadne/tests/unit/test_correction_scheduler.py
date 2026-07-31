import pytest

from ariadne.pose_correction import (
    CorrectionCadenceScheduler,
    CorrectionDemand,
    CorrectionLoadProfile,
    assess_correction_capacity,
)


def demand(
    agent_id: str,
    *,
    error_m: float = 0.01,
    drift_m_s: float = 0.0,
    covariance_m2: float = 0.0,
    age_s: float = 10.0,
    link: float = 1.0,
    queue: float = 0.0,
    tracking_healthy: bool = True,
    orientation_error_rad: float = 0.0,
    orientation_drift_rad_s: float = 0.0,
    orientation_covariance_rad2: float = 0.0,
) -> CorrectionDemand:
    return CorrectionDemand(
        agent_id,
        error_m,
        drift_m_s,
        covariance_m2,
        age_s,
        link,
        queue,
        128,
        tracking_healthy,
        orientation_error_rad,
        orientation_drift_rad_s,
        orientation_covariance_rad2,
    )


def test_scheduler_defers_safe_work_under_queue_pressure() -> None:
    scheduler = CorrectionCadenceScheduler(max_corrections_per_cycle=1)
    schedule = scheduler.schedule(
        (
            demand("clear"),
            demand("loaded", link=0.5, queue=0.8),
        )
    )

    assert schedule.selected_agent_ids == ("clear",)
    decisions = {item.agent_id: item for item in schedule.decisions}
    assert decisions["clear"].reason == "cadence_due"
    assert decisions["loaded"].reason == "not_due"
    assert schedule.selected_payload_bytes == 128


def test_scheduler_never_defers_predicted_target_breaches() -> None:
    scheduler = CorrectionCadenceScheduler(
        evaluation_period_s=2.0,
        max_corrections_per_cycle=1,
    )
    schedule = scheduler.schedule(
        tuple(
            demand(
                agent,
                error_m=0.095,
                drift_m_s=0.01,
                age_s=1.0,
                queue=1.0,
                link=0.1,
            )
            for agent in ("alpha", "bob", "carol")
        )
    )

    assert schedule.selected_agent_ids == ("alpha", "bob", "carol")
    assert schedule.capacity_overridden
    assert all(item.mandatory for item in schedule.decisions)
    assert scheduler.metrics["capacity_overrides"] == 1


def test_scheduler_makes_orientation_deadline_mandatory_when_translation_is_safe() -> None:
    scheduler = CorrectionCadenceScheduler(
        target_orientation_error_rad=0.05,
        evaluation_period_s=0.1,
        max_corrections_per_cycle=1,
    )

    schedule = scheduler.schedule(
        (
            demand(
                "rotating",
                error_m=0.01,
                orientation_error_rad=0.045,
                orientation_drift_rad_s=0.1,
                age_s=0.1,
            ),
        )
    )

    decision = schedule.decisions[0]
    assert schedule.selected_agent_ids == ("rotating",)
    assert decision.mandatory
    assert decision.reason == "orientation_target_deadline"
    assert decision.predicted_next_error_m < 0.1
    assert decision.predicted_next_orientation_error_rad == pytest.approx(0.055)
    assert decision.orientation_deadline_s == pytest.approx(0.05)


def test_scheduler_prioritizes_highest_error_when_optional_capacity_is_full() -> None:
    scheduler = CorrectionCadenceScheduler(max_corrections_per_cycle=1)
    schedule = scheduler.schedule(
        (
            demand("lower", error_m=0.01, age_s=20.0),
            demand("higher", error_m=0.04, age_s=20.0),
        )
    )

    assert schedule.selected_agent_ids == ("higher",)
    decisions = {item.agent_id: item for item in schedule.decisions}
    assert decisions["lower"].reason == "capacity_deferred"
    assert scheduler.metrics["deferred"] == 1


def test_scheduler_rejects_invalid_or_duplicate_demands() -> None:
    with pytest.raises(ValueError, match="invalid"):
        demand("bad", link=1.1)
    scheduler = CorrectionCadenceScheduler()
    duplicate = demand("same")
    with pytest.raises(ValueError, match="unique"):
        scheduler.schedule((duplicate, duplicate))


def test_unhealthy_tracking_requests_relocalization_without_using_capacity() -> None:
    scheduler = CorrectionCadenceScheduler(max_corrections_per_cycle=1)
    schedule = scheduler.schedule(
        (
            demand("unhealthy", error_m=1.0, tracking_healthy=False),
            demand("healthy", error_m=0.2, drift_m_s=0.01),
        )
    )

    assert schedule.selected_agent_ids == ("healthy",)
    assert schedule.relocalization_agent_ids == ("unhealthy",)
    assert next(
        item for item in schedule.decisions if item.agent_id == "unhealthy"
    ).reason == "relocalization_required"
    assert scheduler.metrics["relocalizations"] == 1


def test_capacity_assessment_separates_average_load_from_reaction_bursts() -> None:
    profiles = (
        CorrectionLoadProfile("Alpha", 70.37, 0.1, 4, True, True),
        CorrectionLoadProfile("Bob", 365.60, 0.1, 10, False, False),
        CorrectionLoadProfile("Carol", 528.45, 0.099, 11, False, False),
    )

    current = assess_correction_capacity(
        profiles,
        evaluation_period_s=1.0,
        max_corrections_per_cycle=2,
    )
    faster = assess_correction_capacity(
        profiles,
        evaluation_period_s=0.09,
        max_corrections_per_cycle=2,
    )

    assert current.schedulable_agent_ids == ("Alpha",)
    assert current.relocalization_agent_ids == ("Bob", "Carol")
    assert current.tracking_failure_agent_ids == ("Bob", "Carol")
    assert current.live_pose_failure_agent_ids == ()
    assert current.total_messages_per_minute == pytest.approx(70.37)
    assert current.suppressed_messages_per_minute == pytest.approx(894.05)
    assert current.suppressed_peak_corrections_per_second == 21
    assert current.configured_messages_per_minute_capacity == pytest.approx(120.0)
    assert current.rate_feasible
    assert not current.reaction_time_feasible
    assert not current.burst_feasible
    assert not current.feasible
    assert current.recommended_maximum_evaluation_period_s == pytest.approx(0.1)
    assert current.action == "increase_scheduler_frequency"
    assert faster.feasible
    assert faster.action == "configured_capacity_sufficient"


def test_capacity_assessment_requests_relocalization_when_no_profile_passes() -> None:
    assessment = assess_correction_capacity(
        (
            CorrectionLoadProfile("Alpha", 58.8, 1.0, 1, True, False),
            CorrectionLoadProfile("Bob", 59.5, 1.0, 1, False, False),
        ),
        evaluation_period_s=1.0,
        max_corrections_per_cycle=2,
    )

    assert assessment.schedulable_agent_ids == ()
    assert assessment.relocalization_agent_ids == ("Alpha", "Bob")
    assert assessment.tracking_failure_agent_ids == ("Bob",)
    assert assessment.live_pose_failure_agent_ids == ("Alpha",)
    assert assessment.total_messages_per_minute == 0.0
    assert assessment.suppressed_messages_per_minute == pytest.approx(118.3)
    assert assessment.suppressed_peak_corrections_per_second == 2
    assert assessment.feasible
    assert assessment.action == "relocalize_only"


def test_capacity_profile_rejects_correction_eligibility_without_tracking() -> None:
    with pytest.raises(ValueError, match="invalid"):
        CorrectionLoadProfile("failed", 10.0, 1.0, 1, False, True)
