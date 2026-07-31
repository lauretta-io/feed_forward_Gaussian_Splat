"""Bounded Intelligence-node scheduling for per-Wingman pose corrections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrectionDemand:
    agent_id: str
    estimated_error_m: float
    drift_rate_m_s: float
    covariance_trace_m2: float
    seconds_since_correction: float
    link_quality: float
    queue_utilization: float
    payload_bytes: int
    tracking_healthy: bool = True
    orientation_error_rad: float = 0.0
    orientation_drift_rate_rad_s: float = 0.0
    orientation_covariance_trace_rad2: float = 0.0

    def __post_init__(self) -> None:
        finite = (
            self.estimated_error_m,
            self.drift_rate_m_s,
            self.covariance_trace_m2,
            self.seconds_since_correction,
            self.link_quality,
            self.queue_utilization,
            self.orientation_error_rad,
            self.orientation_drift_rate_rad_s,
            self.orientation_covariance_trace_rad2,
        )
        if (
            not self.agent_id
            or not all(np.isfinite(value) for value in finite)
            or min(
                self.estimated_error_m,
                self.drift_rate_m_s,
                self.covariance_trace_m2,
                self.seconds_since_correction,
                self.orientation_error_rad,
                self.orientation_drift_rate_rad_s,
                self.orientation_covariance_trace_rad2,
            )
            < 0
            or not 0 <= self.link_quality <= 1
            or not 0 <= self.queue_utilization <= 1
            or self.payload_bytes <= 0
            or not isinstance(self.tracking_healthy, bool)
        ):
            raise ValueError("correction demand fields are invalid")

    @property
    def error_bound_m(self) -> float:
        return self.estimated_error_m + float(np.sqrt(self.covariance_trace_m2 / 3.0))

    @property
    def orientation_error_bound_rad(self) -> float:
        return self.orientation_error_rad + float(
            np.sqrt(self.orientation_covariance_trace_rad2 / 3.0)
        )


@dataclass(frozen=True)
class CorrectionDecision:
    agent_id: str
    selected: bool
    mandatory: bool
    error_bound_m: float
    predicted_next_error_m: float
    recommended_interval_s: float
    deadline_s: float
    priority: float
    payload_bytes: int
    reason: str
    orientation_error_bound_rad: float = 0.0
    predicted_next_orientation_error_rad: float = 0.0
    translation_deadline_s: float = 0.0
    orientation_deadline_s: float = 0.0


@dataclass(frozen=True)
class CorrectionSchedule:
    decisions: tuple[CorrectionDecision, ...]
    selected_agent_ids: tuple[str, ...]
    relocalization_agent_ids: tuple[str, ...]
    selected_payload_bytes: int
    capacity_overridden: bool


@dataclass(frozen=True)
class CorrectionLoadProfile:
    agent_id: str
    messages_per_minute: float
    minimum_interval_s: float
    peak_corrections_per_second: int
    tracking_healthy: bool
    correction_eligible: bool

    def __post_init__(self) -> None:
        if (
            not self.agent_id
            or not np.isfinite(self.messages_per_minute)
            or not np.isfinite(self.minimum_interval_s)
            or self.messages_per_minute < 0
            or self.minimum_interval_s <= 0
            or self.peak_corrections_per_second < 0
            or not isinstance(self.tracking_healthy, bool)
            or not isinstance(self.correction_eligible, bool)
            or (self.correction_eligible and not self.tracking_healthy)
        ):
            raise ValueError("correction load profile is invalid")


@dataclass(frozen=True)
class CorrectionCapacityAssessment:
    schedulable_agent_ids: tuple[str, ...]
    relocalization_agent_ids: tuple[str, ...]
    tracking_failure_agent_ids: tuple[str, ...]
    live_pose_failure_agent_ids: tuple[str, ...]
    total_messages_per_minute: float
    suppressed_messages_per_minute: float
    configured_messages_per_minute_capacity: float
    peak_corrections_per_second: int
    suppressed_peak_corrections_per_second: int
    configured_peak_corrections_per_second: float
    rate_feasible: bool
    reaction_time_feasible: bool
    burst_feasible: bool
    feasible: bool
    recommended_maximum_evaluation_period_s: float
    action: str


def assess_correction_capacity(
    profiles: tuple[CorrectionLoadProfile, ...],
    *,
    evaluation_period_s: float,
    max_corrections_per_cycle: int,
) -> CorrectionCapacityAssessment:
    """Compare measured Wingman correction envelopes with scheduler capacity."""
    if (
        not np.isfinite(evaluation_period_s)
        or evaluation_period_s <= 0
        or max_corrections_per_cycle <= 0
        or len({profile.agent_id for profile in profiles}) != len(profiles)
    ):
        raise ValueError("correction capacity controls or profiles are invalid")
    schedulable = tuple(profile for profile in profiles if profile.correction_eligible)
    suppressed = tuple(profile for profile in profiles if not profile.correction_eligible)
    tracking_failure_ids = tuple(
        sorted(profile.agent_id for profile in profiles if not profile.tracking_healthy)
    )
    live_pose_failure_ids = tuple(
        sorted(
            profile.agent_id
            for profile in profiles
            if profile.tracking_healthy and not profile.correction_eligible
        )
    )
    relocalization_ids = tuple(sorted((*tracking_failure_ids, *live_pose_failure_ids)))
    schedulable_ids = tuple(sorted(profile.agent_id for profile in schedulable))
    total_rate = float(sum(profile.messages_per_minute for profile in schedulable))
    suppressed_rate = float(sum(profile.messages_per_minute for profile in suppressed))
    peak_rate = sum(profile.peak_corrections_per_second for profile in schedulable)
    suppressed_peak_rate = sum(
        profile.peak_corrections_per_second for profile in suppressed
    )
    configured_rate = max_corrections_per_cycle * 60.0 / evaluation_period_s
    configured_peak = max_corrections_per_cycle / evaluation_period_s
    rate_feasible = total_rate <= configured_rate
    reaction_feasible = all(
        evaluation_period_s <= profile.minimum_interval_s for profile in schedulable
    )
    burst_feasible = peak_rate <= configured_peak
    feasible = rate_feasible and reaction_feasible and burst_feasible
    if schedulable:
        reaction_limit = min(profile.minimum_interval_s for profile in schedulable)
        average_limit = (
            max_corrections_per_cycle * 60.0 / total_rate
            if total_rate > 0
            else reaction_limit
        )
        burst_limit = (
            max_corrections_per_cycle / peak_rate
            if peak_rate > 0
            else reaction_limit
        )
        recommended_period = min(reaction_limit, average_limit, burst_limit)
    else:
        recommended_period = evaluation_period_s
    if not schedulable:
        action = "relocalize_only"
    elif feasible:
        action = "configured_capacity_sufficient"
    else:
        action = "increase_scheduler_frequency"
    return CorrectionCapacityAssessment(
        schedulable_ids,
        relocalization_ids,
        tracking_failure_ids,
        live_pose_failure_ids,
        total_rate,
        suppressed_rate,
        configured_rate,
        peak_rate,
        suppressed_peak_rate,
        configured_peak,
        rate_feasible,
        reaction_feasible,
        burst_feasible,
        feasible,
        recommended_period,
        action,
    )


class CorrectionCadenceScheduler:
    """Prioritize compact corrections without deferring a predicted SE(3) breach."""

    def __init__(
        self,
        *,
        target_error_m: float = 0.1,
        target_orientation_error_rad: float = 0.05,
        nominal_interval_s: float = 9.0,
        minimum_interval_s: float = 2.0,
        maximum_interval_s: float = 60.0,
        evaluation_period_s: float = 1.0,
        max_corrections_per_cycle: int = 2,
    ) -> None:
        controls = (
            target_error_m,
            target_orientation_error_rad,
            nominal_interval_s,
            minimum_interval_s,
            maximum_interval_s,
            evaluation_period_s,
        )
        if (
            not all(np.isfinite(value) and value > 0 for value in controls)
            or not minimum_interval_s <= nominal_interval_s <= maximum_interval_s
            or max_corrections_per_cycle <= 0
        ):
            raise ValueError("correction scheduling controls are invalid")
        self.target_error_m = target_error_m
        self.target_orientation_error_rad = target_orientation_error_rad
        self.nominal_interval_s = nominal_interval_s
        self.minimum_interval_s = minimum_interval_s
        self.maximum_interval_s = maximum_interval_s
        self.evaluation_period_s = evaluation_period_s
        self.max_corrections_per_cycle = max_corrections_per_cycle
        self.metrics = {
            "cycles": 0,
            "selected": 0,
            "deferred": 0,
            "mandatory": 0,
            "capacity_overrides": 0,
            "payload_bytes": 0,
            "relocalizations": 0,
        }

    def schedule(self, demands: tuple[CorrectionDemand, ...]) -> CorrectionSchedule:
        if len({demand.agent_id for demand in demands}) != len(demands):
            raise ValueError("correction demands must be unique per agent")
        candidates = [self._candidate(demand) for demand in demands]
        mandatory = [
            item
            for item in candidates
            if item.mandatory and item.reason != "relocalization_required"
        ]
        optional = [
            item for item in candidates if not item.mandatory and item.reason == "cadence_due"
        ]
        mandatory.sort(key=lambda item: (-item.priority, item.agent_id))
        optional.sort(key=lambda item: (-item.priority, item.agent_id))
        remaining = max(self.max_corrections_per_cycle - len(mandatory), 0)
        selected_ids = {item.agent_id for item in (*mandatory, *optional[:remaining])}
        decisions = tuple(
            CorrectionDecision(
                item.agent_id,
                item.agent_id in selected_ids,
                item.mandatory,
                item.error_bound_m,
                item.predicted_next_error_m,
                item.recommended_interval_s,
                item.deadline_s,
                item.priority,
                item.payload_bytes,
                item.reason
                if item.agent_id in selected_ids or item.reason != "cadence_due"
                else "capacity_deferred",
                item.orientation_error_bound_rad,
                item.predicted_next_orientation_error_rad,
                item.translation_deadline_s,
                item.orientation_deadline_s,
            )
            for item in sorted(candidates, key=lambda candidate: candidate.agent_id)
        )
        selected_payload_bytes = sum(item.payload_bytes for item in decisions if item.selected)
        relocalization_ids = tuple(
            item.agent_id for item in decisions if item.reason == "relocalization_required"
        )
        capacity_overridden = len(mandatory) > self.max_corrections_per_cycle
        self.metrics["cycles"] += 1
        self.metrics["selected"] += len(selected_ids)
        self.metrics["deferred"] += sum(item.reason == "capacity_deferred" for item in decisions)
        self.metrics["mandatory"] += len(mandatory)
        self.metrics["capacity_overrides"] += int(capacity_overridden)
        self.metrics["payload_bytes"] += selected_payload_bytes
        self.metrics["relocalizations"] += len(relocalization_ids)
        return CorrectionSchedule(
            decisions,
            tuple(sorted(selected_ids)),
            relocalization_ids,
            selected_payload_bytes,
            capacity_overridden,
        )

    def _candidate(self, demand: CorrectionDemand) -> CorrectionDecision:
        error_bound = demand.error_bound_m
        orientation_error_bound = demand.orientation_error_bound_rad
        predicted_next = error_bound + demand.drift_rate_m_s * self.evaluation_period_s
        predicted_next_orientation = (
            orientation_error_bound
            + demand.orientation_drift_rate_rad_s * self.evaluation_period_s
        )
        if not demand.tracking_healthy:
            return CorrectionDecision(
                demand.agent_id,
                False,
                False,
                error_bound,
                predicted_next,
                0.0,
                0.0,
                0.0,
                demand.payload_bytes,
                "relocalization_required",
                orientation_error_bound,
                predicted_next_orientation,
            )
        if error_bound >= self.target_error_m:
            translation_deadline_s = 0.0
        elif demand.drift_rate_m_s > 0:
            translation_deadline_s = (
                self.target_error_m - error_bound
            ) / demand.drift_rate_m_s
        else:
            translation_deadline_s = self.maximum_interval_s
        if orientation_error_bound >= self.target_orientation_error_rad:
            orientation_deadline_s = 0.0
        elif demand.orientation_drift_rate_rad_s > 0:
            orientation_deadline_s = (
                self.target_orientation_error_rad - orientation_error_bound
            ) / demand.orientation_drift_rate_rad_s
        else:
            orientation_deadline_s = self.maximum_interval_s
        deadline_s = min(translation_deadline_s, orientation_deadline_s)
        deadline_from_last_s = demand.seconds_since_correction + deadline_s
        load_factor = 1.0 + demand.queue_utilization + 0.5 * (1.0 - demand.link_quality)
        load_adjusted_interval = self.nominal_interval_s * load_factor
        recommended_interval = min(
            self.maximum_interval_s,
            max(self.minimum_interval_s, load_adjusted_interval),
            max(self.minimum_interval_s, deadline_from_last_s),
        )
        translation_mandatory = predicted_next >= self.target_error_m
        orientation_mandatory = (
            predicted_next_orientation >= self.target_orientation_error_rad
        )
        mandatory = translation_mandatory or orientation_mandatory
        due = demand.seconds_since_correction >= recommended_interval
        priority = (
            predicted_next / self.target_error_m
            + predicted_next_orientation / self.target_orientation_error_rad
            + demand.seconds_since_correction / self.maximum_interval_s
            + (1.0 - demand.link_quality) * 0.1
            - demand.queue_utilization * 0.1
        )
        if translation_mandatory and orientation_mandatory:
            reason = "se3_target_deadline"
        elif orientation_mandatory:
            reason = "orientation_target_deadline"
        elif translation_mandatory:
            reason = "target_deadline"
        elif due:
            reason = "cadence_due"
        else:
            reason = "not_due"
        return CorrectionDecision(
            demand.agent_id,
            False,
            mandatory,
            error_bound,
            predicted_next,
            recommended_interval,
            deadline_s,
            priority,
            demand.payload_bytes,
            reason,
            orientation_error_bound,
            predicted_next_orientation,
            translation_deadline_s,
            orientation_deadline_s,
        )
