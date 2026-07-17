"""V3 display-policy metrics for the frozen sensor observation benchmark."""

from statistics import mean

from evaluation.sensor_fault_observer_benchmark import (
    summarize_sensor_fault_observer_benchmark,
)


REQUIRED_OPERATOR_POSSIBLE_SCENARIOS = (
    "ambiguous_imu_partial_stuck",
    "ambiguous_dvl_partial_stuck",
    "ambiguous_depth_bias",
    "ambiguous_imu_bias",
    "ambiguous_dvl_bias",
    "ambiguous_imu_drift",
    "ambiguous_depth_intermittent_unavailable",
    "ambiguous_imu_intermittent_unavailable",
    "ambiguous_dvl_intermittent_unavailable",
)
OPTIONAL_OPERATOR_POSSIBLE_SCENARIOS = (
    "ambiguous_depth_drift",
    "ambiguous_dvl_drift",
)
REQUIRED_LOG_ONLY_SCENARIOS = (
    "ambiguous_depth_weak_spike",
    "ambiguous_imu_weak_spike",
    "ambiguous_dvl_weak_spike",
)


def sensor_fault_observer_display_policy_metrics(rows):
    """Score the V3 required, optional, and log-only display groups."""

    rows = list(rows)
    required = [
        row for row in rows
        if row["scenario"] in REQUIRED_OPERATOR_POSSIBLE_SCENARIOS
    ]
    optional = [
        row for row in rows
        if row["scenario"] in OPTIONAL_OPERATOR_POSSIBLE_SCENARIOS
    ]
    log_only = [
        row for row in rows
        if row["scenario"] in REQUIRED_LOG_ONLY_SCENARIOS
    ]
    if not required or not optional or not log_only:
        raise ValueError("rows do not cover all V3 display-policy groups")
    return {
        "required_operator_possible_rate": mean(
            row["observer_operator_possible_evidence_observed"]
            for row in required
        ),
        "optional_operator_possible_rate": mean(
            row["observer_operator_possible_evidence_observed"]
            for row in optional
        ),
        "optional_raw_evidence_rate": mean(
            row["observer_possible_evidence_observed"]
            for row in optional
        ),
        "required_log_only_overpromoted_missions": sum(
            row["observer_operator_possible_evidence_observed"]
            for row in log_only
        ),
    }


def summarize_sensor_fault_observer_benchmark_v3(rows):
    """Apply the predeclared V3 display policy to mission-level rows."""

    rows = list(rows)
    base = summarize_sensor_fault_observer_benchmark(rows)
    policy = sensor_fault_observer_display_policy_metrics(rows)
    checks = {
        "strong_direct_recall_at_least_95pct": (
            base["strong_direct_event_recall"] >= 0.95
        ),
        "strong_direct_precision_at_least_95pct": (
            base["strong_direct_event_precision"] >= 0.95
        ),
        "strong_direct_ftc_action_match_at_least_95pct": (
            base["strong_direct_ftc_action_match_rate"] >= 0.95
        ),
        "normal_false_confirmed_missions_zero": (
            base["normal_false_confirmed_missions"] == 0
        ),
        "normal_false_protective_missions_zero": (
            base["normal_false_protective_missions"] == 0
        ),
        "normal_operator_possible_missions_zero": (
            base["normal_possible_observation_missions"] == 0
        ),
        "ambiguous_raw_possible_evidence_at_least_90pct": (
            base["ambiguous_possible_evidence_rate"] >= 0.90
        ),
        "required_operator_possible_rate_at_least_95pct": (
            policy["required_operator_possible_rate"] >= 0.95
        ),
        "required_log_only_overpromoted_missions_zero": (
            policy["required_log_only_overpromoted_missions"] == 0
        ),
        "ambiguous_conflicting_certainty_zero": (
            base["ambiguous_conflicting_certainty_rate"] == 0.0
        ),
        "observer_never_confirms": (
            base["observer_confirmed_sample_count"] == 0
        ),
        "observer_never_requests_protection": (
            base["observer_protective_request_count"] == 0
        ),
        "wrong_thruster_targets_zero": (
            base["wrong_thruster_target_count"] == 0
        ),
    }
    return {
        **base,
        "evaluation_type": "frozen_sensor_fault_observer_v3",
        "display_policy": {
            "required_operator_possible": list(
                REQUIRED_OPERATOR_POSSIBLE_SCENARIOS
            ),
            "optional_operator_possible": list(
                OPTIONAL_OPERATOR_POSSIBLE_SCENARIOS
            ),
            "required_log_only": list(REQUIRED_LOG_ONLY_SCENARIOS),
        },
        **policy,
        "acceptance_checks": checks,
        "all_acceptance_checks_passed": all(checks.values()),
    }
