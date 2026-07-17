"""Development metrics for the log-only ambiguous sensor observer."""

from statistics import mean

from evaluation.sensor_fault_stress_benchmark import (
    SensorFaultStressScenario,
    evaluate_sensor_fault_stress_mission,
    summarize_sensor_fault_stress_benchmark,
)


SENSOR_NAMES = ("depth", "imu", "dvl")
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
LOG_ONLY_POLICY_SCENARIOS = (
    "ambiguous_depth_weak_spike",
    "ambiguous_imu_weak_spike",
    "ambiguous_dvl_weak_spike",
    "ambiguous_depth_drift",
    "ambiguous_dvl_drift",
)


def _inside_fault_window(time_s, scenario, grace_s=0.5):
    return any(
        event.start_time_s <= time_s <= (
            (event.end_time_s or event.start_time_s) + grace_s
        )
        for event in scenario.events
    )


def evaluate_sensor_fault_observer_mission(logs, scenario):
    """Evaluate fast certainty and log-only possibility as separate tiers."""

    if not isinstance(scenario, SensorFaultStressScenario):
        raise TypeError("scenario must be SensorFaultStressScenario")
    logs = list(logs)
    base = evaluate_sensor_fault_stress_mission(logs, scenario)
    target_possible = []
    target_operator_possible = []
    all_possible = []
    operator_possible = []
    observer_confirmed_count = 0
    observer_protective_count = 0
    for log in logs:
        time_s = float(log["time"])
        observations = log.get("sensor_fault_observations", {})
        for sensor in SENSOR_NAMES:
            observation = observations.get(sensor, {})
            if bool(observation.get("confirmed", False)):
                observer_confirmed_count += 1
            if observation.get("state") == "possible_fault":
                all_possible.append((sensor, observation))
                if observation.get("display_level") == "possible":
                    operator_possible.append((sensor, observation))
                if (
                    sensor == scenario.sensor
                    and _inside_fault_window(time_s, scenario)
                ):
                    target_possible.append(observation)
                    if observation.get("display_level") == "possible":
                        target_operator_possible.append(observation)
        summary = log.get("sensor_fault_observation_summary", {})
        if bool(summary.get("protective_action_required", False)):
            observer_protective_count += 1

    hypotheses = sorted({
        str(observation.get("hypothesis"))
        for observation in target_possible
    })
    all_hypotheses = sorted({
        f"{sensor}:{observation.get('hypothesis')}"
        for sensor, observation in all_possible
    })
    possible_observed = bool(target_possible)
    operator_possible_observed = bool(target_operator_possible)
    if scenario.category == "normal":
        display_tier = "none" if not operator_possible else "possible"
    elif scenario.category == "strong_direct" and base["exact_event_detected"]:
        display_tier = "confirmed"
    elif operator_possible_observed:
        display_tier = "possible"
    elif scenario.truth_mode == "intermittent_unavailable" and base[
        "exact_event_detected"
    ]:
        display_tier = "possible"
    else:
        display_tier = "log_only"
    base.update({
        "observer_possible_evidence_observed": possible_observed,
        "observer_operator_possible_evidence_observed": (
            operator_possible_observed
        ),
        "observer_possible_hypotheses": hypotheses,
        "observer_possible_sample_count": len(target_possible),
        "any_observer_possible_sample_count": len(all_possible),
        "observer_all_possible_hypotheses": all_hypotheses,
        "observer_operator_possible_sample_count": len(operator_possible),
        "observer_confirmed_count": observer_confirmed_count,
        "observer_protective_count": observer_protective_count,
        "display_tier": display_tier,
    })
    return base


def summarize_sensor_fault_observer_benchmark(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("rows cannot be empty")
    base = summarize_sensor_fault_stress_benchmark(rows)
    normal = [row for row in rows if row["category"] == "normal"]
    ambiguous = [row for row in rows if row["category"] == "ambiguous"]
    possible_rate = mean(
        row["observer_possible_evidence_observed"] for row in ambiguous
    )
    normal_possible_missions = sum(
        row["observer_operator_possible_sample_count"] > 0 for row in normal
    )
    conflicting_rate = mean(
        row["conflicting_confirmed_event_count"] > 0
        for row in ambiguous
    )
    observer_confirmed = sum(row["observer_confirmed_count"] for row in rows)
    observer_protective = sum(
        row["observer_protective_count"] for row in rows
    )
    possible_display_rate = mean(
        row["observer_operator_possible_evidence_observed"]
        for row in ambiguous
    )
    required_rows = [
        row for row in ambiguous
        if row["scenario"] in REQUIRED_OPERATOR_POSSIBLE_SCENARIOS
    ]
    log_only_rows = [
        row for row in ambiguous
        if row["scenario"] in LOG_ONLY_POLICY_SCENARIOS
    ]
    required_operator_possible_rate = mean(
        row["observer_operator_possible_evidence_observed"]
        for row in required_rows
    )
    log_only_overpromoted_missions = sum(
        row["observer_operator_possible_evidence_observed"]
        for row in log_only_rows
    )
    checks = {
        "strong_direct_recall_at_least_95pct": (
            base["strong_direct_event_recall"] >= 0.95
        ),
        "strong_direct_precision_at_least_95pct": (
            base["strong_direct_event_precision"] >= 0.95
        ),
        "normal_false_confirmed_missions_zero": (
            base["normal_false_confirmed_missions"] == 0
        ),
        "normal_operator_possible_missions_zero": (
            normal_possible_missions == 0
        ),
        "ambiguous_possible_evidence_at_least_70pct": (
            possible_rate >= 0.70
        ),
        "required_operator_possible_rate_at_least_95pct": (
            required_operator_possible_rate >= 0.95
        ),
        "log_only_policy_overpromoted_missions_zero": (
            log_only_overpromoted_missions == 0
        ),
        "ambiguous_conflicting_certainty_zero": conflicting_rate == 0.0,
        "observer_never_confirms": observer_confirmed == 0,
        "observer_never_requests_protection": observer_protective == 0,
        "wrong_thruster_targets_zero": (
            base["wrong_thruster_target_count"] == 0
        ),
    }
    per_scenario = {}
    for name in sorted({row["scenario"] for row in rows}):
        selected = [row for row in rows if row["scenario"] == name]
        tiers = {
            tier: sum(row["display_tier"] == tier for row in selected)
            for tier in ("none", "log_only", "possible", "confirmed")
        }
        per_scenario[name] = {
            "category": selected[0]["category"],
            "sensor": selected[0]["sensor"],
            "truth_mode": selected[0]["truth_mode"],
            "missions": len(selected),
            "fast_exact_event_rate": mean(
                row["exact_event_detected"] for row in selected
            ),
            "raw_possible_evidence_rate": mean(
                row["observer_possible_evidence_observed"]
                for row in selected
            ),
            "operator_possible_rate": mean(
                row["observer_operator_possible_evidence_observed"]
                for row in selected
            ),
            "conflicting_certainty_rate": mean(
                row["conflicting_confirmed_event_count"] > 0
                for row in selected
            ),
            "display_tier_counts": tiers,
        }
    return {
        "evaluation_type": "sensor_fault_observer_development",
        "mission_count": len(rows),
        "strong_direct_event_recall": base["strong_direct_event_recall"],
        "strong_direct_event_precision": base["strong_direct_event_precision"],
        "strong_direct_ftc_action_match_rate": (
            base["strong_direct_ftc_action_match_rate"]
        ),
        "normal_false_confirmed_missions": (
            base["normal_false_confirmed_missions"]
        ),
        "normal_false_protective_missions": (
            base["normal_false_protective_missions"]
        ),
        "normal_possible_observation_missions": normal_possible_missions,
        "ambiguous_possible_evidence_rate": possible_rate,
        "ambiguous_possible_display_rate": possible_display_rate,
        "required_operator_possible_rate": required_operator_possible_rate,
        "log_only_policy_overpromoted_missions": (
            log_only_overpromoted_missions
        ),
        "ambiguous_conflicting_certainty_rate": conflicting_rate,
        "observer_confirmed_sample_count": observer_confirmed,
        "observer_protective_request_count": observer_protective,
        "wrong_thruster_target_count": base["wrong_thruster_target_count"],
        "per_scenario": per_scenario,
        "acceptance_checks": checks,
        "all_acceptance_checks_passed": all(checks.values()),
    }
