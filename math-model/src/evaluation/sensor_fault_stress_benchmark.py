"""Frozen stress scenarios for honest sensor-diagnosis boundary testing.

The benchmark separates directly observable safety facts from ambiguous
maintenance hypotheses. Online code never receives the injected truth.
"""

from dataclasses import dataclass
from statistics import mean

import numpy as np

from evaluation.sensor_fault_benchmark import (
    SENSOR_NAMES,
    extract_confirmed_sensor_events,
)
from sensors.sensor_faults import SensorFaultEvent, SensorFaultMode


STRESS_CATEGORIES = ("normal", "strong_direct", "ambiguous")
PROTECTIVE_ACTIONS = {
    "degraded_operation",
    "targeted_reallocation",
    "safe_hold_or_abort",
    "controlled_ascent",
}


@dataclass(frozen=True)
class SensorFaultStressScenario:
    name: str
    category: str
    sensor: str | None = None
    truth_mode: str = "normal"
    events: tuple[SensorFaultEvent, ...] = ()
    disturbance_scale: float = 1.0

    def __post_init__(self):
        name = str(self.name).strip()
        category = str(self.category)
        sensor = None if self.sensor is None else str(self.sensor)
        events = tuple(self.events)
        scale = float(self.disturbance_scale)
        if not name:
            raise ValueError("name cannot be empty")
        if category not in STRESS_CATEGORIES:
            raise ValueError("invalid stress category")
        if sensor is not None and sensor not in SENSOR_NAMES:
            raise ValueError("sensor must be depth, imu, or dvl")
        if category == "normal":
            if sensor is not None or events or self.truth_mode != "normal":
                raise ValueError("normal scenarios cannot contain a fault")
        else:
            if sensor is None or not events:
                raise ValueError("fault scenarios require a sensor and events")
            if any(event.sensor != sensor for event in events):
                raise ValueError("all events must belong to the scenario sensor")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("disturbance_scale must be finite and positive")
        # Reuse injector validation for ordering and overlap rules.
        from sensors.sensor_faults import SensorFaultInjector
        SensorFaultInjector(events)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "sensor", sensor)
        object.__setattr__(self, "truth_mode", str(self.truth_mode))
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "disturbance_scale", scale)

    @property
    def first_fault_time_s(self):
        return None if not self.events else min(
            event.start_time_s for event in self.events
        )

    @property
    def last_recovery_time_s(self):
        if not self.events:
            return None
        return max(
            event.end_time_s
            if event.end_time_s is not None
            else event.start_time_s
            for event in self.events
        )

    def as_dict(self):
        return {
            "name": self.name,
            "category": self.category,
            "sensor": self.sensor,
            "truth_mode": self.truth_mode,
            "disturbance_scale": self.disturbance_scale,
            "events": [
                {
                    "sensor": event.sensor,
                    "mode": event.mode.value,
                    "start_time_s": event.start_time_s,
                    "end_time_s": event.end_time_s,
                    "channels": list(event.channels),
                    "magnitude": event.magnitude,
                    "event_id": event.event_id,
                }
                for event in self.events
            ],
        }


def _event(name, sensor, mode, start, end, channels=(), magnitude=0.0):
    return SensorFaultEvent(
        sensor=sensor,
        mode=SensorFaultMode(mode),
        start_time_s=start,
        end_time_s=end,
        channels=channels,
        magnitude=magnitude,
        event_id=name,
    )


def default_sensor_fault_stress_scenarios():
    """Return the frozen V1 matrix before seeds or outcomes are inspected."""

    scenarios = [
        SensorFaultStressScenario("normal_nominal", "normal"),
        SensorFaultStressScenario(
            "normal_strong_disturbance",
            "normal",
            disturbance_scale=2.5,
        ),
    ]
    spike_specs = {
        "depth": ((), 2.0),
        "imu": ((2,), 0.8),
        "dvl": ((0,), 2.0),
    }
    for sensor in SENSOR_NAMES:
        for mode in ("unavailable", "stuck", "spike"):
            name = f"strong_{sensor}_{mode}"
            channels, magnitude = (
                spike_specs[sensor] if mode == "spike" else ((), 0.0)
            )
            start, end = (6.0, 6.5) if mode == "spike" else (5.0, 9.0)
            scenarios.append(SensorFaultStressScenario(
                name,
                "strong_direct",
                sensor,
                mode,
                (_event(
                    name, sensor, mode, start, end, channels, magnitude
                ),),
            ))

    weak_spikes = {
        "depth": ((), 0.30),
        "imu": ((2,), 0.15),
        "dvl": ((0,), 0.40),
    }
    for sensor, (channels, magnitude) in weak_spikes.items():
        name = f"ambiguous_{sensor}_weak_spike"
        scenarios.append(SensorFaultStressScenario(
            name,
            "ambiguous",
            sensor,
            "spike",
            (_event(
                name, sensor, "spike", 6.0, 6.5, channels, magnitude
            ),),
        ))

    for sensor, channels in (("imu", (2,)), ("dvl", (0,))):
        name = f"ambiguous_{sensor}_partial_stuck"
        scenarios.append(SensorFaultStressScenario(
            name,
            "ambiguous",
            sensor,
            "stuck",
            (_event(name, sensor, "stuck", 5.0, 9.0, channels),),
        ))

    bias_specs = {
        "depth": ((), 0.35),
        "imu": ((2,), 0.15),
        "dvl": ((0,), 0.30),
    }
    drift_specs = {
        "depth": ((), 0.04),
        "imu": ((2,), 0.02),
        "dvl": ((0,), 0.04),
    }
    for mode, specs in (("bias", bias_specs), ("drift", drift_specs)):
        for sensor, (channels, magnitude) in specs.items():
            name = f"ambiguous_{sensor}_{mode}"
            scenarios.append(SensorFaultStressScenario(
                name,
                "ambiguous",
                sensor,
                mode,
                (_event(
                    name, sensor, mode, 5.0, 11.0, channels, magnitude
                ),),
            ))

    for sensor in SENSOR_NAMES:
        name = f"ambiguous_{sensor}_intermittent_unavailable"
        events = tuple(
            _event(
                f"{name}_{index}",
                sensor,
                "unavailable",
                start,
                start + 0.15,
            )
            for index, start in enumerate((5.0, 6.0, 7.0), start=1)
        )
        scenarios.append(SensorFaultStressScenario(
            name,
            "ambiguous",
            sensor,
            "intermittent_unavailable",
            events,
        ))
    return tuple(scenarios)


def _overlaps_fault_window(time_s, scenario, grace_s=0.25):
    return any(
        event.start_time_s <= time_s <= (
            (event.end_time_s or event.start_time_s) + grace_s
        )
        for event in scenario.events
    )


def _expected_direct_action(scenario):
    if scenario.truth_mode == "spike":
        return "log_only"
    if scenario.sensor == "imu":
        return "safe_hold_or_abort"
    return "degraded_operation"


def evaluate_sensor_fault_stress_mission(
    logs,
    scenario,
    recovery_grace_s=0.5,
):
    """Score one stress mission using truth only in this offline evaluator."""

    if not isinstance(scenario, SensorFaultStressScenario):
        raise TypeError("scenario must be SensorFaultStressScenario")
    logs = list(logs)
    if not logs:
        raise ValueError("logs cannot be empty")
    events = extract_confirmed_sensor_events(logs)
    target_events = [
        event for event in events if event["sensor"] == scenario.sensor
    ]
    window_events = [
        event for event in target_events
        if scenario.events and any(
            event["start_time_s"] <= (
                truth.end_time_s or truth.start_time_s
            ) + 0.5
            and event["end_time_s"] >= truth.start_time_s
            for truth in scenario.events
        )
    ]
    exact_types = {
        truth.mode.value for truth in scenario.events
    }
    exact_events = [
        event for event in window_events
        if event["fault_type"] in exact_types
    ]
    conflicting_events = [
        event for event in window_events
        if event["fault_type"] not in exact_types
    ]

    possible_samples = []
    evidence_samples = []
    if scenario.sensor is not None:
        for log in logs:
            time_s = float(log["time"])
            if not _overlaps_fault_window(time_s, scenario):
                continue
            health = log.get("sensor_health", {}).get(
                scenario.sensor, {}
            )
            fault_type = str(health.get("fault_type", "normal"))
            if fault_type == "normal":
                continue
            evidence_samples.append(log)
            if not bool(health.get("confirmed", False)):
                possible_samples.append(log)

    expected_action = (
        _expected_direct_action(scenario)
        if scenario.category == "strong_direct"
        else None
    )
    action_match = (
        None
        if expected_action is None
        else any(log.get("ftc_action") == expected_action for log in logs)
    )
    recovery_start = (
        None
        if scenario.last_recovery_time_s is None
        else scenario.last_recovery_time_s + recovery_grace_s
    )
    post_recovery = [
        log for log in logs
        if recovery_start is not None and float(log["time"]) >= recovery_start
    ]
    post_recovery_protective = any(
        log.get("ftc_action") in PROTECTIVE_ACTIONS
        for log in post_recovery
    )
    all_protective = any(
        log.get("ftc_action") in PROTECTIVE_ACTIONS for log in logs
    )
    direct_spurious = (
        len(events)
        if scenario.category == "normal"
        else max(0, len(events) - min(1, len(exact_events)))
    )
    return {
        "scenario": scenario.name,
        "category": scenario.category,
        "sensor": scenario.sensor,
        "truth_mode": scenario.truth_mode,
        "disturbance_scale": scenario.disturbance_scale,
        "confirmed_event_count": len(events),
        "target_confirmed_event_count": len(target_events),
        "exact_event_detected": bool(exact_events),
        "exact_event_count": len(exact_events),
        "conflicting_confirmed_event_count": len(conflicting_events),
        "direct_spurious_event_count": direct_spurious,
        "possible_evidence_observed": bool(possible_samples),
        "any_fault_evidence_observed": bool(evidence_samples),
        "expected_ftc_action": expected_action,
        "correct_ftc_action_observed": action_match,
        "protective_action_observed": all_protective,
        "post_recovery_protective_action_observed": (
            post_recovery_protective
        ),
        "wrong_thruster_target_count": sum(
            log.get("ftc_targeted_thruster_name") is not None
            for log in logs
        ),
    }


def summarize_sensor_fault_stress_benchmark(rows):
    """Aggregate safety, certainty, and ambiguous-coverage boundaries."""

    rows = list(rows)
    if not rows:
        raise ValueError("rows cannot be empty")
    normal = [row for row in rows if row["category"] == "normal"]
    direct = [row for row in rows if row["category"] == "strong_direct"]
    ambiguous = [row for row in rows if row["category"] == "ambiguous"]
    if not normal or not direct or not ambiguous:
        raise ValueError("rows must contain all stress categories")

    direct_true = sum(row["exact_event_detected"] for row in direct)
    direct_predictions = sum(
        row["confirmed_event_count"] for row in direct + normal
    )
    direct_recall = direct_true / len(direct)
    direct_precision = direct_true / max(direct_predictions, 1)
    direct_action_match = mean(
        row["correct_ftc_action_observed"] for row in direct
    )
    ambiguous_observation = mean(
        row["any_fault_evidence_observed"] for row in ambiguous
    )
    ambiguous_possible = mean(
        row["possible_evidence_observed"] for row in ambiguous
    )
    ambiguous_exact = mean(
        row["exact_event_detected"] for row in ambiguous
    )
    ambiguous_conflicting = mean(
        row["conflicting_confirmed_event_count"] > 0
        for row in ambiguous
    )
    normal_false_confirmed = sum(
        row["confirmed_event_count"] > 0 for row in normal
    )
    normal_false_protective = sum(
        row["protective_action_observed"] for row in normal
    )
    ambiguous_post_recovery = sum(
        row["post_recovery_protective_action_observed"]
        for row in ambiguous
    )
    wrong_thruster_targets = sum(
        row["wrong_thruster_target_count"] for row in rows
    )
    checks = {
        "strong_direct_recall_at_least_95pct": direct_recall >= 0.95,
        "strong_direct_precision_at_least_95pct": direct_precision >= 0.95,
        "strong_direct_ftc_action_match_at_least_95pct": (
            direct_action_match >= 0.95
        ),
        "normal_false_confirmed_missions_zero": (
            normal_false_confirmed == 0
        ),
        "normal_false_protective_missions_zero": (
            normal_false_protective == 0
        ),
        "ambiguous_evidence_observation_at_least_50pct": (
            ambiguous_observation >= 0.50
        ),
        "ambiguous_conflicting_certainty_at_most_10pct": (
            ambiguous_conflicting <= 0.10
        ),
        "ambiguous_post_recovery_protective_missions_zero": (
            ambiguous_post_recovery == 0
        ),
        "wrong_thruster_targets_zero": wrong_thruster_targets == 0,
    }

    per_scenario = {}
    for name in sorted({row["scenario"] for row in rows}):
        selected = [row for row in rows if row["scenario"] == name]
        per_scenario[name] = {
            "category": selected[0]["category"],
            "sensor": selected[0]["sensor"],
            "truth_mode": selected[0]["truth_mode"],
            "missions": len(selected),
            "exact_event_rate": mean(
                row["exact_event_detected"] for row in selected
            ),
            "any_evidence_rate": mean(
                row["any_fault_evidence_observed"] for row in selected
            ),
            "possible_evidence_rate": mean(
                row["possible_evidence_observed"] for row in selected
            ),
            "conflicting_certainty_rate": mean(
                row["conflicting_confirmed_event_count"] > 0
                for row in selected
            ),
            "post_recovery_protective_missions": sum(
                row["post_recovery_protective_action_observed"]
                for row in selected
            ),
        }

    return {
        "evaluation_type": "frozen_sensor_fault_stress_v1",
        "mission_count": len(rows),
        "normal_mission_count": len(normal),
        "strong_direct_mission_count": len(direct),
        "ambiguous_mission_count": len(ambiguous),
        "strong_direct_event_recall": direct_recall,
        "strong_direct_event_precision": direct_precision,
        "strong_direct_ftc_action_match_rate": direct_action_match,
        "normal_false_confirmed_missions": normal_false_confirmed,
        "normal_false_protective_missions": normal_false_protective,
        "ambiguous_fault_evidence_observation_rate": ambiguous_observation,
        "ambiguous_possible_only_evidence_rate": ambiguous_possible,
        "ambiguous_exact_classification_rate": ambiguous_exact,
        "ambiguous_conflicting_certainty_rate": ambiguous_conflicting,
        "ambiguous_post_recovery_protective_missions": (
            ambiguous_post_recovery
        ),
        "wrong_thruster_target_count": wrong_thruster_targets,
        "certainty_interpretation": {
            "strong_direct": (
                "May be shown as confirmed only when the per-scenario frozen "
                "recall remains at least 95%."
            ),
            "ambiguous": (
                "Must remain possible/log-only unless a later independent "
                "frozen benchmark validates exact classification."
            ),
            "brief_unavailability": (
                "Current-sample unavailability may be certain while hardware "
                "failure remains an unconfirmed maintenance hypothesis."
            ),
        },
        "per_scenario": per_scenario,
        "acceptance_checks": checks,
        "all_acceptance_checks_passed": all(checks.values()),
    }
