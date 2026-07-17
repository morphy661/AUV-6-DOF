"""Scenario definitions and event metrics for sensor-fault evaluation."""

from dataclasses import dataclass
from statistics import mean, median
from typing import Optional

import numpy as np

from sensors.sensor_faults import (
    SensorFaultEvent,
    SensorFaultMode,
)


SENSOR_NAMES = ("depth", "imu", "dvl")
FAULT_MODES = ("unavailable", "stuck", "spike")


@dataclass(frozen=True)
class SensorFaultBenchmarkScenario:
    name: str
    sensor: Optional[str]
    mode: str
    start_time_s: float = 5.0
    duration_s: float = 4.0
    channels: tuple[int, ...] = ()
    magnitude: float = 0.0

    def __post_init__(self):
        name = str(self.name).strip()
        sensor = None if self.sensor is None else str(self.sensor)
        mode = str(self.mode)
        start = float(self.start_time_s)
        duration = float(self.duration_s)
        if not name:
            raise ValueError("name cannot be empty")
        if sensor is None:
            if mode != "normal":
                raise ValueError("a normal scenario must use mode='normal'")
        elif sensor not in SENSOR_NAMES or mode not in FAULT_MODES:
            raise ValueError("invalid sensor or fault mode")
        if not np.isfinite(start) or start < 0.0:
            raise ValueError("start_time_s must be finite and non-negative")
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "sensor", sensor)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "start_time_s", start)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(
            self, "channels", tuple(int(value) for value in self.channels)
        )
        object.__setattr__(self, "magnitude", float(self.magnitude))

    @property
    def is_fault(self):
        return self.sensor is not None

    @property
    def end_time_s(self):
        return self.start_time_s + self.duration_s

    def fault_event(self):
        if not self.is_fault:
            return None
        return SensorFaultEvent(
            sensor=self.sensor,
            mode=SensorFaultMode(self.mode),
            start_time_s=self.start_time_s,
            end_time_s=self.end_time_s,
            channels=self.channels,
            magnitude=self.magnitude,
            event_id=self.name,
        )

    def as_dict(self):
        return {
            "name": self.name,
            "sensor": self.sensor,
            "mode": self.mode,
            "start_time_s": self.start_time_s,
            "duration_s": self.duration_s,
            "channels": list(self.channels),
            "magnitude": self.magnitude,
        }


def default_sensor_fault_scenarios(start_time_s=5.0, duration_s=4.0):
    """Return one normal and nine directly observable fault scenarios."""

    values = [SensorFaultBenchmarkScenario(
        "normal", None, "normal", start_time_s, duration_s
    )]
    spike_definitions = {
        "depth": ((), 2.0),
        "imu": ((2,), 0.8),
        "dvl": ((0,), 2.0),
    }
    for sensor in SENSOR_NAMES:
        for mode in FAULT_MODES:
            channels, magnitude = (
                spike_definitions[sensor]
                if mode == "spike"
                else ((), 0.0)
            )
            values.append(SensorFaultBenchmarkScenario(
                f"{sensor}_{mode}",
                sensor,
                mode,
                start_time_s,
                duration_s,
                channels,
                magnitude,
            ))
    return tuple(values)


def extract_confirmed_sensor_events(logs):
    """Collapse contiguous confirmed samples into operator-level events."""

    events = []
    active = {sensor: None for sensor in SENSOR_NAMES}
    for log in logs:
        time_s = float(log["time"])
        health = log.get("sensor_health", {})
        for sensor in SENSOR_NAMES:
            result = health.get(sensor, {})
            fault_type = str(result.get("fault_type", "normal"))
            confirmed = bool(result.get("confirmed", False))
            confidence = float(result.get("confidence", 0.0))
            key = fault_type if confirmed and fault_type != "normal" else None
            current = active[sensor]
            if current is not None and key != current["fault_type"]:
                events.append(current)
                active[sensor] = None
                current = None
            if key is not None:
                if current is None:
                    active[sensor] = {
                        "sensor": sensor,
                        "fault_type": key,
                        "start_time_s": time_s,
                        "end_time_s": time_s,
                        "max_confidence": confidence,
                        "sample_count": 1,
                    }
                else:
                    current["end_time_s"] = time_s
                    current["max_confidence"] = max(
                        current["max_confidence"], confidence
                    )
                    current["sample_count"] += 1
    events.extend(
        event for event in active.values() if event is not None
    )
    return sorted(events, key=lambda event: (
        event["start_time_s"], event["sensor"]
    ))


def _expected_ftc_action(scenario):
    if not scenario.is_fault:
        return "normal_control"
    if scenario.mode == "spike":
        return "log_only"
    if scenario.sensor == "imu":
        return "safe_hold_or_abort"
    return "degraded_operation"


def evaluate_sensor_fault_mission(
    logs,
    scenario,
    recovery_grace_s=0.5,
    absolute_position_tolerance_m=0.75,
):
    """Evaluate one mission without using injected truth in online logic."""

    if not isinstance(scenario, SensorFaultBenchmarkScenario):
        raise TypeError("scenario must be SensorFaultBenchmarkScenario")
    logs = list(logs)
    if not logs:
        raise ValueError("logs cannot be empty")
    events = extract_confirmed_sensor_events(logs)
    matching = []
    if scenario.is_fault:
        matching = [
            event for event in events
            if event["sensor"] == scenario.sensor
            and event["fault_type"] == scenario.mode
            and event["start_time_s"] <= (
                scenario.end_time_s + recovery_grace_s
            )
            and event["end_time_s"] >= scenario.start_time_s
        ]
    detected = bool(matching)
    matched_event = matching[0] if matching else None
    matched_index = (
        events.index(matched_event) if matched_event is not None else None
    )
    spurious_events = [
        event for index, event in enumerate(events)
        if index != matched_index
    ]

    expected_action = _expected_ftc_action(scenario)
    correct_action_observed = not scenario.is_fault
    if scenario.is_fault:
        correct_action_observed = any(
            log.get("ftc_action") == expected_action
            and log.get("sensor_health", {})
            .get(scenario.sensor, {})
            .get("fault_type") == scenario.mode
            for log in logs
        )
    recovery_logs = [
        log for log in logs
        if float(log["time"]) >= scenario.end_time_s + recovery_grace_s
    ]
    if scenario.is_fault and recovery_logs:
        sensor_health_recovered = bool(
            recovery_logs[-1]
            .get("sensor_health", {})
            .get(scenario.sensor, {})
            .get("fault_type", "normal") == "normal"
        )
    else:
        sensor_health_recovered = not scenario.is_fault

    final_true_position_error_m = float(np.linalg.norm(
        np.asarray(logs[-1]["true_position_error_ned"], dtype=float)
    ))
    estimate_integrity_restored = bool(
        logs[-1].get(
            "horizontal_position_reference", "initial_dead_reckoning"
        ) != "degraded_without_absolute_fix"
    )
    absolute_trajectory_recovered = bool(
        final_true_position_error_m
        <= float(absolute_position_tolerance_m)
    )

    protective_actions = {
        "degraded_operation",
        "targeted_reallocation",
        "safe_hold_or_abort",
        "controlled_ascent",
    }
    return {
        "scenario": scenario.name,
        "sensor": scenario.sensor,
        "mode": scenario.mode,
        "is_fault": scenario.is_fault,
        "exact_event_detected": detected,
        "detection_delay_s": (
            None
            if matched_event is None
            else max(
                0.0,
                matched_event["start_time_s"] - scenario.start_time_s,
            )
        ),
        "confirmed_event_count": len(events),
        "spurious_event_count": len(spurious_events),
        "spurious_events": spurious_events,
        "expected_ftc_action": expected_action,
        "correct_ftc_action_observed": correct_action_observed,
        "sensor_health_recovered": sensor_health_recovered,
        "estimate_integrity_restored": estimate_integrity_restored,
        "absolute_trajectory_recovered": absolute_trajectory_recovered,
        "wrong_thruster_target_count": sum(
            log.get("ftc_targeted_thruster_name") is not None
            for log in logs
        ),
        "protective_action_observed": any(
            log.get("ftc_action") in protective_actions for log in logs
        ),
        "final_true_position_error_m": final_true_position_error_m,
        "maximum_estimation_position_error_m": float(max(
            np.linalg.norm(
                np.asarray(log["estimated_position_ned"], dtype=float)
                - np.asarray(log["position_ned"], dtype=float)
            )
            for log in logs
        )),
    }


def summarize_sensor_fault_benchmark(rows):
    """Aggregate event-level and operational metrics."""

    rows = list(rows)
    if not rows:
        raise ValueError("rows cannot be empty")
    fault_rows = [row for row in rows if row["is_fault"]]
    normal_rows = [row for row in rows if not row["is_fault"]]
    true_positive_events = sum(
        bool(row["exact_event_detected"]) for row in fault_rows
    )
    predicted_events = sum(row["confirmed_event_count"] for row in rows)
    delays = [
        row["detection_delay_s"] for row in fault_rows
        if row["detection_delay_s"] is not None
    ]
    per_scenario = {}
    for scenario in sorted({row["scenario"] for row in rows}):
        selected = [row for row in rows if row["scenario"] == scenario]
        selected_faults = [row for row in selected if row["is_fault"]]
        selected_delays = [
            row["detection_delay_s"] for row in selected_faults
            if row["detection_delay_s"] is not None
        ]
        per_scenario[scenario] = {
            "missions": len(selected),
            "event_recall": (
                None if not selected_faults else mean(
                    row["exact_event_detected"] for row in selected_faults
                )
            ),
            "mean_detection_delay_s": (
                None if not selected_delays else mean(selected_delays)
            ),
            "ftc_action_match_rate": (
                None if not selected_faults else mean(
                    row["correct_ftc_action_observed"]
                    for row in selected_faults
                )
            ),
            "sensor_health_recovery_rate": mean(
                row["sensor_health_recovered"] for row in selected
            ),
            "estimate_integrity_restoration_rate": mean(
                row["estimate_integrity_restored"] for row in selected
            ),
            "absolute_trajectory_recovery_rate": mean(
                row["absolute_trajectory_recovered"] for row in selected
            ),
            "spurious_event_count": sum(
                row["spurious_event_count"] for row in selected
            ),
        }

    event_recall = true_positive_events / max(len(fault_rows), 1)
    event_precision = true_positive_events / max(predicted_events, 1)
    action_match_rate = mean(
        row["correct_ftc_action_observed"] for row in fault_rows
    )
    sensor_health_recovery_rate = mean(
        row["sensor_health_recovered"] for row in fault_rows
    )
    estimate_integrity_restoration_rate = mean(
        row["estimate_integrity_restored"] for row in fault_rows
    )
    absolute_trajectory_recovery_rate = mean(
        row["absolute_trajectory_recovered"] for row in fault_rows
    )
    normal_false_event_missions = sum(
        row["confirmed_event_count"] > 0 for row in normal_rows
    )
    normal_false_protective_missions = sum(
        row["protective_action_observed"] for row in normal_rows
    )
    wrong_thruster_targets = sum(
        row["wrong_thruster_target_count"] for row in rows
    )
    checks = {
        "event_recall_at_least_95pct": event_recall >= 0.95,
        "event_precision_at_least_95pct": event_precision >= 0.95,
        "ftc_action_match_at_least_95pct": action_match_rate >= 0.95,
        "sensor_health_recovery_rate_100pct": (
            sensor_health_recovery_rate == 1.0
        ),
        "normal_false_protective_missions_zero": (
            normal_false_protective_missions == 0
        ),
        "wrong_thruster_targets_zero": wrong_thruster_targets == 0,
    }
    return {
        "evaluation_type": "development_benchmark_not_blind",
        "mission_count": len(rows),
        "fault_mission_count": len(fault_rows),
        "normal_mission_count": len(normal_rows),
        "event_recall": event_recall,
        "event_precision": event_precision,
        "confirmed_event_count": predicted_events,
        "true_positive_event_count": true_positive_events,
        "spurious_event_count": predicted_events - true_positive_events,
        "mean_detection_delay_s": None if not delays else mean(delays),
        "median_detection_delay_s": None if not delays else median(delays),
        "ftc_action_match_rate": action_match_rate,
        "sensor_health_recovery_rate": sensor_health_recovery_rate,
        "estimate_integrity_restoration_rate": (
            estimate_integrity_restoration_rate
        ),
        "absolute_trajectory_recovery_rate": (
            absolute_trajectory_recovery_rate
        ),
        "absolute_trajectory_recovery_is_informational": True,
        "absolute_position_fix_note": (
            "Depth, IMU, and DVL do not provide an absolute horizontal "
            "position fix. Persistent IMU/DVL outages can recover sensor "
            "health while leaving an uncorrected dead-reckoning offset."
        ),
        "normal_false_event_missions": normal_false_event_missions,
        "normal_false_protective_missions": (
            normal_false_protective_missions
        ),
        "wrong_thruster_target_count": wrong_thruster_targets,
        "maximum_estimation_position_error_m": max(
            row["maximum_estimation_position_error_m"] for row in rows
        ),
        "per_scenario": per_scenario,
        "acceptance_checks": checks,
        "all_acceptance_checks_passed": all(checks.values()),
    }
