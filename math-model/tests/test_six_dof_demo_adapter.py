"""Tests for the causal six-DOF operator-display adapter."""

from copy import deepcopy
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from presentation.six_dof_demo_adapter import (
    adapt_log,
    extract_demo_events,
    summarize_demo,
)


def base_log():
    normal_health = {
        "health_state": "healthy", "fault_type": "normal",
        "confidence": 1.0, "trust_level": "trusted", "confirmed": False,
        "recommended_action": "use_sensor",
    }
    normal_observation = {
        "state": "normal", "hypothesis": "normal", "confidence": 0.0,
        "display_level": "background",
    }
    return {
        "time": 1.0,
        "position_ned": np.zeros(3),
        "estimated_position_ned": np.zeros(3),
        "target_position_ned": np.ones(3),
        "euler_rpy": np.zeros(3),
        "estimated_euler_rpy": np.zeros(3),
        "sensor_health": {
            name: deepcopy(normal_health) for name in ("depth", "imu", "dvl")
        },
        "sensor_fault_observations": {
            name: deepcopy(normal_observation)
            for name in ("depth", "imu", "dvl")
        },
        "thruster_names": ("H1", "H2", "H3", "H4", "V1", "V2"),
        "commanded_thruster_forces": np.full(6, 5.0),
        "thruster_force_limits": np.full(6, 10.0),
        "thruster_expected_currents": np.full(6, 2.0),
        "thruster_measured_currents": np.full(6, 2.0),
        "thruster_expected_rpms": np.full(6, 1000.0),
        "thruster_measured_rpms": np.full(6, 1000.0),
        "thruster_telemetry_valid": np.ones(6, dtype=bool),
        "thruster_telemetry_age_s": np.zeros(6),
        "ftc_no_output_scores": np.zeros(6),
        "ftc_estimated_effectiveness_next_step": np.ones(6),
        "ftc_action": "normal_control",
        "ftc_reason": "No fault evidence.",
    }


def test_display_tiers_follow_confirmed_possible_and_background_policy():
    log = base_log()
    log["sensor_health"]["depth"].update({
        "health_state": "confirmed_fault", "fault_type": "unavailable",
        "confirmed": True, "confidence": 1.0,
    })
    log["sensor_fault_observations"]["imu"].update({
        "state": "possible_fault", "hypothesis": "possible_bias_or_drift",
        "confidence": 0.7, "display_level": "possible",
        "candidates": ["bias", "drift"],
    })
    log["sensor_fault_observations"]["dvl"].update({
        "state": "possible_fault", "hypothesis": "possible_weak_spike_or_bias",
        "confidence": 0.45, "display_level": "background",
    })
    frame = adapt_log(log)
    assert frame["sensors"]["depth"]["tier"] == "confirmed"
    assert frame["sensors"]["imu"]["tier"] == "possible"
    assert frame["sensors"]["dvl"]["tier"] == "log_only"
    assert frame["overall_tier"] == "confirmed"


def test_direct_confirmation_overrides_ambiguous_observer_label():
    log = base_log()
    log["sensor_health"]["imu"].update({
        "health_state": "confirmed_fault", "fault_type": "spike",
        "confirmed": True, "confidence": 0.99,
    })
    log["sensor_fault_observations"]["imu"].update({
        "state": "possible_fault", "hypothesis": "possible_bias_or_drift",
        "confidence": 0.8, "display_level": "possible",
    })
    card = adapt_log(log)["sensors"]["imu"]
    assert card["tier"] == "confirmed"
    assert card["label"] == "spike"
    assert card["source"] == "direct_monitor"


def test_privileged_truth_cannot_change_diagnostic_presentation():
    first = base_log()
    second = deepcopy(first)
    second.update({
        "sensor_fault_truth": {
            "depth": {"mode": "unavailable"}, "imu": {"mode": "stuck"},
            "dvl": {"mode": "bias"},
        },
        "thruster_fault_modes": ("normal",) * 4 + ("no_output", "normal"),
        "faulted_thruster_index": 4,
        "thruster_force_efficiencies": np.array([1, 1, 1, 1, 0, 1]),
        "actual_thruster_forces": np.array([5, 5, 5, 5, 0, 5]),
    })
    first_frame = adapt_log(first)
    second_frame = adapt_log(second)
    assert first_frame["sensors"] == second_frame["sensors"]
    assert first_frame["thrusters"] == second_frame["thrusters"]
    assert first_frame["ftc"] == second_frame["ftc"]
    assert first_frame["overall_tier"] == second_frame["overall_tier"]


def test_thruster_target_requires_causal_ftc_output_not_truth_label():
    truth_only = base_log()
    truth_only["faulted_thruster_index"] = 4
    assert all(
        card["tier"] == "normal"
        for card in adapt_log(truth_only)["thrusters"]
    )
    targeted = base_log()
    targeted.update({
        "ftc_action": "targeted_reallocation",
        "ftc_targeted_thruster_index": 5,
        "ftc_targeted_thruster_name": "V1",
        "ftc_estimated_effectiveness_next_step": np.array([1, 1, 1, 1, 0, 1]),
    })
    cards = adapt_log(targeted)["thrusters"]
    assert cards[4]["tier"] == "confirmed"
    assert cards[4]["label"] == "targeted / isolated"


def test_esc_communication_fault_is_log_only_and_recovers_visibly():
    normal = adapt_log(base_log())
    fault_log = base_log()
    fault_log["time"] = 2.0
    fault_log["thruster_telemetry_valid"][5] = False
    fault_log["thruster_telemetry_age_s"][5] = 0.5
    fault_log["ftc_untrusted_esc_channels"] = ("V2",)
    fault_log["ftc_action"] = "log_only"
    fault = adapt_log(fault_log)
    recovery_log = base_log()
    recovery_log["time"] = 3.0
    recovery = adapt_log(recovery_log)

    card = fault["thrusters"][5]
    assert card["tier"] == "log_only"
    assert card["label"] == "ESC telemetry unavailable"
    assert card["telemetry_status"] == "invalid"
    assert fault["ftc"]["untrusted_esc_channels"] == ["V2"]
    events = extract_demo_events([normal, fault, recovery])
    messages = [event["message"] for event in events]
    assert "V2: ESC telemetry unavailable (log_only)" in messages
    assert "V2 evidence cleared" in messages
    summary = summarize_demo([normal, fault, recovery], events)
    assert summary["esc_communication_anomaly_thrusters"] == ["V2"]
    assert summary["esc_communication_anomaly_frame_count"] == 1


def test_event_extraction_records_onsets_recovery_and_ftc_transition_once():
    normal = adapt_log(base_log())
    possible_log = base_log()
    possible_log["time"] = 2.0
    possible_log["sensor_fault_observations"]["dvl"].update({
        "state": "possible_fault", "hypothesis": "possible_bias_or_drift",
        "confidence": 0.7, "display_level": "possible",
    })
    possible = adapt_log(possible_log)
    duplicate = deepcopy(possible)
    duplicate["time_s"] = 2.1
    recovered_log = base_log()
    recovered_log["time"] = 3.0
    recovered = adapt_log(recovered_log)
    targeted_log = base_log()
    targeted_log.update({
        "time": 4.0, "ftc_action": "targeted_reallocation",
        "ftc_targeted_thruster_name": "V1", "ftc_targeted_thruster_index": 5,
        "ftc_estimated_effectiveness_next_step": np.array([1, 1, 1, 1, 0, 1]),
    })
    targeted = adapt_log(targeted_log)
    frames = [normal, possible, duplicate, recovered, targeted]
    events = extract_demo_events(frames)
    messages = [event["message"] for event in events]
    assert messages.count("DVL: possible bias or drift (possible)") == 1
    assert messages.count("DVL returned to normal") == 1
    assert sum(event["category"] == "ftc" for event in events) == 1
    assert summarize_demo(frames, events)["targeted_thrusters"] == ["V1"]


def test_repeated_no_output_candidates_are_grouped_before_confirmation():
    frames = [adapt_log(base_log())]
    for time_s, active in ((2.0, True), (2.1, False), (2.2, True), (2.3, False)):
        log = base_log()
        log["time"] = time_s
        if active:
            log["ftc_no_output_scores"][4] = 0.95
            log["ftc_action"] = "log_only"
        frames.append(adapt_log(log))
    events = extract_demo_events(frames)
    assert sum(event["category"] == "thruster" for event in events) == 1
    assert not any(event["category"] == "ftc" for event in events)


def test_learned_maintenance_location_is_always_presented_as_advisory():
    log = base_log()
    log["maintenance_diagnosis"] = {
        "available": True,
        "updated": True,
        "status": "model_inference",
        "health_level": 3,
        "health_state": "critical_fault",
        "temporal_state": "confirmed",
        "probable_mode_name": "thrust_loss",
        "confirmed_mode_name": "thrust_loss",
        "fault_probability": 0.91,
        "suspected_group": "vertical",
        "group_confidence": 0.77,
        "location_confidence": "medium",
        "candidates": [
            {"index": 5, "name": "V1", "probability": 0.46},
            {"index": 6, "name": "V2", "probability": 0.31},
        ],
        "mode_probabilities": [0.09, 0.10, 0.81],
        "location_probabilities": [0.05, 0.05, 0.06, 0.07, 0.46, 0.31],
        "action": "safety_alert_and_consider_ftc",
        "record_event": True,
        "requires_ftc": True,
    }
    maintenance = adapt_log(log)["maintenance"]
    assert maintenance["tier"] == "possible"
    assert maintenance["model_recommends_ftc"]
    assert [item["name"] for item in maintenance["candidates"]] == ["V1", "V2"]
    assert maintenance["confirmed_mode"] == "thrust_loss"


def test_context_suppressed_model_output_stays_normal_but_auditable():
    log = base_log()
    log["maintenance_diagnosis"] = {
        "available": True,
        "updated": True,
        "status": "model_inference",
        "health_level": 0,
        "health_state": "context_suppressed",
        "probable_mode_name": "normal",
        "fault_probability": 0.0,
        "advisory_gate_active": True,
        "advisory_gate_reasons": [
            "context_stabilization", "guidance_context_change"
        ],
        "advisory_suppressed": True,
        "raw_probable_mode_name": "thrust_loss",
        "raw_fault_probability": 0.88,
        "raw_suspected_group": "horizontal",
    }
    maintenance = adapt_log(log)["maintenance"]
    assert maintenance["tier"] == "normal"
    assert maintenance["advisory_gate_active"]
    assert maintenance["advisory_suppressed"]
    assert maintenance["raw_probable_mode"] == "thrust_loss"
    assert maintenance["raw_fault_probability"] == 0.88
