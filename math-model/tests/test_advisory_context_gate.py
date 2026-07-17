"""Tests for causal gating of learned maintenance advice."""

import sys
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from presentation.advisory_context_gate import (
    AdvisoryContextGate,
    AdvisoryGateConfig,
)


def normal_log(time_s=0.0, context_id=1):
    health = {
        "health_state": "healthy",
        "fault_type": "normal",
        "trust_level": "trusted",
        "confirmed": False,
    }
    observation = {"state": "normal", "display_level": "background"}
    return {
        "time": time_s,
        "guidance_context_id": context_id,
        "sensor_health": {
            name: deepcopy(health) for name in ("depth", "imu", "dvl")
        },
        "sensor_fault_observations": {
            name: deepcopy(observation)
            for name in ("depth", "imu", "dvl")
        },
    }


def test_initial_context_does_not_create_an_extra_gate_transition():
    gate = AdvisoryContextGate(AdvisoryGateConfig(stabilization_time_s=3.0))
    decision = gate.update(normal_log())
    assert not decision.reset_model_context
    assert not decision.active


def test_guidance_change_resets_once_and_holds_until_stable():
    gate = AdvisoryContextGate(AdvisoryGateConfig(stabilization_time_s=3.0))
    gate.update(normal_log(1.0, 1))
    transition = gate.update(normal_log(2.0, 2))
    held = gate.update(normal_log(4.9, 2))
    awaiting_fresh = gate.update(normal_log(5.0, 2))
    released = gate.mark_model_inference(5.0)
    assert transition.reset_model_context
    assert "guidance_context_change" in transition.reasons
    assert held.active and not held.reset_model_context
    assert awaiting_fresh.active
    assert "awaiting_fresh_model_inference" in awaiting_fresh.reasons
    assert not released.active


def test_inference_before_stabilization_does_not_release_held_result():
    gate = AdvisoryContextGate(AdvisoryGateConfig(stabilization_time_s=3.0))
    gate.update(normal_log(1.0, 1))
    gate.update(normal_log(2.0, 2))
    early = gate.mark_model_inference(4.9)
    expired = gate.update(normal_log(5.0, 2))
    released = gate.mark_model_inference(5.1)
    assert early.active
    assert expired.active
    assert "awaiting_fresh_model_inference" in expired.reasons
    assert not released.active


def test_direct_sensor_fault_resets_on_onset_not_every_active_sample():
    gate = AdvisoryContextGate(AdvisoryGateConfig(stabilization_time_s=1.0))
    gate.update(normal_log())
    fault = normal_log(1.0)
    fault["sensor_health"]["imu"].update({
        "health_state": "confirmed_fault",
        "fault_type": "unavailable",
        "trust_level": "untrusted",
        "confirmed": True,
    })
    onset = gate.update(fault)
    still_active = gate.update({**fault, "time": 1.1})
    assert onset.reset_model_context and onset.active
    assert not still_active.reset_model_context and still_active.active
    assert "active_direct_sensor_fault" in still_active.reasons


def test_operator_possible_hypothesis_creates_one_stabilization_reset():
    gate = AdvisoryContextGate(AdvisoryGateConfig(stabilization_time_s=2.0))
    gate.update(normal_log())
    possible = normal_log(1.0)
    possible["sensor_fault_observations"]["dvl"].update({
        "state": "possible_fault", "display_level": "possible"
    })
    onset = gate.update(possible)
    repeated = gate.update({**possible, "time": 1.5})
    assert onset.reset_model_context
    assert "sensor_hypothesis:dvl" in onset.reasons
    assert repeated.active and not repeated.reset_model_context


def test_esc_telemetry_fault_and_recovery_each_reset_model_context():
    gate = AdvisoryContextGate(AdvisoryGateConfig(stabilization_time_s=1.0))
    gate.update(normal_log())
    fault = normal_log(1.0)
    fault["ftc_untrusted_esc_channels"] = ("V2",)

    onset = gate.update(fault)
    active = gate.update({**fault, "time": 1.5})
    recovery = gate.update(normal_log(2.0))
    waiting = gate.update(normal_log(3.0))
    released = gate.mark_model_inference(3.0)

    assert onset.reset_model_context
    assert "esc_telemetry_fault:V2" in onset.reasons
    assert "active_esc_telemetry_fault" in active.reasons
    assert recovery.reset_model_context
    assert "esc_telemetry_recovered:V2" in recovery.reasons
    assert "awaiting_fresh_model_inference" in waiting.reasons
    assert not released.active
