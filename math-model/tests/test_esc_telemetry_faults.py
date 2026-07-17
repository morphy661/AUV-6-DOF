"""Tests for the shared ESC telemetry fault injector."""

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from actuators.esc_telemetry_faults import ESCTelemetryFaultInjector


def packet(time_s, current=2.0, rpm=1000.0):
    return {
        "time": time_s,
        "thruster_measured_currents": np.full(6, current),
        "thruster_measured_rpms": np.full(6, rpm),
        "thruster_measured_voltages": np.full(6, 48.0),
        "thruster_telemetry_valid": np.ones(6, dtype=bool),
        "thruster_telemetry_age_s": np.zeros(6),
    }


def event(mode, **overrides):
    values = {
        "event_id": "esc_event",
        "thruster_name": "V2",
        "mode": mode,
        "start_time_s": 1.0,
        "end_time_s": 2.0,
    }
    values.update(overrides)
    return values


def test_packet_loss_zero_fills_and_marks_only_affected_channel_invalid():
    output = ESCTelemetryFaultInjector([
        event("continuous_packet_loss")
    ]).apply(packet(1.5), copy_log=True)

    assert output["thruster_measured_currents"][5] == 0.0
    assert output["thruster_measured_rpms"][5] == 0.0
    assert not output["thruster_telemetry_valid"][5]
    assert output["thruster_telemetry_age_s"][5] == pytest.approx(0.5)
    assert np.all(output["thruster_telemetry_valid"][:5])


def test_freeze_reuses_last_pre_event_sample_and_recovers_after_event():
    injector = ESCTelemetryFaultInjector([event("communication_freeze")])
    logs = [packet(0.9, 3.0, 1200.0), packet(1.5), packet(2.1, 4.0, 1400.0)]
    output = injector.apply_logs(logs)

    assert output[1]["thruster_measured_currents"][5] == 3.0
    assert output[1]["thruster_measured_rpms"][5] == 1200.0
    assert output[1]["thruster_telemetry_age_s"][5] == pytest.approx(0.5)
    assert output[2]["thruster_measured_currents"][5] == 4.0
    assert output[2]["thruster_telemetry_age_s"][5] == 0.0


def test_voltage_dip_and_quantization_share_the_same_event_contract():
    dip = ESCTelemetryFaultInjector([event(
        "bus_voltage_dip", signal_scale=0.5, voltage_scale=0.6
    )]).apply(packet(1.5), copy_log=True)
    quantized = ESCTelemetryFaultInjector([event(
        "quantization", current_step_a=0.3, rpm_step=175.0
    )]).apply(packet(1.5, 2.05, 1010.0), copy_log=True)

    assert dip["thruster_measured_currents"][5] == 1.0
    assert dip["thruster_measured_voltages"][5] == pytest.approx(28.8)
    assert quantized["thruster_measured_currents"][5] == pytest.approx(2.1)
    assert quantized["thruster_measured_rpms"][5] == pytest.approx(1050.0)


def test_invalid_event_declarations_fail_early():
    with pytest.raises(ValueError):
        ESCTelemetryFaultInjector([event("not_a_mode")])
    with pytest.raises(ValueError):
        ESCTelemetryFaultInjector([event(
            "communication_freeze", end_time_s=0.5
        )])
