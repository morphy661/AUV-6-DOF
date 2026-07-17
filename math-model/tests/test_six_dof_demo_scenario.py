"""Tests for fixed and reproducible-random demonstration fault schedules."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
EXAMPLES_ROOT = PROJECT_ROOT / "examples"
for path in (SRC_ROOT, EXAMPLES_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from demo_six_dof_unified_diagnostics import build_fault_scenario, run_demo


def _manifest(seed, mode="random"):
    _, _, manifest = build_fault_scenario(seed, mode)
    return manifest


def test_random_schedule_is_reproducible_for_one_seed():
    assert _manifest(20260717) == _manifest(20260717)


def test_random_schedule_changes_with_seed_and_preserves_phase_structure():
    first = _manifest(20260717)
    second = _manifest(20260718)
    assert first != second
    for manifest in (first, second):
        events = manifest["sensor_events"]
        assert len(events) == 5
        sensors = {event["sensor"] for event in events}
        assert sensors == {"depth", "imu", "dvl"}
        assert manifest["thruster_fault"]["thruster_name"] in {
            "H1", "H2", "H3", "H4", "V1", "V2"
        }
        assert manifest["thruster_fault"]["mode"] in {
            "no_output", "thrust_loss"
        }
        esc_events = manifest["esc_telemetry_events"]
        assert len(esc_events) == 1
        assert esc_events[0]["thruster_name"] in {
            "H1", "H2", "H3", "H4", "V1", "V2"
        }
        assert esc_events[0]["mode"] in {
            "continuous_packet_loss", "communication_freeze"
        }


def test_fixed_schedule_remains_the_reproducible_reference_story():
    manifest = _manifest(123, mode="fixed")
    assert manifest["injection_mode"] == "fixed"
    assert manifest["thruster_fault"]["thruster_name"] == "V1"
    assert manifest["thruster_fault"]["mode"] == "no_output"
    assert [event["sensor"] for event in manifest["sensor_events"]] == [
        "depth", "dvl", "imu", "imu", "imu"
    ]
    assert manifest["esc_telemetry_events"] == [{
        "event_id": "v2_esc_packet_loss",
        "thruster_name": "V2",
        "mode": "continuous_packet_loss",
        "start_time_s": 15.35,
        "end_time_s": 16.35,
    }]


def test_fixed_demo_logs_esc_loss_without_isolation_then_targets_real_fault():
    logs, frames, events, manifest = run_demo(
        18.0, 0.10, 123, injection_mode="fixed"
    )

    communication = [
        log for log in logs
        if 15.35 <= float(log["time"]) <= 16.35
    ]
    assert communication
    assert all("V2" in log["ftc_untrusted_esc_channels"] for log in communication)
    assert all(log["ftc_targeted_thruster_name"] is None for log in communication)
    assert any(
        frame["thrusters"][5]["label"] == "ESC telemetry unavailable"
        for frame in frames
    )
    assert any(
        log["ftc_targeted_thruster_name"] == "V1" for log in logs
    )
    assert manifest["thruster_fault"]["thruster_name"] == "V1"
