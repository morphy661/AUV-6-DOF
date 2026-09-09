"""AI admission boundaries and the next-step control feedback contract."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ftc.ai_evidence_gate import AIControlEvidenceGate, AIEvidenceGateConfig
from ftc.safety_supervisor import FTCEvidence, FTCSafetySupervisor, FTCSupervisorConfig
from ftc.safety_supervisor import build_rule_based_ftc_evidence
from environment.six_dof_simulator import SixDOFSimulator
from simple_control.six_dof_controller import PoseTarget


def diagnosis(time_s, **changes):
    values = dict(available=True, updated=True, inference_time_s=time_s,
                  confirmed_mode=2, temporal_state="confirmed", health_level=3,
                  raw_mode_probabilities=[.02, .02, .96], suspected_group="vertical")
    values.update(changes)
    return values


def evidence(time_s, **changes):
    return FTCEvidence(time_s, tracking_error_ratio=.65,
                       control_saturation_ratio=.97, **changes)


@pytest.mark.parametrize("changes,verdict", [
    ({"available": False}, "unavailable"),
    ({"advisory_gate_active": True}, "context_blocked"),
    ({"inference_time_s": -1.0}, "stale_or_future_result"),
    ({"inference_time_s": 2.0}, "stale_or_future_result"),
    ({"inference_time_s": float("nan")}, "stale_or_future_result"),
    ({"raw_mode_probabilities": [float("nan"), 0, 1]}, "invalid_result"),
    ({"raw_mode_probabilities": [.1, .7, .2]}, "probability_conflict"),
    ({"temporal_state": "recovering"}, "recovering"),
    ({"temporal_state": "suspected", "confirmed_mode": 0}, "suspected"),
])
def test_invalid_or_unconfirmed_ai_never_adds_control_authority(changes, verdict):
    gate = AIControlEvidenceGate()
    base = evidence(0.)
    fused, report = gate.update(base, diagnosis(0., **changes))
    assert fused is base
    assert report["verdict"] == verdict
    assert not report["accepted"]


def test_repeated_held_or_duplicate_results_cannot_confirm():
    gate = AIControlEvidenceGate()
    gate.update(evidence(0), diagnosis(0))
    for time_s in (.1, .2, .3):
        base = evidence(time_s)
        fused, report = gate.update(base, diagnosis(0, updated=False))
        assert fused is base
        assert report["confirming_updates"] == 1
    fused, report = gate.update(evidence(.4), diagnosis(0))
    assert report["confirming_updates"] == 1
    assert not report["accepted"]
    _, report = gate.update(evidence(.5), diagnosis(.5))
    assert report["accepted"]


def test_compensated_fault_and_saturation_alone_are_record_only():
    for tracking, saturation in ((.1, .97), (.7, .2)):
        gate = AIControlEvidenceGate()
        for t in (0., .5):
            base = FTCEvidence(t, tracking_error_ratio=tracking,
                               control_saturation_ratio=saturation)
            fused, report = gate.update(base, diagnosis(t))
        assert fused is base
        assert report["verdict"] == "record_only"


def test_horizontal_motion_error_can_corroborate_without_isolation():
    gate = AIControlEvidenceGate()
    horizontal_diagnosis = {
        "suspected_group": "horizontal",
    }
    for time_s in (0.0, 0.5):
        base = FTCEvidence(
            time_s,
            tracking_error_ratio=0.1,
            horizontal_motion_error_ratio=1.2,
            control_saturation_ratio=0.9,
        )
        fused, report = gate.update(
            base, diagnosis(time_s, **horizontal_diagnosis)
        )

    assert report["accepted"]
    assert report["supporting_control_signals"] == [
        "control_saturation", "horizontal_motion_error",
    ]
    decision = FTCSafetySupervisor().update(fused)
    assert decision.action.value == "degraded_operation"
    assert decision.guidance_mode == "horizontal_degraded_hold"
    assert decision.targeted_thruster_name is None


def test_direct_esc_path_wins_even_if_ai_is_unavailable_or_wrong():
    scores = np.array([0., .99, 0., 0., 0., 0.])
    base = evidence(0., no_output_scores=scores, excitation_ratios=np.ones(6))
    gate = AIControlEvidenceGate()
    fused, report = gate.update(base, diagnosis(0, available=False))
    assert fused is base
    assert report["verdict"] == "direct_rule_priority"
    cfg = FTCSupervisorConfig(no_output_confirmation_s=0)
    result = FTCSafetySupervisor(cfg).update(fused)
    assert result.targeted_thruster_name == "H2"


def test_accepted_ai_cannot_create_isolation_and_stale_data_revokes_authority():
    gate = AIControlEvidenceGate()
    for time_s in (0., .5):
        fused, report = gate.update(evidence(time_s), diagnosis(time_s))
    assert report["accepted"]
    assert fused.health_level == 2
    assert np.max(fused.no_output_scores) == 0
    assert FTCSafetySupervisor().update(fused).targeted_thruster_name is None
    base = evidence(1.2)
    fused, report = gate.update(base, diagnosis(.5, updated=False))
    assert fused is base
    assert not report["accepted"]
    assert gate.confirming_updates == 0


def test_sensor_guard_is_preserved_and_blocks_ai():
    gate = AIControlEvidenceGate()
    base = evidence(0, sensor_guard_action="safe_hold_or_abort", untrusted_sensors=("imu",))
    fused, report = gate.update(base, diagnosis(0))
    assert fused is base
    assert report["verdict"] == "sensor_or_esc_guard"


def test_ai_changes_supervisor_once_and_control_applies_on_next_step():
    class Observer:
        def reset(self):
            self.count = 0

        def update(self, log):
            assert log["ftc_tracking_error_ratio"] == .65
            self.count += 1
            return diagnosis(log["time"])

    class CountingSupervisor(FTCSafetySupervisor):
        def update(self, value):
            self.count = getattr(self, "count", 0) + 1
            return super().update(value)

    cfg = FTCSupervisorConfig(degraded_confirmation_s=0., critical_confirmation_s=0.)
    observer = Observer()
    supervisor = CountingSupervisor(cfg)
    sim = SixDOFSimulator(
        ftc_supervisor=supervisor, diagnostic_observer=observer,
        ftc_evidence_provider=lambda log: evidence(log["time"]),
        ai_ftc_fusion=AIControlEvidenceGate(supervisor_config=cfg),
    )
    sim.reset()
    target = PoseTarget(np.array([0., 0., 2.]), np.zeros(3))
    first = sim.step(target, .05)
    second = sim.step(target, .05)
    third = sim.step(target, .05)
    assert first["ftc_action"] == "degraded_operation"
    assert second["ftc_action"] == "safe_hold_or_abort"
    assert second["ftc_applied_action"] == first["ftc_action"]
    assert third["ftc_applied_action"] == second["ftc_action"]
    assert second["ai_ftc_fusion"]["accepted"]
    assert "validated_online_ai" in second["ftc_evidence_source"]
    assert observer.count == supervisor.count == 3
    sim.reset()
    assert sim.ai_ftc_fusion.confirming_updates == 0
    assert observer.count == 0


def test_fusion_requires_model_and_supervisor():
    with pytest.raises(ValueError):
        SixDOFSimulator(ai_ftc_fusion=AIControlEvidenceGate())


def test_missing_tracking_input_is_not_reported_as_measured_zero():
    assert not build_rule_based_ftc_evidence({"time": 0.}).tracking_error_available
    assert build_rule_based_ftc_evidence({"time": 0., "tracking_error_ratio": 0.}).tracking_error_available


def test_unavailable_tracking_cannot_corroborate_but_allocation_can():
    gate = AIControlEvidenceGate()
    for t in (0., .5):
        base = evidence(t, tracking_error_available=False)
        fused, report = gate.update(base, diagnosis(t))
    assert fused is base
    assert report["verdict"] == "record_only"
    fused, report = gate.update(replace(base, time_s=1., allocation_residual_ratio=.2), diagnosis(1.))
    assert report["accepted"]
    assert report["supporting_control_signals"] == ["control_saturation", "allocation_residual"]
