import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from environment.six_dof_simulator import SixDOFSimulator
from actuators.six_dof_thruster_faults import (
    SingleThrusterFault,
    SixDOFThrusterFaultMode,
)
from ftc.safety_supervisor import (
    FTCAction,
    FTCEvidence,
    FTCSafetySupervisor,
    FTCSupervisorConfig,
    build_rule_based_ftc_evidence,
)
from simple_control.six_dof_controller import PoseTarget


class SixDOFFTCSupervisorTests(unittest.TestCase):
    @staticmethod
    def config(**overrides):
        values = {
            "no_output_confirmation_s": 1.0,
            "degraded_confirmation_s": 1.0,
            "critical_confirmation_s": 1.0,
            "recovery_confirmation_s": 2.0,
        }
        values.update(overrides)
        return FTCSupervisorConfig(**values)

    @staticmethod
    def evidence(time_s, **overrides):
        values = {
            "time_s": time_s,
            "no_output_scores": np.zeros(6),
            "excitation_ratios": np.zeros(6),
        }
        values.update(overrides)
        return FTCEvidence(**values)

    def test_compensated_thrust_loss_is_log_only(self):
        supervisor = FTCSafetySupervisor(self.config())

        decision = supervisor.update(self.evidence(
            0.0,
            health_level=2,
            confirmed_mode=2,
            fault_probability=0.95,
        ))

        self.assertEqual(decision.action, FTCAction.LOG_ONLY)
        self.assertFalse(decision.intervention_required)

    def test_low_excitation_cannot_isolate_a_thruster(self):
        supervisor = FTCSafetySupervisor(self.config())
        scores = np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.01])

        for time_s in (0.0, 1.0, 2.0):
            decision = supervisor.update(self.evidence(
                time_s,
                health_level=3,
                confirmed_mode=1,
                no_output_scores=scores,
                excitation_ratios=np.full(6, 0.05),
            ))

        self.assertEqual(decision.action, FTCAction.LOG_ONLY)
        self.assertEqual(decision.isolated_thruster_indices, ())

    def test_vertical_excitation_override_does_not_relax_horizontal_gate(self):
        config = self.config(
            no_output_confirmation_s=0.0,
            minimum_excitation_ratio=0.20,
            vertical_minimum_excitation_ratio=0.08,
        )
        excitation = np.full(6, 0.10)

        horizontal = FTCSafetySupervisor(config).update(self.evidence(
            0.0,
            no_output_scores=np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.01]),
            excitation_ratios=excitation,
        ))
        vertical = FTCSafetySupervisor(config).update(self.evidence(
            0.0,
            no_output_scores=np.array([0.01, 0.01, 0.01, 0.01, 0.95, 0.01]),
            excitation_ratios=excitation,
        ))

        self.assertIsNone(horizontal.targeted_thruster_name)
        self.assertEqual(vertical.targeted_thruster_name, "V1")

    def test_vertical_excitation_override_must_be_a_valid_ratio(self):
        with self.assertRaises(ValueError):
            self.config(vertical_minimum_excitation_ratio=1.01)

    def test_default_vertical_gate_matches_low_command_health_threshold(self):
        thresholds = FTCSupervisorConfig().minimum_excitation_ratios
        np.testing.assert_allclose(thresholds[:4], 0.20)
        np.testing.assert_allclose(thresholds[4:], 0.08)

    def test_default_esc_telemetry_gate_is_enabled(self):
        config = FTCSupervisorConfig()

        self.assertTrue(config.require_fresh_esc_telemetry)
        self.assertAlmostEqual(config.maximum_esc_telemetry_age_s, 0.20)

    def test_esc_telemetry_age_must_be_non_negative(self):
        with self.assertRaises(ValueError):
            FTCSupervisorConfig(maximum_esc_telemetry_age_s=-0.01)

    def test_short_vertical_dropout_does_not_complete_confirmation(self):
        supervisor = FTCSafetySupervisor(FTCSupervisorConfig())
        scores = np.array([0.01, 0.01, 0.01, 0.01, 0.95, 0.01])
        excitation = np.full(6, 0.10)

        decisions = [
            supervisor.update(self.evidence(
                time_s,
                no_output_scores=(scores if time_s <= 0.40 else np.zeros(6)),
                excitation_ratios=excitation,
            ))
            for time_s in (0.0, 0.2, 0.4, 0.45, 0.8)
        ]

        self.assertTrue(all(
            decision.targeted_thruster_name is None for decision in decisions
        ))
        self.assertTrue(all(
            not decision.isolated_thruster_indices for decision in decisions
        ))

    def test_direct_no_output_evidence_isolates_and_latches(self):
        supervisor = FTCSafetySupervisor(self.config())
        scores = np.array([0.01, 0.95, 0.01, 0.01, 0.01, 0.01])
        excitation = np.full(6, 0.50)

        first = supervisor.update(self.evidence(
            0.0,
            health_level=3,
            confirmed_mode=1,
            no_output_scores=scores,
            excitation_ratios=excitation,
        ))
        confirmed = supervisor.update(self.evidence(
            1.0,
            health_level=3,
            confirmed_mode=1,
            no_output_scores=scores,
            excitation_ratios=excitation,
        ))
        latched = supervisor.update(self.evidence(2.0))

        self.assertEqual(first.action, FTCAction.LOG_ONLY)
        self.assertEqual(confirmed.action, FTCAction.TARGETED_REALLOCATION)
        self.assertEqual(confirmed.targeted_thruster_name, "H2")
        self.assertEqual(confirmed.isolated_thruster_indices, (2,))
        self.assertEqual(confirmed.estimated_thruster_effectiveness[1], 0.0)
        self.assertEqual(latched.action, FTCAction.TARGETED_REALLOCATION)

    def test_new_direct_isolation_precedes_critical_abort(self):
        supervisor = FTCSafetySupervisor(self.config(
            no_output_confirmation_s=0.0,
            critical_confirmation_s=0.0,
        ))
        decision = supervisor.update(self.evidence(
            0.0,
            health_level=3,
            confirmed_mode=1,
            tracking_error_ratio=1.2,
            no_output_scores=np.array([
                0.95, 0.01, 0.01, 0.01, 0.01, 0.01
            ]),
            excitation_ratios=np.full(6, 0.50),
        ))

        self.assertEqual(decision.action, FTCAction.TARGETED_REALLOCATION)
        self.assertEqual(decision.targeted_thruster_name, "H1")

    def test_new_direct_isolation_precedes_imu_sensor_guard(self):
        supervisor = FTCSafetySupervisor(self.config(
            no_output_confirmation_s=0.0,
        ))
        decision = supervisor.update(self.evidence(
            0.0,
            health_level=3,
            confirmed_mode=1,
            no_output_scores=np.array([
                0.01, 0.95, 0.01, 0.01, 0.01, 0.01
            ]),
            excitation_ratios=np.full(6, 0.50),
            sensor_guard_action="safe_hold_or_abort",
            untrusted_sensors=("imu",),
        ))

        self.assertEqual(decision.action, FTCAction.TARGETED_REALLOCATION)
        self.assertEqual(decision.targeted_thruster_name, "H2")

    def test_imu_sensor_guard_never_guesses_thruster_location(self):
        supervisor = FTCSafetySupervisor(self.config())

        decision = supervisor.update(self.evidence(
            0.0,
            sensor_guard_action="safe_hold_or_abort",
            untrusted_sensors=("imu",),
        ))

        self.assertEqual(decision.action, FTCAction.SAFE_HOLD_OR_ABORT)
        self.assertIsNone(decision.targeted_thruster_name)
        self.assertEqual(decision.isolated_thruster_indices, ())

    def test_critical_control_stress_requires_persistence(self):
        supervisor = FTCSafetySupervisor(self.config())

        first = supervisor.update(self.evidence(
            0.0,
            health_level=2,
            confirmed_mode=2,
            tracking_error_ratio=1.2,
        ))
        confirmed = supervisor.update(self.evidence(
            1.0,
            health_level=2,
            confirmed_mode=2,
            tracking_error_ratio=1.2,
        ))

        self.assertEqual(first.action, FTCAction.LOG_ONLY)
        self.assertEqual(confirmed.action, FTCAction.SAFE_HOLD_OR_ABORT)
        self.assertTrue(confirmed.mission_abort_requested)

    def test_stress_only_critical_uses_longer_abort_timeout(self):
        supervisor = FTCSafetySupervisor(self.config(
            degraded_confirmation_s=1.0,
            stress_only_critical_confirmation_s=5.0,
        ))

        first = supervisor.update(self.evidence(
            0.0, tracking_error_ratio=1.2
        ))
        degraded = supervisor.update(self.evidence(
            1.0, tracking_error_ratio=1.2
        ))
        still_degraded = supervisor.update(self.evidence(
            4.9, tracking_error_ratio=1.2
        ))
        timed_out = supervisor.update(self.evidence(
            5.0, tracking_error_ratio=1.2
        ))

        self.assertEqual(first.action, FTCAction.NORMAL_CONTROL)
        self.assertEqual(degraded.action, FTCAction.DEGRADED_OPERATION)
        self.assertEqual(still_degraded.action, FTCAction.DEGRADED_OPERATION)
        self.assertEqual(timed_out.action, FTCAction.SAFE_HOLD_OR_ABORT)

    def test_improving_stress_only_signal_delays_abort(self):
        supervisor = FTCSafetySupervisor(self.config(
            degraded_confirmation_s=1.0,
            stress_only_critical_confirmation_s=5.0,
            stress_only_recovery_fraction=0.80,
        ))

        supervisor.update(self.evidence(
            0.0, tracking_error_ratio=2.0
        ))
        improving = supervisor.update(self.evidence(
            5.0, tracking_error_ratio=1.2
        ))

        self.assertEqual(improving.action, FTCAction.DEGRADED_OPERATION)
        self.assertFalse(improving.mission_abort_requested)

    def test_short_tracking_excursion_recovers_without_intervention(self):
        supervisor = FTCSafetySupervisor(self.config(
            degraded_confirmation_s=2.0,
            critical_confirmation_s=2.0,
        ))

        first = supervisor.update(self.evidence(
            0.0, tracking_error_ratio=1.2
        ))
        pending = supervisor.update(self.evidence(
            1.5, tracking_error_ratio=1.2
        ))
        recovered = supervisor.update(self.evidence(1.75))

        self.assertEqual(first.action, FTCAction.NORMAL_CONTROL)
        self.assertEqual(pending.action, FTCAction.NORMAL_CONTROL)
        self.assertEqual(recovered.action, FTCAction.NORMAL_CONTROL)
        self.assertFalse(recovered.intervention_required)
        self.assertEqual(recovered.isolated_thruster_indices, ())

    def test_high_command_without_fault_evidence_does_not_degrade(self):
        supervisor = FTCSafetySupervisor(self.config())

        for time_s in (0.0, 1.0, 2.0):
            decision = supervisor.update(self.evidence(
                time_s,
                control_saturation_ratio=0.99,
            ))

        self.assertEqual(decision.action, FTCAction.NORMAL_CONTROL)

    def test_fault_plus_sustained_saturation_enters_degraded_operation(self):
        supervisor = FTCSafetySupervisor(self.config())

        supervisor.update(self.evidence(
            0.0,
            health_level=2,
            confirmed_mode=2,
            control_saturation_ratio=0.85,
        ))
        decision = supervisor.update(self.evidence(
            1.0,
            health_level=2,
            confirmed_mode=2,
            control_saturation_ratio=0.85,
        ))

        self.assertEqual(decision.action, FTCAction.DEGRADED_OPERATION)
        self.assertLess(decision.wrench_scale, 1.0)

    def test_vertical_authority_loss_requests_controlled_ascent(self):
        supervisor = FTCSafetySupervisor(self.config())

        decision = supervisor.update(self.evidence(
            0.0,
            vertical_control_unavailable=True,
        ))

        self.assertEqual(decision.action, FTCAction.CONTROLLED_ASCENT)
        self.assertTrue(decision.controlled_ascent_requested)

    def test_rule_evidence_uses_current_and_rpm_dropout(self):
        log = self.observable_log()
        log["commanded_thruster_forces"][0] = 20.0
        log["thruster_expected_currents"][0] = 2.0
        log["thruster_measured_currents"][0] = 0.0
        log["thruster_expected_rpms"][0] = 1000.0
        log["thruster_measured_rpms"][0] = 0.0

        evidence = build_rule_based_ftc_evidence(log, config=self.config())

        self.assertEqual(evidence.no_output_scores[0], 1.0)
        self.assertEqual(evidence.confirmed_mode, 1)
        self.assertEqual(evidence.health_level, 3)

    def test_invalid_zero_filled_esc_packet_cannot_isolate_thruster(self):
        log = self.observable_log()
        log["commanded_thruster_forces"][4] = 4.0
        log["thruster_expected_currents"][4] = 2.0
        log["thruster_expected_rpms"][4] = 1000.0
        log["thruster_telemetry_valid"] = np.ones(6, dtype=bool)
        log["thruster_telemetry_valid"][4] = False
        log["thruster_telemetry_age_s"] = np.zeros(6)
        log["thruster_telemetry_age_s"][4] = 0.5

        evidence = build_rule_based_ftc_evidence(log)
        decision = FTCSafetySupervisor(FTCSupervisorConfig(
            no_output_confirmation_s=0.0
        )).update(evidence)

        self.assertEqual(evidence.no_output_scores[4], 0.0)
        self.assertEqual(evidence.untrusted_esc_channels, ("V1",))
        self.assertIn("esc_telemetry_guard", evidence.source)
        self.assertEqual(decision.action, FTCAction.LOG_ONLY)
        self.assertIsNone(decision.targeted_thruster_name)

    def test_stale_esc_packet_cannot_isolate_thruster(self):
        log = self.observable_log()
        log["commanded_thruster_forces"][0] = 20.0
        log["thruster_expected_currents"][0] = 2.0
        log["thruster_expected_rpms"][0] = 1000.0
        log["thruster_telemetry_valid"] = np.ones(6, dtype=bool)
        log["thruster_telemetry_age_s"] = np.zeros(6)
        log["thruster_telemetry_age_s"][0] = 0.25

        evidence = build_rule_based_ftc_evidence(log)

        self.assertEqual(evidence.no_output_scores[0], 0.0)
        self.assertEqual(evidence.untrusted_esc_channels, ("H1",))

    def test_valid_fresh_no_output_still_isolates_thruster(self):
        log = self.observable_log()
        log["commanded_thruster_forces"][4] = 4.0
        log["thruster_expected_currents"][4] = 2.0
        log["thruster_expected_rpms"][4] = 1000.0
        log["thruster_telemetry_valid"] = np.ones(6, dtype=bool)
        log["thruster_telemetry_age_s"] = np.zeros(6)

        evidence = build_rule_based_ftc_evidence(log)
        decision = FTCSafetySupervisor(FTCSupervisorConfig(
            no_output_confirmation_s=0.0
        )).update(evidence)

        self.assertEqual(evidence.no_output_scores[4], 1.0)
        self.assertEqual(evidence.untrusted_esc_channels, ())
        self.assertEqual(decision.targeted_thruster_name, "V1")

    def test_legacy_zero_fill_mode_reproduces_old_false_isolation(self):
        log = self.observable_log()
        log["commanded_thruster_forces"][4] = 4.0
        log["thruster_expected_currents"][4] = 2.0
        log["thruster_expected_rpms"][4] = 1000.0
        log["thruster_telemetry_valid"] = np.array(
            [True, True, True, True, False, True]
        )
        log["thruster_telemetry_age_s"] = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0]
        )
        config = FTCSupervisorConfig(
            no_output_confirmation_s=0.0,
            require_fresh_esc_telemetry=False,
        )

        evidence = build_rule_based_ftc_evidence(log, config=config)
        decision = FTCSafetySupervisor(config).update(evidence)

        self.assertEqual(evidence.no_output_scores[4], 1.0)
        self.assertEqual(evidence.untrusted_esc_channels, ())
        self.assertEqual(decision.targeted_thruster_name, "V1")

    def test_esc_telemetry_vectors_are_validated(self):
        log = self.observable_log()
        log["thruster_telemetry_valid"] = np.ones(5, dtype=bool)
        with self.assertRaises(ValueError):
            build_rule_based_ftc_evidence(log)

        log = self.observable_log()
        log["thruster_telemetry_age_s"] = np.array(
            [0.0, 0.0, 0.0, 0.0, -0.1, 0.0]
        )
        with self.assertRaises(ValueError):
            build_rule_based_ftc_evidence(log)

    def test_privileged_truth_cannot_change_ftc_evidence(self):
        log = self.observable_log()
        log["commanded_thruster_forces"][0] = 20.0
        log["thruster_expected_currents"][0] = 2.0
        log["thruster_expected_rpms"][0] = 1000.0
        baseline = build_rule_based_ftc_evidence(log, config=self.config())

        changed = dict(log)
        changed.update({
            "actual_thruster_forces": np.zeros(6),
            "thruster_force_efficiencies": np.zeros(6),
            "faulted_thruster_index": 0,
            "thruster_fault_modes": ("no_output",) + ("normal",) * 5,
            "sensor_fault_truth": {
                "depth": {"active": True, "fault_type": "unavailable"},
                "imu": {"active": True, "fault_type": "stuck"},
                "dvl": {"active": True, "fault_type": "spike"},
            },
        })
        repeated = build_rule_based_ftc_evidence(
            changed, config=self.config()
        )

        np.testing.assert_allclose(
            baseline.no_output_scores, repeated.no_output_scores
        )
        self.assertEqual(baseline.confirmed_mode, repeated.confirmed_mode)
        self.assertEqual(baseline.sensor_guard_action, "none")
        self.assertEqual(repeated.sensor_guard_action, "none")

    def test_observable_sensor_summary_reaches_ftc_evidence(self):
        log = self.observable_log()
        log["sensor_health_summary"] = {
            "ftc_recommendation": "degraded_navigation",
            "untrusted_sensors": ["depth"],
        }

        evidence = build_rule_based_ftc_evidence(log, config=self.config())
        decision = FTCSafetySupervisor(self.config()).update(evidence)

        self.assertEqual(evidence.sensor_guard_action, "degraded_navigation")
        self.assertEqual(evidence.untrusted_sensors, ("depth",))
        self.assertIn("sensor_guard", evidence.source)
        self.assertEqual(decision.action, FTCAction.DEGRADED_OPERATION)
        self.assertIsNone(decision.targeted_thruster_name)

    def test_degraded_position_reference_keeps_navigation_degraded(self):
        log = self.observable_log()
        log["state_estimate_ftc_recommendation"] = "degraded_navigation"

        evidence = build_rule_based_ftc_evidence(log, config=self.config())
        decision = FTCSafetySupervisor(self.config()).update(evidence)

        self.assertEqual(evidence.sensor_guard_action, "degraded_navigation")
        self.assertIn("state_estimator_guard", evidence.source)
        self.assertEqual(decision.action, FTCAction.DEGRADED_OPERATION)
        self.assertIsNone(decision.targeted_thruster_name)

    def test_simulator_applies_supervisor_effectiveness_next_step(self):
        supervisor = FTCSafetySupervisor(self.config(
            no_output_confirmation_s=0.0
        ))

        def provider(log):
            return self.evidence(
                log["time"],
                health_level=3,
                confirmed_mode=1,
                no_output_scores=np.array([
                    0.95, 0.01, 0.01, 0.01, 0.01, 0.01
                ]),
                excitation_ratios=np.full(6, 0.50),
                source="test_observable_provider",
            )

        simulator = SixDOFSimulator(
            ftc_supervisor=supervisor,
            ftc_evidence_provider=provider,
        )
        target = PoseTarget(np.array([5.0, 0.0, 0.0]))
        first = simulator.step(target, dt=0.1)
        second = simulator.step(target, dt=0.1)

        self.assertEqual(first["ftc_action"], "targeted_reallocation")
        self.assertEqual(second["ftc_applied_action"], "targeted_reallocation")
        self.assertEqual(second["allocation_thruster_effectiveness"][0], 0.0)

    def test_observable_closed_loop_rule_isolates_real_no_output(self):
        fault = SingleThrusterFault(
            thruster_name="H1",
            mode=SixDOFThrusterFaultMode.NO_OUTPUT,
            start_time=0.0,
            thrust_efficiency=0.0,
        )
        simulator = SixDOFSimulator(
            fault=fault,
            ftc_supervisor=FTCSafetySupervisor(self.config(
                no_output_confirmation_s=0.5
            )),
        )

        def target_provider(_time_s, _state):
            return PoseTarget(
                np.array([8.0, 4.0, 1.0]),
                np.array([0.0, 0.0, 0.7]),
            )

        logs = simulator.run(2.0, 0.1, target_provider)
        intervention = next(
            log for log in logs
            if log["ftc_action"] == "targeted_reallocation"
        )

        self.assertEqual(intervention["ftc_targeted_thruster_name"], "H1")
        self.assertEqual(intervention["ftc_evidence_source"], "esc_rule")
        self.assertTrue(any(
            log["allocation_thruster_effectiveness"][0] == 0.0
            for log in logs
        ))

    @staticmethod
    def observable_log():
        return {
            "time": 1.0,
            "commanded_thruster_forces": np.zeros(6),
            "thruster_force_limits": np.full(6, 40.0),
            "thruster_expected_currents": np.zeros(6),
            "thruster_measured_currents": np.zeros(6),
            "thruster_expected_rpms": np.zeros(6),
            "thruster_measured_rpms": np.zeros(6),
            "desired_wrench_body": np.zeros(6),
            "allocation_residual_body": np.zeros(6),
        }


if __name__ == "__main__":
    unittest.main()
