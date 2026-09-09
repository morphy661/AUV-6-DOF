import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from actuators.six_dof_thruster_faults import SingleThrusterFault, SixDOFThrusterFaultMode
from actuators.thruster_array import default_six_thruster_array
from environment.six_dof_dynamics import SixDOFState, euler_to_quaternion
from environment.six_dof_simulator import SixDOFSimulator
from ftc.degraded_guidance import FTCGuidanceManager
from ftc.safety_supervisor import (
    FTCEvidence,
    FTCSafetySupervisor,
    FTCSupervisorConfig,
    build_rule_based_ftc_evidence,
)
from simple_control.six_dof_controller import PoseTarget


def evidence_log(**updates):
    values = {
        "time": 1.0,
        "commanded_thruster_forces": np.zeros(6),
        "thruster_force_limits": np.full(6, 40.0),
        "thruster_expected_currents": np.ones(6),
        "thruster_measured_currents": np.ones(6),
        "thruster_expected_rpms": np.full(6, 100.0),
        "thruster_measured_rpms": np.full(6, 100.0),
        "thruster_telemetry_valid": np.ones(6, dtype=bool),
        "thruster_telemetry_age_s": np.zeros(6),
        "desired_wrench_body": np.zeros(6),
        "allocation_residual_body": np.zeros(6),
        "controller_position_ned": np.array([1.0, 2.0, 100.0]),
        "controller_euler_rpy": np.array([0.1, 0.2, 0.3]),
        "target_position_ned": np.array([4.0, 6.0, 102.0]),
        "target_euler_rpy": np.array([0.0, 0.0, 0.5]),
        "ftc_guidance_mode_applied": "normal",
    }
    values.update(updates)
    return values


def max_tilt_deg(logs):
    euler = np.array([row["euler_rpy"] for row in logs])
    return float(np.max(np.degrees(np.arccos(np.clip(
        np.cos(euler[:, 0]) * np.cos(euler[:, 1]), -1.0, 1.0
    )))))


class SixDOFSafetyRecoveryTests(unittest.TestCase):
    def test_tracking_safety_uses_controller_state_not_simulator_truth(self):
        first = build_rule_based_ftc_evidence(evidence_log(
            position_ned=np.array([999.0, 999.0, 999.0]),
            euler_rpy=np.array([1.0, 1.0, 1.0]),
        ))
        second = build_rule_based_ftc_evidence(evidence_log(
            position_ned=np.array([-999.0, -999.0, -999.0]),
            euler_rpy=np.array([-1.0, -1.0, -1.0]),
        ))

        self.assertTrue(first.tracking_error_available)
        self.assertFalse(first.position_tracking_safety_active)
        self.assertAlmostEqual(first.horizontal_tracking_error_m, 5.0)
        self.assertAlmostEqual(first.depth_tracking_error_m, 2.0)
        self.assertAlmostEqual(first.yaw_tracking_error_deg, np.degrees(0.2))
        self.assertAlmostEqual(
            first.tracking_error_ratio,
            first.tilt_angle_deg / FTCSupervisorConfig().tilt_limit_deg,
        )
        self.assertAlmostEqual(
            first.tracking_error_ratio, second.tracking_error_ratio
        )
        self.assertAlmostEqual(first.tilt_angle_deg, second.tilt_angle_deg)

    def test_local_hold_activates_horizontal_and_depth_limits(self):
        evidence = build_rule_based_ftc_evidence(evidence_log(
            controller_euler_rpy=np.zeros(3),
            ftc_guidance_mode_applied="vertical_thruster_hold",
        ))

        self.assertTrue(evidence.position_tracking_safety_active)
        self.assertAlmostEqual(evidence.tracking_error_ratio, 2.0 / 3.0)

    def test_missing_controller_state_keeps_tracking_unavailable(self):
        log = evidence_log()
        del log["controller_position_ned"]

        evidence = build_rule_based_ftc_evidence(log)

        self.assertFalse(evidence.tracking_error_available)
        self.assertIsNone(evidence.horizontal_tracking_error_m)

    def test_vertical_fault_allocation_prioritizes_pitch_over_heave(self):
        array = default_six_thruster_array()
        effectiveness = np.ones(6)
        effectiveness[4] = 0.0
        desired = np.array([0.0, 0.0, 60.0, 0.0, 0.0, 0.0])

        equal = array.allocate(desired, effectiveness)
        priority = array.allocate(
            desired,
            effectiveness,
            wrench_weights=np.array([1.0, 1.0, 0.1, 0.0, 5.0, 1.0]),
        )

        self.assertLess(
            abs(priority.residual_wrench[4]),
            0.01 * abs(equal.residual_wrench[4]),
        )
        self.assertGreater(
            abs(priority.residual_wrench[2]),
            abs(equal.residual_wrench[2]),
        )

    def test_small_residual_is_normalized_by_available_actuator_authority(self):
        array = default_six_thruster_array()
        effectiveness = np.ones(6)
        effectiveness[4] = 0.0
        log = evidence_log(
            allocation_residual_body=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            thruster_allocation_matrix=array.allocation_matrix,
            allocation_thruster_effectiveness=effectiveness,
        )

        evidence = build_rule_based_ftc_evidence(log)

        self.assertAlmostEqual(evidence.allocation_residual_ratio, 1.0 / 40.0)

    def test_deprioritized_heave_residual_relies_on_hold_tracking_safety(self):
        array = default_six_thruster_array()
        effectiveness = np.ones(6)
        effectiveness[4] = 0.0
        log = evidence_log(
            allocation_residual_body=np.array([0.0, 0.0, 32.0, 0.0, 0.0, 0.0]),
            thruster_allocation_matrix=array.allocation_matrix,
            allocation_thruster_effectiveness=effectiveness,
            ftc_allocation_wrench_weights_applied=np.array([
                1.0, 1.0, 0.1, 0.0, 5.0, 1.0,
            ]),
            ftc_guidance_mode_applied="vertical_thruster_hold",
            controller_euler_rpy=np.zeros(3),
            controller_position_ned=np.array([1.0, 2.0, 100.0]),
            target_position_ned=np.array([1.0, 2.0, 100.0]),
        )

        evidence = build_rule_based_ftc_evidence(log)

        self.assertAlmostEqual(evidence.allocation_residual_ratio, 0.08)
        self.assertEqual(evidence.tracking_error_ratio, 0.0)

    def test_guidance_latches_estimated_hold_point_after_vertical_isolation(self):
        config = FTCSupervisorConfig(no_output_confirmation_s=0.0)
        supervisor = FTCSafetySupervisor(config)
        scores = np.array([0.0, 0.0, 0.0, 0.0, 0.97, 0.0])
        decision = supervisor.update(FTCEvidence(
            time_s=1.0,
            no_output_scores=scores,
            excitation_ratios=np.full(6, 0.5),
        ))
        state = SixDOFState(
            position_ned=np.array([3.0, 4.0, 200.0]),
            quaternion_nb=euler_to_quaternion(0.1, -0.1, 0.7),
        )
        requested = PoseTarget(
            np.array([20.0, 5.0, 205.0]),
            np.array([0.0, 0.0, 1.0]),
            guidance_context_id=4,
            linear_velocity_ned=np.array([1.0, 0.0, 0.2]),
        )
        manager = FTCGuidanceManager()

        first = manager.update(requested, state, decision)
        moved = state.copy()
        moved.position_ned += np.array([10.0, 10.0, 10.0])
        second = manager.update(requested, moved, decision)

        self.assertTrue(first.entered_mode)
        self.assertFalse(second.entered_mode)
        self.assertEqual(first.mode, "vertical_thruster_hold")
        np.testing.assert_allclose(first.target.position_ned, state.position_ned)
        np.testing.assert_allclose(second.target.position_ned, state.position_ned)
        np.testing.assert_allclose(first.target.linear_velocity_ned, np.zeros(3))
        self.assertAlmostEqual(first.target.euler_rpy[2], 0.7)
        self.assertEqual(first.target.guidance_context_id, 100004)

    def test_distant_normal_waypoint_does_not_create_tracking_stress(self):
        simulator = SixDOFSimulator(ftc_supervisor=FTCSafetySupervisor())
        target = PoseTarget(np.array([100.0, 50.0, 200.0]))

        logs = simulator.run(3.0, 0.05, lambda _t, _s: target)

        self.assertTrue(all(row["ftc_tracking_error_ratio_available"] for row in logs))
        self.assertTrue(all(not row["ftc_position_tracking_safety_active"] for row in logs))
        self.assertTrue(all(row["ftc_action"] != "safe_hold_or_abort" for row in logs))

    def test_vertical_recovery_prevents_v1_attitude_loss(self):
        def run(recovery):
            simulator = SixDOFSimulator(
                fault=SingleThrusterFault(
                    "V1",
                    SixDOFThrusterFaultMode.NO_OUTPUT,
                    start_time=3.0,
                    thrust_efficiency=0.0,
                ),
                ftc_supervisor=FTCSafetySupervisor(),
                vertical_fault_recovery=recovery,
            )

            def target(time_s, _state):
                return PoseTarget(
                    np.array([1.35 * time_s, 0.0, 200.0 + 0.1 * time_s]),
                    guidance_context_id=1,
                )

            simulator.reset(SixDOFState(
                position_ned=np.array([0.0, 0.0, 200.0])
            ))
            return simulator.run(15.0, 0.05, target)

        legacy = run(False)
        recovered = run(True)

        self.assertGreater(max_tilt_deg(legacy), 30.0)
        self.assertLess(max_tilt_deg(recovered), 10.0)
        self.assertTrue(any(
            row["ftc_guidance_mode_applied"] == "vertical_thruster_hold"
            for row in recovered
        ))


if __name__ == "__main__":
    unittest.main()
