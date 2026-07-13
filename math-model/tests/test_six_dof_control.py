import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from actuators.six_dof_thruster_faults import (
    SingleThrusterFault,
    SixDOFThrusterFaultMode,
    ThrusterActuatorBank,
)
from actuators.thruster_array import default_six_thruster_array
from environment.six_dof_dynamics import SixDOFState
from environment.six_dof_simulator import SixDOFNominalSimulator
from simple_control.six_dof_controller import (
    CascadedSixDOFController,
    PoseTarget,
)


class ThrusterAllocationTests(unittest.TestCase):
    def test_default_layout_has_six_thrusters_and_four_controlled_axes(self):
        array = default_six_thruster_array()
        self.assertEqual(array.names, ["H1", "H2", "H3", "H4", "V1", "V2"])
        self.assertEqual(array.allocation_matrix.shape, (6, 6))
        controlled = array.wrench_weights > 0
        np.testing.assert_array_equal(
            controlled,
            np.array([True, True, True, False, False, True]),
        )
        self.assertEqual(
            np.linalg.matrix_rank(array.allocation_matrix[controlled]),
            4,
        )

    def test_feasible_controlled_wrench_is_reconstructed(self):
        array = default_six_thruster_array()
        desired = np.array([10.0, -5.0, 8.0, 0.0, 0.0, 3.0])
        result = array.allocate(desired)

        np.testing.assert_allclose(result.achieved_wrench, desired, atol=1e-9)
        self.assertFalse(np.any(result.saturated))

    def test_infeasible_wrench_respects_force_limits(self):
        array = default_six_thruster_array()
        result = array.allocate(np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        self.assertTrue(np.any(result.saturated))
        self.assertTrue(np.all(result.thruster_forces <= array.max_forces + 1e-12))
        self.assertTrue(np.all(result.thruster_forces >= array.min_forces - 1e-12))
        self.assertGreater(np.linalg.norm(result.residual_wrench), 0.0)

    def test_roll_and_pitch_are_not_actively_allocated(self):
        array = default_six_thruster_array()
        desired = np.array([0.0, 0.0, 0.0, 2.0, -3.0, 0.0])

        result = array.allocate(desired)

        np.testing.assert_allclose(result.thruster_forces, 0.0, atol=1e-12)
        np.testing.assert_allclose(result.achieved_wrench, 0.0, atol=1e-12)
        np.testing.assert_allclose(result.residual_wrench, desired, atol=1e-12)

    def test_effectiveness_aware_allocation_avoids_failed_thruster(self):
        array = default_six_thruster_array()
        effectiveness = np.ones(6)
        effectiveness[0] = 0.0
        desired = np.array([5.0, -3.0, 4.0, 0.0, 0.0, 1.0])

        result = array.allocate(desired, thruster_effectiveness=effectiveness)

        self.assertAlmostEqual(result.thruster_forces[0], 0.0)
        np.testing.assert_allclose(result.achieved_wrench, desired, atol=1e-9)
        np.testing.assert_allclose(result.thruster_effectiveness, effectiveness)

    def test_invalid_thruster_effectiveness_is_rejected(self):
        array = default_six_thruster_array()

        with self.assertRaisesRegex(ValueError, "within"):
            array.allocate(np.zeros(6), np.array([1, 1, 1, 1, 1, 1.1]))


class ThrusterFaultActuationTests(unittest.TestCase):
    def test_no_output_removes_force_and_current_after_start(self):
        array = default_six_thruster_array()
        fault = SingleThrusterFault(
            "V1", SixDOFThrusterFaultMode.NO_OUTPUT, start_time=5.0
        )
        bank = ThrusterActuatorBank(array, fault=fault)
        commanded = np.full(6, 10.0)

        before = bank.apply(commanded, time_s=4.9)
        after = bank.apply(commanded, time_s=5.0)

        np.testing.assert_allclose(before.actual_forces, commanded)
        self.assertEqual(after.actual_forces[4], 0.0)
        self.assertLess(after.measured_currents[4], after.expected_currents[4])
        np.testing.assert_allclose(after.actual_forces[:4], commanded[:4])

    def test_thrust_loss_preserves_current_but_reduces_force(self):
        array = default_six_thruster_array()
        fault = SingleThrusterFault(
            "V1",
            SixDOFThrusterFaultMode.THRUST_LOSS,
            start_time=0.0,
            thrust_efficiency=0.45,
        )
        bank = ThrusterActuatorBank(array, fault=fault)
        commanded = np.full(6, 10.0)

        result = bank.apply(commanded, time_s=0.0)

        self.assertAlmostEqual(result.actual_forces[4], 4.5)
        self.assertAlmostEqual(
            result.measured_currents[4], result.expected_currents[4]
        )
        self.assertEqual(
            result.fault_modes[4], SixDOFThrusterFaultMode.THRUST_LOSS.value
        )

    def test_horizontal_and_vertical_fault_signatures_match_layout(self):
        array = default_six_thruster_array()
        commanded = np.full(6, 10.0)
        residuals = {}

        for thruster_name in ("H1", "V1"):
            fault = SingleThrusterFault(
                thruster_name,
                SixDOFThrusterFaultMode.NO_OUTPUT,
                start_time=0.0,
            )
            result = ThrusterActuatorBank(array, fault=fault).apply(
                commanded,
                time_s=0.0,
            )
            residuals[thruster_name] = array.wrench_from_forces(
                result.commanded_forces - result.actual_forces
            )

        horizontal_axes = np.abs(residuals["H1"]) > 1e-9
        vertical_axes = np.abs(residuals["V1"]) > 1e-9
        np.testing.assert_array_equal(
            horizontal_axes,
            np.array([True, True, False, False, False, True]),
        )
        np.testing.assert_array_equal(
            vertical_axes,
            np.array([False, False, True, False, True, False]),
        )

    def test_oracle_effectiveness_matches_active_fault(self):
        array = default_six_thruster_array()
        fault = SingleThrusterFault(
            "V2",
            SixDOFThrusterFaultMode.THRUST_LOSS,
            start_time=5.0,
            thrust_efficiency=0.35,
        )
        bank = ThrusterActuatorBank(array, fault=fault)

        np.testing.assert_allclose(bank.force_efficiencies_at(4.9), 1.0)
        expected = np.ones(6)
        expected[5] = 0.35
        np.testing.assert_allclose(bank.force_efficiencies_at(5.0), expected)


class SixDOFControllerTests(unittest.TestCase):
    def test_controller_is_zero_at_level_target(self):
        controller = CascadedSixDOFController()
        state = SixDOFState()
        target = PoseTarget(np.zeros(3), np.zeros(3))

        output = controller.compute(state, target, dt=0.1)

        np.testing.assert_allclose(output.desired_wrench_body, 0.0, atol=1e-12)

    def test_controller_handles_exact_180_degree_yaw_error(self):
        controller = CascadedSixDOFController()
        state = SixDOFState()
        target = PoseTarget(np.zeros(3), np.array([0.0, 0.0, np.pi]))

        output = controller.compute(state, target, dt=0.1)

        self.assertGreater(abs(output.attitude_error_body[2]), 3.0)
        self.assertGreater(abs(output.desired_wrench_body[5]), 1.0)

    def test_nominal_simulator_tracks_static_pose(self):
        simulator = SixDOFNominalSimulator()
        target = PoseTarget(
            position_ned=np.array([2.0, 1.0, 1.0]),
            euler_rpy=np.array([0.0, 0.0, 0.3]),
        )

        simulator.run(
            duration=30.0,
            dt=0.05,
            target_provider=lambda _time, _state: target,
        )

        position_error = np.linalg.norm(
            target.position_ned - simulator.state.position_ned
        )
        yaw_error = np.arctan2(
            np.sin(target.euler_rpy[2] - simulator.state.euler_rpy[2]),
            np.cos(target.euler_rpy[2] - simulator.state.euler_rpy[2]),
        )
        self.assertLess(position_error, 0.15)
        self.assertLess(abs(yaw_error), np.deg2rad(2.0))
        self.assertAlmostEqual(
            np.linalg.norm(simulator.state.quaternion_nb), 1.0, places=12
        )

    def test_faulted_simulator_logs_actual_wrench_loss(self):
        fault = SingleThrusterFault(
            "V1", SixDOFThrusterFaultMode.NO_OUTPUT, start_time=0.0
        )
        simulator = SixDOFNominalSimulator(fault=fault)
        target = PoseTarget(np.array([0.0, 0.0, 1.0]), np.zeros(3))

        log = simulator.step(target, dt=0.05)

        self.assertTrue(log["thruster_fault_active"])
        self.assertEqual(log["actual_thruster_forces"][4], 0.0)
        self.assertGreater(np.linalg.norm(log["actuation_residual_body"]), 0.0)

    def test_ideal_ftc_reallocates_around_known_failure(self):
        fault = SingleThrusterFault(
            "V1", SixDOFThrusterFaultMode.NO_OUTPUT, start_time=0.0
        )
        simulator = SixDOFNominalSimulator(
            fault=fault,
            ideal_fault_tolerant_allocation=True,
        )
        target = PoseTarget(np.array([0.0, 0.0, 1.0]), np.zeros(3))

        log = simulator.step(target, dt=0.05)

        self.assertTrue(log["ftc_active"])
        self.assertEqual(log["allocation_thruster_effectiveness"][4], 0.0)
        self.assertAlmostEqual(log["commanded_thruster_forces"][4], 0.0)
        np.testing.assert_allclose(
            log["actuation_residual_body"],
            0.0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
