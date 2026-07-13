import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from environment.six_dof_dynamics import SixDOFState, euler_to_quaternion
from environment.six_dof_simulator import SixDOFSimulator
from sensors.depth_sensor import DepthSensor
from sensors.dvl_sensor import DVLSensor
from sensors.imu_sensor import IMUSensor
from sensors.six_dof_sensor_suite import SixDOFSensorSuite
from simple_control.six_dof_controller import PoseTarget


class SixDOFSensorTests(unittest.TestCase):
    def setUp(self):
        self.state = SixDOFState(
            position_ned=np.array([1.0, -2.0, 3.5]),
            quaternion_nb=euler_to_quaternion(0.1, -0.2, 0.3),
            body_velocity=np.array([0.8, -0.4, 0.2, 0.05, -0.06, 0.07]),
        )

    def test_zero_noise_imu_reads_six_dof_state(self):
        sensor = IMUSensor(
            attitude_noise_std=0.0,
            gyro_noise_std=0.0,
            accel_noise_std=0.0,
        )
        acceleration = np.array([0.3, -0.2, 0.1])

        packet = sensor.read(
            self.state,
            linear_acceleration_body=acceleration,
            dt=0.05,
        )

        np.testing.assert_allclose(packet["orientation"], self.state.euler_rpy)
        np.testing.assert_allclose(
            packet["angular_velocity"], self.state.body_velocity[3:]
        )
        np.testing.assert_allclose(packet["linear_acceleration"], acceleration)
        self.assertEqual(packet["frame"], "body")

    def test_imu_can_estimate_acceleration_from_velocity_difference(self):
        sensor = IMUSensor(0.0, 0.0, 0.0)
        first = SixDOFState(body_velocity=np.zeros(6))
        second = SixDOFState(
            body_velocity=np.array([1.0, -0.5, 0.25, 0.0, 0.0, 0.0])
        )

        sensor.read(first, dt=0.5)
        packet = sensor.read(second, dt=0.5)

        np.testing.assert_allclose(
            packet["linear_acceleration"],
            np.array([2.0, -1.0, 0.5]),
        )

    def test_zero_noise_dvl_reads_body_linear_velocity(self):
        packet = DVLSensor(velocity_noise_std=0.0).read(self.state)

        self.assertTrue(packet["valid"])
        self.assertEqual(packet["frame"], "body")
        np.testing.assert_allclose(packet["velocity"], self.state.body_velocity[:3])

    def test_dvl_dropout_marks_all_velocity_values_invalid(self):
        packet = DVLSensor(dropout_prob=1.0, seed=4).read(self.state)

        self.assertFalse(packet["valid"])
        self.assertTrue(np.all(np.isnan(packet["velocity"])))

    def test_sensor_suite_is_reproducible_after_reset(self):
        suite = SixDOFSensorSuite(seed=21)

        first = suite.read(
            self.state,
            dt=0.05,
            linear_acceleration_body=np.array([0.1, 0.2, 0.3]),
        )
        suite.reset()
        repeated = suite.read(
            self.state,
            dt=0.05,
            linear_acceleration_body=np.array([0.1, 0.2, 0.3]),
        )

        self.assertEqual(first["depth"], repeated["depth"])
        np.testing.assert_allclose(
            first["imu"]["orientation"], repeated["imu"]["orientation"]
        )
        np.testing.assert_allclose(
            first["dvl"]["velocity"], repeated["dvl"]["velocity"]
        )

    def test_six_dof_simulator_logs_synchronized_sensor_packet(self):
        suite = SixDOFSensorSuite(
            depth_sensor=DepthSensor(noise_std=0.0, drift_std=0.0),
            imu_sensor=IMUSensor(0.0, 0.0, 0.0),
            dvl_sensor=DVLSensor(velocity_noise_std=0.0),
        )
        simulator = SixDOFSimulator(sensor_suite=suite)
        target = PoseTarget(np.array([0.0, 0.0, 1.0]), np.zeros(3))

        log = simulator.step(target, dt=0.05)

        self.assertAlmostEqual(log["depth"], log["position_ned"][2])
        np.testing.assert_allclose(
            log["imu"]["orientation"], log["euler_rpy"], atol=1e-12
        )
        np.testing.assert_allclose(
            log["dvl"]["velocity"], log["body_velocity"][:3]
        )

    def test_legacy_state_interface_remains_supported(self):
        class LegacyState:
            orientation = np.array([0.1, 0.2, 0.3])
            velocity = np.array([1.0, 2.0, 3.0])
            angular_velocity = np.array([0.4, 0.5, 0.6])

        legacy = LegacyState()
        imu = IMUSensor(0.0, 0.0, 0.0).read(legacy, dt=0.1)
        dvl = DVLSensor(0.0).read(legacy)

        np.testing.assert_allclose(imu["orientation"], legacy.orientation)
        np.testing.assert_allclose(imu["angular_velocity"], legacy.angular_velocity)
        np.testing.assert_allclose(dvl["velocity"], legacy.velocity)


if __name__ == "__main__":
    unittest.main()
