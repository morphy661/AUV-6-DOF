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
from sensors.sensor_faults import (
    SensorFaultEvent,
    SensorFaultInjector,
    SensorFaultMode,
)
from ftc.safety_supervisor import FTCSafetySupervisor
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

    def test_dvl_scheduled_dropout_recovers_after_window(self):
        sensor = DVLSensor(
            velocity_noise_std=0.0,
            dropout_windows=((1.0, 2.0),),
            seed=4,
        )
        self.state.time = 1.5
        missing = sensor.read(self.state)
        self.state.time = 2.0
        recovered = sensor.read(self.state)

        self.assertFalse(missing["valid"])
        self.assertEqual(missing["dropout_reason"], "scheduled")
        self.assertTrue(recovered["valid"])
        self.assertIsNone(recovered["dropout_reason"])

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
        self.assertIn("sensor_health", log)
        self.assertTrue(log["sensor_health_summary"]["all_sensors_trusted"])
        self.assertIn("sensor_fault_observations", log)
        self.assertEqual(
            log["sensor_fault_observation_summary"]["ftc_recommendation"],
            "none",
        )

    def test_depth_unavailable_fault_is_injected_and_recovers(self):
        suite = SixDOFSensorSuite(
            depth_sensor=DepthSensor(noise_std=0.0, drift_std=0.0),
            fault_injector=SensorFaultInjector((SensorFaultEvent(
                sensor="depth",
                mode=SensorFaultMode.UNAVAILABLE,
                start_time_s=1.0,
                end_time_s=2.0,
                event_id="depth_dropout",
            ),)),
        )
        self.state.time = 1.5
        failed = suite.read(self.state, dt=0.05)
        self.state.time = 2.0
        recovered = suite.read(self.state, dt=0.05)

        self.assertFalse(failed["depth_valid"])
        self.assertTrue(np.isnan(failed["depth"]))
        self.assertEqual(
            failed["sensor_fault_truth"]["depth"]["fault_type"],
            "unavailable",
        )
        self.assertTrue(recovered["depth_valid"])
        self.assertEqual(
            recovered["sensor_fault_truth"]["depth"]["fault_type"],
            "normal",
        )

    def test_imu_stuck_fault_freezes_observable_packet(self):
        suite = SixDOFSensorSuite(
            imu_sensor=IMUSensor(0.0, 0.0, 0.0),
            fault_injector=SensorFaultInjector((SensorFaultEvent(
                sensor="imu",
                mode=SensorFaultMode.STUCK,
                start_time_s=0.0,
                end_time_s=2.0,
            ),)),
        )
        self.state.time = 0.0
        first = suite.read(
            self.state,
            dt=0.05,
            linear_acceleration_body=np.array([0.1, 0.2, 0.3]),
        )
        changed = SixDOFState(
            position_ned=self.state.position_ned,
            quaternion_nb=euler_to_quaternion(0.4, 0.2, -0.3),
            body_velocity=np.array([1.0, 0.5, -0.2, 0.3, 0.2, 0.1]),
            time=0.5,
        )
        second = suite.read(
            changed,
            dt=0.05,
            linear_acceleration_body=np.array([1.0, 2.0, 3.0]),
        )

        np.testing.assert_allclose(
            second["imu"]["orientation"], first["imu"]["orientation"]
        )
        np.testing.assert_allclose(
            second["imu"]["angular_velocity"],
            first["imu"]["angular_velocity"],
        )
        np.testing.assert_allclose(
            second["imu"]["linear_acceleration"],
            first["imu"]["linear_acceleration"],
        )

    def test_dvl_spike_is_one_shot_and_channel_specific(self):
        suite = SixDOFSensorSuite(
            dvl_sensor=DVLSensor(velocity_noise_std=0.0),
            fault_injector=SensorFaultInjector((SensorFaultEvent(
                sensor="dvl",
                mode=SensorFaultMode.SPIKE,
                start_time_s=1.0,
                end_time_s=1.5,
                channels=(0,),
                magnitude=2.0,
            ),)),
        )
        self.state.time = 1.0
        spiked = suite.read(self.state, dt=0.05)
        self.state.time = 1.05
        normal = suite.read(self.state, dt=0.05)

        self.assertAlmostEqual(
            spiked["dvl"]["vx"], self.state.body_velocity[0] + 2.0
        )
        self.assertAlmostEqual(normal["dvl"]["vx"], self.state.body_velocity[0])
        self.assertEqual(
            normal["sensor_fault_truth"]["dvl"]["fault_type"],
            "normal",
        )

    def test_depth_bias_is_constant_during_interval_and_recovers(self):
        suite = SixDOFSensorSuite(
            depth_sensor=DepthSensor(noise_std=0.0, drift_std=0.0),
            fault_injector=SensorFaultInjector((SensorFaultEvent(
                sensor="depth",
                mode=SensorFaultMode.BIAS,
                start_time_s=1.0,
                end_time_s=2.0,
                magnitude=0.35,
            ),)),
        )
        self.state.time = 1.25
        biased = suite.read(self.state, dt=0.05)
        self.state.time = 2.0
        recovered = suite.read(self.state, dt=0.05)

        self.assertAlmostEqual(biased["depth"], 3.85)
        self.assertEqual(
            biased["sensor_fault_truth"]["depth"]["fault_type"],
            "bias",
        )
        self.assertAlmostEqual(recovered["depth"], 3.5)

    def test_dvl_drift_grows_linearly_on_selected_channel(self):
        suite = SixDOFSensorSuite(
            dvl_sensor=DVLSensor(velocity_noise_std=0.0),
            fault_injector=SensorFaultInjector((SensorFaultEvent(
                sensor="dvl",
                mode=SensorFaultMode.DRIFT,
                start_time_s=1.0,
                end_time_s=4.0,
                channels=(0,),
                magnitude=0.04,
            ),)),
        )
        self.state.time = 1.5
        early = suite.read(self.state, dt=0.05)
        self.state.time = 3.0
        later = suite.read(self.state, dt=0.05)

        self.assertAlmostEqual(early["dvl"]["vx"], 0.82)
        self.assertAlmostEqual(later["dvl"]["vx"], 0.88)
        self.assertAlmostEqual(later["dvl"]["vy"], -0.4)
        self.assertEqual(
            later["sensor_fault_truth"]["dvl"]["fault_type"],
            "drift",
        )

    def test_bias_and_drift_require_nonzero_magnitude(self):
        for mode in (SensorFaultMode.BIAS, SensorFaultMode.DRIFT):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    SensorFaultEvent(
                        sensor="imu",
                        mode=mode,
                        start_time_s=1.0,
                        end_time_s=2.0,
                    )

    def test_depth_unavailability_reaches_ftc_navigation_guard(self):
        suite = SixDOFSensorSuite(
            fault_injector=SensorFaultInjector((SensorFaultEvent(
                sensor="depth",
                mode=SensorFaultMode.UNAVAILABLE,
                start_time_s=0.0,
                end_time_s=1.0,
            ),)),
        )
        simulator = SixDOFSimulator(
            sensor_suite=suite,
            ftc_supervisor=FTCSafetySupervisor(),
        )
        log = simulator.step(
            PoseTarget(np.array([0.0, 0.0, 1.0]), np.zeros(3)),
            dt=0.05,
        )

        self.assertEqual(
            log["sensor_health"]["depth"]["fault_type"],
            "unavailable",
        )
        self.assertEqual(log["ftc_sensor_guard_action"], "degraded_navigation")
        self.assertEqual(log["ftc_action"], "degraded_operation")
        self.assertIsNone(log["ftc_targeted_thruster_index"])

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
