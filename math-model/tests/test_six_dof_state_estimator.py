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
from estimation.six_dof_state_estimator import SixDOFStateEstimator
from ftc.safety_supervisor import FTCSafetySupervisor
from sensors.depth_sensor import DepthSensor
from sensors.dvl_sensor import DVLSensor
from sensors.imu_sensor import IMUSensor
from sensors.six_dof_sensor_suite import SixDOFSensorSuite
from sensors.sensor_faults import (
    SensorFaultEvent,
    SensorFaultInjector,
    SensorFaultMode,
)
from simple_control.six_dof_controller import PoseTarget


def health(**overrides):
    values = {
        sensor: {
            "fault_type": "normal",
            "trust_level": "trusted",
        }
        for sensor in ("depth", "imu", "dvl")
    }
    for sensor, update in overrides.items():
        values[sensor].update(update)
    return values


def packet(
    *,
    depth=0.0,
    orientation=None,
    angular_velocity=None,
    linear_acceleration=None,
    dvl_velocity=None,
    imu_valid=True,
    dvl_valid=True,
    truth=None,
):
    orientation = (
        np.zeros(3) if orientation is None else np.asarray(orientation)
    )
    angular_velocity = (
        np.zeros(3)
        if angular_velocity is None
        else np.asarray(angular_velocity)
    )
    linear_acceleration = (
        np.zeros(3)
        if linear_acceleration is None
        else np.asarray(linear_acceleration)
    )
    dvl_velocity = (
        np.zeros(3)
        if dvl_velocity is None
        else np.asarray(dvl_velocity)
    )
    result = {
        "depth": float(depth),
        "depth_valid": bool(np.isfinite(depth)),
        "imu": {
            "valid": imu_valid,
            "orientation": orientation,
            "angular_velocity": angular_velocity,
            "linear_acceleration": linear_acceleration,
        },
        "dvl": {
            "valid": dvl_valid,
            "velocity": dvl_velocity,
        },
    }
    if truth is not None:
        result["sensor_fault_truth"] = truth
    return result


class SixDOFStateEstimatorTests(unittest.TestCase):
    def test_healthy_measurements_update_all_observable_state_parts(self):
        estimator = SixDOFStateEstimator()
        estimate = estimator.update(
            time_s=1.0,
            dt=1.0,
            sensor_packet=packet(
                depth=2.0,
                orientation=np.array([0.1, -0.2, 0.3]),
                angular_velocity=np.array([0.01, 0.02, 0.03]),
                dvl_velocity=np.array([1.0, 0.5, 0.0]),
            ),
            sensor_health=health(),
        )

        self.assertAlmostEqual(estimate.state.position_ned[2], 2.0)
        np.testing.assert_allclose(
            estimate.state.body_velocity,
            np.array([1.0, 0.5, 0.0, 0.01, 0.02, 0.03]),
        )
        np.testing.assert_allclose(
            estimate.state.quaternion_nb,
            euler_to_quaternion(0.1, -0.2, 0.3),
        )
        self.assertEqual(estimate.quality, "nominal")
        self.assertEqual(estimate.sources["depth"], "depth_sensor")
        self.assertEqual(estimate.sources["linear_velocity"], "dvl")

    def test_depth_spike_is_rejected_and_depth_is_dead_reckoned(self):
        estimator = SixDOFStateEstimator()
        estimator.reset(SixDOFState(
            position_ned=np.array([0.0, 0.0, 5.0]),
            body_velocity=np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0]),
        ))
        estimate = estimator.update(
            time_s=1.0,
            dt=1.0,
            sensor_packet=packet(
                depth=100.0,
                dvl_velocity=np.array([0.0, 0.0, 0.2]),
            ),
            sensor_health=health(depth={
                "fault_type": "spike",
                "trust_level": "degraded",
            }),
        )

        self.assertAlmostEqual(estimate.state.position_ned[2], 5.2)
        self.assertIn("depth", estimate.rejected_sensors)
        self.assertEqual(estimate.quality, "cautious")
        self.assertEqual(
            estimate.sources["depth"], "velocity_dead_reckoning"
        )

    def test_dvl_unavailability_uses_imu_dead_reckoning(self):
        estimator = SixDOFStateEstimator()
        estimator.reset(SixDOFState(
            body_velocity=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ))
        estimate = estimator.update(
            time_s=1.0,
            dt=1.0,
            sensor_packet=packet(
                linear_acceleration=np.array([0.5, 0.0, 0.0]),
                dvl_velocity=np.full(3, np.nan),
                dvl_valid=False,
            ),
            sensor_health=health(dvl={
                "fault_type": "unavailable",
                "trust_level": "untrusted",
            }),
        )

        self.assertAlmostEqual(estimate.state.body_velocity[0], 1.5)
        self.assertEqual(estimate.quality, "degraded")
        self.assertEqual(
            estimate.sources["linear_velocity"], "imu_dead_reckoning"
        )
        self.assertAlmostEqual(estimate.fallback_durations_s["dvl"], 1.0)
        self.assertEqual(
            estimate.horizontal_position_reference,
            "degraded_without_absolute_fix",
        )
        self.assertEqual(
            estimate.ftc_recommendation, "degraded_navigation"
        )

    def test_imu_unavailability_holds_attitude_and_zeroes_rate(self):
        estimator = SixDOFStateEstimator()
        initial = SixDOFState(
            quaternion_nb=euler_to_quaternion(0.0, 0.0, 0.3),
            body_velocity=np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3]),
        )
        estimator.reset(initial)
        estimate = estimator.update(
            time_s=0.1,
            dt=0.1,
            sensor_packet=packet(
                orientation=np.array([1.0, 1.0, 1.0]),
                angular_velocity=np.ones(3),
                imu_valid=False,
            ),
            sensor_health=health(imu={
                "fault_type": "unavailable",
                "trust_level": "untrusted",
            }),
        )

        np.testing.assert_allclose(
            estimate.state.quaternion_nb, initial.quaternion_nb
        )
        np.testing.assert_allclose(estimate.state.body_velocity[3:], 0.0)
        self.assertEqual(estimate.quality, "unsafe")
        self.assertEqual(
            estimate.sources["attitude"], "held_last_attitude"
        )

    def test_privileged_fault_truth_cannot_change_estimate(self):
        first = SixDOFStateEstimator()
        second = SixDOFStateEstimator()
        common = dict(
            depth=1.0,
            orientation=np.array([0.1, 0.2, 0.3]),
            angular_velocity=np.array([0.01, 0.02, 0.03]),
            dvl_velocity=np.array([0.5, -0.2, 0.1]),
        )
        baseline = first.update(
            time_s=0.1,
            dt=0.1,
            sensor_packet=packet(**common),
            sensor_health=health(),
        )
        changed = second.update(
            time_s=0.1,
            dt=0.1,
            sensor_packet=packet(
                **common,
                truth={
                    "depth": {"active": True, "fault_type": "stuck"},
                    "imu": {"active": True, "fault_type": "spike"},
                    "dvl": {
                        "active": True,
                        "fault_type": "unavailable",
                    },
                },
            ),
            sensor_health=health(),
        )

        np.testing.assert_allclose(
            baseline.state.as_vector(), changed.state.as_vector()
        )

    def test_external_horizontal_fix_restores_position_reference(self):
        estimator = SixDOFStateEstimator()
        estimator.update(
            time_s=1.0,
            dt=1.0,
            sensor_packet=packet(
                dvl_velocity=np.full(3, np.nan),
                dvl_valid=False,
            ),
            sensor_health=health(dvl={
                "fault_type": "unavailable",
                "trust_level": "untrusted",
            }),
        )
        estimator.apply_horizontal_position_fix(np.array([4.0, -2.0]))
        estimate = estimator.update(
            time_s=1.1,
            dt=0.1,
            sensor_packet=packet(dvl_velocity=np.zeros(3)),
            sensor_health=health(),
        )

        np.testing.assert_allclose(
            estimate.state.position_ned[:2], np.array([4.0, -2.0])
        )
        self.assertEqual(estimate.quality, "nominal")
        self.assertEqual(
            estimate.horizontal_position_reference,
            "initial_dead_reckoning",
        )

    def test_simulator_uses_previous_causal_estimate_for_control(self):
        suite = SixDOFSensorSuite(
            depth_sensor=DepthSensor(noise_std=0.0, drift_std=0.0),
            imu_sensor=IMUSensor(0.0, 0.0, 0.0),
            dvl_sensor=DVLSensor(velocity_noise_std=0.0),
        )
        simulator = SixDOFSimulator(sensor_suite=suite)
        target = PoseTarget(np.array([2.0, 1.0, 0.5]))

        first = simulator.step(target, dt=0.1)
        second = simulator.step(target, dt=0.1)

        self.assertTrue(first["sensor_feedback_enabled"])
        self.assertEqual(
            first["controller_state_source"], "initial_pose_prior"
        )
        self.assertEqual(
            second["controller_state_source"],
            "sensor_estimate_previous_step",
        )
        np.testing.assert_allclose(
            second["controller_position_ned"],
            first["estimated_position_ned"],
        )
        np.testing.assert_allclose(
            second["controller_body_velocity"],
            first["estimated_body_velocity"],
        )

    def test_closed_loop_sensor_failures_recover_without_thruster_guess(self):
        scenarios = {
            "depth": (
                "degraded", "degraded_operation", "nominal"
            ),
            "dvl": (
                "degraded", "degraded_operation", "degraded"
            ),
            "imu": (
                "unsafe", "safe_hold_or_abort", "degraded"
            ),
        }
        target = PoseTarget(
            np.array([2.0, 1.0, 0.5]),
            np.array([0.0, 0.0, 0.3]),
        )

        for sensor, (quality, action, final_quality) in scenarios.items():
            with self.subTest(sensor=sensor):
                suite = SixDOFSensorSuite(
                    depth_sensor=DepthSensor(
                        noise_std=0.0, drift_std=0.0
                    ),
                    imu_sensor=IMUSensor(0.0, 0.0, 0.0),
                    dvl_sensor=DVLSensor(velocity_noise_std=0.0),
                    fault_injector=SensorFaultInjector((SensorFaultEvent(
                        sensor=sensor,
                        mode=SensorFaultMode.UNAVAILABLE,
                        start_time_s=0.5,
                        end_time_s=1.25,
                    ),)),
                )
                simulator = SixDOFSimulator(
                    sensor_suite=suite,
                    ftc_supervisor=FTCSafetySupervisor(),
                )
                logs = simulator.run(
                    1.75, 0.05, lambda _time, _state: target
                )

                self.assertIn(
                    quality,
                    {log["state_estimate_quality"] for log in logs},
                )
                self.assertIn(action, {log["ftc_action"] for log in logs})
                self.assertFalse(any(
                    log["ftc_targeted_thruster_name"] is not None
                    for log in logs
                ))
                self.assertEqual(
                    logs[-1]["state_estimate_quality"], final_quality
                )
                if sensor in ("dvl", "imu"):
                    self.assertEqual(
                        logs[-1]["ftc_action"], "degraded_operation"
                    )


if __name__ == "__main__":
    unittest.main()
