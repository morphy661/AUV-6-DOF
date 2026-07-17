import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from diagnosis.sensor_health_monitor import SensorHealthMonitor
from ftc.safety_supervisor import (
    FTCAction,
    FTCSafetySupervisor,
    build_rule_based_ftc_evidence,
)


def packet(depth=2.0, orientation=None, gyro=None, acceleration=None, dvl=None):
    orientation = np.zeros(3) if orientation is None else np.asarray(orientation)
    gyro = np.zeros(3) if gyro is None else np.asarray(gyro)
    acceleration = (
        np.zeros(3) if acceleration is None else np.asarray(acceleration)
    )
    dvl = np.zeros(3) if dvl is None else np.asarray(dvl)
    return {
        "depth": float(depth),
        "depth_valid": np.isfinite(depth),
        "imu": {
            "valid": True,
            "orientation": orientation.astype(float),
            "angular_velocity": gyro.astype(float),
            "linear_acceleration": acceleration.astype(float),
        },
        "dvl": {
            "valid": True,
            "velocity": dvl.astype(float),
        },
    }


class SensorHealthMonitorTests(unittest.TestCase):
    @staticmethod
    def observable_log(summary):
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
            "sensor_health_summary": summary,
        }

    def test_unavailable_sensor_is_confirmed_from_observable_packet(self):
        monitor = SensorHealthMonitor()
        failed = packet()
        failed["dvl"] = {
            "valid": False,
            "velocity": np.full(3, np.nan),
        }

        result = monitor.update(0.0, failed)
        summary = monitor.summarize(result)

        self.assertTrue(result["dvl"].confirmed)
        self.assertEqual(result["dvl"].fault_type, "unavailable")
        self.assertEqual(result["dvl"].trust_level, "untrusted")
        self.assertEqual(summary["ftc_recommendation"], "degraded_navigation")

    def test_first_valid_sample_after_unavailability_rebaselines(self):
        monitor = SensorHealthMonitor()
        monitor.update(0.0, packet(depth=2.0))
        monitor.update(0.1, packet(depth=np.nan))

        recovered = monitor.update(0.2, packet(depth=4.0))

        self.assertEqual(recovered["depth"].fault_type, "normal")
        self.assertNotEqual(recovered["depth"].fault_type, "spike")

    def test_stationary_vehicle_does_not_create_false_stuck_fault(self):
        monitor = SensorHealthMonitor()
        current = packet()

        for time_s in (0.0, 0.25, 0.50, 0.75, 1.00):
            result = monitor.update(
                time_s,
                current,
                motion_context={
                    "desired_velocity_ned": np.zeros(3),
                    "desired_angular_velocity_body": np.zeros(3),
                },
            )

        self.assertTrue(all(
            value.fault_type == "normal" for value in result.values()
        ))

    def test_unchanged_sensors_confirm_stuck_only_under_expected_motion(self):
        monitor = SensorHealthMonitor()
        current = packet(acceleration=[0.5, 0.0, 0.0])
        context = {
            "desired_velocity_ned": np.array([0.0, 0.0, 0.2]),
            "desired_angular_velocity_body": np.array([0.1, 0.0, 0.0]),
        }

        for time_s in (0.0, 0.25, 0.50, 0.75, 1.00):
            result = monitor.update(time_s, current, context)

        self.assertEqual(result["depth"].fault_type, "stuck")
        self.assertTrue(result["depth"].confirmed)
        self.assertEqual(result["imu"].fault_type, "stuck")
        self.assertTrue(result["imu"].confirmed)
        self.assertEqual(result["dvl"].fault_type, "stuck")
        self.assertTrue(result["dvl"].confirmed)

    def test_confirmed_stuck_latches_until_reading_changes(self):
        monitor = SensorHealthMonitor()
        current = packet(depth=2.0, acceleration=[0.5, 0.0, 0.0])
        moving = {
            "desired_velocity_ned": np.array([0.0, 0.0, 0.2]),
            "desired_angular_velocity_body": np.zeros(3),
        }
        for time_s in (0.0, 0.25, 0.50, 0.75, 1.00):
            result = monitor.update(time_s, current, moving)
        self.assertTrue(result["depth"].confirmed)

        latched = monitor.update(
            1.25,
            packet(depth=2.0),
            {
                "desired_velocity_ned": np.zeros(3),
                "desired_angular_velocity_body": np.zeros(3),
            },
        )
        recovered = monitor.update(1.50, packet(depth=3.0))

        self.assertEqual(latched["depth"].fault_type, "stuck")
        self.assertTrue(latched["depth"].confirmed)
        self.assertEqual(recovered["depth"].fault_type, "normal")

    def test_large_depth_jump_is_a_sample_rejection_not_thruster_isolation(self):
        monitor = SensorHealthMonitor()
        monitor.update(0.0, packet(depth=2.0))
        result = monitor.update(0.1, packet(depth=4.0))
        summary = monitor.summarize(result)
        evidence = build_rule_based_ftc_evidence(
            self.observable_log(summary)
        )
        decision = FTCSafetySupervisor().update(evidence)

        self.assertEqual(result["depth"].fault_type, "spike")
        self.assertEqual(summary["ftc_recommendation"], "reject_current_sample")
        self.assertEqual(decision.action, FTCAction.LOG_ONLY)
        self.assertEqual(decision.isolated_thruster_indices, ())

    def test_imu_unavailability_requests_safe_hold_without_location_guess(self):
        monitor = SensorHealthMonitor()
        failed = packet()
        failed["imu"]["valid"] = False
        failed["imu"]["orientation"][:] = np.nan
        failed["imu"]["angular_velocity"][:] = np.nan
        failed["imu"]["linear_acceleration"][:] = np.nan
        result = monitor.update(0.0, failed)
        summary = monitor.summarize(result)
        evidence = build_rule_based_ftc_evidence(
            self.observable_log(summary)
        )
        decision = FTCSafetySupervisor().update(evidence)

        self.assertEqual(summary["ftc_recommendation"], "safe_hold_or_abort")
        self.assertEqual(decision.action, FTCAction.SAFE_HOLD_OR_ABORT)
        self.assertEqual(decision.isolated_thruster_indices, ())


if __name__ == "__main__":
    unittest.main()
