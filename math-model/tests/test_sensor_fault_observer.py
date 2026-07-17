import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from diagnosis.sensor_fault_observer import SensorFaultObserver
from diagnosis.sensor_health_monitor import SensorHealthMonitor


def packet(
    *,
    depth=0.0,
    orientation=(0.0, 0.0, 0.0),
    gyro=(0.0, 0.0, 0.0),
    acceleration=(0.0, 0.0, 0.0),
    dvl=(0.0, 0.0, 0.0),
    depth_valid=True,
    imu_valid=True,
    dvl_valid=True,
):
    depth_value = float(depth) if depth_valid else float("nan")
    orientation = np.asarray(orientation, dtype=float)
    gyro = np.asarray(gyro, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    dvl = np.asarray(dvl, dtype=float)
    if not imu_valid:
        orientation = np.full(3, np.nan)
        gyro = np.full(3, np.nan)
        acceleration = np.full(3, np.nan)
    if not dvl_valid:
        dvl = np.full(3, np.nan)
    return {
        "depth": depth_value,
        "depth_valid": depth_valid,
        "imu": {
            "valid": imu_valid,
            "orientation": orientation,
            "angular_velocity": gyro,
            "linear_acceleration": acceleration,
        },
        "dvl": {
            "valid": dvl_valid,
            "velocity": dvl,
        },
    }


class SensorFaultObserverTests(unittest.TestCase):
    def test_weak_jump_is_possible_and_never_protective(self):
        observer = SensorFaultObserver()
        observer.update(0.0, packet(depth=1.0))

        results = observer.update(0.1, packet(depth=1.3))
        result = results["depth"]
        summary = observer.summarize(results)

        self.assertEqual(result.hypothesis, "possible_weak_spike_or_bias")
        self.assertEqual(result.display_level, "background")
        self.assertFalse(result.confirmed)
        self.assertFalse(result.protective_action_required)
        self.assertEqual(summary["ftc_recommendation"], "none")
        self.assertEqual(summary["operator_message_level"], "background")
        self.assertFalse(summary["confirmed_hardware_fault"])

    def test_slow_depth_drift_creates_long_window_observation(self):
        observer = SensorFaultObserver()
        result = None
        for index in range(13):
            time_s = 0.5 * index
            results = observer.update(
                time_s,
                packet(depth=0.04 * time_s),
            )
            result = results["depth"]

        self.assertEqual(result.state, "possible_fault")
        self.assertEqual(result.hypothesis, "possible_bias_or_drift")
        self.assertIn("drift", result.candidates)
        self.assertFalse(result.confirmed)

    def test_partial_imu_channel_stuck_is_possible(self):
        observer = SensorFaultObserver()
        context = {
            "desired_velocity_ned": np.zeros(3),
            "desired_angular_velocity_body": np.array([0.0, 0.0, 1.0]),
        }
        result = None
        for index in range(5):
            time_s = 0.25 * index
            results = observer.update(
                time_s,
                packet(orientation=(0.0, 0.0, 0.0), gyro=(0.0, 0.0, 1.0)),
                motion_context=context,
            )
            result = results["imu"]

        self.assertEqual(result.hypothesis, "possible_partial_stuck")
        self.assertEqual(result.affected_channels, (2,))
        self.assertFalse(result.confirmed)

    def test_partial_stuck_recovery_requests_one_sample_rebaseline(self):
        observer = SensorFaultObserver()
        context = {
            "desired_velocity_ned": np.zeros(3),
            "desired_angular_velocity_body": np.array([0.0, 0.0, 1.0]),
        }
        for index in range(5):
            observer.update(
                0.25 * index,
                packet(orientation=(0.0, 0.0, 0.0), gyro=(0.0, 0.0, 1.0)),
                motion_context=context,
            )

        observer.update(
            1.1,
            packet(orientation=(0.0, 0.0, 0.5), gyro=(0.0, 0.0, 1.0)),
            motion_context=context,
        )

        self.assertEqual(observer.rebaseline_sensors, ("imu",))

    def test_health_monitor_honors_observer_rebaseline(self):
        monitor = SensorHealthMonitor()
        monitor.update(0.0, packet(orientation=(0.0, 0.0, 0.0)))

        results = monitor.update(
            0.1,
            packet(orientation=(0.0, 0.0, 0.5)),
            rebaseline_sensors=("imu",),
        )

        self.assertEqual(results["imu"].fault_type, "normal")
        self.assertFalse(results["imu"].confirmed)

    def test_repeated_unavailability_is_hardware_possibility(self):
        observer = SensorFaultObserver()
        observer.update(0.0, packet())
        observer.update(1.0, packet(dvl_valid=False))
        observer.update(1.2, packet())

        results = observer.update(2.0, packet(dvl_valid=False))
        result = results["dvl"]

        self.assertEqual(
            result.hypothesis,
            "possible_intermittent_unavailability",
        )
        self.assertFalse(result.confirmed)
        self.assertEqual(result.recommended_action, "record_and_observe")
        self.assertFalse(result.protective_action_required)


if __name__ == "__main__":
    unittest.main()
