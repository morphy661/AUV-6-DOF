import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.sensor_fault_benchmark import (
    SensorFaultBenchmarkScenario,
    default_sensor_fault_scenarios,
    evaluate_sensor_fault_mission,
    extract_confirmed_sensor_events,
    summarize_sensor_fault_benchmark,
)


def log_at(
    time_s,
    *,
    sensor=None,
    fault_type="normal",
    confirmed=False,
    ftc_action="normal_control",
    targeted_thruster=None,
    estimate_quality="nominal",
):
    health = {
        name: {
            "fault_type": "normal",
            "confirmed": False,
            "confidence": 1.0,
        }
        for name in ("depth", "imu", "dvl")
    }
    if sensor is not None:
        health[sensor] = {
            "fault_type": fault_type,
            "confirmed": confirmed,
            "confidence": 0.99,
        }
    return {
        "time": float(time_s),
        "sensor_health": health,
        "ftc_action": ftc_action,
        "ftc_targeted_thruster_name": targeted_thruster,
        "state_estimate_quality": estimate_quality,
        "true_position_error_ned": np.zeros(3),
        "estimated_position_ned": np.zeros(3),
        "position_ned": np.zeros(3),
    }


class SensorFaultBenchmarkTests(unittest.TestCase):
    def test_default_matrix_contains_normal_and_nine_faults(self):
        scenarios = default_sensor_fault_scenarios()

        self.assertEqual(len(scenarios), 10)
        self.assertEqual(sum(item.is_fault for item in scenarios), 9)
        self.assertEqual(len({item.name for item in scenarios}), 10)
        self.assertEqual(
            {
                (item.sensor, item.mode)
                for item in scenarios if item.is_fault
            },
            {
                (sensor, mode)
                for sensor in ("depth", "imu", "dvl")
                for mode in ("unavailable", "stuck", "spike")
            },
        )

    def test_contiguous_confirmed_samples_form_one_event(self):
        logs = [
            log_at(0.0),
            log_at(
                0.1,
                sensor="depth",
                fault_type="spike",
                confirmed=True,
                ftc_action="log_only",
            ),
            log_at(
                0.2,
                sensor="depth",
                fault_type="spike",
                confirmed=True,
                ftc_action="log_only",
            ),
            log_at(0.3),
        ]

        events = extract_confirmed_sensor_events(logs)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["sensor"], "depth")
        self.assertEqual(events[0]["fault_type"], "spike")
        self.assertEqual(events[0]["sample_count"], 2)

    def test_mission_evaluation_checks_detection_action_and_recovery(self):
        scenario = SensorFaultBenchmarkScenario(
            "dvl_unavailable", "dvl", "unavailable", 5.0, 4.0
        )
        logs = [
            log_at(4.9),
            log_at(
                5.1,
                sensor="dvl",
                fault_type="unavailable",
                confirmed=True,
                ftc_action="degraded_operation",
                estimate_quality="degraded",
            ),
            log_at(
                8.9,
                sensor="dvl",
                fault_type="unavailable",
                confirmed=True,
                ftc_action="degraded_operation",
                estimate_quality="degraded",
            ),
            log_at(9.6),
        ]

        result = evaluate_sensor_fault_mission(logs, scenario)

        self.assertTrue(result["exact_event_detected"])
        self.assertAlmostEqual(result["detection_delay_s"], 0.1)
        self.assertTrue(result["correct_ftc_action_observed"])
        self.assertTrue(result["sensor_health_recovered"])
        self.assertTrue(result["estimate_integrity_restored"])
        self.assertTrue(result["absolute_trajectory_recovered"])
        self.assertEqual(result["spurious_event_count"], 0)

    def test_summary_separates_event_precision_from_recall(self):
        normal = evaluate_sensor_fault_mission(
            [log_at(0.0), log_at(10.0)],
            SensorFaultBenchmarkScenario(
                "normal", None, "normal", 5.0, 4.0
            ),
        )
        fault = evaluate_sensor_fault_mission(
            [
                log_at(4.9),
                log_at(
                    5.0,
                    sensor="imu",
                    fault_type="unavailable",
                    confirmed=True,
                    ftc_action="safe_hold_or_abort",
                    estimate_quality="unsafe",
                ),
                log_at(9.6),
            ],
            SensorFaultBenchmarkScenario(
                "imu_unavailable", "imu", "unavailable", 5.0, 4.0
            ),
        )

        summary = summarize_sensor_fault_benchmark([normal, fault])

        self.assertEqual(summary["event_recall"], 1.0)
        self.assertEqual(summary["event_precision"], 1.0)
        self.assertEqual(summary["ftc_action_match_rate"], 1.0)
        self.assertEqual(summary["sensor_health_recovery_rate"], 1.0)
        self.assertEqual(summary["wrong_thruster_target_count"], 0)
        self.assertTrue(summary["all_acceptance_checks_passed"])


if __name__ == "__main__":
    unittest.main()
