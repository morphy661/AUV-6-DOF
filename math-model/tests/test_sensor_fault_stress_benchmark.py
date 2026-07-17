import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.sensor_fault_stress_benchmark import (
    SensorFaultStressScenario,
    default_sensor_fault_stress_scenarios,
    evaluate_sensor_fault_stress_mission,
    summarize_sensor_fault_stress_benchmark,
)
from sensors.sensor_faults import SensorFaultEvent, SensorFaultMode


def log_at(
    time_s,
    *,
    sensor=None,
    fault_type="normal",
    confirmed=False,
    action="normal_control",
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
            "confidence": 0.9,
        }
    return {
        "time": float(time_s),
        "sensor_health": health,
        "ftc_action": action,
        "ftc_targeted_thruster_name": None,
    }


class SensorFaultStressBenchmarkTests(unittest.TestCase):
    def test_default_matrix_has_all_three_categories(self):
        scenarios = default_sensor_fault_stress_scenarios()

        self.assertEqual(len(scenarios), 25)
        self.assertEqual(len({item.name for item in scenarios}), 25)
        self.assertEqual(
            {item.category for item in scenarios},
            {"normal", "strong_direct", "ambiguous"},
        )
        self.assertEqual(
            sum(item.category == "strong_direct" for item in scenarios),
            9,
        )

    def test_direct_fault_checks_exact_event_and_action(self):
        scenario = next(
            item for item in default_sensor_fault_stress_scenarios()
            if item.name == "strong_dvl_unavailable"
        )
        result = evaluate_sensor_fault_stress_mission(
            [
                log_at(4.9),
                log_at(
                    5.0,
                    sensor="dvl",
                    fault_type="unavailable",
                    confirmed=True,
                    action="degraded_operation",
                ),
                log_at(9.6),
            ],
            scenario,
        )

        self.assertTrue(result["exact_event_detected"])
        self.assertTrue(result["correct_ftc_action_observed"])
        self.assertEqual(result["conflicting_confirmed_event_count"], 0)
        self.assertFalse(result["post_recovery_protective_action_observed"])

    def test_ambiguous_suspicion_is_possible_evidence_not_certainty(self):
        scenario = next(
            item for item in default_sensor_fault_stress_scenarios()
            if item.name == "ambiguous_depth_bias"
        )
        result = evaluate_sensor_fault_stress_mission(
            [
                log_at(4.9),
                log_at(
                    5.2,
                    sensor="depth",
                    fault_type="stuck",
                    confirmed=False,
                    action="normal_control",
                ),
                log_at(11.6),
            ],
            scenario,
        )

        self.assertTrue(result["any_fault_evidence_observed"])
        self.assertTrue(result["possible_evidence_observed"])
        self.assertFalse(result["exact_event_detected"])
        self.assertEqual(result["conflicting_confirmed_event_count"], 0)

    def test_ambiguous_wrong_confirmed_type_is_counted(self):
        scenario = SensorFaultStressScenario(
            "bias",
            "ambiguous",
            "depth",
            "bias",
            (SensorFaultEvent(
                sensor="depth",
                mode=SensorFaultMode.BIAS,
                start_time_s=5.0,
                end_time_s=9.0,
                magnitude=0.2,
            ),),
        )
        result = evaluate_sensor_fault_stress_mission(
            [
                log_at(4.9),
                log_at(
                    5.1,
                    sensor="depth",
                    fault_type="spike",
                    confirmed=True,
                    action="log_only",
                ),
                log_at(9.6),
            ],
            scenario,
        )

        self.assertEqual(result["conflicting_confirmed_event_count"], 1)

    def test_summary_exposes_ambiguous_coverage_gap(self):
        normal = {
            "scenario": "normal",
            "category": "normal",
            "sensor": None,
            "truth_mode": "normal",
            "confirmed_event_count": 0,
            "exact_event_detected": False,
            "any_fault_evidence_observed": False,
            "possible_evidence_observed": False,
            "conflicting_confirmed_event_count": 0,
            "correct_ftc_action_observed": None,
            "protective_action_observed": False,
            "post_recovery_protective_action_observed": False,
            "wrong_thruster_target_count": 0,
        }
        direct = dict(normal)
        direct.update({
            "scenario": "direct",
            "category": "strong_direct",
            "sensor": "depth",
            "truth_mode": "stuck",
            "confirmed_event_count": 1,
            "exact_event_detected": True,
            "correct_ftc_action_observed": True,
        })
        ambiguous = dict(normal)
        ambiguous.update({
            "scenario": "ambiguous",
            "category": "ambiguous",
            "sensor": "depth",
            "truth_mode": "bias",
        })

        summary = summarize_sensor_fault_stress_benchmark(
            [normal, direct, ambiguous]
        )

        self.assertEqual(summary["strong_direct_event_recall"], 1.0)
        self.assertEqual(summary["strong_direct_event_precision"], 1.0)
        self.assertEqual(
            summary["ambiguous_fault_evidence_observation_rate"], 0.0
        )
        self.assertFalse(summary["all_acceptance_checks_passed"])


if __name__ == "__main__":
    unittest.main()
