import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.sensor_fault_observer_benchmark import (
    LOG_ONLY_POLICY_SCENARIOS,
    REQUIRED_OPERATOR_POSSIBLE_SCENARIOS,
    evaluate_sensor_fault_observer_mission,
    summarize_sensor_fault_observer_benchmark,
)
from evaluation.sensor_fault_stress_benchmark import (
    default_sensor_fault_stress_scenarios,
)


def log_at(time_s, sensor=None, hypothesis="normal"):
    observations = {
        name: {
            "state": "normal",
            "hypothesis": "normal",
            "confirmed": False,
        }
        for name in ("depth", "imu", "dvl")
    }
    if sensor is not None:
        observations[sensor] = {
            "state": "possible_fault",
            "hypothesis": hypothesis,
            "confirmed": False,
            "display_level": "possible",
        }
    return {
        "time": float(time_s),
        "sensor_health": {
            name: {
                "fault_type": "normal",
                "confirmed": False,
                "confidence": 1.0,
            }
            for name in ("depth", "imu", "dvl")
        },
        "sensor_fault_observations": observations,
        "sensor_fault_observation_summary": {
            "protective_action_required": False,
        },
        "ftc_action": "normal_control",
        "ftc_targeted_thruster_name": None,
    }


class SensorFaultObserverBenchmarkTests(unittest.TestCase):
    def test_display_policy_partitions_all_ambiguous_scenarios(self):
        ambiguous_names = {
            item.name for item in default_sensor_fault_stress_scenarios()
            if item.category == "ambiguous"
        }

        self.assertFalse(
            set(REQUIRED_OPERATOR_POSSIBLE_SCENARIOS)
            & set(LOG_ONLY_POLICY_SCENARIOS)
        )
        self.assertEqual(
            ambiguous_names,
            set(REQUIRED_OPERATOR_POSSIBLE_SCENARIOS)
            | set(LOG_ONLY_POLICY_SCENARIOS),
        )

    def test_ambiguous_possible_evidence_sets_possible_display_tier(self):
        scenario = next(
            item for item in default_sensor_fault_stress_scenarios()
            if item.name == "ambiguous_depth_bias"
        )
        row = evaluate_sensor_fault_observer_mission(
            [
                log_at(4.9),
                log_at(5.5, "depth", "possible_bias_or_drift"),
                log_at(11.6),
            ],
            scenario,
        )

        self.assertTrue(row["observer_possible_evidence_observed"])
        self.assertTrue(row["observer_operator_possible_evidence_observed"])
        self.assertEqual(row["display_tier"], "possible")
        self.assertEqual(row["observer_confirmed_count"], 0)
        self.assertEqual(row["observer_protective_count"], 0)

    def test_summary_rejects_missing_ambiguous_coverage(self):
        scenarios = default_sensor_fault_stress_scenarios()
        rows = []
        for scenario in scenarios:
            logs = [log_at(0.0), log_at(18.0)]
            rows.append(evaluate_sensor_fault_observer_mission(logs, scenario))

        summary = summarize_sensor_fault_observer_benchmark(rows)

        self.assertEqual(summary["ambiguous_possible_evidence_rate"], 0.0)
        self.assertFalse(summary["all_acceptance_checks_passed"])


if __name__ == "__main__":
    unittest.main()
