import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.sensor_fault_observer_benchmark_v3 import (
    OPTIONAL_OPERATOR_POSSIBLE_SCENARIOS,
    REQUIRED_LOG_ONLY_SCENARIOS,
    REQUIRED_OPERATOR_POSSIBLE_SCENARIOS,
    sensor_fault_observer_display_policy_metrics,
)
from evaluation.sensor_fault_stress_benchmark import (
    default_sensor_fault_stress_scenarios,
)


def row(scenario, *, operator=False, raw=True):
    return {
        "scenario": scenario,
        "observer_operator_possible_evidence_observed": operator,
        "observer_possible_evidence_observed": raw,
    }


class SensorFaultObserverBenchmarkV3Tests(unittest.TestCase):
    def test_v3_policy_partitions_all_ambiguous_scenarios(self):
        ambiguous = {
            item.name for item in default_sensor_fault_stress_scenarios()
            if item.category == "ambiguous"
        }
        groups = (
            set(REQUIRED_OPERATOR_POSSIBLE_SCENARIOS),
            set(OPTIONAL_OPERATOR_POSSIBLE_SCENARIOS),
            set(REQUIRED_LOG_ONLY_SCENARIOS),
        )

        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        self.assertEqual(ambiguous, set().union(*groups))

    def test_optional_drift_may_be_possible_or_log_only(self):
        rows = [
            row(name, operator=True)
            for name in REQUIRED_OPERATOR_POSSIBLE_SCENARIOS
        ]
        rows.extend((
            row("ambiguous_depth_drift", operator=True),
            row("ambiguous_dvl_drift", operator=False, raw=False),
        ))
        rows.extend(
            row(name, operator=False)
            for name in REQUIRED_LOG_ONLY_SCENARIOS
        )

        metrics = sensor_fault_observer_display_policy_metrics(rows)

        self.assertEqual(metrics["required_operator_possible_rate"], 1.0)
        self.assertEqual(metrics["optional_operator_possible_rate"], 0.5)
        self.assertEqual(metrics["required_log_only_overpromoted_missions"], 0)

    def test_weak_spike_operator_prompt_is_overpromotion(self):
        rows = [
            row(name, operator=True)
            for name in REQUIRED_OPERATOR_POSSIBLE_SCENARIOS
        ]
        rows.extend(
            row(name, operator=False)
            for name in OPTIONAL_OPERATOR_POSSIBLE_SCENARIOS
        )
        rows.extend(
            row(name, operator=(index == 0))
            for index, name in enumerate(REQUIRED_LOG_ONLY_SCENARIOS)
        )

        metrics = sensor_fault_observer_display_policy_metrics(rows)

        self.assertEqual(metrics["required_log_only_overpromoted_missions"], 1)


if __name__ == "__main__":
    unittest.main()
