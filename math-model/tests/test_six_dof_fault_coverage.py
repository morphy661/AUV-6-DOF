import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = PROJECT_ROOT / "examples"
SRC_ROOT = PROJECT_ROOT / "src"
for path in (EXAMPLES_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from actuators.six_dof_thruster_faults import SixDOFThrusterFaultMode
from demo_six_dof_fault_coverage import (
    FAULT_TIME,
    THRUSTER_NAMES,
    scenario_metadata,
    scenarios,
    validate_coverage,
)


class SixThrusterFaultCoverageTests(unittest.TestCase):
    def test_matrix_contains_normal_and_twelve_fault_cases(self):
        cases = scenarios()

        self.assertEqual(len(cases), 13)
        self.assertIsNone(cases["Normal"])
        expected_names = {"Normal"}
        for thruster_name in THRUSTER_NAMES:
            expected_names.add(f"{thruster_name} No Output")
            expected_names.add(f"{thruster_name} Thrust Loss")
        self.assertEqual(set(cases), expected_names)

    def test_each_thruster_has_both_fault_modes(self):
        cases = scenarios()

        for thruster_name in THRUSTER_NAMES:
            no_output = cases[f"{thruster_name} No Output"]
            thrust_loss = cases[f"{thruster_name} Thrust Loss"]
            self.assertEqual(no_output.thruster_name, thruster_name)
            self.assertEqual(no_output.mode, SixDOFThrusterFaultMode.NO_OUTPUT)
            self.assertEqual(thrust_loss.mode, SixDOFThrusterFaultMode.THRUST_LOSS)
            self.assertEqual(no_output.start_time, FAULT_TIME)
            self.assertEqual(thrust_loss.start_time, FAULT_TIME)
            self.assertAlmostEqual(thrust_loss.thrust_efficiency, 0.45)

    def test_scenario_metadata_is_machine_readable(self):
        self.assertEqual(scenario_metadata("Normal"), ("none", "normal"))
        self.assertEqual(
            scenario_metadata("V2 Thrust Loss"),
            ("V2", "thrust_loss"),
        )

    def test_coverage_validator_rejects_missing_cases(self):
        with self.assertRaisesRegex(AssertionError, "expected 13 scenarios"):
            validate_coverage([])


if __name__ == "__main__":
    unittest.main()
