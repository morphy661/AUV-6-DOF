import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.transient_recovery import (
    boundary_transient_scenarios,
    TransientDisturbanceScenario,
    default_transient_scenarios,
    dvl_dropout_boundary_scenarios,
    summarize_transient_recovery,
)


class TransientDisturbanceTests(unittest.TestCase):
    def test_half_sine_pulse_is_finite_and_time_limited(self):
        scenario = TransientDisturbanceScenario(
            "test", 2.0, 1.0, np.array([4.0, 0, 0, 0, 0, 0])
        )

        np.testing.assert_allclose(scenario.wrench_at(1.9), np.zeros(6))
        np.testing.assert_allclose(scenario.wrench_at(2.0), np.zeros(6))
        self.assertAlmostEqual(scenario.wrench_at(2.5)[0], 4.0)
        np.testing.assert_allclose(scenario.wrench_at(3.0), np.zeros(6), atol=1e-12)
        np.testing.assert_allclose(scenario.wrench_at(3.1), np.zeros(6))

    def test_default_scenarios_stay_below_two_second_ftc_confirmation(self):
        scenarios = default_transient_scenarios()

        self.assertEqual(len({scenario.name for scenario in scenarios}), 4)
        self.assertTrue(all(scenario.duration_s < 2.0 for scenario in scenarios))

    def test_boundary_matrix_has_nine_disturbances_and_three_dropouts(self):
        disturbances = boundary_transient_scenarios()
        dropouts = dvl_dropout_boundary_scenarios()

        self.assertEqual(len(disturbances), 9)
        self.assertEqual(
            sorted({scenario.duration_s for scenario in disturbances}),
            [1.0, 2.0, 4.0],
        )
        self.assertEqual(len(dropouts), 3)
        self.assertEqual(dropouts[-1].dropout_window, (15.0, 19.0))

    def test_recovery_summary_requires_response_and_return(self):
        scenario = TransientDisturbanceScenario(
            "test", 5.0, 1.0, np.array([1.0, 0, 0, 0, 0, 0])
        )
        logs = []
        for time_s in np.arange(0.0, 16.0, 0.5):
            position_error = 0.0
            if 5.0 <= time_s <= 7.0:
                position_error = 0.20 * np.sin(
                    np.pi * (time_s - 5.0) / 2.0
                )
            logs.append({
                "time": time_s,
                "position_error_ned": np.array([position_error, 0.0, 0.0]),
                "attitude_error_body": np.zeros(3),
            })

        summary = summarize_transient_recovery(logs, scenario)

        self.assertTrue(summary["response_observed"])
        self.assertTrue(summary["recovered"])
        self.assertAlmostEqual(summary["final_position_error_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
