import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from diagnosis.temporal_fault_decision import (
    TemporalDecisionConfig,
    TemporalFaultDecision,
    apply_temporal_decision_layer,
)


NORMAL = np.array([0.90, 0.05, 0.05])
NO_OUTPUT = np.array([0.05, 0.90, 0.05])
THRUST_LOSS = np.array([0.05, 0.05, 0.90])
H1 = np.array([0.80, 0.04, 0.04, 0.04, 0.04, 0.04])
H2 = np.array([0.04, 0.80, 0.04, 0.04, 0.04, 0.04])


class TemporalFaultDecisionTests(unittest.TestCase):
    def config(self, **overrides):
        values = {
            "enter_fault_probability": 0.70,
            "no_output_confirmation_s": 1.25,
            "thrust_loss_confirmation_s": 2.50,
            "exit_normal_probability": 0.70,
            "recovery_confirmation_s": 2.50,
            "probability_time_constant_s": 0.0,
            "location_probability_threshold": 0.25,
            "location_confirmation_s": 1.25,
        }
        values.update(overrides)
        return TemporalDecisionConfig(**values)

    def test_short_motion_anomaly_returns_to_normal_without_alarm(self):
        decision = TemporalFaultDecision(self.config())

        outputs = [
            decision.update(0.0, NORMAL, H1),
            decision.update(1.25, THRUST_LOSS, H1),
            decision.update(2.50, NORMAL, H1),
        ]

        self.assertEqual([result.mode for result in outputs], [0, 0, 0])
        self.assertEqual(outputs[1].state, "suspected")
        self.assertEqual(outputs[2].state, "normal")

    def test_persistent_thrust_loss_is_confirmed_after_required_time(self):
        decision = TemporalFaultDecision(self.config())

        outputs = [
            decision.update(time_s, THRUST_LOSS, H1)
            for time_s in (0.0, 1.25, 2.50)
        ]

        self.assertEqual([result.mode for result in outputs], [0, 0, 2])
        self.assertEqual(outputs[-1].location, 1)
        self.assertEqual(outputs[-1].state, "confirmed")

    def test_no_output_uses_shorter_confirmation_time(self):
        decision = TemporalFaultDecision(self.config())

        first = decision.update(0.0, NO_OUTPUT, H2)
        second = decision.update(1.25, NO_OUTPUT, H2)

        self.assertEqual(first.mode, 0)
        self.assertEqual((second.mode, second.location), (1, 2))

    def test_confirmed_fault_clears_only_after_sustained_recovery(self):
        decision = TemporalFaultDecision(self.config())
        for time_s in (0.0, 1.25, 2.50):
            decision.update(time_s, THRUST_LOSS, H1)

        recovering = decision.update(3.75, NORMAL, H1)
        still_recovering = decision.update(5.00, NORMAL, H1)
        cleared = decision.update(6.25, NORMAL, H1)

        self.assertEqual(recovering.state, "recovering")
        self.assertEqual(still_recovering.mode, 2)
        self.assertEqual((cleared.state, cleared.mode), ("normal", 0))

    def test_location_changes_require_persistent_evidence(self):
        decision = TemporalFaultDecision(self.config())
        for time_s in (0.0, 1.25, 2.50):
            decision.update(time_s, THRUST_LOSS, H1)

        first_h2 = decision.update(3.75, THRUST_LOSS, H2)
        second_h2 = decision.update(5.00, THRUST_LOSS, H2)

        self.assertEqual(first_h2.location, 1)
        self.assertEqual(second_h2.location, 2)

    def test_zero_location_confirmation_changes_immediately(self):
        decision = TemporalFaultDecision(self.config(
            location_confirmation_s=0.0
        ))
        for time_s in (0.0, 1.25, 2.50):
            decision.update(time_s, THRUST_LOSS, H1)

        changed = decision.update(3.75, THRUST_LOSS, H2)

        self.assertEqual(changed.location, 2)

    def test_each_mission_has_independent_state(self):
        dataset = {
            "mission_ids": np.array([1, 1, 1, 2, 2, 2]),
            "window_end_times": np.array([0.0, 1.25, 2.5] * 2),
        }
        predictions = {
            "mode_true": np.array([2, 2, 2, 0, 0, 0]),
            "location_true": np.array([1, 1, 1, 0, 0, 0]),
            "joint_true": np.array([7, 7, 7, 0, 0, 0]),
            "mode_probabilities": np.vstack([
                THRUST_LOSS, THRUST_LOSS, THRUST_LOSS,
                NORMAL, NORMAL, NORMAL,
            ]),
            "location_probabilities": np.vstack([H1] * 6),
        }

        result = apply_temporal_decision_layer(
            dataset,
            np.arange(6),
            predictions,
            self.config(),
        )

        np.testing.assert_array_equal(result["mode_pred"], [0, 0, 2, 0, 0, 0])
        np.testing.assert_array_equal(result["joint_pred"], [0, 0, 7, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
