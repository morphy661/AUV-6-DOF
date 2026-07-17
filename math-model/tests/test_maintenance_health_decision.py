import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from diagnosis.maintenance_health_decision import (
    MaintenanceHealthDecision,
    apply_maintenance_decision_layer,
    maintenance_event_metrics,
)
from diagnosis.temporal_fault_decision import TemporalDecisionConfig


NORMAL = np.array([0.90, 0.05, 0.05])
NO_OUTPUT = np.array([0.05, 0.90, 0.05])
THRUST_LOSS = np.array([0.05, 0.05, 0.90])
H2_H3 = np.array([0.03, 0.55, 0.31, 0.03, 0.04, 0.04])
UNIFORM_LOCATION = np.ones(6) / 6.0


class MaintenanceHealthDecisionTests(unittest.TestCase):
    @staticmethod
    def temporal_config():
        return TemporalDecisionConfig(
            enter_fault_probability=0.70,
            no_output_confirmation_s=1.0,
            thrust_loss_confirmation_s=2.0,
            exit_normal_probability=0.70,
            recovery_confirmation_s=2.0,
            probability_time_constant_s=0.0,
            location_probability_threshold=0.0,
            location_confirmation_s=0.0,
        )

    def test_short_anomaly_is_recorded_without_ftc(self):
        decision = MaintenanceHealthDecision(self.temporal_config())

        anomaly = decision.update(0.0, THRUST_LOSS, H2_H3)
        recovered = decision.update(1.0, NORMAL, H2_H3)

        self.assertEqual(anomaly.health_state, "transient_observation")
        self.assertTrue(anomaly.record_event)
        self.assertFalse(anomaly.requires_ftc)
        self.assertEqual(recovered.health_state, "normal")

    def test_persistent_thrust_loss_becomes_maintenance_advice(self):
        decision = MaintenanceHealthDecision(self.temporal_config())

        results = [
            decision.update(time_s, THRUST_LOSS, H2_H3)
            for time_s in (0.0, 1.0, 2.0)
        ]

        self.assertEqual(results[-1].health_state, "persistent_degradation")
        self.assertEqual(results[-1].confirmed_mode, 2)
        self.assertFalse(results[-1].requires_ftc)
        self.assertEqual(
            [candidate.name for candidate in results[-1].candidates],
            ["H2", "H3"],
        )

    def test_no_output_is_a_critical_fault(self):
        decision = MaintenanceHealthDecision(self.temporal_config())

        decision.update(0.0, NO_OUTPUT, H2_H3)
        result = decision.update(1.0, NO_OUTPUT, H2_H3)

        self.assertEqual(result.health_state, "critical_fault")
        self.assertTrue(result.requires_ftc)

    def test_operating_stress_escalates_compensated_thrust_loss(self):
        decision = MaintenanceHealthDecision(self.temporal_config())
        decision.update(0.0, THRUST_LOSS, H2_H3)
        decision.update(1.0, THRUST_LOSS, H2_H3)

        result = decision.update(
            2.0,
            THRUST_LOSS,
            H2_H3,
            control_saturation_ratio=0.95,
        )

        self.assertEqual(result.health_state, "critical_fault")
        self.assertTrue(result.requires_ftc)

    def test_uniform_location_evidence_does_not_force_a_group(self):
        decision = MaintenanceHealthDecision(self.temporal_config())

        result = decision.update(0.0, THRUST_LOSS, UNIFORM_LOCATION)

        self.assertEqual(result.suspected_group, "uncertain")
        self.assertEqual(result.location_confidence, "low")

    def test_batch_output_uses_event_and_top2_metrics(self):
        dataset = {
            "mission_ids": np.array([7, 7, 7, 7]),
            "window_end_times": np.array([0.0, 1.0, 2.0, 3.0]),
        }
        predictions = {
            "mode_true": np.array([0, 0, 2, 2]),
            "location_true": np.array([0, 0, 2, 2]),
            "joint_true": np.array([0, 0, 8, 8]),
            "mode_probabilities": np.vstack([THRUST_LOSS] * 4),
            "location_probabilities": np.vstack([H2_H3] * 4),
        }

        decisions = apply_maintenance_decision_layer(
            dataset,
            np.arange(4),
            predictions,
            self.temporal_config(),
        )
        metrics = maintenance_event_metrics(
            dataset, np.arange(4), decisions
        )

        self.assertEqual(len(decisions["maintenance_events"]), 1)
        self.assertEqual(
            decisions["maintenance_events"][0]["category"],
            "persistent_degradation",
        )
        self.assertEqual(metrics["event_recall"], 1.0)
        self.assertEqual(metrics["event_precision"], 1.0)
        self.assertEqual(metrics["top2_location_hit_rate"], 1.0)
        self.assertEqual(metrics["normal_window_advisory_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
