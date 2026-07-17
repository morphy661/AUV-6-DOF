import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from diagnosis.maintenance_ticket_policy import (
    MaintenanceTicketConfig,
    apply_maintenance_ticket_policy,
    maintenance_ticket_metrics,
)


class MaintenanceTicketPolicyTests(unittest.TestCase):
    @staticmethod
    def config(**overrides):
        values = {
            "minimum_excitation_ratio": 0.20,
            "minimum_thrust_loss_motion_evidence": 0.30,
            "minimum_no_output_local_anomaly": 0.70,
            "ticket_confirmation_s": 0.0,
            "thrust_loss_pending_confirmation_s": 0.0,
            "thrust_loss_recovery_cancel_s": 0.0,
            "thrust_loss_recurrence_window_s": 30.0,
            "thrust_loss_recurrence_count": 2,
            "ticket_recovery_s": 0.0,
            "merge_gap_s": 5.0,
        }
        values.update(overrides)
        return MaintenanceTicketConfig(**values)

    @staticmethod
    def dataset(count, mode_true=None, location_true=None):
        return {
            "mission_ids": np.zeros(count, dtype=np.int64),
            "window_end_times": np.arange(count, dtype=float),
            "y_mode": np.asarray(
                mode_true if mode_true is not None else np.zeros(count)
            ),
            "y_location": np.asarray(
                location_true
                if location_true is not None
                else np.zeros(count)
            ),
        }

    @staticmethod
    def decisions(mode_pred, health_level, mode_true=None, location_true=None):
        count = len(mode_pred)
        mode_true = np.asarray(
            mode_true if mode_true is not None else mode_pred,
            dtype=np.int64,
        )
        location_true = np.asarray(
            location_true
            if location_true is not None
            else np.where(mode_true != 0, 1, 0),
            dtype=np.int64,
        )
        location_probabilities = np.tile(
            np.array([0.70, 0.20, 0.03, 0.03, 0.02, 0.02]),
            (count, 1),
        )
        return {
            "mode_true": mode_true,
            "location_true": location_true,
            "joint_true": np.zeros(count, dtype=np.int64),
            "mode_pred": np.asarray(mode_pred, dtype=np.int64),
            "health_level_pred": np.asarray(health_level, dtype=np.int64),
            "candidate_indices": np.tile(np.array([1, 2]), (count, 1)),
            "smoothed_location_probabilities": location_probabilities,
            "maintenance_events": [{"raw": "retained"}],
        }

    @staticmethod
    def evidence(count, excitation=0.0, local=0.0, motion=0.0):
        result = {
            "excitation_ratios": np.zeros((count, 6)),
            "local_anomaly_scores": np.zeros((count, 6)),
            "motion_loss_evidence": np.zeros((count, 6)),
            "saturation_fraction": np.zeros((count, 6)),
        }
        result["excitation_ratios"][:, 0] = excitation
        result["local_anomaly_scores"][:, 0] = local
        result["motion_loss_evidence"][:, 0] = motion
        return result

    def test_low_excitation_thrust_loss_stays_in_raw_log(self):
        decisions = self.decisions([2, 2], [2, 2])
        result = apply_maintenance_ticket_policy(
            self.dataset(2),
            np.arange(2),
            decisions,
            self.config(),
            ticket_evidence=self.evidence(
                2, excitation=0.05, motion=0.80
            ),
        )

        self.assertFalse(np.any(result["maintenance_ticket_active"]))
        self.assertEqual(result["maintenance_tickets"], [])
        self.assertEqual(result["maintenance_events"], [{"raw": "retained"}])

    def test_motion_evidence_is_required_for_thrust_loss_ticket(self):
        decisions = self.decisions([2, 2], [2, 2])
        result = apply_maintenance_ticket_policy(
            self.dataset(2),
            np.arange(2),
            decisions,
            self.config(),
            ticket_evidence=self.evidence(
                2, excitation=0.60, motion=0.10
            ),
        )

        self.assertFalse(np.any(result["maintenance_ticket_active"]))

    def test_dual_evidence_creates_formal_ticket(self):
        true_mode = [0, 0, 2, 2]
        true_location = [0, 0, 1, 1]
        decisions = self.decisions(
            [0, 0, 2, 2],
            [0, 0, 2, 2],
            true_mode,
            true_location,
        )
        evidence = self.evidence(4, excitation=0.60, motion=0.80)
        result = apply_maintenance_ticket_policy(
            self.dataset(4, true_mode, true_location),
            np.arange(4),
            decisions,
            self.config(),
            ticket_evidence=evidence,
        )
        metrics = maintenance_ticket_metrics(
            self.dataset(4, true_mode, true_location),
            np.arange(4),
            result,
        )

        self.assertEqual(len(result["maintenance_tickets"]), 1)
        self.assertEqual(
            result["maintenance_tickets"][0]["fault_mode"],
            "thrust_loss",
        )
        self.assertEqual(metrics["ticket_event_recall"], 1.0)
        self.assertEqual(metrics["ticket_event_precision"], 1.0)

    def test_no_output_requires_local_telemetry_evidence(self):
        decisions = self.decisions([1, 1], [3, 3])
        weak = apply_maintenance_ticket_policy(
            self.dataset(2),
            np.arange(2),
            decisions,
            self.config(),
            ticket_evidence=self.evidence(
                2, excitation=0.60, local=0.20
            ),
        )
        strong = apply_maintenance_ticket_policy(
            self.dataset(2),
            np.arange(2),
            decisions,
            self.config(),
            ticket_evidence=self.evidence(
                2, excitation=0.60, local=0.95
            ),
        )

        self.assertEqual(weak["maintenance_tickets"], [])
        self.assertEqual(len(strong["maintenance_tickets"]), 1)

    def test_short_gap_ticket_segments_are_merged(self):
        decisions = self.decisions(
            [2, 2, 0, 2, 2],
            [2, 2, 0, 2, 2],
        )
        evidence = self.evidence(5, excitation=0.60, motion=0.80)
        result = apply_maintenance_ticket_policy(
            self.dataset(5),
            np.arange(5),
            decisions,
            self.config(merge_gap_s=5.0),
            ticket_evidence=evidence,
        )

        self.assertEqual(len(result["maintenance_tickets"]), 1)
        self.assertEqual(
            result["maintenance_tickets"][0]["merged_segment_count"],
            2,
        )

    def test_wrong_mode_ticket_is_separate_from_maintenance_precision(self):
        true_mode = [0, 0, 2, 2]
        true_location = [0, 0, 1, 1]
        decisions = self.decisions(
            [0, 0, 1, 1],
            [0, 0, 3, 3],
            true_mode,
            true_location,
        )
        result = apply_maintenance_ticket_policy(
            self.dataset(4, true_mode, true_location),
            np.arange(4),
            decisions,
            self.config(),
            ticket_evidence=self.evidence(
                4, excitation=0.60, local=0.95
            ),
        )
        metrics = maintenance_ticket_metrics(
            self.dataset(4, true_mode, true_location),
            np.arange(4),
            result,
        )

        self.assertEqual(metrics["ticket_event_recall"], 1.0)
        self.assertEqual(metrics["ticket_event_precision"], 1.0)
        self.assertEqual(metrics["false_maintenance_tickets"], 0)
        self.assertEqual(metrics["ticket_fault_mode_judgement_rate"], 0.0)

    def test_short_thrust_loss_recovery_keeps_observation_without_ticket(self):
        decisions = self.decisions(
            [2, 2, 2, 0, 0, 0, 0, 0],
            [2, 2, 2, 0, 0, 0, 0, 0],
        )
        result = apply_maintenance_ticket_policy(
            self.dataset(8),
            np.arange(8),
            decisions,
            self.config(
                thrust_loss_pending_confirmation_s=8.0,
                thrust_loss_recovery_cancel_s=3.0,
            ),
            ticket_evidence=self.evidence(
                8, excitation=0.60, motion=0.80
            ),
        )

        self.assertEqual(result["maintenance_tickets"], [])
        self.assertEqual(len(result["maintenance_pending_observations"]), 1)
        self.assertEqual(
            result["maintenance_pending_observations"][0]["status"],
            "recovered_before_confirmation",
        )
        self.assertFalse(np.any(result["maintenance_ticket_active"]))

    def test_persistent_thrust_loss_creates_ticket_after_eight_seconds(self):
        decisions = self.decisions([2] * 10, [2] * 10)
        result = apply_maintenance_ticket_policy(
            self.dataset(10),
            np.arange(10),
            decisions,
            self.config(thrust_loss_pending_confirmation_s=8.0),
            ticket_evidence=self.evidence(
                10, excitation=0.60, motion=0.80
            ),
        )

        self.assertEqual(len(result["maintenance_tickets"]), 1)
        self.assertEqual(
            result["maintenance_tickets"][0]["trigger"],
            "persistent_thrust_loss",
        )
        self.assertEqual(
            result["maintenance_tickets"][0]["start_time_s"], 8.0
        )

    def test_two_brief_thrust_loss_episodes_remain_observations(self):
        decisions = self.decisions(
            [2, 2, 0, 0, 0, 2, 2],
            [2, 2, 0, 0, 0, 2, 2],
        )
        result = apply_maintenance_ticket_policy(
            self.dataset(7),
            np.arange(7),
            decisions,
            self.config(
                thrust_loss_pending_confirmation_s=8.0,
                thrust_loss_recovery_cancel_s=2.0,
            ),
            ticket_evidence=self.evidence(
                7, excitation=0.60, motion=0.80
            ),
        )

        self.assertEqual(result["maintenance_tickets"], [])
        self.assertEqual(len(result["maintenance_pending_observations"]), 2)

    def test_recurrent_thrust_loss_becomes_advisory_not_ticket(self):
        decisions = self.decisions(
            [2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0],
            [2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0],
        )
        result = apply_maintenance_ticket_policy(
            self.dataset(16),
            np.arange(16),
            decisions,
            self.config(
                thrust_loss_pending_confirmation_s=8.0,
                thrust_loss_recovery_cancel_s=2.0,
            ),
            ticket_evidence=self.evidence(
                16, excitation=0.60, motion=0.80
            ),
        )

        self.assertEqual(result["maintenance_tickets"], [])
        self.assertEqual(len(result["maintenance_advisories"]), 1)
        self.assertEqual(
            result["maintenance_advisories"][0]["status"],
            "intermittent_thrust_loss_advisory",
        )
        self.assertEqual(
            result["maintenance_advisories"][0][
                "cumulative_qualified_duration_s"
            ],
            8.0,
        )

    def test_context_change_splits_pending_thrust_loss(self):
        decisions = self.decisions([2] * 10, [2] * 10)
        dataset = self.dataset(10)
        dataset["guidance_context_ids"] = np.array([0] * 5 + [1] * 5)
        dataset["guidance_context_stable"] = np.ones(10, dtype=bool)
        result = apply_maintenance_ticket_policy(
            dataset,
            np.arange(10),
            decisions,
            self.config(thrust_loss_pending_confirmation_s=8.0),
            ticket_evidence=self.evidence(
                10, excitation=0.60, motion=0.80
            ),
        )

        self.assertEqual(result["maintenance_tickets"], [])
        self.assertEqual(result["maintenance_advisories"], [])
        self.assertEqual(
            result["maintenance_pending_observations"][0]["status"],
            "context_transition_observation",
        )
        self.assertEqual(
            result["maintenance_pending_observations"][0][
                "qualified_duration_s"
            ],
            4.0,
        )

    def test_unstable_context_windows_do_not_confirm_thrust_loss(self):
        decisions = self.decisions([2] * 14, [2] * 14)
        dataset = self.dataset(14)
        dataset["guidance_context_ids"] = np.zeros(14, dtype=np.int64)
        dataset["guidance_context_stable"] = np.array(
            [False] * 5 + [True] * 9
        )
        result = apply_maintenance_ticket_policy(
            dataset,
            np.arange(14),
            decisions,
            self.config(thrust_loss_pending_confirmation_s=8.0),
            ticket_evidence=self.evidence(
                14, excitation=0.60, motion=0.80
            ),
        )

        self.assertEqual(len(result["maintenance_tickets"]), 1)
        self.assertEqual(
            result["maintenance_tickets"][0]["start_time_s"], 13.0
        )
        self.assertTrue(np.all(
            result["maintenance_ticket_raw_qualified"][:5]
        ))
        self.assertFalse(np.any(
            result["maintenance_ticket_qualified"][:5]
        ))

    def test_unqualified_gap_is_not_counted_as_confirmation_time(self):
        decisions = self.decisions([2] * 11, [2] * 11)
        evidence = self.evidence(11, excitation=0.60, motion=0.80)
        evidence["motion_loss_evidence"][4, 0] = 0.0
        result = apply_maintenance_ticket_policy(
            self.dataset(11),
            np.arange(11),
            decisions,
            self.config(
                thrust_loss_pending_confirmation_s=8.0,
                thrust_loss_recovery_cancel_s=3.0,
            ),
            ticket_evidence=evidence,
        )

        self.assertEqual(len(result["maintenance_tickets"]), 1)
        self.assertEqual(
            result["maintenance_tickets"][0]["start_time_s"], 10.0
        )

    def test_no_output_supersedes_pending_thrust_loss_immediately(self):
        decisions = self.decisions([2, 2, 1], [2, 2, 3])
        result = apply_maintenance_ticket_policy(
            self.dataset(3),
            np.arange(3),
            decisions,
            self.config(thrust_loss_pending_confirmation_s=8.0),
            ticket_evidence=self.evidence(
                3, excitation=0.60, local=0.95, motion=0.80
            ),
        )

        self.assertEqual(len(result["maintenance_tickets"]), 1)
        self.assertEqual(
            result["maintenance_tickets"][0]["trigger"],
            "direct_no_output",
        )
        self.assertTrue(result["maintenance_ticket_active"][2])
        self.assertEqual(
            result["maintenance_pending_observations"][0]["status"],
            "superseded_by_no_output",
        )


if __name__ == "__main__":
    unittest.main()
