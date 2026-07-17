import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from diagnosis.maintenance_log_policy import (
    MaintenanceLogConfig,
    apply_maintenance_log_policy,
    maintenance_log_metrics,
)


LOCATION = np.array([0.50, 0.20, 0.10, 0.08, 0.07, 0.05])


def decisions_for(
    health_levels,
    modes,
    *,
    ticket_active=None,
    contexts=None,
    context_stable=None,
    mode_true=None,
    raw_event_count=None,
):
    health_levels = np.asarray(health_levels, dtype=np.int64)
    modes = np.asarray(modes, dtype=np.int64)
    sample_count = len(health_levels)
    ticket_active = (
        np.zeros(sample_count, dtype=bool)
        if ticket_active is None
        else np.asarray(ticket_active, dtype=bool)
    )
    contexts = (
        np.zeros(sample_count, dtype=np.int64)
        if contexts is None
        else np.asarray(contexts, dtype=np.int64)
    )
    context_stable = (
        np.ones(sample_count, dtype=bool)
        if context_stable is None
        else np.asarray(context_stable, dtype=bool)
    )
    mode_true = (
        np.zeros(sample_count, dtype=np.int64)
        if mode_true is None
        else np.asarray(mode_true, dtype=np.int64)
    )
    if raw_event_count is None:
        raw_event_count = sample_count
    return {
        "mode_true": mode_true,
        "location_true": np.where(mode_true != 0, 1, 0),
        "joint_true": np.where(mode_true == 1, 1, np.where(mode_true == 2, 7, 0)),
        "health_level_pred": health_levels,
        "mode_pred": modes,
        "probable_mode_pred": np.where(modes == 0, 2, modes),
        "fault_probabilities": np.full(sample_count, 0.80),
        "smoothed_location_probabilities": np.tile(LOCATION, (sample_count, 1)),
        "maintenance_ticket_active": ticket_active,
        "maintenance_ticket_raw_qualified": ticket_active.copy(),
        "maintenance_ticket_excitation": np.full(sample_count, 0.40),
        "maintenance_ticket_independent_evidence": np.full(sample_count, 0.60),
        "maintenance_guidance_context_id": contexts,
        "maintenance_guidance_context_stable": context_stable,
        "maintenance_guidance_context_available": True,
        "maintenance_events": [
            {"raw_event_id": index} for index in range(raw_event_count)
        ],
        "requires_ftc": modes == 1,
    }


class MaintenanceLogPolicyTests(unittest.TestCase):
    def test_same_context_short_recovery_is_merged(self):
        dataset = {
            "mission_ids": np.zeros(5, dtype=np.int64),
            "window_end_times": np.arange(5, dtype=float) * 1.25,
        }
        decisions = decisions_for(
            [2, 2, 0, 2, 2],
            [2, 2, 0, 2, 2],
            raw_event_count=2,
        )

        result = apply_maintenance_log_policy(
            dataset,
            np.arange(5),
            decisions,
            MaintenanceLogConfig(merge_gap_s=5.0),
        )

        self.assertEqual(len(result["maintenance_graded_events"]), 1)
        event = result["maintenance_graded_events"][0]
        self.assertEqual(event["source_segment_count"], 2)
        self.assertEqual(event["log_level"], "observation")
        self.assertFalse(event["attention_required"])
        self.assertEqual(len(result["maintenance_raw_events"]), 2)

    def test_context_change_prevents_event_merge(self):
        dataset = {
            "mission_ids": np.zeros(5, dtype=np.int64),
            "window_end_times": np.arange(5, dtype=float) * 1.25,
        }
        decisions = decisions_for(
            [2, 2, 0, 2, 2],
            [2, 2, 0, 2, 2],
            contexts=[0, 0, 1, 1, 1],
            raw_event_count=2,
        )

        result = apply_maintenance_log_policy(
            dataset, np.arange(5), decisions
        )

        self.assertEqual(len(result["maintenance_graded_events"]), 2)
        self.assertEqual(
            [event["guidance_context_id"] for event in result["maintenance_graded_events"]],
            [0, 1],
        )

    def test_events_are_graded_without_changing_ticket_or_ftc_arrays(self):
        dataset = {
            "mission_ids": np.arange(4, dtype=np.int64),
            "window_end_times": np.zeros(4, dtype=float),
        }
        decisions = decisions_for(
            [1, 2, 2, 3],
            [2, 2, 2, 1],
            ticket_active=[False, False, True, True],
            raw_event_count=4,
        )
        original_tickets = decisions["maintenance_ticket_active"].copy()
        original_ftc = decisions["requires_ftc"].copy()

        result = apply_maintenance_log_policy(
            dataset, np.arange(4), decisions
        )

        self.assertEqual(
            [event["log_level"] for event in result["maintenance_graded_events"]],
            [
                "background_trace",
                "observation",
                "maintenance_advisory",
                "safety_alert",
            ],
        )
        np.testing.assert_array_equal(
            result["maintenance_ticket_active"], original_tickets
        )
        np.testing.assert_array_equal(result["requires_ftc"], original_ftc)

    def test_metrics_separate_retained_events_from_operator_prompts(self):
        dataset = {
            "mission_ids": np.arange(4, dtype=np.int64),
            "window_end_times": np.zeros(4, dtype=float),
        }
        decisions = decisions_for(
            [1, 2, 2, 3],
            [2, 2, 2, 1],
            ticket_active=[False, False, True, True],
            mode_true=[2, 2, 2, 1],
            raw_event_count=4,
        )
        result = apply_maintenance_log_policy(
            dataset, np.arange(4), decisions
        )

        metrics = maintenance_log_metrics(
            dataset, np.arange(4), result
        )

        self.assertEqual(metrics["raw_event_count"], 4)
        self.assertEqual(metrics["operator_attention_event_count"], 2)
        self.assertEqual(metrics["false_operator_attention_events"], 0)
        self.assertEqual(metrics["raw_fault_mission_recall"], 1.0)
        self.assertEqual(metrics["operator_attention_fault_mission_recall"], 0.5)
        self.assertEqual(metrics["no_output_operator_attention_recall"], 1.0)
        self.assertAlmostEqual(
            metrics["thrust_loss_operator_attention_recall"], 1.0 / 3.0
        )


if __name__ == "__main__":
    unittest.main()
