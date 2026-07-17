import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = (
    REPO_ROOT
    / "depth-sensor-fault-detection"
    / "depth_fault_detection"
)
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from cross_validate_six_dof_multitask import (
    build_scenario_stratified_mission_folds,
    select_feature_set,
)
from train_six_dof_multitask import (
    fit_training_statistics,
    operational_test_metrics,
)


class SixDOFCrossValidationTests(unittest.TestCase):
    def test_baseline_ablation_uses_view_and_dynamic_statistics(self):
        original = torch.randn(4, 10, 109)
        dataset = {"X": original}

        selected = select_feature_set(dataset, "baseline")
        mean, std = fit_training_statistics(
            selected["X"], np.arange(4), batch_size=2
        )

        self.assertEqual(tuple(selected["X"].shape), (4, 10, 61))
        self.assertEqual(tuple(mean.shape), (122,))
        self.assertEqual(tuple(std.shape), (122,))
        self.assertEqual(tuple(dataset["X"].shape), (4, 10, 109))

    def test_folds_are_scenario_stratified_and_exclude_test_missions(self):
        metadata = {}
        mission_ids = []
        next_id = 0
        for scenario in ("Normal", "H1 Thrust Loss"):
            for repetition in range(8):
                split = "train" if repetition < 5 else (
                    "validation" if repetition == 5 else "test"
                )
                metadata[next_id] = {
                    "scenario": scenario,
                    "split": split,
                }
                mission_ids.extend([next_id] * 4)
                next_id += 1
        dataset = {
            "mission_metadata": metadata,
            "mission_ids": torch.tensor(mission_ids),
        }

        folds = build_scenario_stratified_mission_folds(
            dataset, fold_count=3, seed=5
        )
        test_missions = {
            mission_id
            for mission_id, values in metadata.items()
            if values["split"] == "test"
        }

        self.assertEqual(len(folds), 3)
        validation_union = set()
        for fold in folds:
            train = set(fold["train_missions"])
            validation = set(fold["validation_missions"])
            self.assertFalse(train & validation)
            self.assertFalse((train | validation) & test_missions)
            validation_union.update(validation)
            scenarios = {
                metadata[mission_id]["scenario"]
                for mission_id in validation
            }
            self.assertEqual(scenarios, {"Normal", "H1 Thrust Loss"})
        expected_in_domain = set(metadata) - test_missions
        self.assertEqual(validation_union, expected_in_domain)

    def test_operational_metrics_use_mission_time_and_severity(self):
        dataset = {
            "mission_ids": torch.tensor([1, 1, 1, 1]),
            "window_end_times": torch.tensor([10.0, 11.0, 12.0, 13.0]),
            "mission_metadata": {
                1: {
                    "parameters": {
                        "fault_start_time_s": 10.5,
                        "thrust_efficiency": 0.40,
                    }
                }
            },
        }
        predictions = {
            "joint_true": np.array([0, 7, 7, 7]),
            "joint_pred": np.array([0, 0, 7, 7]),
        }

        metrics = operational_test_metrics(
            dataset, np.arange(4), predictions
        )

        self.assertEqual(metrics["mode_detection_rate"], 1.0)
        self.assertEqual(metrics["exact_diagnosis_rate"], 1.0)
        self.assertAlmostEqual(metrics["mean_exact_diagnosis_delay_s"], 1.5)
        self.assertAlmostEqual(
            metrics["thrust_loss_exact_accuracy_by_efficiency"]["0.30-0.45"],
            2.0 / 3.0,
        )


if __name__ == "__main__":
    unittest.main()
