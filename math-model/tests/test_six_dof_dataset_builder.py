import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TEST_ROOT = PROJECT_ROOT / "tests"
for path in (SRC_ROOT, TEST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_six_dof_fault_features import observable_log
from utils.six_dof_dataset_builder import (
    add_temporal_differences,
    apply_standardizer,
    build_six_dof_sequence_dataset,
    fit_standardizer,
    stratified_mission_split,
)


def mission_logs(length, fault_mode=None, fault_index=0):
    logs = []
    for index in range(length):
        log = observable_log()
        log["time"] = index * 0.05
        log["depth"] += index * 0.001
        log["guidance_context_id"] = 0 if index < length // 2 else 1
        if fault_mode is not None and index >= length // 2:
            modes = ["normal"] * 6
            modes[fault_index] = fault_mode
            log["thruster_fault_modes"] = tuple(modes)
        logs.append(log)
    return logs


class SixDOFDatasetBuilderTests(unittest.TestCase):
    def test_windows_never_cross_mission_boundaries(self):
        dataset = build_six_dof_sequence_dataset(
            {
                10: mission_logs(30),
                20: mission_logs(30, "no_output", fault_index=2),
            },
            seq_len=10,
            stride=5,
        )

        self.assertEqual(dataset["X"].shape[1:], (10, 109))
        self.assertEqual(set(dataset["mission_ids"].tolist()), {10, 20})
        self.assertIn(3, dataset["y_location"])
        self.assertIn(3, dataset["y_joint"])
        self.assertEqual(
            dataset["guidance_context_ids"].shape,
            dataset["mission_ids"].shape,
        )
        self.assertTrue(np.any(~dataset["guidance_context_stable"]))

    def test_stratified_split_keeps_whole_missions_disjoint(self):
        mission_ids = []
        labels = []
        for scenario in (0, 1, 12):
            for repetition in range(3):
                mission_id = scenario * 10 + repetition
                mission_ids.extend([mission_id] * 8)
                labels.extend([0] * 4 + [scenario] * 4)
        mission_ids = np.asarray(mission_ids)
        labels = np.asarray(labels)

        splits = stratified_mission_split(mission_ids, labels, seed=7)
        split_missions = {
            name: set(mission_ids[indices].tolist())
            for name, indices in splits.items()
        }

        self.assertFalse(split_missions["train"] & split_missions["validation"])
        self.assertFalse(split_missions["train"] & split_missions["test"])
        self.assertFalse(
            split_missions["validation"] & split_missions["test"]
        )
        for indices in splits.values():
            self.assertEqual(set(labels[indices].tolist()), {0, 1, 12})

    def test_standardizer_is_fit_from_training_windows_only(self):
        train = build_six_dof_sequence_dataset(
            {1: mission_logs(30), 2: mission_logs(30, "thrust_loss", 5)},
            seq_len=10,
            stride=5,
        )["X"]
        augmented = add_temporal_differences(train)
        stats = fit_standardizer(train)
        normalized = apply_standardizer(train, stats)

        self.assertEqual(augmented.shape[-1], 218)
        self.assertEqual(stats["mean"].shape, (218,))
        self.assertEqual(normalized.shape, augmented.shape)
        self.assertTrue(np.all(np.isfinite(normalized)))


if __name__ == "__main__":
    unittest.main()
