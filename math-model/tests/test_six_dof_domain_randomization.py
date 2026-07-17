import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = PROJECT_ROOT / "examples"
SRC_ROOT = PROJECT_ROOT / "src"
for path in (EXAMPLES_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_six_dof_fault_dataset import (
    _randomized_dynamics,
    _randomized_thrusters,
    _split_counts,
    _split_for_repetition,
    fixed_split_indices,
)


class SixDOFDomainRandomizationTests(unittest.TestCase):
    def test_twenty_missions_give_fourteen_three_three_split(self):
        self.assertEqual(
            _split_counts(20),
            {"train": 14, "validation": 3, "test": 3},
        )
        splits = [_split_for_repetition(index, 20) for index in range(20)]
        self.assertEqual(splits.count("train"), 14)
        self.assertEqual(splits.count("validation"), 3)
        self.assertEqual(splits.count("test"), 3)

    def test_held_out_mass_scale_is_outside_training_range(self):
        train_dynamics, _ = _randomized_dynamics(
            np.random.default_rng(1), "train"
        )
        test_dynamics, _ = _randomized_dynamics(
            np.random.default_rng(1), "test"
        )

        self.assertGreaterEqual(train_dynamics.config.mass, 45.0)
        self.assertLessEqual(train_dynamics.config.mass, 55.0)
        self.assertTrue(
            test_dynamics.config.mass <= 45.0
            or test_dynamics.config.mass >= 55.0
        )
        self.assertGreater(
            np.min(np.linalg.eigvalsh(test_dynamics.mass_matrix)), 0.0
        )

    def test_randomized_thruster_layout_remains_valid(self):
        array, metadata = _randomized_thrusters(
            np.random.default_rng(8), "test"
        )

        self.assertEqual(array.names, ["H1", "H2", "H3", "H4", "V1", "V2"])
        self.assertEqual(array.allocation_matrix.shape, (6, 6))
        self.assertGreater(metadata["length_m"], 0.0)
        self.assertGreater(metadata["vertical_force_limit_n"], 0.0)

    def test_fixed_split_indices_keep_missions_disjoint(self):
        mission_ids = np.repeat(np.arange(6), 4)
        metadata = {
            0: {"split": "train"},
            1: {"split": "train"},
            2: {"split": "validation"},
            3: {"split": "validation"},
            4: {"split": "test"},
            5: {"split": "test"},
        }

        splits = fixed_split_indices(mission_ids, metadata)
        split_missions = {
            name: set(mission_ids[indices].tolist())
            for name, indices in splits.items()
        }
        self.assertFalse(split_missions["train"] & split_missions["test"])
        self.assertFalse(
            split_missions["validation"] & split_missions["test"]
        )


if __name__ == "__main__":
    unittest.main()
