import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = (
    REPO_ROOT
    / "depth-sensor-fault-detection"
    / "depth_fault_detection"
)
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from model_six_dof_multitask import (
    AUVSixDOFMultiTaskDetector,
    combine_multitask_predictions,
)


class SixDOFMultiTaskModelTests(unittest.TestCase):
    def test_forward_shapes_and_attention_normalization(self):
        model = AUVSixDOFMultiTaskDetector(
            input_dim=218,
            hidden_size=16,
            num_layers=1,
            local_hidden_size=8,
        )
        inputs = torch.randn(4, 100, 218)

        modes, locations, mode_attention, location_attention = model(
            inputs, return_attention=True
        )

        self.assertEqual(tuple(modes.shape), (4, 3))
        self.assertEqual(tuple(locations.shape), (4, 6))
        self.assertEqual(tuple(mode_attention.shape), (4, 100))
        self.assertEqual(tuple(location_attention.shape), (4, 100))
        torch.testing.assert_close(
            mode_attention.sum(dim=1), torch.ones(4), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            location_attention.sum(dim=1),
            torch.ones(4),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertTrue(model.structured_fusion)

    def test_flat_baseline_and_explicit_flat_hybrid_remain_available(self):
        baseline = AUVSixDOFMultiTaskDetector(
            input_dim=122, hidden_size=8, num_layers=1
        )
        hybrid_flat = AUVSixDOFMultiTaskDetector(
            input_dim=218,
            hidden_size=8,
            num_layers=1,
            structured_fusion=False,
        )

        self.assertFalse(baseline.structured_fusion)
        self.assertFalse(hybrid_flat.structured_fusion)
        self.assertEqual(
            tuple(baseline(torch.randn(2, 10, 122))[0].shape),
            (2, 3),
        )
        self.assertEqual(
            tuple(hybrid_flat(torch.randn(2, 10, 218))[1].shape),
            (2, 6),
        )

    def test_joint_prediction_mapping(self):
        modes = torch.tensor([0, 1, 1, 2, 2, 2])
        locations = torch.tensor([0, 1, 6, 1, 6, 0])

        joint = combine_multitask_predictions(modes, locations)

        torch.testing.assert_close(joint, torch.tensor([0, 1, 6, 7, 12, 0]))

    def test_wrong_input_dimension_is_rejected(self):
        model = AUVSixDOFMultiTaskDetector(input_dim=218)
        with self.assertRaisesRegex(ValueError, "feature dimension"):
            model(torch.zeros(2, 100, 40))


if __name__ == "__main__":
    unittest.main()
