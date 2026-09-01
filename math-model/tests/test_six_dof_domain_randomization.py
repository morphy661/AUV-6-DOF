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
    DEPTH_BAND_PROBABILITIES,
    DEPTH_BANDS_M,
    _depth_band_index_for_mission,
    _disturbance_provider,
    _mission_schedule,
    _randomized_dynamics,
    _randomized_thrusters,
    _split_counts,
    _split_for_repetition,
    _target_provider,
    fixed_split_indices,
    run_mission,
)
from actuators.six_dof_thruster_faults import SixDOFThrusterFaultMode


class SixDOFDomainRandomizationTests(unittest.TestCase):
    def test_depth_bands_follow_zero_to_300m_focused_mix(self):
        split_bands = {name: [] for name in ("train", "validation", "test")}
        for scenario_index in range(13):
            for repetition in range(20):
                split = _split_for_repetition(repetition, 20)
                split_bands[split].append(_depth_band_index_for_mission(
                    scenario_index,
                    repetition,
                    20,
                ))

        expected_counts = {
            "train": [45, 64, 55, 18],
            "validation": [10, 13, 12, 4],
            "test": [10, 13, 12, 4],
        }
        for split, bands in split_bands.items():
            counts = np.bincount(bands, minlength=len(DEPTH_BANDS_M))
            self.assertEqual(counts.tolist(), expected_counts[split])
            self.assertAlmostEqual(
                float(np.mean(np.asarray(bands) < 3)),
                0.90,
                delta=0.015,
            )

        np.testing.assert_allclose(
            DEPTH_BAND_PROBABILITIES,
            [0.25, 0.35, 0.30, 0.10],
        )

    def test_mission_schedule_stays_inside_selected_depth_band(self):
        for band_index, (depth_low, depth_high) in enumerate(DEPTH_BANDS_M):
            plan = _mission_schedule(
                duration=90.0,
                rng=np.random.default_rng(100 + band_index),
                depth_band_index=band_index,
            )
            depths = np.array([item[1][2] for item in plan.waypoints])

            self.assertTrue(np.all(depths >= depth_low))
            self.assertTrue(np.all(depths <= depth_high))
            self.assertGreaterEqual(plan.cruise_speed_mps, 1.2)
            self.assertLessEqual(plan.cruise_speed_mps, 1.5)
            self.assertGreaterEqual(plan.vertical_speed_mps, 0.15)
            self.assertLessEqual(plan.vertical_speed_mps, 0.35)

    def test_smooth_target_respects_horizontal_and_vertical_speed_limits(self):
        plan = _mission_schedule(
            duration=90.0,
            rng=np.random.default_rng(44),
            depth_band_index=2,
        )
        provider = _target_provider(plan)
        sample_times = np.linspace(0.0, 90.0, 9001)
        positions = np.array([
            provider(time_s, None).position_ned
            for time_s in sample_times
        ])
        velocities = np.diff(positions, axis=0) / np.diff(sample_times)[:, None]

        horizontal_speeds = np.linalg.norm(velocities[:, :2], axis=1)
        self.assertLessEqual(
            float(np.max(horizontal_speeds)),
            plan.cruise_speed_mps + 1e-3,
        )
        self.assertLessEqual(
            float(np.max(np.abs(velocities[:, 2]))),
            plan.vertical_speed_mps + 1e-3,
        )

    def test_twenty_missions_give_fourteen_three_three_split(self):
        self.assertEqual(
            _split_counts(20),
            {"train": 14, "validation": 3, "test": 3},
        )
        splits = [_split_for_repetition(index, 20) for index in range(20)]
        self.assertEqual(splits.count("train"), 14)
        self.assertEqual(splits.count("validation"), 3)
        self.assertEqual(splits.count("test"), 3)

    def test_independent_splits_share_broad_deployment_range(self):
        train_dynamics, _ = _randomized_dynamics(
            np.random.default_rng(1), "train"
        )
        test_dynamics, _ = _randomized_dynamics(
            np.random.default_rng(1), "test"
        )

        self.assertGreaterEqual(train_dynamics.config.mass, 41.0)
        self.assertLessEqual(train_dynamics.config.mass, 59.0)
        self.assertGreaterEqual(test_dynamics.config.mass, 41.0)
        self.assertLessEqual(test_dynamics.config.mass, 59.0)
        self.assertEqual(
            train_dynamics.config.mass,
            test_dynamics.config.mass,
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

    def test_paired_fault_seed_preserves_environment_parameters(self):
        common = {
            "duration": 12.0,
            "dt": 0.1,
            "seed": 8123,
            "split": "train",
            "depth_band_index": 1,
            "fault_seed": 99123,
            "rpm_noise_std_override": 68.0,
        }
        _, normal = run_mission(
            thruster_name=None,
            fault_mode=None,
            **common,
        )
        _, fault = run_mission(
            thruster_name="H1",
            fault_mode=SixDOFThrusterFaultMode.THRUST_LOSS,
            **common,
        )

        for key in (
            "dynamics",
            "thrusters",
            "sensors",
            "disturbance",
            "mission",
        ):
            self.assertEqual(normal[key], fault[key])
        self.assertIsNone(normal["fault_start_time_s"])
        self.assertIsNone(normal["thrust_efficiency"])
        self.assertIsNotNone(fault["fault_start_time_s"])
        self.assertIsNotNone(fault["thrust_efficiency"])

    def test_rpm_override_changes_only_rpm_metadata(self):
        common = {
            "thruster_name": None,
            "fault_mode": None,
            "duration": 12.0,
            "dt": 0.1,
            "seed": 8456,
            "split": "train",
            "depth_band_index": 2,
            "fault_seed": 99456,
        }
        _, nominal = run_mission(
            rpm_noise_std_override=40.0,
            **common,
        )
        _, high_noise = run_mission(
            rpm_noise_std_override=70.0,
            **common,
        )

        nominal_sensors = dict(nominal["sensors"])
        high_noise_sensors = dict(high_noise["sensors"])
        self.assertEqual(nominal_sensors.pop("rpm_noise_std"), 40.0)
        self.assertEqual(high_noise_sensors.pop("rpm_noise_std"), 70.0)
        self.assertEqual(nominal_sensors, high_noise_sensors)
        for key in ("dynamics", "thrusters", "disturbance", "mission"):
            self.assertEqual(nominal[key], high_noise[key])

    def test_lateral_disturbance_override_changes_only_xy_amplitudes(self):
        _, low = _disturbance_provider(
            np.random.default_rng(771),
            lateral_force_amplitudes_override=(0.25, 0.25),
        )
        _, high = _disturbance_provider(
            np.random.default_rng(771),
            lateral_force_amplitudes_override=(1.40, 1.40),
        )

        np.testing.assert_allclose(low["amplitudes"][:2], [0.25, 0.25])
        np.testing.assert_allclose(high["amplitudes"][:2], [1.40, 1.40])
        np.testing.assert_allclose(
            low["amplitudes"][2:], high["amplitudes"][2:]
        )
        self.assertEqual(low["frequencies_radps"], high["frequencies_radps"])
        self.assertEqual(low["phases_rad"], high["phases_rad"])

    def test_vertical_force_override_preserves_other_thruster_parameters(self):
        _, low = _randomized_thrusters(
            np.random.default_rng(881),
            "train",
            vertical_force_limit_override=34.5,
        )
        _, high = _randomized_thrusters(
            np.random.default_rng(881),
            "train",
            vertical_force_limit_override=45.0,
        )

        self.assertEqual(low.pop("vertical_force_limit_n"), 34.5)
        self.assertEqual(high.pop("vertical_force_limit_n"), 45.0)
        self.assertEqual(low, high)

    def test_mass_override_preserves_other_dynamics_parameters(self):
        _, low = _randomized_dynamics(
            np.random.default_rng(991),
            "train",
            mass_kg_override=45.0,
        )
        _, high = _randomized_dynamics(
            np.random.default_rng(991),
            "train",
            mass_kg_override=58.0,
        )

        self.assertEqual(low.pop("mass_kg"), 45.0)
        self.assertEqual(high.pop("mass_kg"), 58.0)
        self.assertEqual(low, high)


if __name__ == "__main__":
    unittest.main()
