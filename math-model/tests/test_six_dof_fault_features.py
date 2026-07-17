import copy
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.six_dof_feature_extractor import (
    PRIVILEGED_SIMULATOR_FIELDS,
    SIX_DOF_MODEL_INPUT_DIM,
    SIX_DOF_RAW_FEATURE_DIM,
    SIX_DOF_RAW_FEATURE_NAMES,
    extract_six_dof_fault_labels,
    extract_six_dof_features,
)
from actuators.thruster_array import default_six_thruster_array
from diagnosis.thruster_health_monitor import (
    NOMINAL_GENERALIZED_MASS_DIAGONAL,
    extract_thruster_health_features,
)


def observable_log():
    return {
        "time": 1.0,
        "thruster_names": ("H1", "H2", "H3", "H4", "V1", "V2"),
        "depth": 2.2,
        "target_position_ned": np.array([4.0, -1.0, 3.0]),
        "target_euler_rpy": np.array([0.0, 0.0, 0.5]),
        "dvl": {
            "valid": True,
            "velocity": np.array([0.4, -0.2, 0.1]),
        },
        "imu": {
            "orientation": np.array([0.1, -0.1, 0.2]),
            "angular_velocity": np.array([0.01, 0.02, -0.03]),
            "linear_acceleration": np.array([0.2, -0.1, 0.05]),
        },
        "commanded_thruster_forces": np.arange(1.0, 7.0),
        "allocated_wrench_body": np.array([
            55.0, -70.0, 75.0, 4.5, -13.5, 27.0
        ]),
        "thruster_measured_currents": np.arange(2.0, 8.0),
        "thruster_expected_currents": np.arange(1.5, 7.5),
        "thruster_expected_rpms": np.arange(100.0, 700.0, 100.0),
        "thruster_measured_rpms": np.arange(110.0, 710.0, 100.0),
        "thruster_measured_voltages": np.full(6, 47.5),
        "thruster_measured_temperatures": np.arange(20.0, 26.0),
        "thruster_force_limits": np.array([40, 40, 40, 40, 35, 35]),
        "thruster_allocation_matrix": (
            default_six_thruster_array().allocation_matrix
        ),
        "thruster_nominal_voltage": 48.0,
        "thruster_ambient_temperature": 20.0,
        "thruster_saturated": np.array([
            False, True, False, False, True, False
        ]),
        "thruster_fault_modes": (
            "normal", "normal", "normal", "normal", "normal", "normal"
        ),
    }


class SixDOFFaultFeatureTests(unittest.TestCase):
    def test_feature_schema_has_expected_dimensions(self):
        features = extract_six_dof_features(observable_log())

        self.assertEqual(SIX_DOF_RAW_FEATURE_DIM, 109)
        self.assertEqual(SIX_DOF_MODEL_INPUT_DIM, 218)
        self.assertEqual(len(SIX_DOF_RAW_FEATURE_NAMES), 109)
        self.assertEqual(features.shape, (109,))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_physics_response_features_use_nominal_onboard_model(self):
        features = extract_six_dof_features(observable_log())

        np.testing.assert_allclose(
            features[46:52],
            np.array([55.0, -70.0, 75.0, 4.5, -13.5, 27.0]),
        )
        np.testing.assert_allclose(features[52:55], [1.0, -1.0, 1.0])
        np.testing.assert_allclose(
            features[55:58], [-0.8, 0.9, -0.95], atol=1e-7
        )
        np.testing.assert_allclose(features[58:61], [1.0, -1.0, 2.0])

    def test_local_monitor_isolates_no_output_telemetry(self):
        log = observable_log()
        log["commanded_thruster_forces"] = np.full(6, 20.0)
        log["thruster_expected_currents"] = np.full(6, 4.0)
        log["thruster_measured_currents"] = np.full(6, 4.0)
        log["thruster_expected_rpms"] = np.full(6, 2500.0)
        log["thruster_measured_rpms"] = np.full(6, 2500.0)
        log["thruster_measured_currents"][2] = 0.12
        log["thruster_measured_rpms"][2] = 50.0

        health = extract_thruster_health_features(log)

        self.assertGreater(health.local_anomaly_score[2], 0.9)
        self.assertLess(np.max(np.delete(health.local_anomaly_score, 2)), 0.1)

    def test_motion_monitor_projects_thrust_loss_to_commanded_thruster(self):
        previous = observable_log()
        previous["time"] = 0.0
        previous["imu"]["angular_velocity"] = np.zeros(3)
        current = observable_log()
        current["time"] = 0.1
        commands = np.zeros(6)
        commands[4] = 20.0
        matrix = current["thruster_allocation_matrix"]
        command_wrench = matrix @ commands
        actual_wrench = 0.4 * command_wrench
        observed_acceleration = (
            actual_wrench / NOMINAL_GENERALIZED_MASS_DIAGONAL
        )
        current["commanded_thruster_forces"] = commands
        current["allocated_wrench_body"] = command_wrench
        current["imu"]["linear_acceleration"] = observed_acceleration[:3]
        current["imu"]["angular_velocity"] = (
            observed_acceleration[3:] * 0.1
        )
        current["thruster_expected_currents"] = np.ones(6)
        current["thruster_measured_currents"] = np.ones(6)
        current["thruster_expected_rpms"] = np.ones(6) * 2000.0
        current["thruster_measured_rpms"] = np.ones(6) * 2000.0

        health = extract_thruster_health_features(current, previous)

        self.assertGreater(health.motion_loss_evidence[4], 0.25)
        self.assertEqual(np.count_nonzero(health.motion_loss_evidence), 1)

    def test_privileged_truth_cannot_change_features(self):
        baseline_log = observable_log()
        poisoned_log = copy.deepcopy(baseline_log)
        poisoned_log.update({
            "position_ned": np.array([999.0, 999.0, 999.0]),
            "euler_rpy": np.array([2.0, 2.0, 2.0]),
            "body_velocity": np.full(6, 999.0),
            "actual_thruster_forces": np.full(6, -999.0),
            "thruster_force_efficiencies": np.zeros(6),
            "faulted_thruster_index": 5,
            "thruster_fault_active": True,
            "thruster_fault_modes": (
                "normal", "normal", "normal", "normal", "normal", "no_output"
            ),
            "fault_label": 12,
            "true_depth": -500.0,
        })

        np.testing.assert_array_equal(
            extract_six_dof_features(baseline_log),
            extract_six_dof_features(poisoned_log),
        )
        feature_name_text = " ".join(SIX_DOF_RAW_FEATURE_NAMES)
        for field in PRIVILEGED_SIMULATOR_FIELDS:
            self.assertNotIn(field, feature_name_text)

    def test_dvl_dropout_does_not_fall_back_to_true_velocity(self):
        log = observable_log()
        log["dvl"] = {"valid": False, "velocity": np.full(3, np.nan)}
        log["body_velocity"] = np.full(6, 1234.0)

        features = extract_six_dof_features(log)
        self.assertEqual(features[3], 0.0)
        np.testing.assert_array_equal(features[4:7], np.zeros(3))

    def test_multitask_and_joint_labels_cover_thirteen_classes(self):
        normal = extract_six_dof_fault_labels(observable_log())
        self.assertEqual((normal.mode, normal.location, normal.joint), (0, 0, 0))

        no_output_log = observable_log()
        no_output_log["thruster_fault_modes"] = (
            "no_output", "normal", "normal", "normal", "normal", "normal"
        )
        no_output = extract_six_dof_fault_labels(no_output_log)
        self.assertEqual(
            (no_output.mode, no_output.location, no_output.joint),
            (1, 1, 1),
        )

        loss_log = observable_log()
        loss_log["thruster_fault_modes"] = (
            "normal", "normal", "normal", "normal", "normal", "thrust_loss"
        )
        loss = extract_six_dof_fault_labels(loss_log)
        self.assertEqual((loss.mode, loss.location, loss.joint), (2, 6, 12))


if __name__ == "__main__":
    unittest.main()
