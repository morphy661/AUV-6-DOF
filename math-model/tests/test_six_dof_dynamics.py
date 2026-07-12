import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config.six_dof_config import SixDOFConfig
from environment.six_dof_dynamics import (
    SixDOFDynamics,
    SixDOFState,
    euler_to_quaternion,
)


def undamped_config(**overrides):
    parameters = {
        "mass": 50.0,
        "inertia": np.diag([8.0, 9.0, 10.0]),
        "added_mass": np.zeros((6, 6)),
        "linear_damping": np.zeros(6),
        "quadratic_damping": np.zeros(6),
        "weight": 50.0 * 9.81,
        "buoyancy": 50.0 * 9.81,
        "center_of_gravity": np.zeros(3),
        "center_of_buoyancy": np.zeros(3),
    }
    parameters.update(overrides)
    return SixDOFConfig(**parameters)


class SixDOFDynamicsTests(unittest.TestCase):
    def test_neutral_vehicle_remains_at_rest(self):
        dynamics = SixDOFDynamics(config=undamped_config())

        for _ in range(100):
            dynamics.step(np.zeros(6), dt=0.1)

        np.testing.assert_allclose(dynamics.state.position_ned, 0.0, atol=1e-12)
        np.testing.assert_allclose(dynamics.state.body_velocity, 0.0, atol=1e-12)
        np.testing.assert_allclose(
            dynamics.state.quaternion_nb,
            np.array([1.0, 0.0, 0.0, 0.0]),
            atol=1e-12,
        )

    def test_constant_surge_force_matches_analytic_solution(self):
        dynamics = SixDOFDynamics(config=undamped_config())
        wrench = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        for _ in range(100):
            dynamics.step(wrench, dt=0.01)

        self.assertAlmostEqual(dynamics.state.body_velocity[0], 0.2, places=8)
        self.assertAlmostEqual(dynamics.state.position_ned[0], 0.1, places=8)
        np.testing.assert_allclose(dynamics.state.position_ned[1:], 0.0, atol=1e-12)

    def test_positive_heave_force_increases_ned_depth(self):
        dynamics = SixDOFDynamics(config=undamped_config())
        wrench = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])

        for _ in range(100):
            dynamics.step(wrench, dt=0.01)

        self.assertGreater(dynamics.state.position_ned[2], 0.0)
        self.assertAlmostEqual(dynamics.state.position_ned[2], 0.1, places=8)
        self.assertAlmostEqual(dynamics.state.body_velocity[2], 0.2, places=8)

    def test_constant_yaw_moment_changes_heading(self):
        dynamics = SixDOFDynamics(config=undamped_config())
        wrench = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0])

        for _ in range(100):
            dynamics.step(wrench, dt=0.01)

        self.assertAlmostEqual(dynamics.state.body_velocity[5], 0.2, places=8)
        self.assertAlmostEqual(dynamics.state.euler_rpy[2], 0.1, places=6)
        self.assertAlmostEqual(np.linalg.norm(dynamics.state.quaternion_nb), 1.0, places=12)

    def test_positive_roll_generates_restoring_moment(self):
        config = undamped_config(
            center_of_gravity=np.array([0.0, 0.0, 0.05]),
            center_of_buoyancy=np.array([0.0, 0.0, -0.05]),
        )
        state = SixDOFState(
            quaternion_nb=euler_to_quaternion(np.deg2rad(10.0), 0.0, 0.0)
        )
        dynamics = SixDOFDynamics(config=config, initial_state=state)

        mass_matrix = dynamics.mass_matrix
        np.testing.assert_allclose(mass_matrix, mass_matrix.T, atol=1e-12)
        self.assertGreater(np.linalg.norm(mass_matrix[:3, 3:]), 0.0)

        derivative = dynamics.derivatives(state, np.zeros(6))

        self.assertLess(derivative.body_velocity[3], 0.0)
        self.assertAlmostEqual(derivative.body_velocity[4], 0.0, places=12)
        self.assertAlmostEqual(derivative.body_velocity[5], 0.0, places=12)

    def test_hydrodynamic_damping_removes_kinetic_energy(self):
        config = SixDOFConfig(
            mass=50.0,
            inertia=np.diag([8.0, 9.0, 10.0]),
            added_mass=np.zeros((6, 6)),
            linear_damping=np.array([10.0, 12.0, 14.0, 1.0, 1.2, 1.4]),
            quadratic_damping=np.array([4.0, 5.0, 6.0, 0.3, 0.4, 0.5]),
            weight=50.0 * 9.81,
            buoyancy=50.0 * 9.81,
            center_of_gravity=np.zeros(3),
            center_of_buoyancy=np.zeros(3),
        )
        state = SixDOFState(
            body_velocity=np.array([1.0, -0.4, 0.3, 0.2, -0.1, 0.15])
        )
        dynamics = SixDOFDynamics(config=config, initial_state=state)
        initial_energy = dynamics.kinetic_energy()

        for _ in range(100):
            dynamics.step(np.zeros(6), dt=0.01)

        self.assertLess(dynamics.kinetic_energy(), initial_energy)

    def test_coriolis_matrix_is_skew_symmetric(self):
        dynamics = SixDOFDynamics(config=SixDOFConfig())
        velocity = np.array([0.8, -0.3, 0.2, 0.1, -0.2, 0.05])
        matrix = dynamics.coriolis_matrix(velocity)

        np.testing.assert_allclose(matrix + matrix.T, 0.0, atol=1e-12)
        self.assertAlmostEqual(float(velocity @ matrix @ velocity), 0.0, places=12)

    def test_invalid_time_step_is_rejected(self):
        dynamics = SixDOFDynamics(config=undamped_config())
        with self.assertRaises(ValueError):
            dynamics.step(np.zeros(6), dt=0.0)


if __name__ == "__main__":
    unittest.main()
