"""Nominal six-degree-of-freedom underwater-vehicle dynamics.

The model follows the standard marine-craft form

    M nu_dot + C(nu) nu + D(nu) nu = tau + tau_restoring + tau_external
    position_dot = R_nb(q) velocity_body

where ``nu = [u, v, w, p, q, r]`` and all generalized forces are expressed
in the body frame. Navigation coordinates use North-East-Down (NED).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config.six_dof_config import SixDOFConfig


def skew(vector):
    """Return the 3x3 cross-product matrix for a three-vector."""
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


def normalize_quaternion(quaternion):
    """Normalize a scalar-first quaternion ``[w, x, y, z]``."""
    quaternion = np.asarray(quaternion, dtype=float)
    if quaternion.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quaternion.shape}")
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion norm must be finite and non-zero")
    return quaternion / norm


def quaternion_multiply(left, right):
    """Hamilton product for scalar-first quaternions."""
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quaternion_to_rotation_matrix(quaternion):
    """Return the body-to-NED rotation matrix represented by a quaternion."""
    w, x, y, z = normalize_quaternion(quaternion)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])


def euler_to_quaternion(roll, pitch, yaw):
    """Convert roll-pitch-yaw angles to a scalar-first body-to-NED quaternion."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return normalize_quaternion(np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]))


def quaternion_to_euler(quaternion):
    """Convert a scalar-first quaternion to roll-pitch-yaw angles."""
    w, x, y, z = normalize_quaternion(quaternion)

    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))

    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)

    return np.array([roll, pitch, yaw])


@dataclass
class SixDOFState:
    """State of the vehicle using NED position and body-frame velocity."""

    position_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion_nb: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    body_velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))
    time: float = 0.0

    def __post_init__(self):
        self.position_ned = self._finite_vector(
            self.position_ned, 3, "position_ned"
        )
        self.quaternion_nb = normalize_quaternion(self.quaternion_nb)
        self.body_velocity = self._finite_vector(
            self.body_velocity, 6, "body_velocity"
        )
        self.time = float(self.time)
        if not np.isfinite(self.time):
            raise ValueError("time must be finite")

    @staticmethod
    def _finite_vector(values, size, name):
        array = np.asarray(values, dtype=float)
        if array.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array.copy()

    @property
    def euler_rpy(self):
        return quaternion_to_euler(self.quaternion_nb)

    @property
    def rotation_nb(self):
        return quaternion_to_rotation_matrix(self.quaternion_nb)

    def copy(self):
        return SixDOFState(
            position_ned=self.position_ned.copy(),
            quaternion_nb=self.quaternion_nb.copy(),
            body_velocity=self.body_velocity.copy(),
            time=self.time,
        )

    def as_vector(self):
        return np.concatenate([
            self.position_ned,
            self.quaternion_nb,
            self.body_velocity,
        ])

    @classmethod
    def from_vector(cls, values, time=0.0):
        values = np.asarray(values, dtype=float)
        if values.shape != (13,):
            raise ValueError(f"state vector must have shape (13,), got {values.shape}")
        return cls(
            position_ned=values[:3],
            quaternion_nb=values[3:7],
            body_velocity=values[7:13],
            time=time,
        )


@dataclass
class SixDOFDerivative:
    position_ned: np.ndarray
    quaternion_nb: np.ndarray
    body_velocity: np.ndarray

    def as_vector(self):
        return np.concatenate([
            self.position_ned,
            self.quaternion_nb,
            self.body_velocity,
        ])


class SixDOFDynamics:
    """Integrate nominal rigid-body and hydrodynamic AUV motion."""

    def __init__(
        self,
        config: Optional[SixDOFConfig] = None,
        initial_state: Optional[SixDOFState] = None,
    ):
        self.config = config or SixDOFConfig()
        self.mass_matrix = self.config.total_mass_matrix
        if not np.allclose(self.mass_matrix, self.mass_matrix.T, atol=1e-10):
            raise ValueError("total mass matrix must be symmetric")
        if np.min(np.linalg.eigvalsh(self.mass_matrix)) <= 0:
            raise ValueError("total mass matrix must be positive definite")
        self._mass_matrix_inverse = np.linalg.inv(self.mass_matrix)
        self.state = initial_state.copy() if initial_state else SixDOFState()

    def reset(self, state: Optional[SixDOFState] = None):
        self.state = state.copy() if state else SixDOFState()
        return self.state.copy()

    def coriolis_matrix(self, body_velocity):
        """Build an energy-conserving Coriolis/centripetal matrix.

        The construction uses total generalized momentum and is valid for the
        constant symmetric mass matrix used by this Phase-1 model.
        """
        velocity = np.asarray(body_velocity, dtype=float)
        if velocity.shape != (6,):
            raise ValueError("body_velocity must have shape (6,)")

        momentum = self.mass_matrix @ velocity
        linear_momentum = momentum[:3]
        angular_momentum = momentum[3:]

        matrix = np.zeros((6, 6), dtype=float)
        matrix[:3, 3:] = -skew(linear_momentum)
        matrix[3:, :3] = -skew(linear_momentum)
        matrix[3:, 3:] = -skew(angular_momentum)
        return matrix

    def damping_wrench(self, body_velocity):
        """Return the hydrodynamic damping wrench opposing body motion."""
        velocity = np.asarray(body_velocity, dtype=float)
        linear = self.config.linear_damping * velocity
        quadratic = self.config.quadratic_damping * np.abs(velocity) * velocity
        return -(linear + quadratic)

    def restoring_wrench(self, state: Optional[SixDOFState] = None):
        """Return gravity and buoyancy force/moment in the body frame."""
        state = state or self.state
        rotation_bn = state.rotation_nb.T

        gravity_ned = np.array([0.0, 0.0, self.config.weight])
        buoyancy_ned = np.array([0.0, 0.0, -self.config.buoyancy])
        gravity_body = rotation_bn @ gravity_ned
        buoyancy_body = rotation_bn @ buoyancy_ned

        force_body = gravity_body + buoyancy_body
        moment_body = (
            np.cross(self.config.center_of_gravity, gravity_body)
            + np.cross(self.config.center_of_buoyancy, buoyancy_body)
        )
        return np.concatenate([force_body, moment_body])

    def kinetic_energy(self, state: Optional[SixDOFState] = None):
        state = state or self.state
        velocity = state.body_velocity
        return 0.5 * float(velocity @ self.mass_matrix @ velocity)

    def derivatives(
        self,
        state: SixDOFState,
        tau_body,
        disturbance_body=None,
    ):
        tau_body = self._wrench(tau_body, "tau_body")
        if disturbance_body is None:
            disturbance_body = np.zeros(6)
        disturbance_body = self._wrench(disturbance_body, "disturbance_body")

        velocity = state.body_velocity
        linear_velocity = velocity[:3]
        angular_velocity = velocity[3:]

        position_dot = state.rotation_nb @ linear_velocity
        quaternion_dot = 0.5 * quaternion_multiply(
            state.quaternion_nb,
            np.concatenate([[0.0], angular_velocity]),
        )

        coriolis = self.coriolis_matrix(velocity)
        net_wrench = (
            tau_body
            + disturbance_body
            + self.restoring_wrench(state)
            + self.damping_wrench(velocity)
            - coriolis @ velocity
        )
        velocity_dot = self._mass_matrix_inverse @ net_wrench

        return SixDOFDerivative(
            position_ned=position_dot,
            quaternion_nb=quaternion_dot,
            body_velocity=velocity_dot,
        )

    def step(self, tau_body, dt, disturbance_body=None):
        """Advance the state by one RK4 integration step."""
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")

        tau_body = self._wrench(tau_body, "tau_body")
        if disturbance_body is None:
            disturbance_body = np.zeros(6)
        disturbance_body = self._wrench(disturbance_body, "disturbance_body")

        y0 = self.state.as_vector()
        time0 = self.state.time

        def derivative(values, time):
            intermediate = SixDOFState.from_vector(values, time=time)
            return self.derivatives(
                intermediate,
                tau_body=tau_body,
                disturbance_body=disturbance_body,
            ).as_vector()

        k1 = derivative(y0, time0)
        k2 = derivative(y0 + 0.5 * dt * k1, time0 + 0.5 * dt)
        k3 = derivative(y0 + 0.5 * dt * k2, time0 + 0.5 * dt)
        k4 = derivative(y0 + dt * k3, time0 + dt)

        next_values = y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_values[3:7] = normalize_quaternion(next_values[3:7])
        self.state = SixDOFState.from_vector(next_values, time=time0 + dt)
        return self.state.copy()

    @staticmethod
    def _wrench(values, name):
        array = np.asarray(values, dtype=float)
        if array.shape != (6,):
            raise ValueError(f"{name} must have shape (6,), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array.copy()
