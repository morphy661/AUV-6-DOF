"""Simple cascaded position/attitude controller for the nominal 6-DOF AUV."""

from dataclasses import dataclass, field

import numpy as np

from environment.six_dof_dynamics import (
    SixDOFState,
    euler_to_quaternion,
    quaternion_multiply,
)


def _vector(values, size, name):
    array = np.asarray(values, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _clip_by_axis(values, limits):
    return np.clip(values, -limits, limits)


def _attitude_error_body(current_quaternion, target_quaternion):
    """Return the shortest body-frame rotation vector to the target."""
    current_conjugate = np.asarray(current_quaternion, dtype=float).copy()
    current_conjugate[1:] *= -1.0
    error_quaternion = quaternion_multiply(current_conjugate, target_quaternion)
    if error_quaternion[0] < 0.0:
        error_quaternion *= -1.0

    vector_norm = np.linalg.norm(error_quaternion[1:])
    if vector_norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vector_norm, error_quaternion[0])
    return angle * error_quaternion[1:] / vector_norm


@dataclass
class PoseTarget:
    position_ned: np.ndarray
    euler_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.position_ned = _vector(self.position_ned, 3, "position_ned")
        self.euler_rpy = _vector(self.euler_rpy, 3, "euler_rpy")


@dataclass
class SixDOFControllerConfig:
    position_kp: np.ndarray = field(
        default_factory=lambda: np.array([0.55, 0.55, 0.45])
    )
    position_ki: np.ndarray = field(
        default_factory=lambda: np.array([0.015, 0.015, 0.02])
    )
    velocity_kp: np.ndarray = field(
        default_factory=lambda: np.array([65.0, 75.0, 85.0])
    )
    attitude_kp: np.ndarray = field(
        default_factory=lambda: np.array([2.0, 2.0, 1.8])
    )
    attitude_ki: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.015])
    )
    angular_velocity_kp: np.ndarray = field(
        default_factory=lambda: np.array([14.0, 18.0, 18.0])
    )
    max_velocity_ned: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 0.65])
    )
    max_angular_velocity: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.65])
    )
    position_integral_limit: np.ndarray = field(
        default_factory=lambda: np.array([5.0, 5.0, 4.0])
    )
    attitude_integral_limit: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5])
    )

    def __post_init__(self):
        vector_fields = (
            "position_kp",
            "position_ki",
            "velocity_kp",
            "attitude_kp",
            "attitude_ki",
            "angular_velocity_kp",
            "max_velocity_ned",
            "max_angular_velocity",
            "position_integral_limit",
            "attitude_integral_limit",
        )
        for name in vector_fields:
            value = _vector(getattr(self, name), 3, name)
            if np.any(value < 0):
                raise ValueError(f"{name} must be non-negative")
            setattr(self, name, value)
        if np.any(self.max_velocity_ned <= 0):
            raise ValueError("max_velocity_ned must be positive")
        if np.any(self.max_angular_velocity <= 0):
            raise ValueError("max_angular_velocity must be positive")


@dataclass
class ControllerOutput:
    desired_wrench_body: np.ndarray
    desired_velocity_ned: np.ndarray
    desired_angular_velocity_body: np.ndarray
    position_error_ned: np.ndarray
    attitude_error_body: np.ndarray


class CascadedSixDOFController:
    """PI outer loops with proportional velocity/rate inner loops."""

    def __init__(self, config=None):
        self.config = config or SixDOFControllerConfig()
        self.position_error_integral = np.zeros(3)
        self.attitude_error_integral = np.zeros(3)

    def reset(self):
        self.position_error_integral.fill(0.0)
        self.attitude_error_integral.fill(0.0)

    def compute(self, state: SixDOFState, target: PoseTarget, dt):
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        cfg = self.config

        position_error = target.position_ned - state.position_ned
        self.position_error_integral = _clip_by_axis(
            self.position_error_integral + position_error * dt,
            cfg.position_integral_limit,
        )
        desired_velocity_ned = _clip_by_axis(
            cfg.position_kp * position_error
            + cfg.position_ki * self.position_error_integral,
            cfg.max_velocity_ned,
        )

        current_velocity_ned = state.rotation_nb @ state.body_velocity[:3]
        velocity_error_ned = desired_velocity_ned - current_velocity_ned
        desired_force_ned = cfg.velocity_kp * velocity_error_ned
        desired_force_body = state.rotation_nb.T @ desired_force_ned

        target_quaternion = euler_to_quaternion(*target.euler_rpy)
        attitude_error = _attitude_error_body(
            state.quaternion_nb,
            target_quaternion,
        )
        self.attitude_error_integral = _clip_by_axis(
            self.attitude_error_integral + attitude_error * dt,
            cfg.attitude_integral_limit,
        )
        desired_angular_velocity = _clip_by_axis(
            cfg.attitude_kp * attitude_error
            + cfg.attitude_ki * self.attitude_error_integral,
            cfg.max_angular_velocity,
        )
        angular_velocity_error = desired_angular_velocity - state.body_velocity[3:]
        desired_moment_body = cfg.angular_velocity_kp * angular_velocity_error

        return ControllerOutput(
            desired_wrench_body=np.concatenate([
                desired_force_body,
                desired_moment_body,
            ]),
            desired_velocity_ned=desired_velocity_ned,
            desired_angular_velocity_body=desired_angular_velocity,
            position_error_ned=position_error,
            attitude_error_body=attitude_error,
        )
