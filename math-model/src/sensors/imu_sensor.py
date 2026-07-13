"""IMU measurements compatible with legacy and six-DOF AUV states."""

from typing import Optional

import numpy as np


def _vector(values, size, name):
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return array.copy()


class IMUSensor:
    """Measure body attitude, angular rate, and linear acceleration.

    ``SixDOFState`` exposes ``euler_rpy`` and ``body_velocity``. The older AUV
    model exposes ``orientation``, ``velocity``, and ``angular_velocity``; both
    forms remain supported. Linear acceleration can be supplied by the dynamics
    or estimated by finite differencing consecutive linear velocities.
    """

    def __init__(
        self,
        attitude_noise_std=0.002,
        gyro_noise_std=0.001,
        accel_noise_std=0.01,
        seed: Optional[int] = None,
    ):
        self.attitude_noise_std = float(attitude_noise_std)
        self.gyro_noise_std = float(gyro_noise_std)
        self.accel_noise_std = float(accel_noise_std)
        if min(
            self.attitude_noise_std,
            self.gyro_noise_std,
            self.accel_noise_std,
        ) < 0.0:
            raise ValueError("IMU noise standard deviations must be non-negative")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._previous_linear_velocity = None

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self._previous_linear_velocity = None

    @staticmethod
    def _true_orientation(auv_state):
        if hasattr(auv_state, "euler_rpy"):
            return _vector(auv_state.euler_rpy, 3, "euler_rpy")
        if hasattr(auv_state, "orientation"):
            return _vector(auv_state.orientation, 3, "orientation")
        return np.array([
            float(getattr(auv_state, "roll", 0.0)),
            float(getattr(auv_state, "pitch", 0.0)),
            float(getattr(auv_state, "yaw", 0.0)),
        ])

    @staticmethod
    def _true_velocities(auv_state):
        if hasattr(auv_state, "body_velocity"):
            body_velocity = _vector(
                auv_state.body_velocity, 6, "body_velocity"
            )
            return body_velocity[:3], body_velocity[3:]
        linear = _vector(
            getattr(auv_state, "velocity", np.zeros(3)),
            3,
            "velocity",
        )
        angular = _vector(
            getattr(auv_state, "angular_velocity", np.zeros(3)),
            3,
            "angular_velocity",
        )
        return linear, angular

    def read(
        self,
        auv_state,
        linear_acceleration_body=None,
        dt=None,
    ):
        orientation = self._true_orientation(auv_state)
        linear_velocity, angular_velocity = self._true_velocities(auv_state)

        if linear_acceleration_body is not None:
            acceleration = _vector(
                linear_acceleration_body,
                3,
                "linear_acceleration_body",
            )
        elif self._previous_linear_velocity is not None and dt is not None:
            dt = float(dt)
            if not np.isfinite(dt) or dt <= 0.0:
                raise ValueError("dt must be finite and positive")
            acceleration = (
                linear_velocity - self._previous_linear_velocity
            ) / dt
        else:
            acceleration = _vector(
                getattr(auv_state, "linear_acceleration", np.zeros(3)),
                3,
                "linear_acceleration",
            )
        self._previous_linear_velocity = linear_velocity.copy()

        measured_orientation = orientation + self.rng.normal(
            0.0, self.attitude_noise_std, size=3
        )
        measured_angular_velocity = angular_velocity + self.rng.normal(
            0.0, self.gyro_noise_std, size=3
        )
        measured_acceleration = acceleration + self.rng.normal(
            0.0, self.accel_noise_std, size=3
        )
        return {
            "frame": "body",
            "orientation": measured_orientation,
            "roll": float(measured_orientation[0]),
            "pitch": float(measured_orientation[1]),
            "yaw": float(measured_orientation[2]),
            "angular_velocity": measured_angular_velocity,
            "linear_acceleration": measured_acceleration,
        }
