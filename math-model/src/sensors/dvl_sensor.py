"""DVL velocity measurements for legacy and six-DOF AUV states."""

from typing import Optional

import numpy as np


class DVLSensor:
    """Measure three-axis body velocity with optional dropout and noise."""

    def __init__(
        self,
        velocity_noise_std=0.02,
        dropout_prob=0.0,
        seed: Optional[int] = None,
    ):
        self.velocity_noise_std = float(velocity_noise_std)
        self.dropout_prob = float(dropout_prob)
        if self.velocity_noise_std < 0.0:
            raise ValueError("velocity_noise_std must be non-negative")
        if not 0.0 <= self.dropout_prob <= 1.0:
            raise ValueError("dropout_prob must be within [0, 1]")
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def _true_velocity(auv_state):
        if hasattr(auv_state, "body_velocity"):
            velocity = np.asarray(auv_state.body_velocity, dtype=float)
            if velocity.shape != (6,):
                raise ValueError("body_velocity must have shape (6,)")
            velocity = velocity[:3]
        else:
            velocity = np.asarray(
                getattr(auv_state, "velocity", np.zeros(3)),
                dtype=float,
            )
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
            raise ValueError("DVL velocity must be a finite vector with shape (3,)")
        return velocity.copy()

    def read(self, auv_state):
        if self.rng.random() < self.dropout_prob:
            return {
                "valid": False,
                "frame": "body",
                "velocity": np.full(3, np.nan),
                "vx": np.nan,
                "vy": np.nan,
                "vz": np.nan,
                "speed": np.nan,
            }

        velocity = self._true_velocity(auv_state)
        measured_velocity = velocity + self.rng.normal(
            0.0,
            self.velocity_noise_std,
            size=3,
        )
        return {
            "valid": True,
            "frame": "body",
            "velocity": measured_velocity,
            "vx": float(measured_velocity[0]),
            "vy": float(measured_velocity[1]),
            "vz": float(measured_velocity[2]),
            "speed": float(np.linalg.norm(measured_velocity)),
        }
