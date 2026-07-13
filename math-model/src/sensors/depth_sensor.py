"""Pressure-depth sensor with bias, random walk, and measurement noise."""

from typing import Optional

import numpy as np


class DepthSensor:
    """Measure positive-down depth in metres for the NED convention."""

    def __init__(
        self,
        noise_std: float = 0.05,
        bias: float = 0.0,
        drift_std: float = 0.001,
        seed: Optional[int] = None,
    ):
        self.noise_std = float(noise_std)
        self.bias = float(bias)
        self.drift_std = float(drift_std)
        if self.noise_std < 0.0 or self.drift_std < 0.0:
            raise ValueError("depth noise values must be non-negative")
        if not np.isfinite(self.bias):
            raise ValueError("bias must be finite")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.current_drift = 0.0

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.current_drift = 0.0

    def measure(self, true_depth: float, dt: float = 1.0) -> float:
        true_depth = float(true_depth)
        dt = float(dt)
        if not np.isfinite(true_depth):
            raise ValueError("true_depth must be finite")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        self.current_drift += self.rng.normal(
            0.0,
            self.drift_std * np.sqrt(dt),
        )
        noise = self.rng.normal(0.0, self.noise_std)
        return float(true_depth + self.bias + self.current_drift + noise)
