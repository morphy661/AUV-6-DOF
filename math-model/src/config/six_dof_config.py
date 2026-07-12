"""Configuration objects for the nominal six-degree-of-freedom AUV model.

Coordinate convention
---------------------
Navigation frame: North-East-Down (NED), so positive z means increasing depth.
Body frame: x forward, y starboard, z down.
Generalized body velocity: ``[u, v, w, p, q, r]``.
Generalized body wrench: ``[X, Y, Z, K, M, N]``.
"""

from dataclasses import dataclass, field

import numpy as np


def _vector(values, size, name):
    array = np.asarray(values, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _matrix(values, size, name):
    array = np.asarray(values, dtype=float)
    if array.shape == (size,):
        array = np.diag(array)
    if array.shape != (size, size):
        raise ValueError(
            f"{name} must have shape ({size},) or ({size}, {size}), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(array, array.T, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    return array.copy()


def _skew(vector):
    x, y, z = vector
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


@dataclass
class SixDOFConfig:
    """Physical parameters for a first-principles nominal 6-DOF AUV model.

    ``inertia`` is the rigid-body rotational inertia about the body-frame
    origin. Phase 1 assumes that the body origin is the mass-matrix reference
    point. The centre-of-gravity offset is included in the rigid-body mass
    coupling, while gravity and buoyancy offsets generate hydrostatic moments.
    """

    mass: float = 50.0
    inertia: np.ndarray = field(
        default_factory=lambda: np.diag([4.0, 12.0, 12.0])
    )
    added_mass: np.ndarray = field(
        default_factory=lambda: np.diag([5.0, 20.0, 25.0, 0.5, 1.5, 1.5])
    )
    linear_damping: np.ndarray = field(
        default_factory=lambda: np.array([15.0, 30.0, 35.0, 2.0, 5.0, 5.0])
    )
    quadratic_damping: np.ndarray = field(
        default_factory=lambda: np.array([8.0, 18.0, 22.0, 1.0, 2.5, 2.5])
    )
    weight: float = 50.0 * 9.81
    buoyancy: float = 50.0 * 9.81
    center_of_gravity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.02])
    )
    center_of_buoyancy: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -0.02])
    )

    def __post_init__(self):
        self.mass = float(self.mass)
        self.weight = float(self.weight)
        self.buoyancy = float(self.buoyancy)

        if not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("mass must be finite and positive")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("weight must be finite and non-negative")
        if not np.isfinite(self.buoyancy) or self.buoyancy < 0:
            raise ValueError("buoyancy must be finite and non-negative")

        self.inertia = _matrix(self.inertia, 3, "inertia")
        self.added_mass = _matrix(self.added_mass, 6, "added_mass")
        self.linear_damping = _vector(
            self.linear_damping, 6, "linear_damping"
        )
        self.quadratic_damping = _vector(
            self.quadratic_damping, 6, "quadratic_damping"
        )
        self.center_of_gravity = _vector(
            self.center_of_gravity, 3, "center_of_gravity"
        )
        self.center_of_buoyancy = _vector(
            self.center_of_buoyancy, 3, "center_of_buoyancy"
        )

        if np.any(self.linear_damping < 0):
            raise ValueError("linear_damping must be non-negative")
        if np.any(self.quadratic_damping < 0):
            raise ValueError("quadratic_damping must be non-negative")
        if np.min(np.linalg.eigvalsh(self.inertia)) <= 0:
            raise ValueError("inertia must be positive definite")
        if np.min(np.linalg.eigvalsh(self.added_mass)) < -1e-10:
            raise ValueError("added_mass must be positive semidefinite")

    @property
    def rigid_body_mass_matrix(self):
        matrix = np.zeros((6, 6), dtype=float)
        matrix[:3, :3] = self.mass * np.eye(3)
        cg_skew = _skew(self.center_of_gravity)
        matrix[:3, 3:] = -self.mass * cg_skew
        matrix[3:, :3] = self.mass * cg_skew
        matrix[3:, 3:] = self.inertia
        return matrix

    @property
    def total_mass_matrix(self):
        """Return rigid-body plus added-mass inertia."""
        return self.rigid_body_mass_matrix + self.added_mass
