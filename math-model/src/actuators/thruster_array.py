"""Fixed thruster geometry and bounded body-wrench control allocation."""

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


def _vector(values, size, name):
    array = np.asarray(values, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


@dataclass
class ThrusterSpec:
    """Geometry and reversible force limits for one thruster.

    Position and direction are expressed in the AUV body frame. A positive
    force acts along ``direction_body``; a negative force reverses it.
    """

    name: str
    position_body: np.ndarray
    direction_body: np.ndarray
    min_force: float
    max_force: float

    def __post_init__(self):
        self.name = str(self.name)
        if not self.name:
            raise ValueError("thruster name cannot be empty")
        self.position_body = _vector(
            self.position_body, 3, f"{self.name}.position_body"
        )
        direction = _vector(
            self.direction_body, 3, f"{self.name}.direction_body"
        )
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-12:
            raise ValueError(f"{self.name}.direction_body cannot be zero")
        self.direction_body = direction / direction_norm
        self.min_force = float(self.min_force)
        self.max_force = float(self.max_force)
        if not np.isfinite(self.min_force) or not np.isfinite(self.max_force):
            raise ValueError("thruster force limits must be finite")
        if self.min_force >= self.max_force:
            raise ValueError("min_force must be smaller than max_force")

    @property
    def wrench_column(self):
        moment = np.cross(self.position_body, self.direction_body)
        return np.concatenate([self.direction_body, moment])


@dataclass
class AllocationResult:
    desired_wrench: np.ndarray
    achieved_wrench: np.ndarray
    residual_wrench: np.ndarray
    thruster_forces: np.ndarray
    saturated: np.ndarray


class ThrusterArray:
    """Map a desired body wrench to bounded individual thruster forces.

    Zero-valued wrench weights explicitly mark passively stabilised axes. This
    supports vehicles such as KYUBIC/Tuna-Sand2: their dynamics still have six
    degrees of freedom, while the thrusters actively control only surge, sway,
    heave, and yaw.
    """

    def __init__(
        self,
        thrusters: Iterable[ThrusterSpec],
        wrench_weights: Optional[np.ndarray] = None,
    ):
        self.thrusters = list(thrusters)
        if not self.thrusters:
            raise ValueError("at least one thruster is required")
        names = [thruster.name for thruster in self.thrusters]
        if len(set(names)) != len(names):
            raise ValueError("thruster names must be unique")

        self.allocation_matrix = np.column_stack([
            thruster.wrench_column for thruster in self.thrusters
        ])
        self.min_forces = np.array([
            thruster.min_force for thruster in self.thrusters
        ])
        self.max_forces = np.array([
            thruster.max_force for thruster in self.thrusters
        ])
        self.wrench_weights = (
            np.ones(6)
            if wrench_weights is None
            else _vector(wrench_weights, 6, "wrench_weights")
        )
        if np.any(self.wrench_weights < 0):
            raise ValueError("wrench_weights must be non-negative")
        controlled = self.wrench_weights > 0
        if not np.any(controlled):
            raise ValueError("at least one wrench axis must have positive weight")
        controlled_matrix = self.allocation_matrix[controlled, :]
        if np.linalg.matrix_rank(controlled_matrix) < int(np.sum(controlled)):
            raise ValueError(
                "thruster geometry must independently control every weighted axis"
            )

    @property
    def names(self):
        return [thruster.name for thruster in self.thrusters]

    def wrench_from_forces(self, thruster_forces):
        forces = _vector(thruster_forces, len(self.thrusters), "thruster_forces")
        return self.allocation_matrix @ forces

    def allocate(self, desired_wrench):
        """Allocate wrench with an active-set bounded least-squares method."""
        desired = _vector(desired_wrench, 6, "desired_wrench")
        count = len(self.thrusters)
        forces = np.zeros(count)
        free = list(range(count))
        fixed = []
        weight_matrix = np.diag(self.wrench_weights)

        for _ in range(count + 1):
            if not free:
                break

            fixed_wrench = (
                self.allocation_matrix[:, fixed] @ forces[fixed]
                if fixed
                else np.zeros(6)
            )
            remaining = desired - fixed_wrench
            free_matrix = self.allocation_matrix[:, free]
            solution, *_ = np.linalg.lstsq(
                weight_matrix @ free_matrix,
                weight_matrix @ remaining,
                rcond=None,
            )
            forces[free] = solution

            violations = []
            for local_index, thruster_index in enumerate(free):
                value = solution[local_index]
                low = self.min_forces[thruster_index]
                high = self.max_forces[thruster_index]
                if value < low:
                    scale = max(abs(low), 1e-12)
                    violations.append(((low - value) / scale, thruster_index, low))
                elif value > high:
                    scale = max(abs(high), 1e-12)
                    violations.append(((value - high) / scale, thruster_index, high))

            if not violations:
                break

            _, worst_index, bound = max(violations, key=lambda item: item[0])
            forces[worst_index] = bound
            free.remove(worst_index)
            fixed.append(worst_index)

        forces = np.clip(forces, self.min_forces, self.max_forces)
        achieved = self.wrench_from_forces(forces)
        saturated = np.logical_or(
            np.isclose(forces, self.min_forces, atol=1e-9),
            np.isclose(forces, self.max_forces, atol=1e-9),
        )
        return AllocationResult(
            desired_wrench=desired,
            achieved_wrench=achieved,
            residual_wrench=desired - achieved,
            thruster_forces=forces,
            saturated=saturated,
        )


def default_six_thruster_array(
    length=1.2,
    width=0.6,
    horizontal_force_limit=40.0,
    vertical_force_limit=35.0,
):
    """Return a KYUBIC-style four-horizontal/two-vertical AUV layout.

    The actively allocated axes are X, Y, Z, and N (surge, sway, heave, yaw).
    Roll and pitch remain part of the six-DOF dynamics but are stabilised by
    hydrostatic restoring moments instead of direct control allocation.
    """
    half_length = float(length) / 2.0
    half_width = float(width) / 2.0
    if half_length <= 0 or half_width <= 0:
        raise ValueError("length and width must be positive")
    horizontal_force_limit = float(horizontal_force_limit)
    vertical_force_limit = float(vertical_force_limit)
    if horizontal_force_limit <= 0 or vertical_force_limit <= 0:
        raise ValueError("thruster force limits must be positive")

    diagonal = 1.0 / np.sqrt(2.0)
    horizontal_positions = [
        [half_length, half_width, 0.0],
        [half_length, -half_width, 0.0],
        [-half_length, half_width, 0.0],
        [-half_length, -half_width, 0.0],
    ]
    horizontal_directions = [
        [diagonal, -diagonal, 0.0],
        [diagonal, diagonal, 0.0],
        [diagonal, diagonal, 0.0],
        [diagonal, -diagonal, 0.0],
    ]
    vertical_positions = [
        [half_length, 0.0, 0.0],
        [-half_length, 0.0, 0.0],
    ]

    thrusters = []
    for index, (position, direction) in enumerate(
        zip(horizontal_positions, horizontal_directions), start=1
    ):
        thrusters.append(ThrusterSpec(
            name=f"H{index}",
            position_body=np.array(position),
            direction_body=np.array(direction),
            min_force=-horizontal_force_limit,
            max_force=horizontal_force_limit,
        ))

    for index, position in enumerate(vertical_positions, start=1):
        thrusters.append(ThrusterSpec(
            name=f"V{index}",
            position_body=np.array(position),
            direction_body=np.array([0.0, 0.0, 1.0]),
            min_force=-vertical_force_limit,
            max_force=vertical_force_limit,
        ))

    return ThrusterArray(
        thrusters,
        wrench_weights=np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
    )
