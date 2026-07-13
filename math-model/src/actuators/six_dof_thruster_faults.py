"""Single-thruster actuator faults and deterministic electrical telemetry."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from actuators.thruster_array import ThrusterArray


class SixDOFThrusterFaultMode(str, Enum):
    NORMAL = "normal"
    NO_OUTPUT = "no_output"
    THRUST_LOSS = "thrust_loss"


@dataclass
class SingleThrusterFault:
    """A single fault activated at a configured simulation time."""

    thruster_name: str
    mode: SixDOFThrusterFaultMode
    start_time: float
    thrust_efficiency: float = 0.45

    def __post_init__(self):
        self.thruster_name = str(self.thruster_name)
        if not self.thruster_name:
            raise ValueError("thruster_name cannot be empty")
        self.mode = SixDOFThrusterFaultMode(self.mode)
        self.start_time = float(self.start_time)
        self.thrust_efficiency = float(self.thrust_efficiency)
        if not np.isfinite(self.start_time) or self.start_time < 0:
            raise ValueError("start_time must be finite and non-negative")
        if not 0.0 <= self.thrust_efficiency <= 1.0:
            raise ValueError("thrust_efficiency must be between zero and one")

    def is_active(self, time_s):
        return (
            self.mode is not SixDOFThrusterFaultMode.NORMAL
            and float(time_s) >= self.start_time
        )


@dataclass
class ThrusterActuationResult:
    commanded_forces: np.ndarray
    actual_forces: np.ndarray
    expected_currents: np.ndarray
    measured_currents: np.ndarray
    force_efficiencies: np.ndarray
    fault_modes: tuple
    fault_active: bool
    faulted_thruster_index: Optional[int]


class ThrusterActuatorBank:
    """Convert allocated forces into actual forces and motor-current signals.

    The first version is deterministic so six-degree-of-freedom coupling can be
    validated before sensor noise is added. ``NO_OUTPUT`` removes force and
    produces near-zero current. ``THRUST_LOSS`` reduces useful force while
    keeping motor current close to its expected value.
    """

    def __init__(
        self,
        thruster_array: ThrusterArray,
        fault: Optional[SingleThrusterFault] = None,
        idle_current=0.4,
        current_gain=8.0,
        no_output_current_fraction=0.03,
    ):
        self.thruster_array = thruster_array
        self.fault = fault
        self.idle_current = float(idle_current)
        self.current_gain = float(current_gain)
        self.no_output_current_fraction = float(no_output_current_fraction)
        if self.idle_current < 0 or self.current_gain < 0:
            raise ValueError("current model parameters must be non-negative")
        if not 0.0 <= self.no_output_current_fraction <= 1.0:
            raise ValueError("no_output_current_fraction must be in [0, 1]")

        self._fault_index = None
        if fault is not None:
            if fault.thruster_name not in self.thruster_array.names:
                raise ValueError(
                    f"unknown thruster {fault.thruster_name!r}; "
                    f"available={self.thruster_array.names}"
                )
            self._fault_index = self.thruster_array.names.index(
                fault.thruster_name
            )

    def _expected_currents(self, commanded_forces):
        force_limits = np.maximum(
            np.abs(self.thruster_array.min_forces),
            np.abs(self.thruster_array.max_forces),
        )
        activity = np.abs(commanded_forces) / force_limits
        return np.where(
            activity > 1e-12,
            self.idle_current + self.current_gain * activity,
            0.0,
        )

    def force_efficiencies_at(self, time_s):
        """Return the true per-thruster effectiveness for oracle FTC studies."""
        efficiencies = np.ones(len(self.thruster_array.thrusters))
        if self.fault is None or not self.fault.is_active(time_s):
            return efficiencies
        if self.fault.mode is SixDOFThrusterFaultMode.NO_OUTPUT:
            efficiencies[self._fault_index] = 0.0
        elif self.fault.mode is SixDOFThrusterFaultMode.THRUST_LOSS:
            efficiencies[self._fault_index] = self.fault.thrust_efficiency
        return efficiencies

    def apply(self, commanded_forces, time_s):
        commanded = np.asarray(commanded_forces, dtype=float)
        expected_shape = (len(self.thruster_array.thrusters),)
        if commanded.shape != expected_shape:
            raise ValueError(
                f"commanded_forces must have shape {expected_shape}, "
                f"got {commanded.shape}"
            )
        if not np.all(np.isfinite(commanded)):
            raise ValueError("commanded_forces must be finite")

        actual = commanded.copy()
        expected_currents = self._expected_currents(commanded)
        measured_currents = expected_currents.copy()
        efficiencies = np.ones_like(commanded)
        modes = [SixDOFThrusterFaultMode.NORMAL.value] * len(commanded)
        fault_active = self.fault is not None and self.fault.is_active(time_s)

        if fault_active:
            index = self._fault_index
            modes[index] = self.fault.mode.value
            if self.fault.mode is SixDOFThrusterFaultMode.NO_OUTPUT:
                efficiencies[index] = 0.0
                actual[index] = 0.0
                measured_currents[index] = (
                    self.no_output_current_fraction * expected_currents[index]
                )
            elif self.fault.mode is SixDOFThrusterFaultMode.THRUST_LOSS:
                efficiencies[index] = self.fault.thrust_efficiency
                actual[index] = self.fault.thrust_efficiency * commanded[index]

        return ThrusterActuationResult(
            commanded_forces=commanded.copy(),
            actual_forces=actual,
            expected_currents=expected_currents,
            measured_currents=measured_currents,
            force_efficiencies=efficiencies,
            fault_modes=tuple(modes),
            fault_active=bool(fault_active),
            faulted_thruster_index=self._fault_index,
        )
