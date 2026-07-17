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
    expected_rpms: np.ndarray
    measured_rpms: np.ndarray
    measured_voltages: np.ndarray
    measured_temperatures: np.ndarray
    telemetry_valid: np.ndarray
    telemetry_age_s: np.ndarray
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
        current_noise_std=0.0,
        max_rpm=3500.0,
        no_output_rpm_fraction=0.02,
        rpm_noise_std=0.0,
        nominal_voltage=48.0,
        voltage_droop_per_amp=0.02,
        voltage_noise_std=0.0,
        ambient_temperature=20.0,
        full_load_temperature_rise=30.0,
        thermal_time_constant=45.0,
        temperature_noise_std=0.0,
        seed=None,
    ):
        self.thruster_array = thruster_array
        self.fault = fault
        self.idle_current = float(idle_current)
        self.current_gain = float(current_gain)
        self.no_output_current_fraction = float(no_output_current_fraction)
        self.current_noise_std = float(current_noise_std)
        self.max_rpm = float(max_rpm)
        self.no_output_rpm_fraction = float(no_output_rpm_fraction)
        self.rpm_noise_std = float(rpm_noise_std)
        self.nominal_voltage = float(nominal_voltage)
        self.voltage_droop_per_amp = float(voltage_droop_per_amp)
        self.voltage_noise_std = float(voltage_noise_std)
        self.ambient_temperature = float(ambient_temperature)
        self.full_load_temperature_rise = float(full_load_temperature_rise)
        self.thermal_time_constant = float(thermal_time_constant)
        self.temperature_noise_std = float(temperature_noise_std)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if self.idle_current < 0 or self.current_gain < 0:
            raise ValueError("current model parameters must be non-negative")
        if not 0.0 <= self.no_output_current_fraction <= 1.0:
            raise ValueError("no_output_current_fraction must be in [0, 1]")
        if not np.isfinite(self.current_noise_std) or self.current_noise_std < 0:
            raise ValueError("current_noise_std must be finite and non-negative")
        positive_parameters = {
            "max_rpm": self.max_rpm,
            "nominal_voltage": self.nominal_voltage,
            "thermal_time_constant": self.thermal_time_constant,
        }
        for name, value in positive_parameters.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        bounded_parameters = {
            "no_output_rpm_fraction": self.no_output_rpm_fraction,
        }
        for name, value in bounded_parameters.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        non_negative_parameters = {
            "rpm_noise_std": self.rpm_noise_std,
            "voltage_droop_per_amp": self.voltage_droop_per_amp,
            "voltage_noise_std": self.voltage_noise_std,
            "full_load_temperature_rise": self.full_load_temperature_rise,
            "temperature_noise_std": self.temperature_noise_std,
        }
        for name, value in non_negative_parameters.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(self.ambient_temperature):
            raise ValueError("ambient_temperature must be finite")

        self._motor_temperatures = np.full(
            len(self.thruster_array.thrusters),
            self.ambient_temperature,
            dtype=float,
        )

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

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self._motor_temperatures.fill(self.ambient_temperature)

    def _command_activity(self, commanded_forces):
        force_limits = np.maximum(
            np.abs(self.thruster_array.min_forces),
            np.abs(self.thruster_array.max_forces),
        )
        return np.clip(np.abs(commanded_forces) / force_limits, 0.0, 1.0)

    def _expected_currents(self, commanded_forces):
        activity = self._command_activity(commanded_forces)
        return np.where(
            activity > 1e-12,
            self.idle_current + self.current_gain * activity,
            0.0,
        )

    def _expected_rpms(self, commanded_forces):
        activity = self._command_activity(commanded_forces)
        return (
            np.sign(commanded_forces)
            * self.max_rpm
            * np.sqrt(activity)
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

    def apply(self, commanded_forces, time_s, dt=0.05):
        commanded = np.asarray(commanded_forces, dtype=float)
        expected_shape = (len(self.thruster_array.thrusters),)
        if commanded.shape != expected_shape:
            raise ValueError(
                f"commanded_forces must have shape {expected_shape}, "
                f"got {commanded.shape}"
            )
        if not np.all(np.isfinite(commanded)):
            raise ValueError("commanded_forces must be finite")
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        actual = commanded.copy()
        expected_currents = self._expected_currents(commanded)
        measured_currents = expected_currents.copy()
        expected_rpms = self._expected_rpms(commanded)
        measured_rpms = expected_rpms.copy()
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
                measured_rpms[index] = (
                    self.no_output_rpm_fraction * expected_rpms[index]
                )
            elif self.fault.mode is SixDOFThrusterFaultMode.THRUST_LOSS:
                efficiencies[index] = self.fault.thrust_efficiency
                actual[index] = self.fault.thrust_efficiency * commanded[index]

        if self.current_noise_std > 0.0:
            measured_currents += self.rng.normal(
                0.0,
                self.current_noise_std,
                size=len(measured_currents),
            )
            measured_currents = np.maximum(measured_currents, 0.0)

        if self.rpm_noise_std > 0.0:
            measured_rpms += self.rng.normal(
                0.0,
                self.rpm_noise_std,
                size=len(measured_rpms),
            )

        bus_voltage = (
            self.nominal_voltage
            - self.voltage_droop_per_amp * float(np.sum(measured_currents))
        )
        measured_voltages = np.full(len(commanded), bus_voltage, dtype=float)
        if self.voltage_noise_std > 0.0:
            measured_voltages += self.rng.normal(
                0.0,
                self.voltage_noise_std,
                size=len(measured_voltages),
            )
        measured_voltages = np.maximum(measured_voltages, 0.0)

        rated_current = max(self.idle_current + self.current_gain, 1e-9)
        current_load = np.clip(measured_currents / rated_current, 0.0, 1.5)
        target_temperatures = (
            self.ambient_temperature
            + self.full_load_temperature_rise * current_load ** 2
        )
        thermal_fraction = 1.0 - np.exp(-dt / self.thermal_time_constant)
        self._motor_temperatures += thermal_fraction * (
            target_temperatures - self._motor_temperatures
        )
        measured_temperatures = self._motor_temperatures.copy()
        if self.temperature_noise_std > 0.0:
            measured_temperatures += self.rng.normal(
                0.0,
                self.temperature_noise_std,
                size=len(measured_temperatures),
            )

        return ThrusterActuationResult(
            commanded_forces=commanded.copy(),
            actual_forces=actual,
            expected_currents=expected_currents,
            measured_currents=measured_currents,
            expected_rpms=expected_rpms,
            measured_rpms=measured_rpms,
            measured_voltages=measured_voltages,
            measured_temperatures=measured_temperatures,
            telemetry_valid=np.ones(len(commanded), dtype=bool),
            telemetry_age_s=np.zeros(len(commanded), dtype=float),
            force_efficiencies=efficiencies,
            fault_modes=tuple(modes),
            fault_active=bool(fault_active),
            faulted_thruster_index=self._fault_index,
        )
