"""Deployable local-telemetry and vehicle-response thruster health features.

The monitor deliberately uses only signals available onboard: commands, ESC
telemetry, calibrated thruster geometry, IMU acceleration, and IMU angular
rate.  Simulator-only actual force, efficiency, and fault labels are never
read here.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from actuators.thruster_array import default_six_thruster_array


THRUSTER_COUNT = 6
NOMINAL_GENERALIZED_MASS_DIAGONAL = np.array([
    55.0, 70.0, 75.0, 4.5, 13.5, 13.5,
])
NOMINAL_WRENCH_SCALE = np.array([40.0, 40.0, 35.0, 5.0, 22.0, 20.0])
DEFAULT_ALLOCATION_MATRIX = default_six_thruster_array().allocation_matrix


@dataclass(frozen=True)
class ThrusterHealthFeatures:
    """Six-channel diagnostic evidence from local and vehicle sensors."""

    current_ratio: np.ndarray
    rpm_ratio: np.ndarray
    local_anomaly_score: np.ndarray
    motion_loss_evidence: np.ndarray


def _finite_float(value: Any, default: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return float(default)
    return converted if np.isfinite(converted) else float(default)


def _finite_vector(value: Any, size: int, default: float = 0.0) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.full(size, default, dtype=float)
    if array.shape != (size,):
        return np.full(size, default, dtype=float)
    return np.where(np.isfinite(array), array, default).astype(float)


def _allocation_matrix(log: Mapping[str, Any]) -> np.ndarray:
    try:
        matrix = np.asarray(
            log.get("thruster_allocation_matrix", DEFAULT_ALLOCATION_MATRIX),
            dtype=float,
        )
    except (TypeError, ValueError):
        return DEFAULT_ALLOCATION_MATRIX
    if matrix.shape != (6, THRUSTER_COUNT) or not np.all(np.isfinite(matrix)):
        return DEFAULT_ALLOCATION_MATRIX
    return matrix


def _imu_packet(log: Mapping[str, Any]) -> Mapping[str, Any]:
    imu = log.get("imu", {})
    return imu if isinstance(imu, Mapping) else {}


def _local_telemetry_features(log: Mapping[str, Any]):
    commands = _finite_vector(
        log.get("commanded_thruster_forces", log.get("thruster_forces")),
        THRUSTER_COUNT,
    )
    force_limits = np.maximum(
        _finite_vector(log.get("thruster_force_limits"), THRUSTER_COUNT, 40.0),
        1e-6,
    )
    command_activity = np.abs(commands) / force_limits
    command_active = command_activity >= 0.08

    current_available = (
        "thruster_expected_currents" in log
        and "thruster_measured_currents" in log
    )
    expected_currents = _finite_vector(
        log.get("thruster_expected_currents"), THRUSTER_COUNT
    )
    measured_currents = _finite_vector(
        log.get("thruster_measured_currents"), THRUSTER_COUNT
    )
    rpm_available = (
        "thruster_expected_rpms" in log
        and "thruster_measured_rpms" in log
    )
    expected_rpms = _finite_vector(
        log.get("thruster_expected_rpms"), THRUSTER_COUNT
    )
    measured_rpms = _finite_vector(
        log.get("thruster_measured_rpms"), THRUSTER_COUNT
    )

    current_ratio = np.ones(THRUSTER_COUNT)
    rpm_ratio = np.ones(THRUSTER_COUNT)
    if current_available:
        current_ratio[command_active] = (
            np.abs(measured_currents[command_active])
            / np.maximum(np.abs(expected_currents[command_active]), 1e-6)
        )
    if rpm_available:
        rpm_ratio[command_active] = (
            np.abs(measured_rpms[command_active])
            / np.maximum(np.abs(expected_rpms[command_active]), 1e-6)
        )
    current_ratio = np.clip(current_ratio, 0.0, 3.0)
    rpm_ratio = np.clip(rpm_ratio, 0.0, 3.0)

    current_deficit = np.clip((0.60 - current_ratio) / 0.60, 0.0, 1.0)
    rpm_deficit = np.clip((0.60 - rpm_ratio) / 0.60, 0.0, 1.0)
    overcurrent = np.clip((current_ratio - 1.35) / 0.65, 0.0, 1.0)

    measured_voltages = _finite_vector(
        log.get("thruster_measured_voltages"), THRUSTER_COUNT, 48.0
    )
    nominal_voltage = max(
        _finite_float(log.get("thruster_nominal_voltage"), 48.0), 1e-6
    )
    voltage_ratio = measured_voltages / nominal_voltage
    undervoltage = np.clip((0.85 - voltage_ratio) / 0.15, 0.0, 1.0)

    temperatures = _finite_vector(
        log.get("thruster_measured_temperatures"), THRUSTER_COUNT, 20.0
    )
    ambient_temperature = _finite_float(
        log.get("thruster_ambient_temperature"), 20.0
    )
    overtemperature = np.clip(
        (temperatures - (ambient_temperature + 45.0)) / 20.0,
        0.0,
        1.0,
    )

    local_anomaly = np.maximum.reduce([
        0.5 * (current_deficit + rpm_deficit),
        overcurrent,
        undervoltage,
        overtemperature,
    ])
    local_anomaly[~command_active] = 0.0
    return commands, force_limits, current_ratio, rpm_ratio, local_anomaly


def _motion_loss_evidence(
    log: Mapping[str, Any],
    previous_log: Optional[Mapping[str, Any]],
    commands: np.ndarray,
    force_limits: np.ndarray,
) -> np.ndarray:
    if previous_log is None:
        return np.zeros(THRUSTER_COUNT)

    dt = _finite_float(log.get("time"), 0.0) - _finite_float(
        previous_log.get("time"), 0.0
    )
    if dt <= 1e-9:
        return np.zeros(THRUSTER_COUNT)

    imu = _imu_packet(log)
    previous_imu = _imu_packet(previous_log)
    linear_acceleration = _finite_vector(
        imu.get("linear_acceleration"), 3
    )
    angular_velocity = _finite_vector(imu.get("angular_velocity"), 3)
    previous_angular_velocity = _finite_vector(
        previous_imu.get("angular_velocity"), 3
    )
    angular_acceleration = (
        angular_velocity - previous_angular_velocity
    ) / dt
    observed_acceleration = np.concatenate([
        linear_acceleration,
        angular_acceleration,
    ])
    commanded_wrench = _finite_vector(
        log.get("allocated_wrench_body", log.get("desired_wrench_body")),
        6,
    )
    residual_wrench = (
        commanded_wrench
        - NOMINAL_GENERALIZED_MASS_DIAGONAL * observed_acceleration
    )
    residual_scaled = residual_wrench / NOMINAL_WRENCH_SCALE

    matrix = _allocation_matrix(log)
    evidence = np.zeros(THRUSTER_COUNT)
    activity = np.abs(commands) / np.maximum(force_limits, 1e-6)
    for index in range(THRUSTER_COUNT):
        if activity[index] < 0.08:
            continue
        signature = (
            matrix[:, index]
            * force_limits[index]
            * np.sign(commands[index])
            / NOMINAL_WRENCH_SCALE
        )
        denominator = float(signature @ signature)
        if denominator > 1e-12:
            evidence[index] = float(residual_scaled @ signature) / denominator
    return np.clip(evidence, 0.0, 2.0)


def extract_thruster_health_features(
    log: Mapping[str, Any],
    previous_log: Optional[Mapping[str, Any]] = None,
) -> ThrusterHealthFeatures:
    """Return local telemetry ratios and motion-confirmation evidence."""

    (
        commands,
        force_limits,
        current_ratio,
        rpm_ratio,
        local_anomaly,
    ) = _local_telemetry_features(log)
    motion_evidence = _motion_loss_evidence(
        log,
        previous_log,
        commands,
        force_limits,
    )
    return ThrusterHealthFeatures(
        current_ratio=current_ratio.astype(np.float32),
        rpm_ratio=rpm_ratio.astype(np.float32),
        local_anomaly_score=local_anomaly.astype(np.float32),
        motion_loss_evidence=motion_evidence.astype(np.float32),
    )
