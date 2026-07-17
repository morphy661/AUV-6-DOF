"""Leakage-safe observable features and labels for six-thruster diagnosis.

Feature extraction and label extraction are deliberately separate.  The
feature path only reads signals that can exist on the vehicle at inference
time.  Simulator truth remains available to evaluation code, but is never
consulted by :func:`extract_six_dof_features`.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from diagnosis.thruster_health_monitor import (
    extract_thruster_health_features,
)


THRUSTER_NAMES = ("H1", "H2", "H3", "H4", "V1", "V2")
FAULT_MODE_NAMES = ("normal", "no_output", "thrust_loss")
FAULT_LOCATION_NAMES = ("none",) + THRUSTER_NAMES
JOINT_FAULT_NAMES = (
    ("Normal",)
    + tuple(f"{name} No Output" for name in THRUSTER_NAMES)
    + tuple(f"{name} Thrust Loss" for name in THRUSTER_NAMES)
)

# These simulator fields may be useful as labels or offline evaluation truth,
# but must never enter a deployable fault-detection feature vector.
PRIVILEGED_SIMULATOR_FIELDS = frozenset({
    "position_ned",
    "euler_rpy",
    "body_velocity",
    "position_error_ned",
    "attitude_error_body",
    "achieved_wrench_body",
    "actuation_residual_body",
    "actual_thruster_forces",
    "thruster_force_efficiencies",
    "thruster_fault_modes",
    "thruster_fault_active",
    "faulted_thruster_index",
    "allocation_thruster_effectiveness",
    "ftc_active",
    "fault_label",
    "true_depth",
})


SIX_DOF_BASE_FEATURE_NAMES = (
    "depth_m",
    "target_depth_m",
    "depth_tracking_error_m",
    "dvl_valid",
    "dvl_u_mps",
    "dvl_v_mps",
    "dvl_w_mps",
    "imu_roll_rad",
    "imu_pitch_rad",
    "imu_yaw_rad",
    "imu_p_radps",
    "imu_q_radps",
    "imu_r_radps",
    "imu_accel_x_mps2",
    "imu_accel_y_mps2",
    "imu_accel_z_mps2",
    "target_roll_rad",
    "target_pitch_rad",
    "target_yaw_rad",
    "attitude_error_roll_rad",
    "attitude_error_pitch_rad",
    "attitude_error_yaw_rad",
    *(f"{name}_command_force_n" for name in THRUSTER_NAMES),
    *(f"{name}_measured_current_a" for name in THRUSTER_NAMES),
    *(f"{name}_current_residual_a" for name in THRUSTER_NAMES),
    *(f"{name}_command_saturated" for name in THRUSTER_NAMES),
)
PHYSICS_RESPONSE_FEATURE_NAMES = (
    "command_wrench_x_n",
    "command_wrench_y_n",
    "command_wrench_z_n",
    "command_moment_k_nm",
    "command_moment_m_nm",
    "command_moment_n_nm",
    "nominal_expected_accel_x_mps2",
    "nominal_expected_accel_y_mps2",
    "nominal_expected_accel_z_mps2",
    "linear_response_residual_x_mps2",
    "linear_response_residual_y_mps2",
    "linear_response_residual_z_mps2",
    "nominal_expected_angular_accel_p_radps2",
    "nominal_expected_angular_accel_q_radps2",
    "nominal_expected_angular_accel_r_radps2",
)
THRUSTER_TELEMETRY_FEATURE_NAMES = (
    *(f"{name}_measured_rpm" for name in THRUSTER_NAMES),
    *(f"{name}_rpm_residual" for name in THRUSTER_NAMES),
    *(f"{name}_measured_voltage_v" for name in THRUSTER_NAMES),
    *(f"{name}_motor_temperature_c" for name in THRUSTER_NAMES),
)
HYBRID_DIAGNOSIS_FEATURE_NAMES = (
    *(f"{name}_current_ratio" for name in THRUSTER_NAMES),
    *(f"{name}_rpm_ratio" for name in THRUSTER_NAMES),
    *(f"{name}_local_anomaly_score" for name in THRUSTER_NAMES),
    *(f"{name}_motion_loss_evidence" for name in THRUSTER_NAMES),
)
SIX_DOF_RAW_FEATURE_NAMES = (
    SIX_DOF_BASE_FEATURE_NAMES
    + PHYSICS_RESPONSE_FEATURE_NAMES
    + THRUSTER_TELEMETRY_FEATURE_NAMES
    + HYBRID_DIAGNOSIS_FEATURE_NAMES
)
SIX_DOF_RAW_FEATURE_DIM = len(SIX_DOF_RAW_FEATURE_NAMES)
SIX_DOF_MODEL_INPUT_DIM = 2 * SIX_DOF_RAW_FEATURE_DIM

# Diagonal of the nominal rigid-body plus added-mass matrix.  It is an onboard
# model parameter, not simulator truth from the randomized mission.
NOMINAL_GENERALIZED_MASS_DIAGONAL = np.array([
    55.0, 70.0, 75.0, 4.5, 13.5, 13.5
])


@dataclass(frozen=True)
class SixDOFFaultLabels:
    """Multi-task labels plus a convenient 13-class comparison label."""

    mode: int
    location: int
    joint: int


def _finite_float(value: Any, default: float = 0.0) -> float:
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


def _angle_error(target: np.ndarray, measured: np.ndarray) -> np.ndarray:
    difference = target - measured
    return np.arctan2(np.sin(difference), np.cos(difference))


def _validate_thruster_order(log: Mapping[str, Any]) -> None:
    names = tuple(log.get("thruster_names", THRUSTER_NAMES))
    if names != THRUSTER_NAMES:
        raise ValueError(
            f"expected thruster order {THRUSTER_NAMES}, got {names}"
        )


def extract_six_dof_features(
    log: Mapping[str, Any],
    previous_log: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return 109 raw features using only inference-time observable signals.

    DVL dropout is represented by a validity bit and zero-filled velocity.  It
    never falls back to the simulator's true body velocity.  Likewise, attitude
    only comes from the IMU packet, not ``euler_rpy`` simulation truth.
    """

    _validate_thruster_order(log)

    depth = _finite_float(log.get("depth", 0.0))
    target_position = _finite_vector(log.get("target_position_ned"), 3)
    target_attitude = _finite_vector(log.get("target_euler_rpy"), 3)

    dvl = log.get("dvl", {})
    dvl = dvl if isinstance(dvl, Mapping) else {}
    dvl_valid = bool(dvl.get("valid", False))
    if dvl_valid:
        dvl_velocity = _finite_vector(
            dvl.get("velocity", [
                dvl.get("vx", 0.0),
                dvl.get("vy", 0.0),
                dvl.get("vz", 0.0),
            ]),
            3,
        )
    else:
        dvl_velocity = np.zeros(3)

    imu = log.get("imu", {})
    imu = imu if isinstance(imu, Mapping) else {}
    imu_attitude = _finite_vector(
        imu.get("orientation", [
            imu.get("roll", 0.0),
            imu.get("pitch", 0.0),
            imu.get("yaw", 0.0),
        ]),
        3,
    )
    angular_velocity = _finite_vector(imu.get("angular_velocity"), 3)
    acceleration = _finite_vector(imu.get("linear_acceleration"), 3)

    commanded_forces = _finite_vector(
        log.get("commanded_thruster_forces", log.get("thruster_forces")),
        len(THRUSTER_NAMES),
    )
    measured_currents = _finite_vector(
        log.get("thruster_measured_currents"),
        len(THRUSTER_NAMES),
    )
    expected_currents = _finite_vector(
        log.get("thruster_expected_currents"),
        len(THRUSTER_NAMES),
    )
    current_residuals = measured_currents - expected_currents
    measured_rpms = _finite_vector(
        log.get("thruster_measured_rpms"),
        len(THRUSTER_NAMES),
    )
    expected_rpms = _finite_vector(
        log.get("thruster_expected_rpms"),
        len(THRUSTER_NAMES),
    )
    rpm_residuals = measured_rpms - expected_rpms
    measured_voltages = _finite_vector(
        log.get("thruster_measured_voltages"),
        len(THRUSTER_NAMES),
        default=48.0,
    )
    measured_temperatures = _finite_vector(
        log.get("thruster_measured_temperatures"),
        len(THRUSTER_NAMES),
        default=20.0,
    )
    saturated = _finite_vector(
        log.get("thruster_saturated"),
        len(THRUSTER_NAMES),
    )
    commanded_wrench = _finite_vector(
        log.get("allocated_wrench_body", log.get("desired_wrench_body")),
        6,
    )
    nominal_expected_acceleration = (
        commanded_wrench / NOMINAL_GENERALIZED_MASS_DIAGONAL
    )
    linear_response_residual = (
        acceleration - nominal_expected_acceleration[:3]
    )
    health = extract_thruster_health_features(log, previous_log=previous_log)

    features = np.concatenate([
        np.array([
            depth,
            target_position[2],
            target_position[2] - depth,
            float(dvl_valid),
        ]),
        dvl_velocity,
        imu_attitude,
        angular_velocity,
        acceleration,
        target_attitude,
        _angle_error(target_attitude, imu_attitude),
        commanded_forces,
        measured_currents,
        current_residuals,
        saturated,
        commanded_wrench,
        nominal_expected_acceleration[:3],
        linear_response_residual,
        nominal_expected_acceleration[3:],
        measured_rpms,
        rpm_residuals,
        measured_voltages,
        measured_temperatures,
        health.current_ratio,
        health.rpm_ratio,
        health.local_anomaly_score,
        health.motion_loss_evidence,
    ]).astype(np.float32)

    if features.shape != (SIX_DOF_RAW_FEATURE_DIM,):
        raise RuntimeError(
            f"six-DOF feature shape {features.shape} does not match "
            f"({SIX_DOF_RAW_FEATURE_DIM},)"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("six-DOF feature vector contains non-finite values")
    return features


def extract_six_dof_fault_labels(
    log: Mapping[str, Any],
) -> SixDOFFaultLabels:
    """Extract labels from simulator truth on a path isolated from features."""

    _validate_thruster_order(log)
    modes: Sequence[str] = tuple(log.get("thruster_fault_modes", ()))
    if len(modes) != len(THRUSTER_NAMES):
        raise ValueError(
            "thruster_fault_modes must contain one label per thruster"
        )

    active = [
        (index, str(mode))
        for index, mode in enumerate(modes)
        if str(mode) != "normal"
    ]
    if not active:
        return SixDOFFaultLabels(mode=0, location=0, joint=0)
    if len(active) != 1:
        raise ValueError("only single-thruster fault labels are supported")

    thruster_index, mode_name = active[0]
    if mode_name not in FAULT_MODE_NAMES[1:]:
        raise ValueError(f"unsupported six-DOF fault mode {mode_name!r}")
    mode_label = FAULT_MODE_NAMES.index(mode_name)
    location_label = thruster_index + 1
    joint_label = (
        location_label
        if mode_name == "no_output"
        else len(THRUSTER_NAMES) + location_label
    )
    return SixDOFFaultLabels(
        mode=mode_label,
        location=location_label,
        joint=joint_label,
    )
