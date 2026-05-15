import numpy as np


RAW_FEATURE_NAMES = [
    # Depth / control intention
    "depth",
    "target_z",
    "tracking_error",

    # Thruster / vertical motion
    "cmd_vz",
    "actual_vz",
    "velocity_residual",

    # Motor current
    "measured_current",
    "expected_current",
    "current_residual",

    # DVL velocity
    "dvl_vx",
    "dvl_vy",
    "dvl_vz",
    "dvl_speed",

    # IMU attitude and acceleration
    "imu_roll",
    "imu_pitch",
    "imu_yaw",
    "imu_acc_x",
    "imu_acc_y",
    "imu_acc_z",

    # Battery state
    "battery_voltage",
]


RAW_FEATURE_DIM = len(RAW_FEATURE_NAMES)
MODEL_INPUT_DIM = RAW_FEATURE_DIM * 2  # raw features + first-order temporal differences


def safe_float(value, default=0.0):
    """Convert a value to a finite float with a safe fallback."""
    try:
        value = float(value)
        if np.isfinite(value):
            return value
        return default
    except (TypeError, ValueError):
        return default


def _safe_array_get(array_like, index, default=0.0):
    """Safely read one value from a list / tuple / numpy array."""
    try:
        return safe_float(array_like[index], default=default)
    except (TypeError, IndexError, KeyError):
        return default


def extract_ai_features(log):
    """
    Extract the Stage-2 multi-sensor AI feature vector from one simulator log packet.

    This function must be shared by:
    1. dataset generation
    2. model training
    3. online inference in main.py

    Raw feature dimension: 20
    Final model input dimension after adding temporal differences: 40
    """

    # --------------------------------------------------
    # 1. Depth and tracking intention
    # --------------------------------------------------
    depth = safe_float(log.get("depth", 0.0))
    target_z = safe_float(log.get("target_z", 0.0))
    tracking_error = target_z - depth

    # --------------------------------------------------
    # 2. Thruster and vertical velocity
    # --------------------------------------------------
    thruster = log.get("thruster", {}) or {}

    cmd_vz = safe_float(thruster.get("cmd_vz", 0.0))
    actual_vz = safe_float(thruster.get("actual_vz", 0.0))
    velocity_residual = cmd_vz - actual_vz

    # --------------------------------------------------
    # 3. Current sensor
    # Prefer independent CurrentSensor output when available.
    # Fall back to the synchronized thruster fields otherwise.
    # --------------------------------------------------
    current_sensor = log.get("current_sensor", {}) or {}

    measured_current = safe_float(
        current_sensor.get("measured_current", thruster.get("current", 0.0))
    )

    expected_current = safe_float(
        current_sensor.get("expected_current", thruster.get("expected_current", 0.0))
    )

    current_residual = safe_float(
        current_sensor.get("current_residual", measured_current - expected_current)
    )

    # --------------------------------------------------
    # 4. DVL sensor
    # If DVL is invalid or missing, use conservative fallback values.
    # --------------------------------------------------
    dvl = log.get("dvl", {}) or {}

    if isinstance(dvl, dict) and dvl.get("valid", False):
        dvl_vx = safe_float(dvl.get("vx", 0.0))
        dvl_vy = safe_float(dvl.get("vy", 0.0))
        dvl_vz = safe_float(dvl.get("vz", actual_vz))
        dvl_speed = safe_float(dvl.get("speed", abs(actual_vz)))
    else:
        dvl_vx = 0.0
        dvl_vy = 0.0
        dvl_vz = actual_vz
        dvl_speed = abs(actual_vz)

    # --------------------------------------------------
    # 5. IMU sensor
    # Support both the new flat IMU packet and the old compatibility packet.
    # --------------------------------------------------
    imu = log.get("imu", {}) or {}

    imu_roll = safe_float(imu.get("roll", 0.0))
    imu_pitch = safe_float(imu.get("pitch", 0.0))
    imu_yaw = safe_float(imu.get("yaw", 0.0))

    # Backward compatibility:
    # old simulator packet may contain orientation array.
    if imu_roll == 0.0 and imu_pitch == 0.0 and imu_yaw == 0.0 and "orientation" in imu:
        orientation = imu.get("orientation", np.zeros(3))
        imu_roll = _safe_array_get(orientation, 0, default=0.0)
        imu_pitch = _safe_array_get(orientation, 1, default=0.0)
        imu_yaw = _safe_array_get(orientation, 2, default=0.0)

    linear_acceleration = imu.get("linear_acceleration", np.zeros(3))
    imu_acc_x = _safe_array_get(linear_acceleration, 0, default=0.0)
    imu_acc_y = _safe_array_get(linear_acceleration, 1, default=0.0)
    imu_acc_z = _safe_array_get(linear_acceleration, 2, default=0.0)

    # --------------------------------------------------
    # 6. Battery sensor
    # --------------------------------------------------
    battery = log.get("battery", {}) or {}
    battery_voltage = safe_float(battery.get("voltage", 0.0))

    return [
        depth,
        target_z,
        tracking_error,

        cmd_vz,
        actual_vz,
        velocity_residual,

        measured_current,
        expected_current,
        current_residual,

        dvl_vx,
        dvl_vy,
        dvl_vz,
        dvl_speed,

        imu_roll,
        imu_pitch,
        imu_yaw,
        imu_acc_x,
        imu_acc_y,
        imu_acc_z,

        battery_voltage,
    ]