"""Lightweight causal state estimation with sensor-health fallbacks.

This is intentionally smaller than a production INS/EKF. It establishes the
correct simulation boundary first: the controller receives a state assembled
from observable depth, IMU, and DVL packets, while simulator truth remains
available only for evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from environment.six_dof_dynamics import (
    SixDOFState,
    euler_to_quaternion,
    quaternion_to_rotation_matrix,
)


SENSOR_NAMES = ("depth", "imu", "dvl")
ESTIMATE_QUALITIES = ("nominal", "cautious", "degraded", "unsafe")


def _finite_vector(values, size, name):
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {size} finite values")
    return vector.copy()


@dataclass
class SixDOFStateEstimatorConfig:
    """Bounds and fallback behavior for the lightweight estimator."""

    max_abs_linear_velocity_mps: np.ndarray = field(
        default_factory=lambda: np.array([3.0, 3.0, 2.0])
    )
    max_abs_angular_velocity_radps: np.ndarray = field(
        default_factory=lambda: np.array([2.0, 2.0, 2.0])
    )
    max_abs_linear_acceleration_mps2: np.ndarray = field(
        default_factory=lambda: np.array([5.0, 5.0, 5.0])
    )
    no_inertial_velocity_decay_s: float = 2.0
    horizontal_integrity_loss_confirmation_s: float = 0.5

    def __post_init__(self):
        for name in (
            "max_abs_linear_velocity_mps",
            "max_abs_angular_velocity_radps",
            "max_abs_linear_acceleration_mps2",
        ):
            value = _finite_vector(getattr(self, name), 3, name)
            if np.any(value <= 0.0):
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        decay = float(self.no_inertial_velocity_decay_s)
        if not np.isfinite(decay) or decay <= 0.0:
            raise ValueError(
                "no_inertial_velocity_decay_s must be finite and positive"
            )
        self.no_inertial_velocity_decay_s = decay
        confirmation = float(
            self.horizontal_integrity_loss_confirmation_s
        )
        if not np.isfinite(confirmation) or confirmation < 0.0:
            raise ValueError(
                "horizontal_integrity_loss_confirmation_s must be finite "
                "and non-negative"
            )
        self.horizontal_integrity_loss_confirmation_s = confirmation


@dataclass(frozen=True)
class SixDOFStateEstimate:
    state: SixDOFState
    quality: str
    sources: Mapping[str, str]
    excluded_sensors: tuple[str, ...]
    rejected_sensors: tuple[str, ...]
    fallback_durations_s: Mapping[str, float]
    horizontal_position_reference: str
    ftc_recommendation: str

    def __post_init__(self):
        if not isinstance(self.state, SixDOFState):
            raise TypeError("state must be SixDOFState")
        if self.quality not in ESTIMATE_QUALITIES:
            raise ValueError(
                f"quality must be one of {ESTIMATE_QUALITIES}"
            )


def _health_value(health, sensor, name, default):
    if not isinstance(health, Mapping) or sensor not in health:
        return default
    result = health[sensor]
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


class SixDOFStateEstimator:
    """Fuse healthy measurements and fall back without reading truth labels."""

    def __init__(self, config: Optional[SixDOFStateEstimatorConfig] = None):
        self.config = config or SixDOFStateEstimatorConfig()
        self.reset()

    def reset(self, initial_state: Optional[SixDOFState] = None):
        self.state = (
            initial_state.copy() if initial_state is not None else SixDOFState()
        )
        self.last_time = float(self.state.time)
        self.fallback_durations_s = {
            sensor: 0.0 for sensor in SENSOR_NAMES
        }
        self.horizontal_position_reference_degraded = False
        self.last_estimate = SixDOFStateEstimate(
            state=self.state.copy(),
            quality="nominal",
            sources={
                "position_xy": "initial_pose_prior",
                "depth": "initial_pose_prior",
                "attitude": "initial_pose_prior",
                "linear_velocity": "initial_state_prior",
                "angular_velocity": "initial_state_prior",
            },
            excluded_sensors=(),
            rejected_sensors=(),
            fallback_durations_s=dict(self.fallback_durations_s),
            horizontal_position_reference="initial_dead_reckoning",
            ftc_recommendation="none",
        )
        return self.last_estimate

    def apply_horizontal_position_fix(self, north_east_m):
        """Re-anchor horizontal position from an external absolute fix."""

        north_east = _finite_vector(
            north_east_m, 2, "north_east_m"
        )
        self.state.position_ned[:2] = north_east
        self.horizontal_position_reference_degraded = False
        return self.state.copy()

    @staticmethod
    def _sensor_status(health, sensor):
        fault_type = str(_health_value(
            health, sensor, "fault_type", "normal"
        ))
        trust_level = str(_health_value(
            health, sensor, "trust_level", "trusted"
        ))
        rejected = fault_type == "spike"
        excluded = trust_level == "untrusted" or rejected
        return fault_type, trust_level, excluded, rejected

    @staticmethod
    def _packet_values(sensor_packet):
        depth = float(sensor_packet.get("depth", np.nan))
        depth_valid = bool(
            sensor_packet.get("depth_valid", np.isfinite(depth))
        ) and np.isfinite(depth)

        imu = sensor_packet.get("imu", {})
        orientation = np.asarray(
            imu.get("orientation", np.full(3, np.nan)), dtype=float
        )
        angular_velocity = np.asarray(
            imu.get("angular_velocity", np.full(3, np.nan)), dtype=float
        )
        linear_acceleration = np.asarray(
            imu.get("linear_acceleration", np.full(3, np.nan)), dtype=float
        )
        imu_valid = bool(imu.get("valid", True)) and all((
            orientation.shape == (3,),
            angular_velocity.shape == (3,),
            linear_acceleration.shape == (3,),
            np.all(np.isfinite(orientation)),
            np.all(np.isfinite(angular_velocity)),
            np.all(np.isfinite(linear_acceleration)),
        ))

        dvl = sensor_packet.get("dvl", {})
        dvl_velocity = np.asarray(
            dvl.get("velocity", np.full(3, np.nan)), dtype=float
        )
        dvl_valid = bool(dvl.get("valid", False)) and bool(
            dvl_velocity.shape == (3,)
            and np.all(np.isfinite(dvl_velocity))
        )
        return {
            "depth": (depth, depth_valid),
            "imu": (
                orientation,
                angular_velocity,
                linear_acceleration,
                imu_valid,
            ),
            "dvl": (dvl_velocity, dvl_valid),
        }

    def update(
        self,
        *,
        time_s: float,
        dt: float,
        sensor_packet: Mapping[str, Any],
        sensor_health: Optional[Mapping[str, Any]] = None,
    ):
        time_s = float(time_s)
        dt = float(dt)
        if not np.isfinite(time_s) or time_s < self.last_time:
            raise ValueError("time_s must be finite and non-decreasing")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not isinstance(sensor_packet, Mapping):
            raise TypeError("sensor_packet must be a mapping")

        packet = self._packet_values(sensor_packet)
        excluded = []
        rejected = []
        statuses = {}
        for sensor in SENSOR_NAMES:
            status = self._sensor_status(sensor_health, sensor)
            statuses[sensor] = status
            if status[2]:
                excluded.append(sensor)
            if status[3]:
                rejected.append(sensor)

        previous = self.state
        config = self.config
        orientation, measured_rate, measured_acceleration, imu_valid = (
            packet["imu"]
        )
        imu_usable = imu_valid and not statuses["imu"][2]
        if imu_usable:
            quaternion = euler_to_quaternion(*orientation)
            angular_velocity = np.clip(
                measured_rate,
                -config.max_abs_angular_velocity_radps,
                config.max_abs_angular_velocity_radps,
            )
            linear_acceleration = np.clip(
                measured_acceleration,
                -config.max_abs_linear_acceleration_mps2,
                config.max_abs_linear_acceleration_mps2,
            )
            self.fallback_durations_s["imu"] = 0.0
            attitude_source = "imu"
            angular_velocity_source = "imu"
        else:
            quaternion = previous.quaternion_nb.copy()
            angular_velocity = np.zeros(3)
            linear_acceleration = np.zeros(3)
            self.fallback_durations_s["imu"] += dt
            attitude_source = "held_last_attitude"
            angular_velocity_source = "zero_rate_safe_hold"

        measured_velocity, dvl_valid = packet["dvl"]
        dvl_usable = dvl_valid and not statuses["dvl"][2]
        if dvl_usable:
            linear_velocity = np.clip(
                measured_velocity,
                -config.max_abs_linear_velocity_mps,
                config.max_abs_linear_velocity_mps,
            )
            self.fallback_durations_s["dvl"] = 0.0
            linear_velocity_source = "dvl"
        elif imu_usable:
            linear_velocity = np.clip(
                previous.body_velocity[:3] + linear_acceleration * dt,
                -config.max_abs_linear_velocity_mps,
                config.max_abs_linear_velocity_mps,
            )
            self.fallback_durations_s["dvl"] += dt
            linear_velocity_source = "imu_dead_reckoning"
        else:
            decay = np.exp(-dt / config.no_inertial_velocity_decay_s)
            linear_velocity = previous.body_velocity[:3] * decay
            self.fallback_durations_s["dvl"] += dt
            linear_velocity_source = "decaying_velocity_hold"

        rotation_nb = quaternion_to_rotation_matrix(quaternion)
        position = previous.position_ned + (
            rotation_nb @ linear_velocity
        ) * dt
        position_xy_source = (
            "dvl_dead_reckoning"
            if dvl_usable
            else (
                "imu_dead_reckoning"
                if imu_usable
                else "decaying_velocity_hold"
            )
        )

        measured_depth, depth_valid = packet["depth"]
        depth_usable = depth_valid and not statuses["depth"][2]
        if depth_usable:
            position[2] = measured_depth
            self.fallback_durations_s["depth"] = 0.0
            depth_source = "depth_sensor"
        else:
            self.fallback_durations_s["depth"] += dt
            depth_source = (
                "velocity_dead_reckoning"
                if dvl_usable or imu_usable
                else "held_depth"
            )

        body_velocity = np.concatenate([
            linear_velocity,
            angular_velocity,
        ])
        self.state = SixDOFState(
            position_ned=position,
            quaternion_nb=quaternion,
            body_velocity=body_velocity,
            time=time_s,
        )
        self.last_time = time_s

        untrusted = {
            sensor for sensor in SENSOR_NAMES
            if statuses[sensor][1] == "untrusted"
        }
        degraded = {
            sensor for sensor in SENSOR_NAMES
            if statuses[sensor][1] == "degraded"
        }
        integrity_confirmation = (
            self.config.horizontal_integrity_loss_confirmation_s
        )
        if any(
            statuses[sensor][1] == "untrusted"
            and self.fallback_durations_s[sensor]
            >= integrity_confirmation
            for sensor in ("imu", "dvl")
        ):
            self.horizontal_position_reference_degraded = True
        if "imu" in untrusted:
            quality = "unsafe"
        elif untrusted:
            quality = "degraded"
        elif degraded:
            quality = "cautious"
        elif self.horizontal_position_reference_degraded:
            quality = "degraded"
        else:
            quality = "nominal"

        horizontal_reference = (
            "degraded_without_absolute_fix"
            if self.horizontal_position_reference_degraded
            else "initial_dead_reckoning"
        )
        ftc_recommendation = (
            "degraded_navigation"
            if self.horizontal_position_reference_degraded
            else "none"
        )

        self.last_estimate = SixDOFStateEstimate(
            state=self.state.copy(),
            quality=quality,
            sources={
                "position_xy": position_xy_source,
                "depth": depth_source,
                "attitude": attitude_source,
                "linear_velocity": linear_velocity_source,
                "angular_velocity": angular_velocity_source,
            },
            excluded_sensors=tuple(sorted(excluded)),
            rejected_sensors=tuple(sorted(rejected)),
            fallback_durations_s=dict(self.fallback_durations_s),
            horizontal_position_reference=horizontal_reference,
            ftc_recommendation=ftc_recommendation,
        )
        return self.last_estimate
