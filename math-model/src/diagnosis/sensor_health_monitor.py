"""Causal rule-based health monitoring for depth, IMU, and DVL sensors."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


SENSOR_NAMES = ("depth", "imu", "dvl")
SENSOR_FAULT_TYPES = ("normal", "unavailable", "stuck", "spike")
SENSOR_GUARD_ACTIONS = (
    "none",
    "observe",
    "reject_current_sample",
    "degraded_navigation",
    "safe_hold_or_abort",
)


@dataclass(frozen=True)
class SensorHealthConfig:
    stuck_confirmation_s: float = 0.75
    spike_hold_s: float = 0.25
    depth_stuck_epsilon_m: float = 1e-5
    imu_stuck_epsilon: float = 1e-7
    dvl_stuck_epsilon_mps: float = 1e-6
    depth_motion_threshold_mps: float = 0.05
    imu_motion_threshold_radps: float = 0.05
    dvl_acceleration_threshold_mps2: float = 0.20
    dvl_command_change_threshold_mps: float = 0.05
    depth_spike_threshold_m: float = 0.75
    imu_orientation_spike_threshold_rad: float = 0.35
    imu_gyro_spike_threshold_radps: float = 1.50
    imu_accel_spike_threshold_mps2: float = 4.00
    dvl_spike_threshold_mps: float = 1.00

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class SensorHealthResult:
    sensor: str
    time_s: float
    health_state: str
    fault_type: str
    confidence: float
    trust_level: str
    confirmed: bool
    recommended_action: str
    evidence: str

    def to_dict(self):
        return asdict(self)


def _finite_vector(values, size):
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,):
        return np.full(size, np.nan), False
    return vector.copy(), bool(np.all(np.isfinite(vector)))


class SensorHealthMonitor:
    """Detect directly observable unavailability, stuck, and spike faults.

    Stuck detection requires independent expected-motion evidence. This avoids
    declaring a stationary, correctly operating sensor stuck merely because
    the AUV is holding position.
    """

    def __init__(self, config=None):
        self.config = config or SensorHealthConfig()
        self.reset()

    def reset(self):
        self.last_time = None
        self.previous_values = {sensor: None for sensor in SENSOR_NAMES}
        self.stuck_start = {sensor: None for sensor in SENSOR_NAMES}
        self.confirmed_stuck = {sensor: False for sensor in SENSOR_NAMES}
        self.rebaseline_on_valid = {
            sensor: False for sensor in SENSOR_NAMES
        }
        self.spike_hold_until = {sensor: None for sensor in SENSOR_NAMES}
        self.previous_desired_velocity = None
        return self._normal_results(0.0)

    @staticmethod
    def _normal_result(sensor, time_s):
        return SensorHealthResult(
            sensor=sensor,
            time_s=float(time_s),
            health_state="healthy",
            fault_type="normal",
            confidence=1.0,
            trust_level="trusted",
            confirmed=False,
            recommended_action="use_sensor",
            evidence="Observable values are finite and no direct fault signature is active.",
        )

    def _normal_results(self, time_s):
        return {
            sensor: self._normal_result(sensor, time_s)
            for sensor in SENSOR_NAMES
        }

    @staticmethod
    def _unavailable_result(sensor, time_s):
        return SensorHealthResult(
            sensor=sensor,
            time_s=float(time_s),
            health_state="confirmed_fault",
            fault_type="unavailable",
            confidence=1.0,
            trust_level="untrusted",
            confirmed=True,
            recommended_action="exclude_sensor",
            evidence="Validity flag is false or the observable packet contains non-finite values.",
        )

    @staticmethod
    def _spike_result(sensor, time_s):
        return SensorHealthResult(
            sensor=sensor,
            time_s=float(time_s),
            health_state="confirmed_fault",
            fault_type="spike",
            confidence=0.99,
            trust_level="degraded",
            confirmed=True,
            recommended_action="reject_current_sample",
            evidence="A one-step change exceeded the sensor-specific physical jump threshold.",
        )

    def _stuck_result(self, sensor, time_s, elapsed_s):
        confirmed = elapsed_s >= self.config.stuck_confirmation_s
        if confirmed:
            return SensorHealthResult(
                sensor=sensor,
                time_s=float(time_s),
                health_state="confirmed_fault",
                fault_type="stuck",
                confidence=0.98,
                trust_level="untrusted",
                confirmed=True,
                recommended_action="exclude_sensor",
                evidence=(
                    "The reading stayed unchanged for the confirmation interval while independent onboard signals expected motion."
                ),
            )
        ratio = elapsed_s / max(self.config.stuck_confirmation_s, 1e-12)
        return SensorHealthResult(
            sensor=sensor,
            time_s=float(time_s),
            health_state="suspected_fault",
            fault_type="stuck",
            confidence=float(np.clip(0.20 + 0.70 * ratio, 0.20, 0.90)),
            trust_level="degraded",
            confirmed=False,
            recommended_action="observe",
            evidence="The reading is unchanged under expected motion; confirmation time has not elapsed.",
        )

    @staticmethod
    def _observable_values(packet):
        depth = float(packet.get("depth", np.nan))
        depth_valid = bool(packet.get("depth_valid", np.isfinite(depth)))
        imu = packet.get("imu", {})
        orientation, orientation_valid = _finite_vector(
            imu.get("orientation", np.full(3, np.nan)), 3
        )
        gyro, gyro_valid = _finite_vector(
            imu.get("angular_velocity", np.full(3, np.nan)), 3
        )
        acceleration, acceleration_valid = _finite_vector(
            imu.get("linear_acceleration", np.full(3, np.nan)), 3
        )
        imu_values = np.concatenate([orientation, gyro, acceleration])
        imu_valid = bool(imu.get("valid", True)) and all((
            orientation_valid, gyro_valid, acceleration_valid
        ))
        dvl = packet.get("dvl", {})
        dvl_values, dvl_finite = _finite_vector(
            dvl.get("velocity", np.full(3, np.nan)), 3
        )
        dvl_valid = bool(dvl.get("valid", False)) and dvl_finite
        return {
            "depth": (np.array([depth]), depth_valid and np.isfinite(depth)),
            "imu": (imu_values, imu_valid),
            "dvl": (dvl_values, dvl_valid),
        }

    def _expected_motion(self, packet, motion_context):
        context = motion_context or {}
        desired_velocity, desired_velocity_valid = _finite_vector(
            context.get("desired_velocity_ned", np.zeros(3)), 3
        )
        desired_angular, desired_angular_valid = _finite_vector(
            context.get("desired_angular_velocity_body", np.zeros(3)), 3
        )
        if not desired_velocity_valid:
            desired_velocity = np.zeros(3)
        if not desired_angular_valid:
            desired_angular = np.zeros(3)

        dvl = packet.get("dvl", {})
        dvl_velocity, dvl_valid = _finite_vector(
            dvl.get("velocity", np.full(3, np.nan)), 3
        )
        dvl_valid = bool(dvl.get("valid", False)) and dvl_valid
        imu = packet.get("imu", {})
        acceleration, acceleration_valid = _finite_vector(
            imu.get("linear_acceleration", np.full(3, np.nan)), 3
        )
        imu_valid = bool(imu.get("valid", True)) and acceleration_valid

        depth_expected = bool(
            abs(desired_velocity[2]) >= self.config.depth_motion_threshold_mps
            or (
                dvl_valid
                and abs(dvl_velocity[2])
                >= self.config.depth_motion_threshold_mps
            )
        )
        imu_expected = bool(
            np.linalg.norm(desired_angular)
            >= self.config.imu_motion_threshold_radps
        )
        command_change = 0.0
        if self.previous_desired_velocity is not None:
            command_change = float(np.linalg.norm(
                desired_velocity - self.previous_desired_velocity
            ))
        dvl_expected = bool(
            command_change >= self.config.dvl_command_change_threshold_mps
            or (
                imu_valid
                and np.linalg.norm(acceleration)
                >= self.config.dvl_acceleration_threshold_mps2
            )
        )
        self.previous_desired_velocity = desired_velocity.copy()
        return {
            "depth": depth_expected,
            "imu": imu_expected,
            "dvl": dvl_expected,
        }

    def _is_spike(self, sensor, current, previous):
        delta = np.abs(current - previous)
        if sensor == "depth":
            return bool(delta[0] >= self.config.depth_spike_threshold_m)
        if sensor == "imu":
            return bool(
                np.max(delta[:3])
                >= self.config.imu_orientation_spike_threshold_rad
                or np.max(delta[3:6])
                >= self.config.imu_gyro_spike_threshold_radps
                or np.max(delta[6:9])
                >= self.config.imu_accel_spike_threshold_mps2
            )
        return bool(np.max(delta) >= self.config.dvl_spike_threshold_mps)

    def _is_unchanged(self, sensor, current, previous):
        epsilon = {
            "depth": self.config.depth_stuck_epsilon_m,
            "imu": self.config.imu_stuck_epsilon,
            "dvl": self.config.dvl_stuck_epsilon_mps,
        }[sensor]
        return bool(np.max(np.abs(current - previous)) <= epsilon)

    def update(
        self,
        time_s,
        sensor_packet: Mapping[str, Any],
        motion_context=None,
        rebaseline_sensors=(),
    ):
        time_s = float(time_s)
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        if self.last_time is not None and time_s < self.last_time:
            raise ValueError("sensor health time must be non-decreasing")
        self.last_time = time_s
        observable = self._observable_values(sensor_packet)
        expected_motion = self._expected_motion(
            sensor_packet, motion_context
        )
        rebaseline_sensors = frozenset(
            str(sensor) for sensor in rebaseline_sensors
        )
        if any(sensor not in SENSOR_NAMES for sensor in rebaseline_sensors):
            raise ValueError("rebaseline_sensors contains an unknown sensor")
        results = {}

        for sensor in SENSOR_NAMES:
            current, valid = observable[sensor]
            previous = self.previous_values[sensor]
            if not valid:
                self.stuck_start[sensor] = None
                self.confirmed_stuck[sensor] = False
                self.rebaseline_on_valid[sensor] = True
                results[sensor] = self._unavailable_result(sensor, time_s)
                continue

            if self.rebaseline_on_valid[sensor]:
                # The first finite sample after an unavailable interval is a
                # new baseline, not a physical one-step spike.
                self.rebaseline_on_valid[sensor] = False
                self.previous_values[sensor] = current.copy()
                self.spike_hold_until[sensor] = None
                results[sensor] = self._normal_result(sensor, time_s)
                continue

            if sensor in rebaseline_sensors:
                # A separate causal observer saw a previously frozen subset
                # of channels start moving again. The recovery discontinuity
                # establishes a new baseline and must not become a second,
                # falsely certain spike event.
                self.confirmed_stuck[sensor] = False
                self.stuck_start[sensor] = None
                self.previous_values[sensor] = current.copy()
                self.spike_hold_until[sensor] = None
                results[sensor] = self._normal_result(sensor, time_s)
                continue

            if self.confirmed_stuck[sensor]:
                if previous is not None and self._is_unchanged(
                    sensor, current, previous
                ):
                    elapsed = max(
                        self.config.stuck_confirmation_s,
                        time_s - self.stuck_start[sensor],
                    )
                    results[sensor] = self._stuck_result(
                        sensor, time_s, elapsed
                    )
                    self.previous_values[sensor] = current.copy()
                    continue
                # A changed sample ends the latched stuck episode. Rebaseline
                # once so the recovery jump is not reported as a new spike.
                self.confirmed_stuck[sensor] = False
                self.stuck_start[sensor] = None
                self.previous_values[sensor] = current.copy()
                results[sensor] = self._normal_result(sensor, time_s)
                continue

            hold_until = self.spike_hold_until[sensor]
            if hold_until is not None and time_s <= hold_until:
                results[sensor] = self._spike_result(sensor, time_s)
                self.previous_values[sensor] = current.copy()
                continue

            if previous is not None and self._is_spike(
                sensor, current, previous
            ):
                self.spike_hold_until[sensor] = (
                    time_s + self.config.spike_hold_s
                )
                self.stuck_start[sensor] = None
                results[sensor] = self._spike_result(sensor, time_s)
                self.previous_values[sensor] = current.copy()
                continue

            if (
                previous is not None
                and expected_motion[sensor]
                and self._is_unchanged(sensor, current, previous)
            ):
                if self.stuck_start[sensor] is None:
                    self.stuck_start[sensor] = time_s
                elapsed = time_s - self.stuck_start[sensor]
                results[sensor] = self._stuck_result(
                    sensor, time_s, elapsed
                )
                if results[sensor].confirmed:
                    self.confirmed_stuck[sensor] = True
            else:
                self.stuck_start[sensor] = None
                results[sensor] = self._normal_result(sensor, time_s)
            self.previous_values[sensor] = current.copy()
        return results

    @staticmethod
    def summarize(results):
        if set(results) != set(SENSOR_NAMES):
            raise ValueError("results must contain depth, imu, and dvl")
        untrusted = sorted(
            sensor for sensor, result in results.items()
            if result.trust_level == "untrusted"
        )
        degraded = sorted(
            sensor for sensor, result in results.items()
            if result.trust_level == "degraded"
        )
        active_faults = [
            {
                "sensor": sensor,
                "fault_type": result.fault_type,
                "confidence": result.confidence,
                "confirmed": result.confirmed,
            }
            for sensor, result in results.items()
            if result.fault_type != "normal"
        ]
        if "imu" in untrusted:
            recommendation = "safe_hold_or_abort"
        elif any(sensor in untrusted for sensor in ("depth", "dvl")):
            recommendation = "degraded_navigation"
        elif any(
            result.fault_type == "spike" for result in results.values()
        ):
            recommendation = "reject_current_sample"
        elif active_faults:
            recommendation = "observe"
        else:
            recommendation = "none"
        return {
            "untrusted_sensors": untrusted,
            "degraded_sensors": degraded,
            "active_faults": active_faults,
            "ftc_recommendation": recommendation,
            "all_sensors_trusted": not untrusted and not degraded,
        }
