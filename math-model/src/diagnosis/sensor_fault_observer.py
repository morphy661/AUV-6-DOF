"""Causal log-only observation layer for ambiguous sensor anomalies.

This layer never confirms a hardware failure and never requests an FTC
intervention. It records weak, persistent, or intermittent evidence for the
operator while the direct SensorHealthMonitor retains the fast safety path.
"""

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


SENSOR_NAMES = ("depth", "imu", "dvl")
POSSIBLE_SENSOR_FAULT_TYPES = (
    "normal",
    "possible_weak_spike_or_bias",
    "possible_bias_or_drift",
    "possible_partial_stuck",
    "possible_intermittent_unavailability",
)


@dataclass(frozen=True)
class SensorFaultObserverConfig:
    consistency_window_s: float = 5.0
    consistency_confirmation_s: float = 0.50
    observation_hold_s: float = 3.0
    partial_stuck_confirmation_s: float = 0.75
    intermittent_window_s: float = 10.0
    intermittent_min_episodes: int = 2
    depth_weak_jump_m: float = 0.20
    imu_orientation_weak_jump_rad: float = 0.08
    imu_gyro_weak_jump_radps: float = 0.50
    imu_accel_weak_jump_mps2: float = 1.50
    dvl_weak_jump_mps: float = 0.30
    depth_strong_jump_m: float = 0.75
    imu_orientation_strong_jump_rad: float = 0.35
    imu_gyro_strong_jump_radps: float = 1.50
    imu_accel_strong_jump_mps2: float = 4.00
    dvl_strong_jump_mps: float = 1.00
    depth_consistency_residual_m: float = 0.16
    imu_consistency_residual_rad: float = 0.07
    dvl_consistency_residual_mps: float = 0.16
    depth_drift_rate_mps: float = 0.02
    imu_drift_rate_radps: float = 0.01
    dvl_drift_rate_mps2: float = 0.02
    depth_stuck_epsilon_m: float = 1e-5
    imu_stuck_epsilon: float = 1e-7
    dvl_stuck_epsilon_mps: float = 1e-6
    depth_motion_threshold_mps: float = 0.05
    imu_motion_threshold_radps: float = 0.05
    dvl_acceleration_threshold_mps2: float = 0.20
    dvl_command_change_threshold_mps: float = 0.05

    def __post_init__(self):
        values = asdict(self)
        episodes = int(values.pop("intermittent_min_episodes"))
        if episodes < 2:
            raise ValueError("intermittent_min_episodes must be at least two")
        for name, value in values.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class SensorFaultObservation:
    sensor: str
    time_s: float
    state: str
    hypothesis: str
    confidence: float
    candidates: tuple[str, ...]
    affected_channels: tuple[int, ...]
    display_level: str
    confirmed: bool
    recommended_action: str
    protective_action_required: bool
    evidence: str

    def to_dict(self):
        result = asdict(self)
        result["candidates"] = list(self.candidates)
        result["affected_channels"] = list(self.affected_channels)
        return result


def _vector(values, size):
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        return np.full(size, np.nan), False
    return result.copy(), True


def _body_to_ned_matrix(euler_rpy):
    roll, pitch, yaw = np.asarray(euler_rpy, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _wrap_angle(values):
    return (np.asarray(values, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _euler_rates(euler_rpy, body_rates):
    roll, pitch, _ = np.asarray(euler_rpy, dtype=float)
    p_rate, q_rate, r_rate = np.asarray(body_rates, dtype=float)
    cos_pitch = np.cos(pitch)
    if abs(cos_pitch) < 1e-6:
        cos_pitch = np.copysign(1e-6, cos_pitch if cos_pitch != 0.0 else 1.0)
    sin_roll, cos_roll = np.sin(roll), np.cos(roll)
    tan_pitch = np.sin(pitch) / cos_pitch
    return np.array([
        p_rate + q_rate * sin_roll * tan_pitch + r_rate * cos_roll * tan_pitch,
        q_rate * cos_roll - r_rate * sin_roll,
        q_rate * sin_roll / cos_pitch + r_rate * cos_roll / cos_pitch,
    ])


class SensorFaultObserver:
    """Record possible sensor faults without entering the FTC safety path."""

    def __init__(self, config=None):
        self.config = config or SensorFaultObserverConfig()
        self.reset()

    def reset(self):
        self.last_time = None
        self.previous_values = {sensor: None for sensor in SENSOR_NAMES}
        self.previous_desired_velocity = None
        self.channel_stuck_start = {
            "depth": [None],
            "imu": [None] * 9,
            "dvl": [None] * 3,
        }
        self.latched_partial_stuck = {
            "depth": np.zeros(1, dtype=bool),
            "imu": np.zeros(9, dtype=bool),
            "dvl": np.zeros(3, dtype=bool),
        }
        self.invalid_active = {sensor: False for sensor in SENSOR_NAMES}
        self.invalid_episode_times = {
            sensor: deque() for sensor in SENSOR_NAMES
        }
        self.histories = {sensor: deque() for sensor in SENSOR_NAMES}
        self.integration_totals = {
            "depth": np.zeros(1),
            "imu": np.zeros(3),
            "dvl": np.zeros(3),
        }
        self.residual_start = {sensor: None for sensor in SENSOR_NAMES}
        self.residual_history = {sensor: deque() for sensor in SENSOR_NAMES}
        self.held_observation = {sensor: None for sensor in SENSOR_NAMES}
        self.hold_until = {sensor: None for sensor in SENSOR_NAMES}
        self.rebaseline_sensors = ()
        return self._normal_results(0.0)

    @staticmethod
    def _normal(sensor, time_s):
        return SensorFaultObservation(
            sensor=sensor,
            time_s=float(time_s),
            state="normal",
            hypothesis="normal",
            confidence=0.0,
            candidates=(),
            affected_channels=(),
            display_level="background",
            confirmed=False,
            recommended_action="none",
            protective_action_required=False,
            evidence="No persistent ambiguous sensor evidence is active.",
        )

    def _normal_results(self, time_s):
        return {
            sensor: self._normal(sensor, time_s) for sensor in SENSOR_NAMES
        }

    @staticmethod
    def _possible(
        sensor,
        time_s,
        hypothesis,
        confidence,
        candidates,
        evidence,
        channels=(),
        display_level="possible",
    ):
        return SensorFaultObservation(
            sensor=sensor,
            time_s=float(time_s),
            state="possible_fault",
            hypothesis=hypothesis,
            confidence=float(np.clip(confidence, 0.0, 0.95)),
            candidates=tuple(candidates),
            affected_channels=tuple(int(value) for value in channels),
            display_level=str(display_level),
            confirmed=False,
            recommended_action="record_and_observe",
            protective_action_required=False,
            evidence=str(evidence),
        )

    @staticmethod
    def _observables(packet):
        depth = float(packet.get("depth", np.nan))
        depth_valid = bool(packet.get("depth_valid", np.isfinite(depth)))
        imu = packet.get("imu", {})
        orientation, orientation_valid = _vector(
            imu.get("orientation", np.full(3, np.nan)), 3
        )
        gyro, gyro_valid = _vector(
            imu.get("angular_velocity", np.full(3, np.nan)), 3
        )
        acceleration, acceleration_valid = _vector(
            imu.get("linear_acceleration", np.full(3, np.nan)), 3
        )
        imu_values = np.concatenate([orientation, gyro, acceleration])
        imu_valid = bool(imu.get("valid", True)) and all((
            orientation_valid, gyro_valid, acceleration_valid
        ))
        dvl = packet.get("dvl", {})
        dvl_values, dvl_finite = _vector(
            dvl.get("velocity", np.full(3, np.nan)), 3
        )
        dvl_valid = bool(dvl.get("valid", False)) and dvl_finite
        return {
            "depth": (np.array([depth]), depth_valid and np.isfinite(depth)),
            "imu": (imu_values, imu_valid),
            "dvl": (dvl_values, dvl_valid),
        }

    def _expected_channel_motion(self, observable, motion_context):
        context = motion_context or {}
        desired_velocity, desired_velocity_valid = _vector(
            context.get("desired_velocity_ned", np.zeros(3)), 3
        )
        desired_angular, desired_angular_valid = _vector(
            context.get("desired_angular_velocity_body", np.zeros(3)), 3
        )
        if not desired_velocity_valid:
            desired_velocity = np.zeros(3)
        if not desired_angular_valid:
            desired_angular = np.zeros(3)
        imu_values, imu_valid = observable["imu"]
        dvl_values, dvl_valid = observable["dvl"]
        depth_expected = bool(
            abs(desired_velocity[2]) >= self.config.depth_motion_threshold_mps
            or (
                dvl_valid
                and abs(dvl_values[2])
                >= self.config.depth_motion_threshold_mps
            )
        )
        imu_expected = np.zeros(9, dtype=bool)
        if imu_valid:
            imu_expected[:3] = (
                np.maximum(np.abs(imu_values[3:6]), np.abs(desired_angular))
                >= self.config.imu_motion_threshold_radps
            )
        command_change = np.zeros(3)
        if self.previous_desired_velocity is not None:
            command_change = np.abs(
                desired_velocity - self.previous_desired_velocity
            )
        dvl_expected = command_change >= (
            self.config.dvl_command_change_threshold_mps
        )
        if imu_valid:
            dvl_expected |= np.abs(imu_values[6:9]) >= (
                self.config.dvl_acceleration_threshold_mps2
            )
        self.previous_desired_velocity = desired_velocity.copy()
        return {
            "depth": np.array([depth_expected], dtype=bool),
            "imu": imu_expected,
            "dvl": dvl_expected,
        }

    def _weak_jump(self, sensor, current, previous):
        delta = np.abs(current - previous)
        if sensor == "depth":
            return (
                self.config.depth_weak_jump_m <= delta[0]
                < self.config.depth_strong_jump_m
            )
        if sensor == "imu":
            groups = (
                (
                    delta[:3],
                    self.config.imu_orientation_weak_jump_rad,
                    self.config.imu_orientation_strong_jump_rad,
                ),
                (
                    delta[3:6],
                    self.config.imu_gyro_weak_jump_radps,
                    self.config.imu_gyro_strong_jump_radps,
                ),
                (
                    delta[6:9],
                    self.config.imu_accel_weak_jump_mps2,
                    self.config.imu_accel_strong_jump_mps2,
                ),
            )
            return any(
                np.max(values) >= weak and np.max(values) < strong
                for values, weak, strong in groups
            )
        maximum = float(np.max(delta))
        return (
            self.config.dvl_weak_jump_mps <= maximum
            < self.config.dvl_strong_jump_mps
        )

    def _partial_stuck_channels(
        self, sensor, time_s, current, previous, expected
    ):
        if previous is None:
            return (), ()
        epsilon = {
            "depth": self.config.depth_stuck_epsilon_m,
            "imu": self.config.imu_stuck_epsilon,
            "dvl": self.config.dvl_stuck_epsilon_mps,
        }[sensor]
        supported = np.asarray(expected, dtype=bool)
        unchanged = np.abs(current - previous) <= epsilon
        recovered = []
        for channel in range(len(current)):
            if self.latched_partial_stuck[sensor][channel] and not unchanged[channel]:
                recovered.append(channel)
                self.latched_partial_stuck[sensor][channel] = False
                self.channel_stuck_start[sensor][channel] = None
            elif supported[channel] and unchanged[channel]:
                if self.channel_stuck_start[sensor][channel] is None:
                    self.channel_stuck_start[sensor][channel] = time_s
                elapsed = time_s - self.channel_stuck_start[sensor][channel]
                if elapsed >= self.config.partial_stuck_confirmation_s:
                    self.latched_partial_stuck[sensor][channel] = True
            elif not self.latched_partial_stuck[sensor][channel]:
                self.channel_stuck_start[sensor][channel] = None
        latched = np.flatnonzero(self.latched_partial_stuck[sensor])
        # Full-vector stuck belongs to the direct safety monitor.
        if len(latched) == len(current):
            return (), tuple(recovered)
        return tuple(int(value) for value in latched), tuple(recovered)

    def _record_invalid_episode(self, sensor, time_s, valid):
        episodes = self.invalid_episode_times[sensor]
        while episodes and time_s - episodes[0] > self.config.intermittent_window_s:
            episodes.popleft()
        if not valid:
            if not self.invalid_active[sensor]:
                episodes.append(time_s)
                self.invalid_active[sensor] = True
            return len(episodes)
        self.invalid_active[sensor] = False
        return len(episodes)

    def _update_integrals(self, dt, observable):
        depth_values, depth_valid = observable["depth"]
        imu_values, imu_valid = observable["imu"]
        dvl_values, dvl_valid = observable["dvl"]
        if imu_valid and dvl_valid:
            velocity_ned = _body_to_ned_matrix(imu_values[:3]) @ dvl_values
            self.integration_totals["depth"][0] += velocity_ned[2] * dt
        if imu_valid:
            self.integration_totals["imu"] += _euler_rates(
                imu_values[:3], imu_values[3:6]
            ) * dt
            self.integration_totals["dvl"] += imu_values[6:9] * dt
        return {
            "depth": depth_valid and imu_valid and dvl_valid,
            "imu": imu_valid,
            "dvl": dvl_valid and imu_valid,
        }

    def _consistency_residual(self, sensor, time_s, values, valid):
        history = self.histories[sensor]
        if not valid:
            history.clear()
            self.residual_history[sensor].clear()
            self.residual_start[sensor] = None
            return None, None
        measured = values[:3] if sensor == "imu" else values
        history.append((
            time_s,
            measured.copy(),
            self.integration_totals[sensor].copy(),
        ))
        cutoff = time_s - self.config.consistency_window_s
        while len(history) > 1 and history[1][0] <= cutoff:
            history.popleft()
        oldest_time, oldest_value, oldest_integral = history[0]
        span = time_s - oldest_time
        if span < 0.8 * self.config.consistency_window_s:
            return None, None
        residual = measured - oldest_value - (
            self.integration_totals[sensor] - oldest_integral
        )
        if sensor == "imu":
            residual = _wrap_angle(residual)
        scalar = float(residual[np.argmax(np.abs(residual))])
        residuals = self.residual_history[sensor]
        residuals.append((time_s, scalar))
        while residuals and time_s - residuals[0][0] > 1.5:
            residuals.popleft()
        rate = 0.0
        if len(residuals) >= 2:
            elapsed = residuals[-1][0] - residuals[0][0]
            if elapsed > 0.0:
                rate = (residuals[-1][1] - residuals[0][1]) / elapsed
        return residual, float(rate)

    def _residual_observation(self, sensor, time_s, residual, rate):
        if residual is None:
            return None
        threshold = {
            "depth": self.config.depth_consistency_residual_m,
            "imu": self.config.imu_consistency_residual_rad,
            "dvl": self.config.dvl_consistency_residual_mps,
        }[sensor]
        drift_threshold = {
            "depth": self.config.depth_drift_rate_mps,
            "imu": self.config.imu_drift_rate_radps,
            "dvl": self.config.dvl_drift_rate_mps2,
        }[sensor]
        maximum = float(np.max(np.abs(residual)))
        if maximum < threshold:
            self.residual_start[sensor] = None
            return None
        if self.residual_start[sensor] is None:
            self.residual_start[sensor] = time_s
            return None
        elapsed = time_s - self.residual_start[sensor]
        if elapsed < self.config.consistency_confirmation_s:
            return None
        channel = int(np.argmax(np.abs(residual)))
        if abs(rate) >= drift_threshold:
            candidates = ("drift", "bias", "model_mismatch")
        else:
            candidates = ("bias", "drift", "model_mismatch")
        confidence = 0.55 + 0.25 * min(maximum / max(threshold, 1e-12) - 1.0, 1.0)
        return self._possible(
            sensor,
            time_s,
            "possible_bias_or_drift",
            confidence,
            candidates,
            "A multi-second sensor/kinematic consistency residual persisted; the root cause is not directly observable.",
            (channel,),
        )

    def _hold(self, sensor, observation, time_s):
        if observation is not None:
            self.held_observation[sensor] = observation
            self.hold_until[sensor] = time_s + self.config.observation_hold_s
            return observation
        held = self.held_observation[sensor]
        until = self.hold_until[sensor]
        if held is None or until is None or time_s > until:
            self.held_observation[sensor] = None
            self.hold_until[sensor] = None
            return self._normal(sensor, time_s)
        return SensorFaultObservation(
            sensor=held.sensor,
            time_s=float(time_s),
            state=held.state,
            hypothesis=held.hypothesis,
            confidence=max(0.25, held.confidence * 0.95),
            candidates=held.candidates,
            affected_channels=held.affected_channels,
            display_level=held.display_level,
            confirmed=False,
            recommended_action=held.recommended_action,
            protective_action_required=False,
            evidence="Recent ambiguous evidence is retained briefly for event grouping and operator review.",
        )

    def update(
        self,
        time_s,
        sensor_packet: Mapping[str, Any],
        motion_context=None,
    ):
        time_s = float(time_s)
        if not np.isfinite(time_s):
            raise ValueError("sensor observer time must be finite")
        if self.last_time is not None and time_s < self.last_time:
            raise ValueError("sensor observer time must be non-decreasing")
        dt = 0.0 if self.last_time is None else time_s - self.last_time
        self.last_time = time_s
        observable = self._observables(sensor_packet)
        expected = self._expected_channel_motion(observable, motion_context)
        consistency_valid = self._update_integrals(dt, observable)
        results = {}
        rebaseline = set()

        for sensor in SENSOR_NAMES:
            current, valid = observable[sensor]
            episodes = self._record_invalid_episode(sensor, time_s, valid)
            if not valid:
                self.previous_values[sensor] = None
                self.histories[sensor].clear()
                self.residual_history[sensor].clear()
                observation = None
                if episodes >= self.config.intermittent_min_episodes:
                    observation = self._possible(
                        sensor,
                        time_s,
                        "possible_intermittent_unavailability",
                        0.70,
                        ("intermittent_sensor_fault", "communication_loss"),
                        "Multiple unavailable episodes occurred inside the observation window; current sample loss is certain but persistent hardware failure is not.",
                    )
                results[sensor] = self._hold(sensor, observation, time_s)
                continue

            previous = self.previous_values[sensor]
            partial_channels, recovered_channels = self._partial_stuck_channels(
                sensor, time_s, current, previous, expected[sensor]
            )
            if recovered_channels:
                rebaseline.add(sensor)
            residual, rate = self._consistency_residual(
                sensor,
                time_s,
                current,
                consistency_valid[sensor],
            )
            observation = None
            if episodes >= self.config.intermittent_min_episodes:
                observation = self._possible(
                    sensor,
                    time_s,
                    "possible_intermittent_unavailability",
                    0.70,
                    ("intermittent_sensor_fault", "communication_loss"),
                    "Repeated unavailable episodes were observed recently; record the pattern without declaring permanent hardware failure.",
                )
            elif partial_channels:
                observation = self._possible(
                    sensor,
                    time_s,
                    "possible_partial_stuck",
                    0.78,
                    ("partial_channel_stuck", "channel_specific_fault"),
                    "Only part of the sensor vector remained exactly unchanged while independent signals expected channel motion.",
                    partial_channels,
                )
            else:
                observation = self._residual_observation(
                    sensor, time_s, residual, rate
                )
                if (
                    observation is None
                    and previous is not None
                    and self._weak_jump(sensor, current, previous)
                ):
                    channel = int(np.argmax(np.abs(current - previous)))
                    observation = self._possible(
                        sensor,
                        time_s,
                        "possible_weak_spike_or_bias",
                        0.45,
                        ("weak_spike", "bias_onset", "disturbance"),
                        "A sub-threshold one-step change was observed; it is logged as possible evidence rather than a confirmed fault.",
                        (channel,),
                        display_level="background",
                    )
            results[sensor] = self._hold(sensor, observation, time_s)
            self.previous_values[sensor] = current.copy()
        self.rebaseline_sensors = tuple(sorted(rebaseline))
        return results

    @staticmethod
    def summarize(results):
        if set(results) != set(SENSOR_NAMES):
            raise ValueError("results must contain depth, imu, and dvl")
        active = [
            {
                "sensor": sensor,
                "hypothesis": observation.hypothesis,
                "confidence": observation.confidence,
                "candidates": list(observation.candidates),
                "affected_channels": list(observation.affected_channels),
                "display_level": observation.display_level,
            }
            for sensor, observation in results.items()
            if observation.state == "possible_fault"
        ]
        operator_possible = any(
            observation.display_level == "possible"
            for observation in results.values()
            if observation.state == "possible_fault"
        )
        return {
            "active_possible_faults": active,
            "operator_message_level": (
                "possible"
                if operator_possible
                else ("background" if active else "none")
            ),
            "recommended_action": (
                "record_and_observe" if active else "none"
            ),
            "ftc_recommendation": "none",
            "protective_action_required": False,
            "confirmed_hardware_fault": False,
        }
