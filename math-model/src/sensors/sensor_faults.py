"""Deterministic fault schedules for six-DOF onboard sensors.

Injected truth is returned only for simulation evaluation. Online health
monitors must use the observable packet values and validity flags instead.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


SENSOR_VECTOR_SIZES = {"depth": 1, "imu": 9, "dvl": 3}


class SensorFaultMode(str, Enum):
    NORMAL = "normal"
    UNAVAILABLE = "unavailable"
    STUCK = "stuck"
    SPIKE = "spike"
    BIAS = "bias"
    DRIFT = "drift"


@dataclass(frozen=True)
class SensorFaultEvent:
    """One causal sensor-fault interval or one-shot spike."""

    sensor: str
    mode: SensorFaultMode
    start_time_s: float
    end_time_s: Optional[float] = None
    channels: tuple[int, ...] = ()
    magnitude: float = 0.0
    event_id: str = ""

    def __post_init__(self):
        sensor = str(self.sensor).lower()
        if sensor not in SENSOR_VECTOR_SIZES:
            raise ValueError("sensor must be depth, imu, or dvl")
        object.__setattr__(self, "sensor", sensor)
        mode = (
            self.mode
            if isinstance(self.mode, SensorFaultMode)
            else SensorFaultMode(str(self.mode))
        )
        object.__setattr__(self, "mode", mode)
        start = float(self.start_time_s)
        if not np.isfinite(start) or start < 0.0:
            raise ValueError("start_time_s must be finite and non-negative")
        object.__setattr__(self, "start_time_s", start)
        if self.end_time_s is not None:
            end = float(self.end_time_s)
            if not np.isfinite(end) or end <= start:
                raise ValueError("end_time_s must be finite and after start")
            object.__setattr__(self, "end_time_s", end)
        channels = tuple(int(channel) for channel in self.channels)
        size = SENSOR_VECTOR_SIZES[sensor]
        if len(set(channels)) != len(channels) or any(
            channel < 0 or channel >= size for channel in channels
        ):
            raise ValueError(f"channels must be unique indices within [0, {size})")
        object.__setattr__(self, "channels", channels)
        magnitude = float(self.magnitude)
        if not np.isfinite(magnitude):
            raise ValueError("magnitude must be finite")
        if mode in (
            SensorFaultMode.SPIKE,
            SensorFaultMode.BIAS,
            SensorFaultMode.DRIFT,
        ) and magnitude == 0.0:
            raise ValueError(f"{mode.value} magnitude must be non-zero")
        object.__setattr__(self, "magnitude", magnitude)

    def is_active(self, time_s):
        time_s = float(time_s)
        if time_s < self.start_time_s:
            return False
        return self.end_time_s is None or time_s < self.end_time_s


class SensorFaultInjector:
    """Apply scheduled faults to synchronized depth, IMU, and DVL packets."""

    def __init__(self, events=()):
        self.events = tuple(sorted(
            events,
            key=lambda event: (event.start_time_s, event.sensor),
        ))
        if not all(isinstance(event, SensorFaultEvent) for event in self.events):
            raise TypeError("events must contain SensorFaultEvent instances")
        self._validate_non_overlapping_events()
        self.reset()

    def _validate_non_overlapping_events(self):
        for sensor in SENSOR_VECTOR_SIZES:
            events = [event for event in self.events if event.sensor == sensor]
            for first, second in zip(events, events[1:]):
                if first.mode == SensorFaultMode.SPIKE:
                    if second.start_time_s == first.start_time_s:
                        raise ValueError(
                            f"overlapping fault events are not allowed for {sensor}"
                        )
                    continue
                first_end = (
                    first.end_time_s
                    if first.end_time_s is not None
                    else float("inf")
                )
                if second.start_time_s < first_end:
                    raise ValueError(
                        f"overlapping fault events are not allowed for {sensor}"
                    )

    def reset(self):
        self._stuck_values = {}
        self._spike_emitted = set()

    def _event_key(self, event):
        return self.events.index(event)

    def _event_at(self, sensor, time_s):
        for event in self.events:
            if event.sensor != sensor or not event.is_active(time_s):
                continue
            key = self._event_key(event)
            if event.mode == SensorFaultMode.SPIKE and key in self._spike_emitted:
                continue
            return key, event
        return None, None

    @staticmethod
    def _truth(event):
        if event is None:
            return {
                "active": False,
                "fault_type": SensorFaultMode.NORMAL.value,
                "event_id": None,
                "channels": [],
            }
        return {
            "active": True,
            "fault_type": event.mode.value,
            "event_id": event.event_id or None,
            "channels": list(event.channels),
        }

    @staticmethod
    def _imu_vector(packet):
        return np.concatenate([
            np.asarray(packet["orientation"], dtype=float),
            np.asarray(packet["angular_velocity"], dtype=float),
            np.asarray(packet["linear_acceleration"], dtype=float),
        ])

    @staticmethod
    def _write_imu_vector(packet, values):
        result = dict(packet)
        result["orientation"] = values[:3].copy()
        result["roll"] = float(values[0])
        result["pitch"] = float(values[1])
        result["yaw"] = float(values[2])
        result["angular_velocity"] = values[3:6].copy()
        result["linear_acceleration"] = values[6:9].copy()
        return result

    @staticmethod
    def _channels(event, size):
        return event.channels or tuple(range(size))

    def _apply_vector_event(self, values, key, event, time_s):
        result = np.asarray(values, dtype=float).copy()
        channels = self._channels(event, len(result))
        if event.mode == SensorFaultMode.STUCK:
            if key not in self._stuck_values:
                self._stuck_values[key] = result.copy()
            result[list(channels)] = self._stuck_values[key][list(channels)]
        elif event.mode == SensorFaultMode.SPIKE:
            result[list(channels)] += event.magnitude
            self._spike_emitted.add(key)
        elif event.mode == SensorFaultMode.BIAS:
            result[list(channels)] += event.magnitude
        elif event.mode == SensorFaultMode.DRIFT:
            elapsed_s = max(0.0, float(time_s) - event.start_time_s)
            result[list(channels)] += event.magnitude * elapsed_s
        return result

    def apply(self, *, time_s, depth, imu, dvl):
        time_s = float(time_s)
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        imu_result = dict(imu)
        imu_result.setdefault("valid", True)
        dvl_result = dict(dvl)
        depth_result = float(depth)
        truth = {}

        key, event = self._event_at("depth", time_s)
        truth["depth"] = self._truth(event)
        if event is not None:
            if event.mode == SensorFaultMode.UNAVAILABLE:
                depth_result = float("nan")
            else:
                depth_result = float(self._apply_vector_event(
                    np.array([depth_result]), key, event, time_s
                )[0])

        key, event = self._event_at("imu", time_s)
        truth["imu"] = self._truth(event)
        if event is not None:
            if event.mode == SensorFaultMode.UNAVAILABLE:
                values = np.full(9, np.nan)
                imu_result = self._write_imu_vector(imu_result, values)
                imu_result["valid"] = False
            else:
                values = self._apply_vector_event(
                    self._imu_vector(imu_result), key, event, time_s
                )
                imu_result = self._write_imu_vector(imu_result, values)

        key, event = self._event_at("dvl", time_s)
        truth["dvl"] = self._truth(event)
        if event is not None:
            if event.mode == SensorFaultMode.UNAVAILABLE:
                velocity = np.full(3, np.nan)
                dvl_result.update({
                    "valid": False,
                    "dropout_reason": "fault_injected",
                    "velocity": velocity,
                    "vx": np.nan,
                    "vy": np.nan,
                    "vz": np.nan,
                    "speed": np.nan,
                })
            else:
                velocity = self._apply_vector_event(
                    dvl_result["velocity"], key, event, time_s
                )
                dvl_result.update({
                    "valid": True,
                    "dropout_reason": None,
                    "velocity": velocity,
                    "vx": float(velocity[0]),
                    "vy": float(velocity[1]),
                    "vz": float(velocity[2]),
                    "speed": float(np.linalg.norm(velocity)),
                })

        return {
            "depth": depth_result,
            "depth_valid": bool(np.isfinite(depth_result)),
            "imu": imu_result,
            "dvl": dvl_result,
            "sensor_fault_truth": truth,
        }
