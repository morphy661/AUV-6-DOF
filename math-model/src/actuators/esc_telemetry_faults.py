"""Reusable causal fault injection for six-channel ESC telemetry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


THRUSTER_NAMES = ("H1", "H2", "H3", "H4", "V1", "V2")
FAULT_MODES = (
    "continuous_packet_loss",
    "communication_freeze",
    "bus_voltage_dip",
    "quantization",
)


def _six_vector(log, name, default, dtype=float):
    vector = np.asarray(log.get(name, default), dtype=dtype)
    if vector.shape != (6,):
        raise ValueError(f"{name} must contain six values")
    if dtype is not bool and not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain six finite values")
    return vector.copy()


class ESCTelemetryFaultInjector:
    """Apply declared link or measurement faults to sequential log packets."""

    def __init__(self, events: Sequence[Mapping]):
        self.events = tuple(dict(event) for event in events)
        event_ids = set()
        for event in self.events:
            event_id = str(event.get("event_id", ""))
            name = str(event.get("thruster_name", ""))
            mode = str(event.get("mode", ""))
            start = float(event.get("start_time_s", -1.0))
            stop = float(event.get("end_time_s", -1.0))
            if not event_id or event_id in event_ids:
                raise ValueError("ESC telemetry event_id must be unique")
            if name not in THRUSTER_NAMES:
                raise ValueError(f"unknown ESC thruster: {name}")
            if mode not in FAULT_MODES:
                raise ValueError(f"unknown ESC telemetry mode: {mode}")
            if not np.isfinite(start) or not np.isfinite(stop) or stop < start:
                raise ValueError("ESC telemetry event times are invalid")
            event_ids.add(event_id)
        self.reset()

    def reset(self):
        self._held_samples = {}

    def apply(self, log: Mapping, *, copy_log=False):
        output = dict(log) if copy_log else log
        time_s = float(output.get("time", 0.0))
        if not np.isfinite(time_s):
            raise ValueError("log time must be finite")
        currents = _six_vector(
            output, "thruster_measured_currents", np.zeros(6)
        )
        rpms = _six_vector(output, "thruster_measured_rpms", np.zeros(6))
        voltages = _six_vector(
            output, "thruster_measured_voltages", np.full(6, 48.0)
        )
        valid = _six_vector(
            output, "thruster_telemetry_valid", np.ones(6), dtype=bool
        )
        age_s = _six_vector(
            output, "thruster_telemetry_age_s", np.zeros(6)
        )
        for event in self.events:
            start = float(event["start_time_s"])
            stop = float(event["end_time_s"])
            index = THRUSTER_NAMES.index(str(event["thruster_name"]))
            event_id = str(event["event_id"])
            if time_s < start - 1e-9:
                self._held_samples[event_id] = (
                    float(currents[index]), float(rpms[index])
                )
                continue
            if time_s > stop + 1e-9:
                continue
            elapsed = max(0.0, time_s - start)
            mode = str(event["mode"])
            if mode == "continuous_packet_loss":
                currents[index] = 0.0
                rpms[index] = 0.0
                valid[index] = False
                age_s[index] = elapsed
            elif mode == "communication_freeze":
                held = self._held_samples.setdefault(
                    event_id, (float(currents[index]), float(rpms[index]))
                )
                currents[index], rpms[index] = held
                valid[index] = True
                age_s[index] = elapsed
            elif mode == "bus_voltage_dip":
                scale = float(event.get("signal_scale", 0.65))
                voltages[index] *= float(event.get("voltage_scale", 0.55))
                currents[index] *= scale
                rpms[index] *= scale
            else:
                current_step = float(event.get("current_step_a", 0.25))
                rpm_step = float(event.get("rpm_step", 150.0))
                if current_step <= 0.0 or rpm_step <= 0.0:
                    raise ValueError("ESC quantization steps must be positive")
                currents[index] = np.round(currents[index] / current_step) * current_step
                rpms[index] = np.round(rpms[index] / rpm_step) * rpm_step
        output["thruster_measured_currents"] = currents
        output["thruster_measured_rpms"] = rpms
        output["thruster_measured_voltages"] = voltages
        output["thruster_telemetry_valid"] = valid
        output["thruster_telemetry_age_s"] = age_s
        return output

    def apply_logs(self, logs):
        self.reset()
        return [self.apply(log, copy_log=True) for log in logs]
