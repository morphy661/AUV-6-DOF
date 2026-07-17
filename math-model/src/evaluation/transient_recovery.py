"""Deterministic short-disturbance scenarios and recovery measurements."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimedDVLDropoutScenario:
    """A finite, known evaluation interval with no valid DVL velocity."""

    name: str
    start_time_s: float
    duration_s: float

    def __post_init__(self):
        name = str(self.name).strip()
        start = float(self.start_time_s)
        duration = float(self.duration_s)
        if not name:
            raise ValueError("name cannot be empty")
        if not np.isfinite(start) or start < 0.0:
            raise ValueError("start_time_s must be finite and non-negative")
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "start_time_s", start)
        object.__setattr__(self, "duration_s", duration)

    @property
    def end_time_s(self):
        return self.start_time_s + self.duration_s

    @property
    def dropout_window(self):
        return (self.start_time_s, self.end_time_s)

    def as_dict(self):
        return {
            "name": self.name,
            "type": "dvl_dropout",
            "start_time_s": self.start_time_s,
            "duration_s": self.duration_s,
        }


@dataclass(frozen=True)
class TransientDisturbanceScenario:
    """A finite half-sine body-wrench pulse with no actuator fault."""

    name: str
    start_time_s: float
    duration_s: float
    peak_wrench_body: np.ndarray

    def __post_init__(self):
        name = str(self.name).strip()
        if not name:
            raise ValueError("name cannot be empty")
        start = float(self.start_time_s)
        duration = float(self.duration_s)
        wrench = np.asarray(self.peak_wrench_body, dtype=float)
        if not np.isfinite(start) or start < 0.0:
            raise ValueError("start_time_s must be finite and non-negative")
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
            raise ValueError("peak_wrench_body must contain six finite values")
        if not np.any(np.abs(wrench) > 0.0):
            raise ValueError("peak_wrench_body cannot be all zero")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "start_time_s", start)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "peak_wrench_body", wrench.copy())

    @property
    def end_time_s(self):
        return self.start_time_s + self.duration_s

    def wrench_at(self, time_s):
        """Return the causal pulse value at one simulation timestamp."""

        time_s = float(time_s)
        if time_s < self.start_time_s or time_s > self.end_time_s:
            return np.zeros(6, dtype=float)
        phase = (time_s - self.start_time_s) / self.duration_s
        return np.sin(np.pi * phase) * self.peak_wrench_body

    def as_dict(self):
        return {
            "name": self.name,
            "type": "body_wrench_pulse",
            "start_time_s": self.start_time_s,
            "duration_s": self.duration_s,
            "peak_wrench_body": self.peak_wrench_body.tolist(),
        }


def default_transient_scenarios(start_time_s=15.0):
    """Cover brief surge, sway/yaw, heave, and multi-axis disturbances."""

    start = float(start_time_s)
    return (
        TransientDisturbanceScenario(
            "surge_current_pulse",
            start,
            1.00,
            np.array([42.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        TransientDisturbanceScenario(
            "sway_yaw_current_pulse",
            start,
            1.25,
            np.array([0.0, 38.0, 0.0, 0.0, 0.0, 10.0]),
        ),
        TransientDisturbanceScenario(
            "vertical_current_pulse",
            start,
            1.25,
            np.array([0.0, 0.0, 42.0, 0.0, 0.0, 0.0]),
        ),
        TransientDisturbanceScenario(
            "multi_axis_turbulence_pulse",
            start,
            1.75,
            np.array([28.0, -30.0, 24.0, 0.0, 0.0, 8.0]),
        ),
    )


def boundary_transient_scenarios(
    start_time_s=15.0,
    durations=(1.0, 2.0, 4.0),
    intensity_scales=(0.5, 1.0, 1.5),
):
    """Return the compact duration/intensity multi-axis boundary matrix."""

    base = np.array([28.0, -30.0, 24.0, 0.0, 0.0, 8.0])
    labels = {0.5: "weak", 1.0: "medium", 1.5: "strong"}
    scenarios = []
    for duration in durations:
        for scale in intensity_scales:
            label = labels.get(float(scale), f"x{float(scale):g}")
            scenarios.append(TransientDisturbanceScenario(
                f"multi_axis_{label}_{float(duration):g}s",
                start_time_s,
                duration,
                float(scale) * base,
            ))
    return tuple(scenarios)


def dvl_dropout_boundary_scenarios(
    start_time_s=15.0,
    durations=(1.0, 2.0, 4.0),
):
    """Return scheduled complete-DVL-loss boundary cases."""

    return tuple(
        TimedDVLDropoutScenario(
            f"dvl_dropout_{float(duration):g}s",
            start_time_s,
            duration,
        )
        for duration in durations
    )


def summarize_transient_recovery(
    logs,
    scenario,
    *,
    final_window_s=5.0,
    minimum_response=0.02,
    maximum_remaining_fraction=0.25,
    absolute_final_tolerance=0.05,
):
    """Measure whether pose error rises during the pulse and then recovers."""

    if not isinstance(scenario, TransientDisturbanceScenario):
        raise TypeError("scenario must be a TransientDisturbanceScenario")
    logs = list(logs)
    if not logs:
        raise ValueError("logs cannot be empty")
    times = np.asarray([log["time"] for log in logs], dtype=float)
    position_errors = np.asarray([
        np.linalg.norm(np.asarray(log["position_error_ned"], dtype=float))
        for log in logs
    ])
    attitude_errors = np.asarray([
        np.linalg.norm(np.asarray(log["attitude_error_body"], dtype=float))
        for log in logs
    ])
    # Position and attitude are normalized only to form a common recovery
    # indicator; the physical-unit peak values are reported separately.
    combined = np.sqrt(
        (position_errors / 0.25) ** 2 + (attitude_errors / 0.15) ** 2
    )
    pre_mask = (
        (times >= max(times[0], scenario.start_time_s - final_window_s))
        & (times < scenario.start_time_s)
    )
    response_mask = (
        (times >= scenario.start_time_s)
        & (times <= scenario.end_time_s + final_window_s)
    )
    final_mask = times >= max(times[0], times[-1] - final_window_s)
    if not np.any(pre_mask) or not np.any(response_mask) or not np.any(final_mask):
        raise ValueError("logs do not cover pre-event, response, and recovery windows")

    baseline = float(np.median(combined[pre_mask]))
    peak = float(np.max(combined[response_mask]))
    final = float(np.median(combined[final_mask]))
    excursion = max(0.0, peak - baseline)
    remaining = max(0.0, final - baseline)
    remaining_fraction = (
        remaining / excursion if excursion > 1e-12 else 1.0
    )
    response_observed = excursion >= float(minimum_response)
    recovered = bool(
        response_observed
        and (
            remaining <= float(absolute_final_tolerance)
            or remaining_fraction <= float(maximum_remaining_fraction)
        )
    )
    return {
        "scenario": scenario.name,
        "response_observed": response_observed,
        "recovered": recovered,
        "baseline_normalized_pose_error": baseline,
        "peak_normalized_pose_error": peak,
        "final_normalized_pose_error": final,
        "remaining_excursion_fraction": remaining_fraction,
        "peak_position_error_m": float(np.max(position_errors[response_mask])),
        "final_position_error_m": float(np.median(position_errors[final_mask])),
        "peak_attitude_error_rad": float(np.max(attitude_errors[response_mask])),
        "final_attitude_error_rad": float(np.median(attitude_errors[final_mask])),
    }
