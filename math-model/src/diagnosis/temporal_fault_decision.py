"""Causal persistence and recovery logic for AUV fault probabilities."""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class TemporalDecisionConfig:
    """Time-based thresholds for fault entry, location, and recovery."""

    enter_fault_probability: float = 0.75
    no_output_confirmation_s: float = 1.25
    thrust_loss_confirmation_s: float = 2.50
    exit_normal_probability: float = 0.70
    recovery_confirmation_s: float = 3.75
    probability_time_constant_s: float = 1.25
    location_probability_threshold: float = 0.25
    location_confirmation_s: float = 1.25

    def __post_init__(self):
        probabilities = {
            "enter_fault_probability": self.enter_fault_probability,
            "exit_normal_probability": self.exit_normal_probability,
            "location_probability_threshold": (
                self.location_probability_threshold
            ),
        }
        for name, value in probabilities.items():
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        durations = {
            "no_output_confirmation_s": self.no_output_confirmation_s,
            "thrust_loss_confirmation_s": self.thrust_loss_confirmation_s,
            "recovery_confirmation_s": self.recovery_confirmation_s,
            "probability_time_constant_s": self.probability_time_constant_s,
            "location_confirmation_s": self.location_confirmation_s,
        }
        for name, value in durations.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class TemporalDecisionResult:
    state: str
    mode: int
    location: int
    smoothed_mode_probabilities: np.ndarray
    smoothed_location_probabilities: np.ndarray


def _probability_vector(values, size, name):
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite probabilities")
    if np.any(array < 0.0):
        raise ValueError(f"{name} cannot contain negative values")
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total probability")
    return array / total


class TemporalFaultDecision:
    """Four-state causal decision layer: normal, suspected, confirmed, recovery."""

    def __init__(self, config=None):
        self.config = config or TemporalDecisionConfig()
        self.reset()

    def reset(self):
        self.state = "normal"
        self.last_time = None
        self.mode_probabilities = None
        self.location_probabilities = None
        self.candidate_mode = 0
        self.candidate_mode_start = None
        self.confirmed_mode = 0
        self.confirmed_location = 0
        self.recovery_start = None
        self.location_candidate = 0
        self.location_candidate_start = None
        self.mode_switch_candidate = 0
        self.mode_switch_start = None

    def _confirmation_duration(self, mode):
        return (
            self.config.no_output_confirmation_s
            if int(mode) == 1
            else self.config.thrust_loss_confirmation_s
        )

    def _smooth(self, time_s, mode_probabilities, location_probabilities):
        if self.last_time is None:
            self.mode_probabilities = mode_probabilities
            self.location_probabilities = location_probabilities
            self.last_time = time_s
            return
        if time_s < self.last_time:
            raise ValueError("time_s must be non-decreasing")
        dt = time_s - self.last_time
        tau = self.config.probability_time_constant_s
        alpha = 1.0 if tau <= 0.0 else 1.0 - np.exp(-dt / tau)
        self.mode_probabilities = (
            (1.0 - alpha) * self.mode_probabilities
            + alpha * mode_probabilities
        )
        self.location_probabilities = (
            (1.0 - alpha) * self.location_probabilities
            + alpha * location_probabilities
        )
        self.last_time = time_s

    def _clear_to_normal(self):
        self.state = "normal"
        self.candidate_mode = 0
        self.candidate_mode_start = None
        self.confirmed_mode = 0
        self.confirmed_location = 0
        self.recovery_start = None
        self.location_candidate = 0
        self.location_candidate_start = None
        self.mode_switch_candidate = 0
        self.mode_switch_start = None

    def _update_unconfirmed(self, time_s, best_fault_mode, fault_probability):
        if fault_probability < self.config.enter_fault_probability:
            self._clear_to_normal()
            return
        if best_fault_mode != self.candidate_mode:
            self.candidate_mode = best_fault_mode
            self.candidate_mode_start = time_s
        self.state = "suspected"
        duration = self._confirmation_duration(self.candidate_mode)
        if time_s - self.candidate_mode_start >= duration:
            self.confirmed_mode = self.candidate_mode
            self.confirmed_location = int(
                np.argmax(self.location_probabilities)
            ) + 1
            self.state = "confirmed"
            self.recovery_start = None

    def _update_mode_switch(self, time_s, best_fault_mode, fault_probability):
        if (
            best_fault_mode == self.confirmed_mode
            or fault_probability < self.config.enter_fault_probability
        ):
            self.mode_switch_candidate = 0
            self.mode_switch_start = None
            return
        if best_fault_mode != self.mode_switch_candidate:
            self.mode_switch_candidate = best_fault_mode
            self.mode_switch_start = time_s
            if self._confirmation_duration(best_fault_mode) <= 0.0:
                self.confirmed_mode = best_fault_mode
                self.mode_switch_candidate = 0
                self.mode_switch_start = None
            return
        if (
            time_s - self.mode_switch_start
            >= self._confirmation_duration(best_fault_mode)
        ):
            self.confirmed_mode = best_fault_mode
            self.mode_switch_candidate = 0
            self.mode_switch_start = None

    def _update_location(self, time_s):
        proposed = int(np.argmax(self.location_probabilities)) + 1
        confidence = float(self.location_probabilities[proposed - 1])
        if proposed == self.confirmed_location:
            self.location_candidate = 0
            self.location_candidate_start = None
            return
        if confidence < self.config.location_probability_threshold:
            self.location_candidate = 0
            self.location_candidate_start = None
            return
        if proposed != self.location_candidate:
            self.location_candidate = proposed
            self.location_candidate_start = time_s
            if self.config.location_confirmation_s <= 0.0:
                self.confirmed_location = proposed
                self.location_candidate = 0
                self.location_candidate_start = None
            return
        if (
            time_s - self.location_candidate_start
            >= self.config.location_confirmation_s
        ):
            self.confirmed_location = proposed
            self.location_candidate = 0
            self.location_candidate_start = None

    def _update_confirmed(
        self,
        time_s,
        best_fault_mode,
        fault_probability,
    ):
        normal_probability = float(self.mode_probabilities[0])
        if normal_probability >= self.config.exit_normal_probability:
            if self.recovery_start is None:
                self.recovery_start = time_s
            self.state = "recovering"
            if (
                time_s - self.recovery_start
                >= self.config.recovery_confirmation_s
            ):
                self._clear_to_normal()
                return
        else:
            self.recovery_start = None
            self.state = "confirmed"

        self._update_mode_switch(
            time_s,
            best_fault_mode,
            fault_probability,
        )
        self._update_location(time_s)

    def update(self, time_s, mode_probabilities, location_probabilities):
        time_s = float(time_s)
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        mode_probabilities = _probability_vector(
            mode_probabilities, 3, "mode_probabilities"
        )
        location_probabilities = _probability_vector(
            location_probabilities, 6, "location_probabilities"
        )
        self._smooth(time_s, mode_probabilities, location_probabilities)

        best_fault_mode = int(np.argmax(self.mode_probabilities[1:])) + 1
        fault_probability = 1.0 - float(self.mode_probabilities[0])
        if self.confirmed_mode == 0:
            self._update_unconfirmed(
                time_s,
                best_fault_mode,
                fault_probability,
            )
        else:
            self._update_confirmed(
                time_s,
                best_fault_mode,
                fault_probability,
            )

        output_mode = self.confirmed_mode
        output_location = self.confirmed_location if output_mode != 0 else 0
        return TemporalDecisionResult(
            state=self.state,
            mode=output_mode,
            location=output_location,
            smoothed_mode_probabilities=self.mode_probabilities.copy(),
            smoothed_location_probabilities=(
                self.location_probabilities.copy()
            ),
        )


def _as_numpy(values):
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def apply_temporal_decision_layer(
    dataset: Mapping[str, Any],
    indices,
    predictions: Mapping[str, np.ndarray],
    config: TemporalDecisionConfig,
):
    """Apply an independent causal state machine to each mission."""

    indices = np.asarray(indices, dtype=np.int64)
    mission_ids = _as_numpy(dataset["mission_ids"])[indices]
    end_times = _as_numpy(dataset["window_end_times"])[indices]
    mode_probabilities = np.asarray(
        predictions["mode_probabilities"], dtype=float
    )
    location_probabilities = np.asarray(
        predictions["location_probabilities"], dtype=float
    )
    sample_count = len(indices)
    if mode_probabilities.shape != (sample_count, 3):
        raise ValueError("mode_probabilities have incompatible shape")
    if location_probabilities.shape != (sample_count, 6):
        raise ValueError("location_probabilities have incompatible shape")

    mode_pred = np.zeros(sample_count, dtype=np.int64)
    location_pred = np.zeros(sample_count, dtype=np.int64)
    states = np.empty(sample_count, dtype=object)
    for mission_id in np.unique(mission_ids):
        positions = np.flatnonzero(mission_ids == mission_id)
        positions = positions[np.argsort(end_times[positions], kind="stable")]
        decision = TemporalFaultDecision(config)
        for position in positions:
            result = decision.update(
                end_times[position],
                mode_probabilities[position],
                location_probabilities[position],
            )
            mode_pred[position] = result.mode
            location_pred[position] = result.location
            states[position] = result.state

    joint_pred = np.zeros(sample_count, dtype=np.int64)
    fault_mask = mode_pred != 0
    joint_pred[fault_mask] = location_pred[fault_mask]
    thrust_loss_mask = mode_pred == 2
    joint_pred[thrust_loss_mask] += 6
    return {
        "mode_true": np.asarray(predictions["mode_true"], dtype=np.int64),
        "location_true": np.asarray(
            predictions["location_true"], dtype=np.int64
        ),
        "joint_true": np.asarray(predictions["joint_true"], dtype=np.int64),
        "mode_pred": mode_pred,
        "location_pred": location_pred,
        "joint_pred": joint_pred,
        "decision_states": states,
        "mode_probabilities": mode_probabilities,
        "location_probabilities": location_probabilities,
    }
