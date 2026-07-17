"""Causal context gate for learned six-DOF maintenance advice.

The gate only controls whether learned advice is presented to the operator. It
does not modify sensor diagnosis, direct thruster evidence, or FTC decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping


SENSOR_NAMES = ("depth", "imu", "dvl")
THRUSTER_NAMES = ("H1", "H2", "H3", "H4", "V1", "V2")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _context_id(log: Mapping[str, Any]) -> int | None:
    value = log.get("guidance_context_id")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(number) or number < 0.0 or number != int(number):
        return None
    return int(number)


def _direct_sensor_faults(log: Mapping[str, Any]) -> frozenset[str]:
    health = _mapping(log.get("sensor_health"))
    active = set()
    for sensor in SENSOR_NAMES:
        state = _mapping(health.get(sensor))
        if (
            bool(state.get("confirmed", False))
            or str(state.get("fault_type", "normal")) != "normal"
            or str(state.get("health_state", "healthy")) != "healthy"
            or str(state.get("trust_level", "trusted")) == "untrusted"
        ):
            active.add(sensor)
    return frozenset(active)


def _operator_sensor_hypotheses(log: Mapping[str, Any]) -> frozenset[str]:
    observations = _mapping(log.get("sensor_fault_observations"))
    active = set()
    for sensor in SENSOR_NAMES:
        state = _mapping(observations.get(sensor))
        if (
            str(state.get("state", "normal")) == "possible_fault"
            and str(state.get("display_level", "background")) == "possible"
        ):
            active.add(sensor)
    return frozenset(active)


def _untrusted_esc_channels(log: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        str(value)
        for value in log.get("ftc_untrusted_esc_channels", ())
        if str(value) in THRUSTER_NAMES
    )


@dataclass(frozen=True)
class AdvisoryGateConfig:
    """Configuration for causal learned-advice stabilization."""

    stabilization_time_s: float = 3.0

    def __post_init__(self):
        if not isfinite(self.stabilization_time_s):
            raise ValueError("stabilization_time_s must be finite")
        if self.stabilization_time_s < 0.0:
            raise ValueError("stabilization_time_s must be non-negative")


@dataclass(frozen=True)
class AdvisoryGateDecision:
    reset_model_context: bool
    active: bool
    reasons: tuple[str, ...]
    suppress_until_s: float


class AdvisoryContextGate:
    """Detect causal context transitions and hold learned advice temporarily."""

    def __init__(self, config: AdvisoryGateConfig | None = None):
        self.config = config or AdvisoryGateConfig()
        self.reset()

    def reset(self):
        self._guidance_context_id = None
        self._direct_faults = frozenset()
        self._operator_hypotheses = frozenset()
        self._untrusted_esc_channels = frozenset()
        self._suppress_until_s = float("-inf")
        self._last_transition_reasons: tuple[str, ...] = ()
        self._fresh_inference_required = False

    def _decision(self, time_s, *, reset_model_context=False):
        reasons = []
        if self._direct_faults:
            reasons.append("active_direct_sensor_fault")
        if self._untrusted_esc_channels:
            reasons.append("active_esc_telemetry_fault")
        if time_s < self._suppress_until_s:
            reasons.append("context_stabilization")
            reasons.extend(self._last_transition_reasons)
        elif self._fresh_inference_required:
            reasons.append("awaiting_fresh_model_inference")
        return AdvisoryGateDecision(
            reset_model_context=bool(reset_model_context),
            active=bool(reasons),
            reasons=tuple(dict.fromkeys(reasons)),
            suppress_until_s=float(self._suppress_until_s),
        )

    def update(self, log: Mapping[str, Any]) -> AdvisoryGateDecision:
        time_s = float(log.get("time", 0.0))
        if not isfinite(time_s):
            raise ValueError("log time must be finite")

        transitions = []
        context_id = _context_id(log)
        if self._guidance_context_id is None:
            self._guidance_context_id = context_id
        elif context_id is not None and context_id != self._guidance_context_id:
            transitions.append("guidance_context_change")
            self._guidance_context_id = context_id

        direct_faults = _direct_sensor_faults(log)
        direct_onsets = direct_faults - self._direct_faults
        if direct_onsets:
            transitions.append(
                "direct_sensor_fault:" + ",".join(sorted(direct_onsets))
            )
        self._direct_faults = direct_faults

        hypotheses = _operator_sensor_hypotheses(log)
        hypothesis_onsets = hypotheses - self._operator_hypotheses
        if hypothesis_onsets:
            transitions.append(
                "sensor_hypothesis:" + ",".join(sorted(hypothesis_onsets))
            )
        self._operator_hypotheses = hypotheses

        untrusted_esc = _untrusted_esc_channels(log)
        esc_onsets = untrusted_esc - self._untrusted_esc_channels
        esc_recoveries = self._untrusted_esc_channels - untrusted_esc
        if esc_onsets:
            transitions.append(
                "esc_telemetry_fault:" + ",".join(sorted(esc_onsets))
            )
        if esc_recoveries:
            transitions.append(
                "esc_telemetry_recovered:" + ",".join(
                    sorted(esc_recoveries)
                )
            )
        self._untrusted_esc_channels = untrusted_esc

        if transitions:
            self._last_transition_reasons = tuple(transitions)
            self._fresh_inference_required = True
            self._suppress_until_s = max(
                self._suppress_until_s,
                time_s + self.config.stabilization_time_s,
            )

        return self._decision(
            time_s, reset_model_context=bool(transitions)
        )

    def mark_model_inference(self, time_s: float) -> AdvisoryGateDecision:
        """Release a stable gate only after inference on post-transition data."""

        time_s = float(time_s)
        if not isfinite(time_s):
            raise ValueError("inference time must be finite")
        if (
            time_s >= self._suppress_until_s
            and not self._direct_faults
            and not self._untrusted_esc_channels
        ):
            self._fresh_inference_required = False
        return self._decision(time_s)
