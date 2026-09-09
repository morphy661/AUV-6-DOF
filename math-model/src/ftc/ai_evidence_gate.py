"""Validate online model evidence before admitting it to the FTC supervisor."""

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from ftc.safety_supervisor import FTCSupervisorConfig


@dataclass(frozen=True)
class AIEvidenceGateConfig:
    # The probability threshold is inherited from the frozen temporal policy.
    minimum_mode_probability: float = 0.85
    maximum_result_age_s: float = 0.60
    minimum_confirming_updates: int = 2

    def __post_init__(self):
        if not np.isfinite(self.minimum_mode_probability) or not 0 <= self.minimum_mode_probability <= 1:
            raise ValueError("minimum_mode_probability must be in [0, 1]")
        if not np.isfinite(self.maximum_result_age_s) or self.maximum_result_age_s <= 0:
            raise ValueError("maximum_result_age_s must be positive")
        if self.minimum_confirming_updates < 2 or int(self.minimum_confirming_updates) != self.minimum_confirming_updates:
            raise ValueError("minimum_confirming_updates must be an integer >= 2")


class AIControlEvidenceGate:
    """Admit persistent AI evidence only with observable control stress.

    Never changes direct ESC scores, sensor safety guards or isolation targets.
    A verdict is a control-policy assessment, not ground-truth confirmation.
    """

    def __init__(self, config=None, supervisor_config=None):
        self.config = config or AIEvidenceGateConfig()
        self.supervisor_config = supervisor_config or FTCSupervisorConfig()
        self.reset()

    def reset(self):
        self.mode = 0
        self.confirming_updates = 0
        self.last_inference_time = None

    def update(self, evidence, diagnosis):
        report = {
            "verdict": "unavailable", "accepted": False,
            "reason": "No usable online model result.",
            "time_s": evidence.time_s, "inference_time_s": None,
            "result_age_s": None, "confirmed_mode": 0,
            "confirming_updates": 0, "mode_probability": None,
            "control_corroborated": False,
            "tracking_error_available": bool(evidence.tracking_error_available),
            "supporting_control_signals": [],
            "location_used_for_isolation": False,
        }

        def reject(verdict, reason, *, reset=True):
            if reset:
                self.reset()
            report.update(verdict=verdict, reason=reason,
                          confirming_updates=self.confirming_updates)
            return evidence, report

        cfg = self.supervisor_config
        if np.max(evidence.no_output_scores) >= cfg.no_output_score_threshold:
            return reject("direct_rule_priority", "Direct ESC evidence follows its own confirmation timer.")
        if not isinstance(diagnosis, Mapping) or not diagnosis.get("available", False):
            return reject("unavailable", "Model window is unavailable or warming up.")
        if diagnosis.get("advisory_gate_active", False) or diagnosis.get("advisory_suppressed", False):
            return reject("context_blocked", "Current model context is suppressed.")
        if evidence.untrusted_esc_channels or evidence.untrusted_sensors or evidence.sensor_guard_action != "none":
            return reject("sensor_or_esc_guard", "Sensor or ESC trust is not sufficient for AI corroboration.")
        try:
            inference_time = float(diagnosis["inference_time_s"])
            mode = int(diagnosis.get("confirmed_mode", 0))
            probabilities = np.asarray(diagnosis["raw_mode_probabilities"], dtype=float)
        except (KeyError, TypeError, ValueError, OverflowError):
            return reject("invalid_result", "Missing or invalid model time/probabilities.")
        age = evidence.time_s - inference_time
        if not np.isfinite(inference_time) or age < -1e-9 or age > self.config.maximum_result_age_s + 1e-9:
            return reject("stale_or_future_result", "Model result is too old or from a future time.")
        report.update(inference_time_s=inference_time, result_age_s=max(age, 0.0))
        if probabilities.shape != (3,) or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
            return reject("invalid_result", "Expected a normalized three-class probability vector.")
        if diagnosis.get("temporal_state") == "recovering":
            return reject("recovering", "Model is observing recovery; no fresh fault authorization.")
        if mode not in (1, 2) or diagnosis.get("temporal_state") != "confirmed":
            verdict = "suspected" if diagnosis.get("health_level", 0) else "normal"
            return reject(verdict, "Awaiting temporal fault-mode confirmation.")
        report.update(confirmed_mode=mode, mode_probability=float(probabilities[mode]))
        if np.argmax(probabilities) != mode or probabilities[mode] < self.config.minimum_mode_probability:
            return reject("probability_conflict", "Fresh model probability does not support the latched fault mode.")
        if self.last_inference_time is not None:
            if inference_time < self.last_inference_time:
                return reject("out_of_order_result", "Model inference times moved backwards.")
            if inference_time - self.last_inference_time > self.config.maximum_result_age_s + 1e-9:
                self.reset()
        if mode != self.mode:
            self.reset()
            self.mode = mode
        if diagnosis.get("updated", False) and (
            self.last_inference_time is None or inference_time > self.last_inference_time
        ):
            self.confirming_updates += 1
            self.last_inference_time = inference_time
        report["confirming_updates"] = self.confirming_updates
        if self.confirming_updates < self.config.minimum_confirming_updates:
            return reject("suspected", "Need distinct consecutive confirmed inference updates.", reset=False)
        control_corroborated = bool(
            evidence.control_saturation_ratio >= cfg.degraded_control_saturation_ratio
            and ((evidence.tracking_error_available and evidence.tracking_error_ratio >= cfg.degraded_tracking_error_ratio)
                 or evidence.allocation_residual_ratio >= cfg.degraded_allocation_residual_ratio)
        )
        report["control_corroborated"] = control_corroborated
        if control_corroborated:
            report["supporting_control_signals"].append("control_saturation")
            if evidence.tracking_error_available and evidence.tracking_error_ratio >= cfg.degraded_tracking_error_ratio:
                report["supporting_control_signals"].append("tracking_error")
            if evidence.allocation_residual_ratio >= cfg.degraded_allocation_residual_ratio:
                report["supporting_control_signals"].append("allocation_residual")
        if not control_corroborated:
            return reject("record_only", "Persistent model fault; control response does not justify added FTC authority.", reset=False)
        report.update(verdict="accepted_control_evidence", accepted=True,
                      reason="Persistent model mode plus saturation and tracking/allocation stress.")
        return replace(
            evidence, health_level=max(evidence.health_level, 2),
            confirmed_mode=mode, fault_probability=max(evidence.fault_probability, float(probabilities[mode])),
            probable_group=str(diagnosis.get("suspected_group", "uncertain")),
            source=evidence.source + "+validated_online_ai",
        ), report
