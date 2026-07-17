"""Maintenance-oriented health decisions for the six-thruster AUV.

This layer deliberately separates two questions:

1. Is there a persistent vehicle/actuator anomaly that should be recorded?
2. Which thruster should a human inspect first?

The second answer is advisory.  It is represented by a group and Top-K
candidate list instead of a forced exact-thruster diagnosis.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .temporal_fault_decision import (
    TemporalDecisionConfig,
    TemporalFaultDecision,
)


HEALTH_LEVEL_NAMES = (
    "normal",
    "transient_observation",
    "persistent_degradation",
    "critical_fault",
)
FAULT_MODE_NAMES = ("normal", "no_output", "thrust_loss")
THRUSTER_NAMES = ("H1", "H2", "H3", "H4", "V1", "V2")


@dataclass(frozen=True)
class MaintenanceDecisionConfig:
    """Thresholds for maintenance advice and safety escalation."""

    top_k: int = 2
    group_confidence_threshold: float = 0.60
    group_margin_threshold: float = 0.15
    medium_location_probability: float = 0.30
    medium_location_margin: float = 0.05
    high_location_probability: float = 0.55
    high_location_margin: float = 0.20
    critical_tracking_error_ratio: float = 1.0
    critical_control_saturation_ratio: float = 0.90
    no_output_is_critical: bool = True

    def __post_init__(self):
        if not 1 <= int(self.top_k) <= len(THRUSTER_NAMES):
            raise ValueError("top_k must be between 1 and 6")
        probabilities = {
            "group_confidence_threshold": self.group_confidence_threshold,
            "group_margin_threshold": self.group_margin_threshold,
            "medium_location_probability": self.medium_location_probability,
            "medium_location_margin": self.medium_location_margin,
            "high_location_probability": self.high_location_probability,
            "high_location_margin": self.high_location_margin,
            "critical_control_saturation_ratio": (
                self.critical_control_saturation_ratio
            ),
        }
        for name, value in probabilities.items():
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if (
            not np.isfinite(self.critical_tracking_error_ratio)
            or self.critical_tracking_error_ratio < 0.0
        ):
            raise ValueError(
                "critical_tracking_error_ratio must be finite and non-negative"
            )


@dataclass(frozen=True)
class ThrusterCandidate:
    index: int
    name: str
    probability: float


@dataclass(frozen=True)
class MaintenanceDecisionResult:
    health_level: int
    health_state: str
    temporal_state: str
    confirmed_mode: int
    probable_mode: int
    fault_probability: float
    suspected_group: str
    group_confidence: float
    candidates: tuple[ThrusterCandidate, ...]
    location_confidence: str
    action: str
    record_event: bool
    requires_ftc: bool
    smoothed_mode_probabilities: np.ndarray
    smoothed_location_probabilities: np.ndarray


def _normalise_probabilities(values, size, name):
    probabilities = np.asarray(values, dtype=float)
    if probabilities.shape != (size,) or not np.all(
        np.isfinite(probabilities)
    ):
        raise ValueError(f"{name} must contain {size} finite probabilities")
    if np.any(probabilities < 0.0) or probabilities.sum() <= 0.0:
        raise ValueError(f"{name} must contain non-negative probability mass")
    return probabilities / probabilities.sum()


def _location_summary(probabilities, config):
    probabilities = _normalise_probabilities(
        probabilities, len(THRUSTER_NAMES), "location_probabilities"
    )
    ranking = np.argsort(-probabilities, kind="stable")
    candidates = tuple(
        ThrusterCandidate(
            index=int(index) + 1,
            name=THRUSTER_NAMES[int(index)],
            probability=float(probabilities[index]),
        )
        for index in ranking[: config.top_k]
    )

    top_probability = candidates[0].probability
    runner_up_probability = (
        candidates[1].probability if len(candidates) > 1 else 0.0
    )
    location_margin = top_probability - runner_up_probability
    if (
        top_probability >= config.high_location_probability
        and location_margin >= config.high_location_margin
    ):
        location_confidence = "high"
    elif (
        top_probability >= config.medium_location_probability
        and location_margin >= config.medium_location_margin
    ):
        location_confidence = "medium"
    else:
        location_confidence = "low"

    # Correct for the unequal group sizes (four horizontal and two vertical).
    # A uniform six-way distribution therefore yields 0.5/0.5, not a false
    # horizontal preference of 4/6.
    group_evidence = np.asarray([
        probabilities[:4].mean(),
        probabilities[4:].mean(),
    ])
    group_probabilities = group_evidence / group_evidence.sum()
    best_group = int(np.argmax(group_probabilities))
    group_confidence = float(group_probabilities[best_group])
    group_margin = float(abs(group_probabilities[0] - group_probabilities[1]))
    if (
        group_confidence >= config.group_confidence_threshold
        and group_margin >= config.group_margin_threshold
    ):
        suspected_group = ("horizontal", "vertical")[best_group]
    else:
        suspected_group = "uncertain"
    return candidates, suspected_group, group_confidence, location_confidence


class MaintenanceHealthDecision:
    """Convert temporal detector output into maintenance and safety actions."""

    def __init__(self, temporal_config=None, maintenance_config=None):
        self.temporal = TemporalFaultDecision(
            temporal_config or TemporalDecisionConfig()
        )
        self.config = maintenance_config or MaintenanceDecisionConfig()

    def reset(self):
        self.temporal.reset()

    @staticmethod
    def _operating_ratio(value, name):
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return value

    def update(
        self,
        time_s,
        mode_probabilities,
        location_probabilities,
        *,
        tracking_error_ratio=0.0,
        control_saturation_ratio=0.0,
    ):
        """Update one causal decision window.

        Ratios are optional deployment signals.  ``tracking_error_ratio`` is
        tracking error divided by its safe limit; ``control_saturation_ratio``
        is the largest absolute allocated thrust divided by its limit.
        """

        tracking_error_ratio = self._operating_ratio(
            tracking_error_ratio, "tracking_error_ratio"
        )
        control_saturation_ratio = self._operating_ratio(
            control_saturation_ratio, "control_saturation_ratio"
        )
        temporal = self.temporal.update(
            time_s, mode_probabilities, location_probabilities
        )
        probable_mode = int(
            np.argmax(temporal.smoothed_mode_probabilities[1:])
        ) + 1
        fault_probability = 1.0 - float(
            temporal.smoothed_mode_probabilities[0]
        )

        candidates, group, group_confidence, location_confidence = (
            _location_summary(
                temporal.smoothed_location_probabilities,
                self.config,
            )
        )
        if temporal.state == "normal":
            health_level = 0
            action = "none"
        elif temporal.state == "suspected":
            health_level = 1
            action = "record_and_observe"
        else:
            critical = (
                (
                    self.config.no_output_is_critical
                    and temporal.mode == 1
                )
                or tracking_error_ratio
                >= self.config.critical_tracking_error_ratio
                or control_saturation_ratio
                >= self.config.critical_control_saturation_ratio
            )
            if critical:
                health_level = 3
                action = "safety_alert_and_consider_ftc"
            else:
                health_level = 2
                action = "log_for_post_mission_inspection"

        if health_level == 0:
            candidates = tuple()
            group = "none"
            group_confidence = 0.0
            location_confidence = "none"
        return MaintenanceDecisionResult(
            health_level=health_level,
            health_state=HEALTH_LEVEL_NAMES[health_level],
            temporal_state=temporal.state,
            confirmed_mode=temporal.mode,
            probable_mode=(probable_mode if health_level != 0 else 0),
            fault_probability=fault_probability,
            suspected_group=group,
            group_confidence=group_confidence,
            candidates=candidates,
            location_confidence=location_confidence,
            action=action,
            record_event=health_level != 0,
            requires_ftc=health_level == 3,
            smoothed_mode_probabilities=(
                temporal.smoothed_mode_probabilities.copy()
            ),
            smoothed_location_probabilities=(
                temporal.smoothed_location_probabilities.copy()
            ),
        )


def _as_numpy(values):
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _event_summary(
    mission_id,
    positions,
    end_times,
    health_levels,
    mode_predictions,
    probable_modes,
    fault_probabilities,
    location_probabilities,
    config,
):
    average_location = location_probabilities[positions].mean(axis=0)
    candidates, group, group_confidence, location_confidence = (
        _location_summary(average_location, config)
    )
    confirmed = mode_predictions[positions]
    confirmed = confirmed[confirmed != 0]
    if len(confirmed):
        counts = np.bincount(confirmed, minlength=len(FAULT_MODE_NAMES))
        mode = int(np.argmax(counts[1:])) + 1
    else:
        probable = probable_modes[positions]
        counts = np.bincount(probable, minlength=len(FAULT_MODE_NAMES))
        mode = int(np.argmax(counts[1:])) + 1
    peak_level = int(np.max(health_levels[positions]))
    category = HEALTH_LEVEL_NAMES[peak_level]
    return {
        "mission_id": int(mission_id),
        "start_time_s": float(end_times[positions[0]]),
        "end_time_s": float(end_times[positions[-1]]),
        "duration_s": float(
            end_times[positions[-1]] - end_times[positions[0]]
        ),
        "peak_health_level": peak_level,
        "category": category,
        "probable_fault_mode": FAULT_MODE_NAMES[mode],
        "maximum_fault_probability": float(
            np.max(fault_probabilities[positions])
        ),
        "suspected_group": group,
        "group_confidence": group_confidence,
        "location_confidence": location_confidence,
        "inspection_candidates": [
            {
                "index": candidate.index,
                "name": candidate.name,
                "probability": candidate.probability,
            }
            for candidate in candidates
        ],
        "requires_ftc": peak_level == 3,
    }


def apply_maintenance_decision_layer(
    dataset: Mapping[str, Any],
    indices,
    predictions: Mapping[str, np.ndarray],
    temporal_config: TemporalDecisionConfig,
    maintenance_config=None,
    *,
    tracking_error_ratios=None,
    control_saturation_ratios=None,
):
    """Apply independent maintenance monitors and build an event log."""

    config = maintenance_config or MaintenanceDecisionConfig()
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

    def context_array(values, name):
        if values is None:
            return np.zeros(sample_count, dtype=float)
        result = np.asarray(values, dtype=float)
        if result.shape != (sample_count,) or np.any(result < 0.0) or not np.all(
            np.isfinite(result)
        ):
            raise ValueError(f"{name} must contain one non-negative value per sample")
        return result

    tracking_error_ratios = context_array(
        tracking_error_ratios, "tracking_error_ratios"
    )
    control_saturation_ratios = context_array(
        control_saturation_ratios, "control_saturation_ratios"
    )

    health_levels = np.zeros(sample_count, dtype=np.int64)
    health_states = np.empty(sample_count, dtype=object)
    temporal_states = np.empty(sample_count, dtype=object)
    mode_pred = np.zeros(sample_count, dtype=np.int64)
    probable_mode_pred = np.zeros(sample_count, dtype=np.int64)
    location_pred = np.zeros(sample_count, dtype=np.int64)
    fault_probabilities = np.zeros(sample_count, dtype=float)
    suspected_groups = np.empty(sample_count, dtype=object)
    group_confidences = np.zeros(sample_count, dtype=float)
    location_confidences = np.empty(sample_count, dtype=object)
    candidate_indices = np.zeros((sample_count, config.top_k), dtype=np.int64)
    candidate_probabilities = np.zeros(
        (sample_count, config.top_k), dtype=float
    )
    actions = np.empty(sample_count, dtype=object)
    requires_ftc = np.zeros(sample_count, dtype=bool)
    smoothed_mode = np.zeros_like(mode_probabilities)
    smoothed_location = np.zeros_like(location_probabilities)

    events = []
    for mission_id in np.unique(mission_ids):
        positions = np.flatnonzero(mission_ids == mission_id)
        positions = positions[np.argsort(end_times[positions], kind="stable")]
        decision = MaintenanceHealthDecision(temporal_config, config)
        for position in positions:
            result = decision.update(
                end_times[position],
                mode_probabilities[position],
                location_probabilities[position],
                tracking_error_ratio=tracking_error_ratios[position],
                control_saturation_ratio=control_saturation_ratios[position],
            )
            health_levels[position] = result.health_level
            health_states[position] = result.health_state
            temporal_states[position] = result.temporal_state
            mode_pred[position] = result.confirmed_mode
            probable_mode_pred[position] = result.probable_mode
            if result.confirmed_mode:
                location_pred[position] = int(
                    np.argmax(result.smoothed_location_probabilities)
                ) + 1
            fault_probabilities[position] = result.fault_probability
            suspected_groups[position] = result.suspected_group
            group_confidences[position] = result.group_confidence
            location_confidences[position] = result.location_confidence
            actions[position] = result.action
            requires_ftc[position] = result.requires_ftc
            smoothed_mode[position] = result.smoothed_mode_probabilities
            smoothed_location[position] = (
                result.smoothed_location_probabilities
            )
            for rank, candidate in enumerate(result.candidates):
                candidate_indices[position, rank] = candidate.index
                candidate_probabilities[position, rank] = candidate.probability

        active_positions = []
        for position in positions:
            if health_levels[position] != 0:
                active_positions.append(position)
            elif active_positions:
                events.append(_event_summary(
                    mission_id,
                    np.asarray(active_positions, dtype=np.int64),
                    end_times,
                    health_levels,
                    mode_pred,
                    probable_mode_pred,
                    fault_probabilities,
                    smoothed_location,
                    config,
                ))
                active_positions = []
        if active_positions:
            events.append(_event_summary(
                mission_id,
                np.asarray(active_positions, dtype=np.int64),
                end_times,
                health_levels,
                mode_pred,
                probable_mode_pred,
                fault_probabilities,
                smoothed_location,
                config,
            ))

    joint_pred = np.zeros(sample_count, dtype=np.int64)
    fault_mask = mode_pred != 0
    joint_pred[fault_mask] = location_pred[fault_mask]
    joint_pred[mode_pred == 2] += 6
    return {
        "mode_true": np.asarray(predictions["mode_true"], dtype=np.int64),
        "location_true": np.asarray(
            predictions["location_true"], dtype=np.int64
        ),
        "joint_true": np.asarray(predictions["joint_true"], dtype=np.int64),
        "mode_pred": mode_pred,
        "probable_mode_pred": probable_mode_pred,
        "location_pred": location_pred,
        "joint_pred": joint_pred,
        "health_level_pred": health_levels,
        "health_state_pred": health_states,
        "temporal_state_pred": temporal_states,
        "fault_probabilities": fault_probabilities,
        "suspected_groups": suspected_groups,
        "group_confidences": group_confidences,
        "location_confidences": location_confidences,
        "candidate_indices": candidate_indices,
        "candidate_probabilities": candidate_probabilities,
        "maintenance_actions": actions,
        "requires_ftc": requires_ftc,
        "mode_probabilities": mode_probabilities,
        "location_probabilities": location_probabilities,
        "smoothed_mode_probabilities": smoothed_mode,
        "smoothed_location_probabilities": smoothed_location,
        "maintenance_events": events,
    }


def _segment_count(mask):
    mask = np.asarray(mask, dtype=bool)
    if not len(mask):
        return 0
    return int(mask[0]) + int(np.sum(mask[1:] & ~mask[:-1]))


def maintenance_event_metrics(dataset, indices, decisions):
    """Evaluate event detection and inspection usefulness, not exact identity."""

    indices = np.asarray(indices, dtype=np.int64)
    mission_ids = _as_numpy(dataset["mission_ids"])[indices]
    end_times = _as_numpy(dataset["window_end_times"])[indices]
    mode_true = np.asarray(decisions["mode_true"], dtype=np.int64)
    location_true = np.asarray(decisions["location_true"], dtype=np.int64)
    mode_pred = np.asarray(decisions["mode_pred"], dtype=np.int64)
    health_level = np.asarray(
        decisions["health_level_pred"], dtype=np.int64
    )
    requires_ftc = np.asarray(decisions["requires_ftc"], dtype=bool)
    location_probabilities = np.asarray(
        decisions["smoothed_location_probabilities"], dtype=float
    )

    fault_missions = 0
    detected_fault_missions = 0
    correctly_judged_mode_missions = 0
    no_output_missions = 0
    missed_no_output_missions = 0
    false_alarm_events = 0
    normal_exposure_s = 0.0
    normal_windows = 0
    observed_normal_windows = 0
    advisory_normal_windows = 0
    detection_delays = []
    alert_fragments = []
    top1_hits = []
    top2_hits = []
    group_hits = []

    for mission_id in np.unique(mission_ids):
        positions = np.flatnonzero(mission_ids == mission_id)
        positions = positions[np.argsort(end_times[positions], kind="stable")]
        times = end_times[positions]
        true_modes = mode_true[positions]
        true_locations = location_true[positions]
        levels = health_level[positions]
        predicted_modes = mode_pred[positions]
        ftc_flags = requires_ftc[positions]
        normal_mask = true_modes == 0
        advisory_mask = levels >= 2
        observation_mask = levels >= 1

        normal_windows += int(np.sum(normal_mask))
        observed_normal_windows += int(np.sum(normal_mask & observation_mask))
        advisory_normal_windows += int(np.sum(normal_mask & advisory_mask))
        false_alarm_events += _segment_count(normal_mask & advisory_mask)
        differences = np.diff(times)
        positive_differences = differences[differences > 0.0]
        step = (
            float(np.median(positive_differences))
            if len(positive_differences)
            else 0.0
        )
        normal_exposure_s += float(np.sum(normal_mask)) * step

        targets = np.unique(true_modes[true_modes != 0])
        if not len(targets):
            continue
        target_mode = int(targets[0])
        fault_missions += 1
        fault_positions = np.flatnonzero(true_modes == target_mode)
        first_fault_position = int(fault_positions[0])
        fault_start = float(times[first_fault_position])
        post_fault = times >= fault_start
        detected = post_fault & advisory_mask
        fragments = _segment_count(detected)
        if not np.any(detected):
            if target_mode == 1:
                no_output_missions += 1
                missed_no_output_missions += 1
            continue

        detected_fault_missions += 1
        alert_fragments.append(fragments)
        first_detection = int(np.flatnonzero(detected)[0])
        detection_delays.append(
            max(0.0, float(times[first_detection] - fault_start))
        )
        if np.any(detected & (predicted_modes == target_mode)):
            correctly_judged_mode_missions += 1

        if target_mode == 1:
            no_output_missions += 1
            if not np.any(post_fault & ftc_flags):
                missed_no_output_missions += 1

        target_locations = np.unique(
            true_locations[true_locations != 0]
        )
        if len(target_locations):
            target_location = int(target_locations[0])
            average_location = location_probabilities[positions][detected].mean(
                axis=0
            )
            ranking = np.argsort(-average_location, kind="stable") + 1
            top1_hits.append(float(ranking[0] == target_location))
            top2_hits.append(float(target_location in ranking[:2]))
            target_group = "horizontal" if target_location <= 4 else "vertical"
            group_evidence = np.asarray([
                average_location[:4].mean(),
                average_location[4:].mean(),
            ])
            predicted_group = ("horizontal", "vertical")[
                int(np.argmax(group_evidence))
            ]
            group_hits.append(float(predicted_group == target_group))

    denominator = detected_fault_missions + false_alarm_events
    return {
        "event_recall": (
            detected_fault_missions / max(fault_missions, 1)
        ),
        "event_precision": (
            detected_fault_missions / denominator if denominator else None
        ),
        "fault_mode_judgement_rate": (
            correctly_judged_mode_missions / max(fault_missions, 1)
        ),
        "severe_no_output_miss_rate": (
            missed_no_output_missions / no_output_missions
            if no_output_missions else None
        ),
        "false_advisory_events_per_hour": (
            false_alarm_events / (normal_exposure_s / 3600.0)
            if normal_exposure_s > 0.0 else None
        ),
        "normal_window_observation_rate": (
            observed_normal_windows / max(normal_windows, 1)
        ),
        "normal_window_advisory_rate": (
            advisory_normal_windows / max(normal_windows, 1)
        ),
        "mean_detection_delay_s": (
            float(np.mean(detection_delays)) if detection_delays else None
        ),
        "median_detection_delay_s": (
            float(np.median(detection_delays)) if detection_delays else None
        ),
        "stable_single_alert_rate": (
            float(np.mean(np.asarray(alert_fragments) == 1))
            if alert_fragments else None
        ),
        "mean_alert_fragments_per_detected_mission": (
            float(np.mean(alert_fragments)) if alert_fragments else None
        ),
        "probable_group_accuracy": (
            float(np.mean(group_hits)) if group_hits else None
        ),
        "top2_location_hit_rate": (
            float(np.mean(top2_hits)) if top2_hits else None
        ),
        "top1_location_hit_rate_for_reference": (
            float(np.mean(top1_hits)) if top1_hits else None
        ),
        "detected_fault_missions": detected_fault_missions,
        "total_fault_missions": fault_missions,
        "false_advisory_events": false_alarm_events,
    }
