"""Merge and grade raw health events without changing safety decisions.

The detector, maintenance-ticket policy, and FTC supervisor remain the sources
of truth for diagnosis and control.  This module only changes how retained
health events are grouped and presented to an operator.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .maintenance_health_decision import (
    MaintenanceDecisionConfig,
    _event_summary,
)


LOG_LEVEL_NAMES = (
    "normal",
    "background_trace",
    "observation",
    "maintenance_advisory",
    "safety_alert",
)


@dataclass(frozen=True)
class MaintenanceLogConfig:
    """Presentation-only settings for the retained health log."""

    merge_gap_s: float = 5.0

    def __post_init__(self):
        if not np.isfinite(self.merge_gap_s) or self.merge_gap_s < 0.0:
            raise ValueError("merge_gap_s must be finite and non-negative")


def _as_numpy(values):
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _decision_array(decisions, name, sample_count, *, default, dtype):
    values = decisions.get(name)
    if values is None:
        return np.full(sample_count, default, dtype=dtype)
    result = np.asarray(values, dtype=dtype)
    if result.shape != (sample_count,):
        raise ValueError(f"{name} must contain one value per sample")
    return result


def _context_arrays(dataset, indices, decisions, sample_count):
    context_values = decisions.get("maintenance_guidance_context_id")
    if context_values is None and "guidance_context_ids" in dataset:
        context_values = _as_numpy(dataset["guidance_context_ids"])[indices]
    if context_values is None:
        context_values = np.zeros(sample_count, dtype=np.int64)
    contexts = np.asarray(context_values, dtype=float)
    if (
        contexts.shape != (sample_count,)
        or not np.all(np.isfinite(contexts))
        or np.any(contexts < 0.0)
        or not np.all(contexts == np.floor(contexts))
    ):
        raise ValueError("guidance contexts must be non-negative integers")

    stable_values = decisions.get("maintenance_guidance_context_stable")
    if stable_values is None and "guidance_context_stable" in dataset:
        stable_values = _as_numpy(dataset["guidance_context_stable"])[indices]
    if stable_values is None:
        stable_values = np.ones(sample_count, dtype=bool)
    stable = np.asarray(stable_values, dtype=bool)
    if stable.shape != (sample_count,):
        raise ValueError("guidance context stability has invalid shape")

    available = bool(decisions.get(
        "maintenance_guidance_context_available",
        "guidance_context_ids" in dataset,
    ))
    return contexts.astype(np.int64), stable, available


def _dominant_mode(mode_pred, probable_mode, positions):
    confirmed = mode_pred[positions]
    confirmed = confirmed[confirmed != 0]
    source = confirmed if len(confirmed) else probable_mode[positions]
    source = source[source != 0]
    if not len(source):
        return 0
    counts = Counter(int(value) for value in source)
    return max(counts, key=lambda value: (counts[value], -value))


def _base_segments(
    positions,
    health_levels,
):
    segments = []
    current = []

    def flush():
        nonlocal current
        if current:
            segments.append(np.asarray(current, dtype=np.int64))
            current = []

    for position in positions:
        if health_levels[position] == 0:
            flush()
            continue
        current.append(int(position))
    flush()
    return segments


def _merge_segments(
    segments,
    mission_id,
    end_times,
    mode_pred,
    probable_mode,
    contexts,
    context_stable,
    context_available,
    merge_gap_s,
):
    records = []
    for positions in segments:
        context_ids = np.unique(contexts[positions])
        context_id = int(context_ids[0]) if len(context_ids) == 1 else None
        records.append({
            "mission_id": int(mission_id),
            "positions": positions,
            "mode": _dominant_mode(mode_pred, probable_mode, positions),
            "context_id": context_id,
            "context_stable": bool(np.all(context_stable[positions])),
            "context_available": bool(context_available),
            "source_segment_count": 1,
        })

    merged = []
    for record in records:
        if not merged:
            merged.append(record)
            continue
        previous = merged[-1]
        gap_s = float(
            end_times[record["positions"][0]]
            - end_times[previous["positions"][-1]]
        )
        same_context = (
            previous["context_id"] is not None
            and previous["context_id"] == record["context_id"]
            and (
                not context_available
                or (
                    previous["context_stable"]
                    and record["context_stable"]
                )
            )
        )
        if (
            previous["mode"] == record["mode"]
            and same_context
            and gap_s <= merge_gap_s
        ):
            previous["positions"] = np.concatenate([
                previous["positions"], record["positions"]
            ])
            previous["source_segment_count"] += 1
        else:
            merged.append(record)
    return merged


def _log_level(mode, peak_health_level, formal_ticket_active):
    if formal_ticket_active and mode == 1:
        return 4, "formal_no_output_ticket"
    if formal_ticket_active:
        return 3, "formal_thrust_loss_ticket"
    if peak_health_level >= 2:
        return 2, "persistent_without_formal_ticket"
    return 1, "short_unconfirmed_anomaly"


def apply_maintenance_log_policy(
    dataset: Mapping[str, Any],
    indices,
    decisions: Mapping[str, Any],
    config=None,
    *,
    maintenance_config=None,
):
    """Build a merged, graded view while retaining every raw event."""

    config = config or MaintenanceLogConfig()
    maintenance_config = maintenance_config or MaintenanceDecisionConfig()
    indices = np.asarray(indices, dtype=np.int64)
    sample_count = len(indices)
    mission_ids = _as_numpy(dataset["mission_ids"])[indices]
    end_times = _as_numpy(dataset["window_end_times"])[indices]
    health_levels = _decision_array(
        decisions, "health_level_pred", sample_count, default=0, dtype=np.int64
    )
    mode_pred = _decision_array(
        decisions, "mode_pred", sample_count, default=0, dtype=np.int64
    )
    probable_mode = _decision_array(
        decisions,
        "probable_mode_pred",
        sample_count,
        default=0,
        dtype=np.int64,
    )
    formal_ticket = _decision_array(
        decisions,
        "maintenance_ticket_active",
        sample_count,
        default=False,
        dtype=bool,
    )
    ticket_qualified = _decision_array(
        decisions,
        "maintenance_ticket_raw_qualified",
        sample_count,
        default=False,
        dtype=bool,
    )
    excitation = _decision_array(
        decisions,
        "maintenance_ticket_excitation",
        sample_count,
        default=0.0,
        dtype=float,
    )
    independent = _decision_array(
        decisions,
        "maintenance_ticket_independent_evidence",
        sample_count,
        default=0.0,
        dtype=float,
    )
    contexts, context_stable, context_available = _context_arrays(
        dataset, indices, decisions, sample_count
    )
    fault_probabilities = np.asarray(
        decisions["fault_probabilities"], dtype=float
    )
    location_probabilities = np.asarray(
        decisions["smoothed_location_probabilities"], dtype=float
    )
    if fault_probabilities.shape != (sample_count,):
        raise ValueError("fault_probabilities have incompatible shape")
    if location_probabilities.shape != (sample_count, 6):
        raise ValueError("smoothed_location_probabilities have incompatible shape")

    merged_records = []
    for mission_id in np.unique(mission_ids):
        positions = np.flatnonzero(mission_ids == mission_id)
        positions = positions[np.argsort(end_times[positions], kind="stable")]
        # A continuously active anomaly remains one display event even if the
        # guidance target changes. Context is used only when deciding whether
        # two already-separated episodes may be merged.
        segments = _base_segments(positions, health_levels)
        merged_records.extend(_merge_segments(
            segments,
            mission_id,
            end_times,
            mode_pred,
            probable_mode,
            contexts,
            context_stable,
            context_available,
            config.merge_gap_s,
        ))

    event_membership = np.full(sample_count, -1, dtype=np.int64)
    log_levels = np.zeros(sample_count, dtype=np.int64)
    events = []
    for event_id, record in enumerate(merged_records):
        positions = record["positions"]
        mission_positions = np.flatnonzero(
            mission_ids == record["mission_id"]
        )
        mission_times = np.sort(end_times[mission_positions])
        positive_steps = np.diff(mission_times)
        positive_steps = positive_steps[positive_steps > 0.0]
        step_s = float(np.median(positive_steps)) if len(positive_steps) else 0.0
        event = _event_summary(
            record["mission_id"],
            positions,
            end_times,
            health_levels,
            mode_pred,
            probable_mode,
            fault_probabilities,
            location_probabilities,
            maintenance_config,
        )
        mode = record["mode"]
        formal_active = bool(np.any(formal_ticket[positions]))
        level, reason = _log_level(
            mode,
            int(event["peak_health_level"]),
            formal_active,
        )
        event.update({
            "event_id": int(event_id),
            "log_level": LOG_LEVEL_NAMES[level],
            "log_level_rank": int(level),
            "attention_required": bool(level >= 3),
            "grading_reason": reason,
            "active_window_count": int(len(positions)),
            "active_duration_s": float(len(positions) * step_s),
            "source_segment_count": int(record["source_segment_count"]),
            "guidance_context_id": record["context_id"],
            "guidance_context_stable": bool(record["context_stable"]),
            "guidance_context_available": bool(record["context_available"]),
            "qualified_window_count": int(np.sum(ticket_qualified[positions])),
            "maximum_excitation_ratio": float(np.max(excitation[positions])),
            "maximum_independent_evidence": float(np.max(independent[positions])),
        })
        if level == 4:
            event["operator_action"] = "immediate_safety_attention"
        elif level == 3:
            event["operator_action"] = "post_mission_maintenance_review"
        elif level == 2:
            event["operator_action"] = "retain_in_collapsed_observation_log"
        else:
            event["operator_action"] = "archive_as_background_trace"
        events.append(event)
        event_membership[positions] = event_id
        log_levels[positions] = level

    result = dict(decisions)
    result.update({
        "maintenance_raw_events": list(decisions.get("maintenance_events", [])),
        "maintenance_graded_events": events,
        "maintenance_operator_events": [
            event for event in events if event["attention_required"]
        ],
        "maintenance_observation_events": [
            event for event in events if event["log_level_rank"] == 2
        ],
        "maintenance_trace_events": [
            event for event in events if event["log_level_rank"] == 1
        ],
        "maintenance_log_level_pred": log_levels,
        "maintenance_log_event_membership": event_membership,
        "maintenance_log_config": config,
    })
    return result


def maintenance_log_metrics(dataset, indices, decisions):
    """Measure compression and operator-facing usefulness by event level."""

    indices = np.asarray(indices, dtype=np.int64)
    mission_ids = _as_numpy(dataset["mission_ids"])[indices]
    end_times = _as_numpy(dataset["window_end_times"])[indices]
    mode_true = np.asarray(decisions["mode_true"], dtype=np.int64)
    health_levels = np.asarray(decisions["health_level_pred"], dtype=np.int64)
    log_levels = np.asarray(
        decisions["maintenance_log_level_pred"], dtype=np.int64
    )
    membership = np.asarray(
        decisions["maintenance_log_event_membership"], dtype=np.int64
    )
    sample_count = len(indices)
    for name, values in (
        ("mode_true", mode_true),
        ("health_level_pred", health_levels),
        ("maintenance_log_level_pred", log_levels),
        ("maintenance_log_event_membership", membership),
    ):
        if values.shape != (sample_count,):
            raise ValueError(f"{name} has incompatible shape")

    fault_missions = 0
    raw_detected_missions = 0
    review_detected_missions = 0
    operator_detected_missions = 0
    no_output_missions = 0
    no_output_operator_missions = 0
    thrust_loss_missions = 0
    thrust_loss_operator_missions = 0
    normal_windows = 0
    normal_operator_windows = 0

    for mission_id in np.unique(mission_ids):
        positions = np.flatnonzero(mission_ids == mission_id)
        positions = positions[np.argsort(end_times[positions], kind="stable")]
        true_modes = mode_true[positions]
        targets = np.unique(true_modes[true_modes != 0])
        normal_windows += int(np.sum(true_modes == 0))
        normal_operator_windows += int(np.sum(
            (true_modes == 0) & (log_levels[positions] >= 3)
        ))
        if not len(targets):
            continue
        target = int(targets[0])
        fault_missions += 1
        fault_start = float(end_times[positions[np.flatnonzero(
            true_modes == target
        )[0]]])
        post_fault = end_times[positions] >= fault_start
        if np.any(post_fault & (health_levels[positions] >= 1)):
            raw_detected_missions += 1
        if np.any(post_fault & (log_levels[positions] >= 2)):
            review_detected_missions += 1
        operator_detected = bool(np.any(
            post_fault & (log_levels[positions] >= 3)
        ))
        operator_detected_missions += int(operator_detected)
        if target == 1:
            no_output_missions += 1
            no_output_operator_missions += int(operator_detected)
        elif target == 2:
            thrust_loss_missions += 1
            thrust_loss_operator_missions += int(operator_detected)

    true_operator_events = 0
    false_operator_events = 0
    false_retained_events = 0
    for event in decisions["maintenance_graded_events"]:
        positions = np.flatnonzero(membership == int(event["event_id"]))
        contains_fault = bool(np.any(mode_true[positions] != 0))
        if event["attention_required"]:
            if contains_fault:
                true_operator_events += 1
            else:
                false_operator_events += 1
        elif not contains_fault:
            false_retained_events += 1

    events = decisions["maintenance_graded_events"]
    raw_events = decisions.get("maintenance_raw_events", [])
    counts = Counter(event["log_level"] for event in events)
    operator_denominator = true_operator_events + false_operator_events
    return {
        "raw_event_count": int(len(raw_events)),
        "merged_event_count": int(len(events)),
        "event_count_reduction": int(len(raw_events) - len(events)),
        "event_compression_ratio": (
            float(1.0 - len(events) / len(raw_events)) if raw_events else 0.0
        ),
        "background_trace_count": int(counts["background_trace"]),
        "observation_count": int(counts["observation"]),
        "maintenance_advisory_count": int(counts["maintenance_advisory"]),
        "safety_alert_count": int(counts["safety_alert"]),
        "operator_attention_event_count": int(
            counts["maintenance_advisory"] + counts["safety_alert"]
        ),
        "operator_event_precision": (
            true_operator_events / operator_denominator
            if operator_denominator else None
        ),
        "false_operator_attention_events": int(false_operator_events),
        "false_events_retained_without_prompt": int(false_retained_events),
        "raw_fault_mission_recall": (
            raw_detected_missions / max(fault_missions, 1)
        ),
        "review_log_fault_mission_recall": (
            review_detected_missions / max(fault_missions, 1)
        ),
        "operator_attention_fault_mission_recall": (
            operator_detected_missions / max(fault_missions, 1)
        ),
        "no_output_operator_attention_recall": (
            no_output_operator_missions / no_output_missions
            if no_output_missions else None
        ),
        "thrust_loss_operator_attention_recall": (
            thrust_loss_operator_missions / thrust_loss_missions
            if thrust_loss_missions else None
        ),
        "normal_window_operator_attention_rate": (
            normal_operator_windows / max(normal_windows, 1)
        ),
    }
