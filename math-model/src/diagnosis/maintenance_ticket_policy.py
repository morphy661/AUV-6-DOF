"""Turn raw health observations into conservative maintenance tickets.

Raw anomaly events are deliberately retained. A formal maintenance ticket is
created only when the temporal diagnosis is supported by sufficient thruster
excitation and a second observable evidence channel.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


THRUSTER_NAMES = ("H1", "H2", "H3", "H4", "V1", "V2")


@dataclass(frozen=True)
class MaintenanceTicketConfig:
    minimum_excitation_ratio: float = 0.10
    minimum_thrust_loss_motion_evidence: float = 0.20
    minimum_no_output_local_anomaly: float = 0.70
    ticket_confirmation_s: float = 2.50
    thrust_loss_pending_confirmation_s: float = 8.00
    thrust_loss_recovery_cancel_s: float = 3.75
    thrust_loss_recurrence_window_s: float = 30.0
    thrust_loss_recurrence_count: int = 2
    ticket_recovery_s: float = 2.50
    merge_gap_s: float = 30.0
    evidence_tail_steps: int = 20

    def __post_init__(self):
        probabilities = (
            "minimum_excitation_ratio",
            "minimum_no_output_local_anomaly",
        )
        for name in probabilities:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if (
            not np.isfinite(self.minimum_thrust_loss_motion_evidence)
            or self.minimum_thrust_loss_motion_evidence < 0.0
        ):
            raise ValueError(
                "minimum_thrust_loss_motion_evidence must be non-negative"
            )
        for name in (
            "ticket_confirmation_s",
            "thrust_loss_pending_confirmation_s",
            "thrust_loss_recovery_cancel_s",
            "thrust_loss_recurrence_window_s",
            "ticket_recovery_s",
            "merge_gap_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if int(self.thrust_loss_recurrence_count) < 2:
            raise ValueError(
                "thrust_loss_recurrence_count must be at least two"
            )
        if int(self.evidence_tail_steps) <= 0:
            raise ValueError("evidence_tail_steps must be positive")


def _as_numpy(values):
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _feature_indices(feature_names, suffix):
    names = list(feature_names)
    return [names.index(f"{thruster}_{suffix}") for thruster in THRUSTER_NAMES]


def extract_maintenance_ticket_evidence(
    dataset: Mapping[str, Any],
    indices,
    config=None,
):
    """Extract causal command, local telemetry, and motion evidence."""

    config = config or MaintenanceTicketConfig()
    indices = np.asarray(indices, dtype=np.int64)
    windows = _as_numpy(dataset["X"])[indices]
    feature_names = dataset["feature_names"]
    command_indices = _feature_indices(feature_names, "command_force_n")
    local_indices = _feature_indices(feature_names, "local_anomaly_score")
    motion_indices = _feature_indices(feature_names, "motion_loss_evidence")
    saturation_indices = _feature_indices(feature_names, "command_saturated")
    tail = windows[:, -min(config.evidence_tail_steps, windows.shape[1]):]

    mission_ids = _as_numpy(dataset["mission_ids"])[indices]
    metadata = dataset.get("mission_metadata", {})
    force_limits = np.zeros((len(indices), 6), dtype=float)
    for position, mission_id in enumerate(mission_ids):
        mission = metadata.get(int(mission_id), metadata.get(str(int(mission_id)), {}))
        thrusters = mission.get("parameters", {}).get("thrusters", {})
        horizontal = float(thrusters.get("horizontal_force_limit_n", 40.0))
        vertical = float(thrusters.get("vertical_force_limit_n", 35.0))
        force_limits[position] = [horizontal] * 4 + [vertical] * 2

    command_rms = np.sqrt(np.mean(tail[:, :, command_indices] ** 2, axis=1))
    excitation = np.clip(
        command_rms / np.maximum(force_limits, 1e-6), 0.0, 1.0
    )
    local_anomaly = np.mean(tail[:, :, local_indices], axis=1)
    motion_evidence = np.mean(tail[:, :, motion_indices], axis=1)
    saturation_fraction = np.mean(tail[:, :, saturation_indices], axis=1)
    return {
        "excitation_ratios": excitation,
        "local_anomaly_scores": local_anomaly,
        "motion_loss_evidence": motion_evidence,
        "saturation_fraction": saturation_fraction,
    }


def _validate_evidence(evidence, sample_count):
    result = {}
    for name in (
        "excitation_ratios",
        "local_anomaly_scores",
        "motion_loss_evidence",
    ):
        values = np.asarray(evidence[name], dtype=float)
        if values.shape != (sample_count, 6) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must have shape ({sample_count}, 6)")
        result[name] = values
    saturation = np.asarray(
        evidence.get("saturation_fraction", np.zeros((sample_count, 6))),
        dtype=float,
    )
    if saturation.shape != (sample_count, 6) or not np.all(np.isfinite(saturation)):
        raise ValueError(
            f"saturation_fraction must have shape ({sample_count}, 6)"
        )
    result["saturation_fraction"] = saturation
    return result


def _guidance_context(dataset, indices, sample_count):
    """Return onboard guidance context without adding it to model features."""

    if "guidance_context_ids" not in dataset:
        return (
            np.zeros(sample_count, dtype=np.int64),
            np.ones(sample_count, dtype=bool),
            False,
        )
    all_contexts = _as_numpy(dataset["guidance_context_ids"])
    if all_contexts.ndim != 1:
        raise ValueError("guidance_context_ids must be one-dimensional")
    contexts = np.asarray(all_contexts[indices], dtype=float)
    if (
        contexts.shape != (sample_count,)
        or not np.all(np.isfinite(contexts))
        or np.any(contexts < 0.0)
        or not np.all(contexts == np.floor(contexts))
    ):
        raise ValueError(
            "selected guidance_context_ids must be non-negative integers"
        )
    stable_source = dataset.get("guidance_context_stable")
    if stable_source is None:
        stable = np.ones(sample_count, dtype=bool)
    else:
        all_stable = _as_numpy(stable_source)
        if all_stable.ndim != 1:
            raise ValueError(
                "guidance_context_stable must be one-dimensional"
            )
        stable = np.asarray(all_stable[indices], dtype=bool)
        if stable.shape != (sample_count,):
            raise ValueError(
                "selected guidance_context_stable has invalid shape"
            )
    return contexts.astype(np.int64), stable, True


def _candidate_evidence(decisions, evidence, position):
    indices = np.asarray(decisions["candidate_indices"])[position]
    indices = indices[indices > 0] - 1
    if not len(indices):
        probabilities = np.asarray(
            decisions["smoothed_location_probabilities"]
        )[position]
        indices = np.argsort(-probabilities, kind="stable")[:2]
    excitation = float(np.max(evidence["excitation_ratios"][position, indices]))
    local = float(np.max(evidence["local_anomaly_scores"][position, indices]))
    motion = float(np.max(evidence["motion_loss_evidence"][position, indices]))
    return excitation, local, motion


def _qualify_windows(decisions, evidence, config):
    sample_count = len(decisions["health_level_pred"])
    qualified = np.zeros(sample_count, dtype=bool)
    qualified_mode = np.zeros(sample_count, dtype=np.int64)
    excitation = np.zeros(sample_count, dtype=float)
    independent = np.zeros(sample_count, dtype=float)
    for position in range(sample_count):
        health_level = int(decisions["health_level_pred"][position])
        mode = int(decisions["mode_pred"][position])
        command_evidence, local, motion = _candidate_evidence(
            decisions, evidence, position
        )
        excitation[position] = command_evidence
        if mode == 1:
            independent[position] = local
            qualified[position] = (
                health_level >= 3
                and command_evidence >= config.minimum_excitation_ratio
                and local >= config.minimum_no_output_local_anomaly
            )
        elif mode == 2:
            independent[position] = motion
            qualified[position] = (
                health_level >= 2
                and command_evidence >= config.minimum_excitation_ratio
                and motion
                >= config.minimum_thrust_loss_motion_evidence
            )
        if qualified[position]:
            qualified_mode[position] = mode
    return qualified, qualified_mode, excitation, independent


def _group_and_candidates(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / max(float(probabilities.sum()), 1e-12)
    ranking = np.argsort(-probabilities, kind="stable")
    group_evidence = np.array([
        probabilities[:4].mean(), probabilities[4:].mean()
    ])
    group_probability = group_evidence / group_evidence.sum()
    best_group = int(np.argmax(group_probability))
    group = ("horizontal", "vertical")[best_group]
    if float(group_probability[best_group]) < 0.60:
        group = "uncertain"
    return group, [
        {
            "index": int(index) + 1,
            "name": THRUSTER_NAMES[int(index)],
            "probability": float(probabilities[index]),
        }
        for index in ranking[:2]
    ]


def _ticket_record(
    mission_id,
    positions,
    end_times,
    modes,
    decisions,
    excitation,
    independent,
    triggers,
):
    ticket_modes = modes[positions]
    counts = np.bincount(ticket_modes, minlength=3)
    mode = int(np.argmax(counts[1:])) + 1
    probabilities = np.asarray(
        decisions["smoothed_location_probabilities"]
    )[positions].mean(axis=0)
    group, candidates = _group_and_candidates(probabilities)
    trigger_values = [
        str(value) for value in np.asarray(triggers, dtype=object)[positions]
        if value
    ]
    trigger = (
        Counter(trigger_values).most_common(1)[0][0]
        if trigger_values else "unspecified"
    )
    return {
        "mission_id": int(mission_id),
        "start_time_s": float(end_times[positions[0]]),
        "end_time_s": float(end_times[positions[-1]]),
        "duration_s": float(
            end_times[positions[-1]] - end_times[positions[0]]
        ),
        "fault_mode": ("no_output" if mode == 1 else "thrust_loss"),
        "trigger": trigger,
        "suspected_group": group,
        "inspection_candidates": candidates,
        "maximum_excitation_ratio": float(np.max(excitation[positions])),
        "maximum_independent_evidence": float(
            np.max(independent[positions])
        ),
        "merged_segment_count": 1,
    }


def _pending_observation_record(
    mission_id,
    positions,
    end_times,
    decisions,
    excitation,
    independent,
    status,
    *,
    guidance_context_id=None,
    qualified_duration_s=None,
    cumulative_qualified_duration_s=None,
):
    positions = np.asarray(positions, dtype=np.int64)
    probabilities = np.asarray(
        decisions["smoothed_location_probabilities"]
    )[positions].mean(axis=0)
    group, candidates = _group_and_candidates(probabilities)
    return {
        "mission_id": int(mission_id),
        "start_time_s": float(end_times[positions[0]]),
        "end_time_s": float(end_times[positions[-1]]),
        "duration_s": float(
            end_times[positions[-1]] - end_times[positions[0]]
        ),
        "fault_mode": "thrust_loss",
        "status": str(status),
        "guidance_context_id": (
            None if guidance_context_id is None
            else int(guidance_context_id)
        ),
        "qualified_duration_s": (
            float(end_times[positions[-1]] - end_times[positions[0]])
            if qualified_duration_s is None
            else float(qualified_duration_s)
        ),
        "cumulative_qualified_duration_s": (
            None if cumulative_qualified_duration_s is None
            else float(cumulative_qualified_duration_s)
        ),
        "suspected_group": group,
        "inspection_candidates": candidates,
        "maximum_excitation_ratio": float(np.max(excitation[positions])),
        "maximum_independent_evidence": float(
            np.max(independent[positions])
        ),
    }


def _merge_tickets(tickets, merge_gap_s):
    merged = []
    for ticket in sorted(
        tickets, key=lambda item: (item["mission_id"], item["start_time_s"])
    ):
        if (
            merged
            and merged[-1]["mission_id"] == ticket["mission_id"]
            and merged[-1]["fault_mode"] == ticket["fault_mode"]
            and ticket["start_time_s"] - merged[-1]["end_time_s"]
            <= merge_gap_s
        ):
            previous = merged[-1]
            previous["end_time_s"] = ticket["end_time_s"]
            previous["duration_s"] = (
                previous["end_time_s"] - previous["start_time_s"]
            )
            previous["maximum_excitation_ratio"] = max(
                previous["maximum_excitation_ratio"],
                ticket["maximum_excitation_ratio"],
            )
            previous["maximum_independent_evidence"] = max(
                previous["maximum_independent_evidence"],
                ticket["maximum_independent_evidence"],
            )
            if previous.get("trigger") != ticket.get("trigger"):
                previous["trigger"] = "mixed"
            previous["merged_segment_count"] += 1
        else:
            merged.append(dict(ticket))
    return merged


def apply_maintenance_ticket_policy(
    dataset: Mapping[str, Any],
    indices,
    decisions: Mapping[str, Any],
    config=None,
    *,
    ticket_evidence=None,
):
    """Keep raw events while generating separately qualified tickets."""

    config = config or MaintenanceTicketConfig()
    indices = np.asarray(indices, dtype=np.int64)
    sample_count = len(indices)
    evidence = _validate_evidence(
        ticket_evidence
        if ticket_evidence is not None
        else extract_maintenance_ticket_evidence(dataset, indices, config),
        sample_count,
    )
    qualified, candidate_mode, excitation, independent = _qualify_windows(
        decisions, evidence, config
    )
    mission_ids = _as_numpy(dataset["mission_ids"])[indices]
    end_times = _as_numpy(dataset["window_end_times"])[indices]
    guidance_context_ids, context_stable, context_available = (
        _guidance_context(dataset, indices, sample_count)
    )
    ticket_qualified = qualified.copy()
    ticket_qualified[(candidate_mode == 2) & ~context_stable] = False
    ticket_active = np.zeros(sample_count, dtype=bool)
    ticket_mode = np.zeros(sample_count, dtype=np.int64)
    ticket_pending = np.zeros(sample_count, dtype=bool)
    ticket_trigger = np.full(sample_count, "", dtype=object)
    tickets = []
    pending_observations = []

    for mission_id in np.unique(mission_ids):
        positions = np.flatnonzero(mission_ids == mission_id)
        positions = positions[np.argsort(end_times[positions], kind="stable")]
        active = False
        active_mode = 0
        active_trigger = ""
        pending = None
        recovery_start = None
        recent_thrust_loss_episodes = []
        last_context_id = None
        for position in positions:
            time_s = float(end_times[position])
            raw_is_qualified = bool(qualified[position])
            is_qualified = bool(ticket_qualified[position])
            mode = int(candidate_mode[position])
            context_id = int(guidance_context_ids[position])
            context_changed = bool(
                context_available
                and last_context_id is not None
                and context_id != last_context_id
            )
            if context_changed:
                recent_thrust_loss_episodes = []
                if (
                    pending is not None
                    and pending["mode"] == 2
                    and pending["positions"]
                ):
                    pending_observations.append(
                        _pending_observation_record(
                            mission_id,
                            pending["positions"],
                            end_times,
                            decisions,
                            excitation,
                            independent,
                            "context_transition_observation",
                            guidance_context_id=pending["context_id"],
                            qualified_duration_s=(
                                pending["qualified_duration_s"]
                            ),
                        )
                    )
                    pending = None
            last_context_id = context_id

            if active:
                if raw_is_qualified:
                    recovery_start = None
                    if mode == 1:
                        active_mode = 1
                        active_trigger = "direct_no_output"
                else:
                    if recovery_start is None:
                        recovery_start = time_s
                    if time_s - recovery_start >= config.ticket_recovery_s:
                        active = False
                        active_mode = 0
                        active_trigger = ""
                        recovery_start = None
                ticket_active[position] = active
                ticket_mode[position] = active_mode
                ticket_trigger[position] = active_trigger
                continue

            # A direct no-output diagnosis always supersedes a slower pending
            # thrust-loss observation, so the urgent path is never delayed.
            if (
                pending is not None
                and pending["mode"] == 2
                and raw_is_qualified
                and mode == 1
            ):
                if pending["positions"]:
                    pending_observations.append(_pending_observation_record(
                        mission_id,
                        pending["positions"],
                        end_times,
                        decisions,
                        excitation,
                        independent,
                        "superseded_by_no_output",
                        guidance_context_id=pending["context_id"],
                        qualified_duration_s=(
                            pending["qualified_duration_s"]
                        ),
                    ))
                pending = None

            if pending is not None and pending["mode"] == 2:
                if is_qualified and mode == 2:
                    if pending["previous_qualified"]:
                        pending["qualified_duration_s"] += max(
                            0.0,
                            time_s - pending["last_window_time_s"],
                        )
                    pending["previous_qualified"] = True
                    pending["last_window_time_s"] = time_s
                    pending["recovery_start_s"] = None
                    pending["positions"].append(position)
                    pending["last_qualified_time_s"] = time_s
                    if (
                        pending["qualified_duration_s"]
                        >= config.thrust_loss_pending_confirmation_s
                    ):
                        active = True
                        active_mode = 2
                        active_trigger = "persistent_thrust_loss"
                        pending = None
                else:
                    pending["previous_qualified"] = False
                    pending["last_window_time_s"] = time_s
                    if pending["recovery_start_s"] is None:
                        pending["recovery_start_s"] = time_s
                    if (
                        time_s - pending["recovery_start_s"]
                        >= config.thrust_loss_recovery_cancel_s
                    ):
                        recent_thrust_loss_episodes = [
                            episode
                            for episode in recent_thrust_loss_episodes
                            if (
                                episode["context_id"]
                                == pending["context_id"]
                                and 0.0 <= (
                                    pending["last_qualified_time_s"]
                                    - episode["end_time_s"]
                                )
                                <= config.thrust_loss_recurrence_window_s
                            )
                        ]
                        current_episode = {
                            "context_id": pending["context_id"],
                            "end_time_s": pending[
                                "last_qualified_time_s"
                            ],
                            "qualified_duration_s": pending[
                                "qualified_duration_s"
                            ],
                        }
                        combined = (
                            recent_thrust_loss_episodes
                            + [current_episode]
                        )
                        cumulative_duration_s = float(sum(
                            episode["qualified_duration_s"]
                            for episode in combined
                        ))
                        intermittent_advisory = bool(
                            len(combined)
                            >= config.thrust_loss_recurrence_count
                            and cumulative_duration_s
                            >= config.thrust_loss_pending_confirmation_s
                        )
                        if pending["positions"]:
                            pending_observations.append(
                                _pending_observation_record(
                                    mission_id,
                                    pending["positions"],
                                    end_times,
                                    decisions,
                                    excitation,
                                    independent,
                                    (
                                        "intermittent_thrust_loss_advisory"
                                        if intermittent_advisory
                                        else "recovered_before_confirmation"
                                    ),
                                    guidance_context_id=(
                                        pending["context_id"]
                                    ),
                                    qualified_duration_s=(
                                        pending["qualified_duration_s"]
                                    ),
                                    cumulative_qualified_duration_s=(
                                        cumulative_duration_s
                                    ),
                                )
                            )
                        recent_thrust_loss_episodes.append(
                            current_episode
                        )
                        pending = None

                if active:
                    ticket_active[position] = True
                    ticket_mode[position] = active_mode
                    ticket_trigger[position] = active_trigger
                    continue
                if pending is not None and pending["mode"] == 2:
                    ticket_pending[position] = True
                    continue

            if pending is not None and pending["mode"] == 1:
                if raw_is_qualified and mode == 1:
                    if (
                        time_s - pending["start_time_s"]
                        >= config.ticket_confirmation_s
                    ):
                        active = True
                        active_mode = 1
                        active_trigger = "direct_no_output"
                        pending = None
                else:
                    pending = None
                if active:
                    ticket_active[position] = True
                    ticket_mode[position] = active_mode
                    ticket_trigger[position] = active_trigger
                    continue
                if pending is not None and pending["mode"] == 1:
                    ticket_pending[position] = True
                    continue

            if raw_is_qualified and mode == 1:
                pending = {"mode": 1, "start_time_s": time_s}
                if config.ticket_confirmation_s <= 0.0:
                    active = True
                    active_mode = 1
                    active_trigger = "direct_no_output"
                    pending = None
            elif is_qualified and mode == 2:
                pending = {
                    "mode": 2,
                    "start_time_s": time_s,
                    "positions": [position],
                    "last_qualified_time_s": time_s,
                    "recovery_start_s": None,
                    "qualified_duration_s": 0.0,
                    "last_window_time_s": time_s,
                    "previous_qualified": True,
                    "context_id": context_id,
                }
                if config.thrust_loss_pending_confirmation_s <= 0.0:
                    active = True
                    active_mode = 2
                    active_trigger = "persistent_thrust_loss"
                    pending = None

            ticket_active[position] = active
            ticket_mode[position] = active_mode
            ticket_pending[position] = pending is not None
            ticket_trigger[position] = active_trigger

        if (
            pending is not None
            and pending["mode"] == 2
            and pending["positions"]
        ):
            pending_observations.append(_pending_observation_record(
                mission_id,
                pending["positions"],
                end_times,
                decisions,
                excitation,
                independent,
                "pending_at_mission_end",
                guidance_context_id=pending["context_id"],
                qualified_duration_s=pending["qualified_duration_s"],
            ))

        segment = []
        for position in positions:
            if ticket_active[position]:
                segment.append(position)
            elif segment:
                segment_positions = np.asarray(segment, dtype=np.int64)
                tickets.append(_ticket_record(
                    mission_id,
                    segment_positions,
                    end_times,
                    ticket_mode,
                    decisions,
                    excitation,
                    independent,
                    ticket_trigger,
                ))
                segment = []
        if segment:
            segment_positions = np.asarray(segment, dtype=np.int64)
            tickets.append(_ticket_record(
                mission_id,
                segment_positions,
                end_times,
                ticket_mode,
                decisions,
                excitation,
                independent,
                ticket_trigger,
            ))

    result = dict(decisions)
    result.update({
        "maintenance_ticket_active": ticket_active,
        "maintenance_ticket_mode": ticket_mode,
        "maintenance_ticket_pending": ticket_pending,
        "maintenance_ticket_trigger": ticket_trigger,
        "maintenance_ticket_raw_qualified": qualified,
        "maintenance_ticket_qualified": ticket_qualified,
        "maintenance_guidance_context_id": guidance_context_ids,
        "maintenance_guidance_context_stable": context_stable,
        "maintenance_guidance_context_available": context_available,
        "maintenance_ticket_excitation": excitation,
        "maintenance_ticket_independent_evidence": independent,
        "maintenance_tickets": _merge_tickets(tickets, config.merge_gap_s),
        "maintenance_pending_observations": pending_observations,
        "maintenance_advisories": [
            observation for observation in pending_observations
            if observation["status"]
            == "intermittent_thrust_loss_advisory"
        ],
        "maintenance_ticket_config": config,
    })
    return result


def maintenance_ticket_metrics(dataset, indices, decisions):
    """Evaluate formal tickets separately from raw health observations."""

    indices = np.asarray(indices, dtype=np.int64)
    mission_ids = _as_numpy(dataset["mission_ids"])[indices]
    end_times = _as_numpy(dataset["window_end_times"])[indices]
    mode_true = np.asarray(decisions["mode_true"], dtype=np.int64)
    location_true = np.asarray(decisions["location_true"], dtype=np.int64)
    ticket_active = np.asarray(
        decisions["maintenance_ticket_active"], dtype=bool
    )
    ticket_mode = np.asarray(
        decisions["maintenance_ticket_mode"], dtype=np.int64
    )
    location_probabilities = np.asarray(
        decisions["smoothed_location_probabilities"], dtype=float
    )

    total_by_mode = {1: 0, 2: 0}
    detected_by_mode = {1: 0, 2: 0}
    detected_missions = 0
    correctly_judged_mode_missions = 0
    total_fault_missions = 0
    false_tickets = 0
    normal_exposure_s = 0.0
    delays = []
    top2_hits = []
    group_hits = []
    ticket_counts = []
    observation_status_counts = Counter(
        str(observation.get("status", "unspecified"))
        for observation in decisions.get(
            "maintenance_pending_observations", []
        )
    )

    tickets_by_mission = {}
    for ticket in decisions["maintenance_tickets"]:
        tickets_by_mission.setdefault(ticket["mission_id"], []).append(ticket)

    for mission_id in np.unique(mission_ids):
        positions = np.flatnonzero(mission_ids == mission_id)
        positions = positions[np.argsort(end_times[positions], kind="stable")]
        times = end_times[positions]
        true_modes = mode_true[positions]
        normal_mask = true_modes == 0
        differences = np.diff(times)
        positive = differences[differences > 0.0]
        step = float(np.median(positive)) if len(positive) else 0.0
        normal_exposure_s += float(np.sum(normal_mask)) * step

        targets = np.unique(true_modes[true_modes != 0])
        target_mode = int(targets[0]) if len(targets) else 0
        first_fault_time = (
            float(times[np.flatnonzero(true_modes == target_mode)[0]])
            if target_mode else None
        )
        mission_tickets = tickets_by_mission.get(int(mission_id), [])
        valid_tickets = []
        for ticket in mission_tickets:
            if target_mode and ticket["end_time_s"] >= first_fault_time:
                valid_tickets.append(ticket)
            else:
                false_tickets += 1

        if not target_mode:
            continue
        total_fault_missions += 1
        total_by_mode[target_mode] += 1
        post_fault = times >= first_fault_time
        detected = ticket_active[positions] & post_fault
        if not np.any(detected):
            continue
        detected_missions += 1
        detected_by_mode[target_mode] += 1
        if np.any(detected & (ticket_mode[positions] == target_mode)):
            correctly_judged_mode_missions += 1
        first = int(np.flatnonzero(detected)[0])
        delays.append(max(0.0, float(times[first] - first_fault_time)))
        ticket_counts.append(len(valid_tickets))

        true_locations = np.unique(
            location_true[positions][true_modes != 0]
        )
        true_locations = true_locations[true_locations != 0]
        if len(true_locations):
            target_location = int(true_locations[0])
            average = location_probabilities[positions][detected].mean(axis=0)
            ranking = np.argsort(-average, kind="stable") + 1
            top2_hits.append(float(target_location in ranking[:2]))
            target_group = "horizontal" if target_location <= 4 else "vertical"
            group_evidence = [average[:4].mean(), average[4:].mean()]
            predicted_group = ("horizontal", "vertical")[
                int(np.argmax(group_evidence))
            ]
            group_hits.append(float(predicted_group == target_group))

    denominator = detected_missions + false_tickets
    return {
        "ticket_event_recall": (
            detected_missions / max(total_fault_missions, 1)
        ),
        "ticket_event_precision": (
            detected_missions / denominator if denominator else None
        ),
        "no_output_ticket_recall": (
            detected_by_mode[1] / total_by_mode[1]
            if total_by_mode[1] else None
        ),
        "thrust_loss_ticket_recall": (
            detected_by_mode[2] / total_by_mode[2]
            if total_by_mode[2] else None
        ),
        "ticket_fault_mode_judgement_rate": (
            correctly_judged_mode_missions
            / max(total_fault_missions, 1)
        ),
        "false_tickets_per_hour": (
            false_tickets / (normal_exposure_s / 3600.0)
            if normal_exposure_s > 0.0 else None
        ),
        "false_maintenance_tickets": false_tickets,
        "formal_maintenance_ticket_count": len(
            decisions["maintenance_tickets"]
        ),
        "maintenance_observation_count": int(sum(
            observation_status_counts.values()
        )),
        "recovered_before_confirmation_observations": int(
            observation_status_counts["recovered_before_confirmation"]
        ),
        "pending_at_mission_end_observations": int(
            observation_status_counts["pending_at_mission_end"]
        ),
        "superseded_by_no_output_observations": int(
            observation_status_counts["superseded_by_no_output"]
        ),
        "context_transition_observations": int(
            observation_status_counts["context_transition_observation"]
        ),
        "intermittent_thrust_loss_advisories": int(
            observation_status_counts[
                "intermittent_thrust_loss_advisory"
            ]
        ),
        "mean_ticket_detection_delay_s": (
            float(np.mean(delays)) if delays else None
        ),
        "median_ticket_detection_delay_s": (
            float(np.median(delays)) if delays else None
        ),
        "single_ticket_per_detected_mission_rate": (
            float(np.mean(np.asarray(ticket_counts) == 1))
            if ticket_counts else None
        ),
        "ticket_probable_group_accuracy": (
            float(np.mean(group_hits)) if group_hits else None
        ),
        "ticket_top2_location_hit_rate": (
            float(np.mean(top2_hits)) if top2_hits else None
        ),
        "detected_ticket_missions": detected_missions,
        "total_fault_missions": total_fault_missions,
    }
