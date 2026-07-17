"""Convert six-DOF simulator logs into a stable operator-display contract.

The adapter deliberately separates simulation truth used to draw vehicle motion
from causal diagnostic fields used to classify faults. In particular, injected
fault labels, actual thrust, and injected effectiveness never influence sensor
or thruster presentation tiers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SENSOR_NAMES = ("depth", "imu", "dvl")
DEFAULT_THRUSTER_NAMES = ("H1", "H2", "H3", "H4", "V1", "V2")
TIER_RANK = {"normal": 0, "log_only": 1, "possible": 2, "confirmed": 3}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _vector(value: Any, size: int, default: float = 0.0) -> list[float]:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        vector = np.full(size, default, dtype=float)
    if vector.shape != (size,):
        vector = np.full(size, default, dtype=float)
    vector = np.where(np.isfinite(vector), vector, default)
    return [float(item) for item in vector]


def _bool_vector(value: Any, size: int, default: bool = True) -> list[bool]:
    try:
        vector = np.asarray(value)
    except (TypeError, ValueError):
        vector = np.full(size, default, dtype=bool)
    if vector.shape != (size,):
        vector = np.full(size, default, dtype=bool)
    if vector.dtype != np.bool_:
        try:
            numeric = np.asarray(vector, dtype=float)
            valid_numeric = np.isfinite(numeric) & (
                (numeric == 0.0) | (numeric == 1.0)
            )
            if not np.all(valid_numeric):
                return [bool(default)] * size
            vector = numeric.astype(bool)
        except (TypeError, ValueError):
            return [bool(default)] * size
    return [bool(item) for item in vector]


def _tier_max(*tiers: str) -> str:
    return max(tiers, key=lambda tier: TIER_RANK.get(tier, -1))


def _humanize(value: Any) -> str:
    text = str(value or "normal").strip().replace("_", " ")
    return text if text else "normal"


def _sensor_card(log: Mapping[str, Any], sensor: str) -> dict[str, Any]:
    health = _mapping(_mapping(log.get("sensor_health")).get(sensor))
    observation = _mapping(
        _mapping(log.get("sensor_fault_observations")).get(sensor)
    )
    fault_type = str(health.get("fault_type", "normal"))
    health_state = str(health.get("health_state", "healthy"))
    confirmed = bool(health.get("confirmed", False))
    direct_active = fault_type != "normal" or health_state != "healthy"
    direct_tier = (
        "confirmed" if confirmed else ("possible" if direct_active else "normal")
    )
    observer_active = observation.get("state") == "possible_fault"
    observer_tier = (
        "possible"
        if observer_active and observation.get("display_level") == "possible"
        else ("log_only" if observer_active else "normal")
    )
    tier = _tier_max(direct_tier, observer_tier)

    if TIER_RANK[direct_tier] >= TIER_RANK[observer_tier] and direct_active:
        label = _humanize(fault_type)
        confidence = _finite_float(health.get("confidence"), 0.0)
        source = "direct_monitor"
        action = str(health.get("recommended_action", "observe"))
        evidence = str(health.get("evidence", ""))
        candidates: list[str] = []
        affected_channels: list[int] = []
    elif observer_active:
        label = _humanize(observation.get("hypothesis", "possible fault"))
        confidence = _finite_float(observation.get("confidence"), 0.0)
        source = "long_horizon_observer"
        action = str(observation.get("recommended_action", "record_and_observe"))
        evidence = str(observation.get("evidence", ""))
        candidates = [str(value) for value in observation.get("candidates", ())]
        affected_channels = [
            int(value) for value in observation.get("affected_channels", ())
        ]
    else:
        label = "normal"
        confidence = 1.0
        source = "direct_monitor"
        action = "use_sensor"
        evidence = ""
        candidates = []
        affected_channels = []
    return {
        "name": sensor,
        "tier": tier,
        "label": label,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "source": source,
        "trust_level": str(health.get("trust_level", "trusted")),
        "action": action,
        "candidates": candidates,
        "affected_channels": affected_channels,
        "evidence": evidence,
    }


def _thruster_cards(log: Mapping[str, Any]) -> list[dict[str, Any]]:
    names_raw = log.get("thruster_names", DEFAULT_THRUSTER_NAMES)
    names = tuple(str(value) for value in names_raw)
    if len(names) != 6:
        names = DEFAULT_THRUSTER_NAMES
    commanded = _vector(log.get("commanded_thruster_forces"), 6)
    limits = [max(abs(value), 1e-6) for value in _vector(
        log.get("thruster_force_limits"), 6, default=1.0
    )]
    expected_current = _vector(log.get("thruster_expected_currents"), 6)
    measured_current = _vector(log.get("thruster_measured_currents"), 6)
    expected_rpm = _vector(log.get("thruster_expected_rpms"), 6)
    measured_rpm = _vector(log.get("thruster_measured_rpms"), 6)
    telemetry_valid = _bool_vector(
        log.get("thruster_telemetry_valid"), 6, default=True
    )
    telemetry_age_s = [
        max(0.0, value)
        for value in _vector(log.get("thruster_telemetry_age_s"), 6)
    ]
    untrusted_esc = {
        str(value)
        for value in log.get("ftc_untrusted_esc_channels", ())
        if str(value) in names
    }
    scores = [
        float(np.clip(value, 0.0, 1.0))
        for value in _vector(log.get("ftc_no_output_scores"), 6)
    ]
    next_effectiveness = _vector(
        log.get("ftc_estimated_effectiveness_next_step"), 6, default=1.0
    )
    target_name = log.get("ftc_targeted_thruster_name")
    target_index = log.get("ftc_targeted_thruster_index")
    try:
        target_index = None if target_index is None else int(target_index) - 1
    except (TypeError, ValueError):
        target_index = None

    cards = []
    for index, name in enumerate(names):
        targeted = bool(name == target_name or index == target_index)
        isolated = next_effectiveness[index] < 0.5
        telemetry_untrusted = bool(
            name in untrusted_esc or not telemetry_valid[index]
        )
        telemetry_status = (
            "invalid"
            if not telemetry_valid[index]
            else ("stale" if telemetry_untrusted else "fresh")
        )
        if targeted or isolated:
            tier, label = "confirmed", "targeted / isolated"
        elif telemetry_untrusted:
            tier = "log_only"
            label = (
                "ESC telemetry unavailable"
                if telemetry_status == "invalid"
                else "ESC telemetry stale"
            )
        elif scores[index] >= 0.80:
            tier, label = "possible", "no-output evidence"
        elif scores[index] >= 0.35:
            tier, label = "log_only", "weak dropout evidence"
        else:
            tier, label = "normal", "normal"
        cards.append({
            "name": name,
            "tier": tier,
            "label": label,
            "commanded_force_n": commanded[index],
            "excitation_ratio": float(np.clip(
                abs(commanded[index]) / limits[index], 0.0, 1.0
            )),
            "expected_current_a": expected_current[index],
            "measured_current_a": measured_current[index],
            "expected_rpm": expected_rpm[index],
            "measured_rpm": measured_rpm[index],
            "telemetry_valid": telemetry_valid[index],
            "telemetry_age_s": telemetry_age_s[index],
            "telemetry_status": telemetry_status,
            "no_output_score": scores[index],
            "estimated_effectiveness": float(np.clip(
                next_effectiveness[index], 0.0, 1.0
            )),
        })
    return cards


def _maintenance_card(log: Mapping[str, Any]) -> dict[str, Any]:
    diagnosis = _mapping(log.get("maintenance_diagnosis"))
    available = bool(diagnosis.get("available", False))
    health_level = int(_finite_float(diagnosis.get("health_level"), 0.0))
    if not available or health_level <= 0:
        tier = "normal"
    elif health_level == 1:
        tier = "log_only"
    else:
        # Learned location remains advisory even at a high maintenance level.
        tier = "possible"
    candidates = []
    for candidate in diagnosis.get("candidates", ()):
        candidate = _mapping(candidate)
        name = str(candidate.get("name", ""))
        if name not in DEFAULT_THRUSTER_NAMES:
            continue
        candidates.append({
            "index": int(_finite_float(candidate.get("index"), 0.0)),
            "name": name,
            "probability": float(np.clip(
                _finite_float(candidate.get("probability"), 0.0), 0.0, 1.0
            )),
        })
    return {
        "available": available,
        "updated": bool(diagnosis.get("updated", False)),
        "status": str(diagnosis.get("status", "not_available")),
        "tier": tier,
        "health_level": health_level,
        "health_state": str(diagnosis.get("health_state", "normal")),
        "temporal_state": str(diagnosis.get("temporal_state", "normal")),
        "probable_mode": str(
            diagnosis.get("probable_mode_name", "normal")
        ),
        "confirmed_mode": str(
            diagnosis.get("confirmed_mode_name", "normal")
        ),
        "fault_probability": float(np.clip(
            _finite_float(diagnosis.get("fault_probability"), 0.0), 0.0, 1.0
        )),
        "suspected_group": str(diagnosis.get("suspected_group", "none")),
        "group_confidence": float(np.clip(
            _finite_float(diagnosis.get("group_confidence"), 0.0), 0.0, 1.0
        )),
        "location_confidence": str(
            diagnosis.get("location_confidence", "none")
        ),
        "candidates": candidates,
        "mode_probabilities": _vector(
            diagnosis.get("mode_probabilities"), 3
        ),
        "location_probabilities": _vector(
            diagnosis.get("location_probabilities"), 6
        ),
        "action": str(diagnosis.get("action", "none")),
        "record_event": bool(diagnosis.get("record_event", False)),
        "advisory_gate_active": bool(
            diagnosis.get("advisory_gate_active", False)
        ),
        "advisory_gate_reasons": [
            str(value) for value in diagnosis.get("advisory_gate_reasons", ())
        ],
        "advisory_suppressed": bool(
            diagnosis.get("advisory_suppressed", False)
        ),
        "raw_fault_probability": float(np.clip(
            _finite_float(diagnosis.get("raw_fault_probability"), 0.0),
            0.0,
            1.0,
        )),
        "raw_probable_mode": str(
            diagnosis.get("raw_probable_mode_name", "normal")
        ),
        "raw_suspected_group": str(
            diagnosis.get("raw_suspected_group", "none")
        ),
        # This field reports model advice only; FTC still uses its own evidence.
        "model_recommends_ftc": bool(diagnosis.get("requires_ftc", False)),
    }


def adapt_log(log: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt one simulator log without using privileged diagnosis labels."""

    if not isinstance(log, Mapping):
        raise TypeError("log must be a mapping")
    sensors = {name: _sensor_card(log, name) for name in SENSOR_NAMES}
    thrusters = _thruster_cards(log)
    maintenance = _maintenance_card(log)
    ftc_action = str(log.get("ftc_action", "normal_control"))
    ftc_target = log.get("ftc_targeted_thruster_name")
    ftc_tier = (
        "confirmed"
        if ftc_action in {
            "targeted_reallocation", "safe_hold_or_abort", "controlled_ascent"
        }
        else (
            "possible" if ftc_action == "degraded_operation"
            else ("log_only" if ftc_action == "log_only" else "normal")
        )
    )
    overall_tier = _tier_max(
        ftc_tier,
        maintenance["tier"],
        *(card["tier"] for card in sensors.values()),
        *(card["tier"] for card in thrusters),
    )
    position = _vector(log.get("position_ned"), 3)
    estimated_position = _vector(log.get("estimated_position_ned", position), 3)
    return {
        "time_s": _finite_float(log.get("time"), 0.0),
        "pose": {
            "position_ned_m": position,
            "estimated_position_ned_m": estimated_position,
            "target_position_ned_m": _vector(log.get("target_position_ned"), 3),
            "euler_rpy_rad": _vector(log.get("euler_rpy"), 3),
            "estimated_euler_rpy_rad": _vector(
                log.get("estimated_euler_rpy", log.get("euler_rpy")), 3
            ),
        },
        "sensors": sensors,
        "thrusters": thrusters,
        "maintenance": maintenance,
        "ftc": {
            "tier": ftc_tier,
            "action": ftc_action,
            "applied_action": str(log.get("ftc_applied_action", "normal_control")),
            "reason": str(log.get("ftc_reason", "")),
            "target_thruster": None if ftc_target is None else str(ftc_target),
            "intervention_requested": bool(
                log.get("ftc_intervention_requested", False)
            ),
            "sensor_guard_action": str(log.get("ftc_sensor_guard_action", "none")),
            "untrusted_sensors": [
                str(value) for value in log.get("ftc_untrusted_sensors", ())
            ],
            "untrusted_esc_channels": [
                str(value)
                for value in log.get("ftc_untrusted_esc_channels", ())
                if str(value) in DEFAULT_THRUSTER_NAMES
            ],
        },
        "estimator": {
            "quality": str(log.get("state_estimate_quality", "not_available")),
            "sources": {
                str(key): str(value)
                for key, value in _mapping(log.get("state_estimate_sources")).items()
            },
            "excluded_sensors": [
                str(value)
                for value in log.get("state_estimate_excluded_sensors", ())
            ],
            "rejected_sensors": [
                str(value)
                for value in log.get("state_estimate_rejected_sensors", ())
            ],
            "horizontal_reference": str(
                log.get("horizontal_position_reference", "unknown")
            ),
        },
        "overall_tier": overall_tier,
    }


def adapt_logs(logs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frames = [adapt_log(log) for log in logs]
    if not frames:
        raise ValueError("logs cannot be empty")
    return frames


def extract_demo_events(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract operator-relevant state transitions from adapted frames."""

    events: list[dict[str, Any]] = []
    previous_sensor = {name: ("normal", "normal") for name in SENSOR_NAMES}
    previous_thruster = {
        name: ("normal", "normal") for name in DEFAULT_THRUSTER_NAMES
    }
    seen_thruster_evidence = set()
    previous_ftc = ("normal_control", None)
    previous_maintenance = ("normal", "normal", "none")
    for frame in frames:
        time_s = _finite_float(frame.get("time_s"), 0.0)
        sensors = _mapping(frame.get("sensors"))
        for name in SENSOR_NAMES:
            card = _mapping(sensors.get(name))
            current = (
                str(card.get("tier", "normal")),
                str(card.get("label", "normal")),
            )
            if current != previous_sensor[name]:
                if current[0] == "normal":
                    message, level = f"{name.upper()} returned to normal", "normal"
                else:
                    message = f"{name.upper()}: {current[1]} ({current[0]})"
                    level = current[0]
                events.append({
                    "time_s": time_s, "category": "sensor", "source": name,
                    "level": level, "message": message,
                })
                previous_sensor[name] = current
        for card_value in frame.get("thrusters", ()):
            card = _mapping(card_value)
            name = str(card.get("name", "unknown"))
            if name not in previous_thruster:
                previous_thruster[name] = ("normal", "normal")
            current = (
                str(card.get("tier", "normal")),
                str(card.get("label", "normal")),
            )
            if current != previous_thruster[name]:
                should_emit = False
                if current[0] == "confirmed":
                    should_emit = True
                elif current[0] in ("possible", "log_only"):
                    signature = (name, current[1])
                    should_emit = signature not in seen_thruster_evidence
                    seen_thruster_evidence.add(signature)
                elif previous_thruster[name][0] == "confirmed":
                    should_emit = True
                elif (
                    current[0] == "normal"
                    and previous_thruster[name][1].startswith("ESC telemetry")
                ):
                    should_emit = True
                if current[0] == "normal":
                    message, level = f"{name} evidence cleared", "normal"
                else:
                    message = f"{name}: {current[1]} ({current[0]})"
                    level = current[0]
                if should_emit:
                    events.append({
                        "time_s": time_s, "category": "thruster",
                        "source": name, "level": level, "message": message,
                    })
                previous_thruster[name] = current
        ftc = _mapping(frame.get("ftc"))
        maintenance = _mapping(frame.get("maintenance"))
        current_maintenance = (
            str(maintenance.get("tier", "normal")),
            str(maintenance.get("probable_mode", "normal")),
            str(maintenance.get("suspected_group", "none")),
        )
        if current_maintenance != previous_maintenance:
            if current_maintenance[0] == "normal":
                message = "Model maintenance advice returned to normal"
                level = "normal"
            else:
                message = (
                    "Model advice: possible "
                    f"{current_maintenance[1].replace('_', ' ')} / "
                    f"{current_maintenance[2]} group"
                )
                level = current_maintenance[0]
            events.append({
                "time_s": time_s,
                "category": "maintenance",
                "source": "bilstm_attention",
                "level": level,
                "message": message,
            })
            previous_maintenance = current_maintenance
        current_ftc = (
            str(ftc.get("action", "normal_control")),
            ftc.get("target_thruster"),
        )
        action = current_ftc[0]
        if action == "log_only":
            continue
        if current_ftc != previous_ftc:
            events.append({
                "time_s": time_s,
                "category": "ftc",
                "source": "supervisor",
                "level": str(ftc.get("tier", "normal")),
                "message": (
                    f"FTC: {current_ftc[0].replace('_', ' ')}"
                    + ("" if current_ftc[1] is None else f" -> {current_ftc[1]}")
                ),
            })
            previous_ftc = current_ftc
    return events


def summarize_demo(
    frames: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not frames:
        raise ValueError("frames cannot be empty")
    events = extract_demo_events(frames) if events is None else list(events)
    tier_counts = {tier: 0 for tier in TIER_RANK}
    for frame in frames:
        tier = str(frame.get("overall_tier", "normal"))
        tier_counts[tier if tier in tier_counts else "normal"] += 1
    confirmed_sensors = sorted({
        name
        for frame in frames
        for name, card in _mapping(frame.get("sensors")).items()
        if _mapping(card).get("tier") == "confirmed"
    })
    possible_sensors = sorted({
        name
        for frame in frames
        for name, card in _mapping(frame.get("sensors")).items()
        if _mapping(card).get("tier") == "possible"
    })
    targeted_thrusters = sorted({
        str(_mapping(frame.get("ftc")).get("target_thruster"))
        for frame in frames
        if _mapping(frame.get("ftc")).get("target_thruster") is not None
    })
    model_candidate_thrusters = sorted({
        candidate["name"]
        for frame in frames
        for candidate in _mapping(frame.get("maintenance")).get(
            "candidates", ()
        )
    })
    esc_communication_anomaly_thrusters = sorted({
        str(card.get("name"))
        for frame in frames
        for card in frame.get("thrusters", ())
        if str(card.get("telemetry_status", "fresh")) != "fresh"
    })
    return {
        "duration_s": _finite_float(frames[-1].get("time_s"), 0.0),
        "frame_count": len(frames),
        "event_count": len(events),
        "overall_tier_frame_counts": tier_counts,
        "confirmed_sensors": confirmed_sensors,
        "possible_sensors": possible_sensors,
        "targeted_thrusters": targeted_thrusters,
        "model_candidate_thrusters": model_candidate_thrusters,
        "esc_communication_anomaly_thrusters": (
            esc_communication_anomaly_thrusters
        ),
        "esc_communication_anomaly_frame_count": sum(
            any(
                str(card.get("telemetry_status", "fresh")) != "fresh"
                for card in frame.get("thrusters", ())
            )
            for frame in frames
        ),
        "model_advisory_frame_count": sum(
            _mapping(frame.get("maintenance")).get("tier")
            in ("log_only", "possible")
            for frame in frames
        ),
        "ftc_actions": sorted({
            str(_mapping(frame.get("ftc")).get("action", "normal_control"))
            for frame in frames
        }),
    }
