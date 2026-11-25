"""Display-only provenance, never a diagnosis or control input.

Unknown legacy provenance stays unknown. A finite held value is not a fresh
measurement. Body-frame DVL velocity and NED position have different validity.
"""
from collections.abc import Mapping
import math


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def estimate_observability(metadata):
    metadata = _mapping(metadata)
    sources = _mapping(metadata.get("sources"))
    health = _mapping(metadata.get("sensor_health"))
    blocked = set(metadata.get("excluded_sensors", ())) | set(metadata.get("rejected_sensors", ()))

    def channel(key, sensor, measured, fallback):
        source = str(sources.get(key, "unknown"))
        trust = str(_mapping(health.get(sensor)).get("trust_level", "unknown"))
        if source in measured and sensor not in blocked:
            status = {"trusted": "measured", "degraded": "degraded", "untrusted": "unavailable"}.get(trust, "unknown")
        elif source in fallback:
            status = "degraded"
        elif source.startswith("initial_"):
            status = "prior"
        elif source in {"held_last_attitude", "zero_rate_safe_hold", "decaying_velocity_hold", "held_depth"} or sensor in blocked:
            status = "unavailable"
        else:
            status = "unknown"
        return {"status": status, "source": source}

    attitude = channel("attitude", "imu", {"imu"}, set())
    rate = channel("angular_velocity", "imu", {"imu"}, set())
    velocity = channel("linear_velocity", "dvl", {"dvl"}, {"imu_dead_reckoning"})
    depth = channel("depth", "depth", {"depth_sensor"}, {"velocity_dead_reckoning"})
    reference = str(metadata.get("horizontal_reference", "unknown"))
    xy_source = str(sources.get("position_xy", "unknown"))
    if xy_source == "initial_pose_prior":
        xy_status = "prior"
    elif xy_source in {"dvl_dead_reckoning", "imu_dead_reckoning", "decaying_velocity_hold"}:
        xy_status = (
            "dead_reckoning" if reference == "initial_dead_reckoning"
            and attitude["status"] == "measured"
            and velocity["status"] == "measured" else "degraded"
        )
    else:
        xy_status = "unknown"
    return {
        "time_s": _finite(metadata.get("time_s")),
        "attitude": attitude,
        "angular_rate": rate,
        "horizontal_velocity_body": velocity,
        "horizontal_position_ned": {"status": xy_status, "source": xy_source, "reference": reference},
        "depth": depth,
    }


def display_observability(log):
    """Preserve both epochs, and mask metrics if either end is untrusted."""
    current = estimate_observability({
        "time_s": log.get("time"),
        "sources": log.get("state_estimate_sources"),
        "excluded_sensors": log.get("state_estimate_excluded_sensors", ()),
        "rejected_sensors": log.get("state_estimate_rejected_sensors", ()),
        "horizontal_reference": log.get("horizontal_position_reference"),
        "sensor_health": log.get("sensor_health"),
    })
    controller = estimate_observability(log.get("controller_estimate_metadata"))
    if not log.get("sensor_feedback_enabled", False) or log.get("controller_state_source") != "sensor_estimate_previous_step":
        controller = estimate_observability(None)
    def measured(key):
        return all(x[key]["status"] == "measured" for x in (current, controller))
    valid_xy = all(x["horizontal_position_ned"]["status"] == "dead_reckoning" for x in (current, controller))
    conditions = {
        "tilt_angle_deg": measured("attitude"),
        "yaw_tracking_error_deg": measured("attitude"),
        "yaw_rate_tracking_error_deg_s": measured("angular_rate"),
        "depth_tracking_error_m": measured("depth"),
        "horizontal_tracking_error_m": valid_xy,
        "cross_track_error_m": valid_xy,
    }
    metrics = {key: _finite(log.get("ftc_" + key)) if valid else None for key, valid in conditions.items()}
    mode = str(log.get("ftc_guidance_mode_applied", "normal"))
    return {
        "current_estimate": current, "controller_input": controller,
        "last_external_fix_s": _finite(_mapping(log.get("mission_recovery")).get("accepted_fix_time_s")),
        "metrics_basis": "previous_step_controller_input_checked_against_current_health",
        "metrics": metrics,
        "mission_paused": bool(log.get("ftc_mission_paused_applied", False)) or mode.startswith("safety_pause") or mode in {"attitude_feedback_guard", "horizontal_degraded_hold", "vertical_thruster_hold"},
        "legacy_provenance_missing": not isinstance(log.get("controller_estimate_metadata"), Mapping),
    }


def safety_display_lines(frame):
    """One display contract for summary and technical layouts; no global SAFE."""
    obs = _mapping(frame.get("observability"))
    metrics = _mapping(obs.get("metrics"))
    current = _mapping(obs.get("current_estimate"))
    controller = _mapping(obs.get("controller_input"))
    def number(key, decimals):
        value = _finite(metrics.get(key))
        return "--" if value is None else f"{value:.{decimals}f}"
    def state(key):
        now = str(_mapping(current.get(key)).get("status", "unknown"))
        before = str(_mapping(controller.get(key)).get("status", "unknown"))
        if obs.get("legacy_provenance_missing", False):
            return "?"
        if now == "measured" and before in {"unavailable", "degraded"}:
            return "recovering"
        if now == "measured" and before in {"unknown", "prior"}:
            return "prior/?"
        return {"measured": "OK", "dead_reckoning": "DR", "unavailable": "LOST", "degraded": "DEG", "unknown": "?", "prior": "prior"}.get(now, now)
    heading = "PAUSED" if obs.get("mission_paused", False) else "OBSERVED"
    line1 = (f"{heading} | tilt {number('tilt_angle_deg', 1)}/35 deg | "
             f"yaw {number('yaw_tracking_error_deg', 1)}/25 deg | "
             f"dz {number('depth_tracking_error_m', 2)}/3 m")
    last_fix = _finite(obs.get("last_external_fix_s"))
    fix_note = "no abs fix" if last_fix is None else f"last fix {last_fix:.1f}s"
    line2 = (f"Att {state('attitude')} | rate {state('angular_rate')} | "
             f"V(body) {state('horizontal_velocity_body')} | XY {state('horizontal_position_ned')} "
             f"({fix_note}) | dXY {number('horizontal_tracking_error_m', 2)}/8 m")
    exceeded = any(metrics.get(k) is not None and metrics[k] >= limit for k, limit in (
        ("tilt_angle_deg", 35), ("yaw_tracking_error_deg", 25),
        ("depth_tracking_error_m", 3), ("horizontal_tracking_error_m", 8)))
    incomplete = any(value is None for value in metrics.values())
    tier = "confirmed" if exceeded else "possible" if incomplete or obs.get("mission_paused") else "normal"
    return line1, line2, tier
