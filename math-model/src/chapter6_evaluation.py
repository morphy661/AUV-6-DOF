"""Batch closed-loop evaluation and thesis artifact export for Chapter 6."""

import argparse
import csv
import json
import os
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from faults.system_faults import FaultType
from main import (
    DEFAULT_RANDOM_SEED,
    execute_mission,
    get_recovery_action,
    merge_fault_for_monitoring_and_ftc,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "chapter6_evaluation"
FAULT_NAME_TO_TYPE = {fault.name: fault for fault in FaultType}
FTC_NAME_TO_ID = {
    "NO_FAULT": 0,
    "NORMAL": 0,
    "BIAS": 1,
    "DRIFT": 2,
    "STUCK": 3,
    "SPIKE": 4,
    "NOISE_INCREASE": 5,
    "NOISE": 5,
    "INCREASED NOISE": 5,
    "THRUSTER_ENTANGLED": 6,
    "ENTANGLED": 6,
    "THRUSTER_THRUST_LOSS": 6,
    "THRUST_LOSS": 6,
    "THRUSTER_NO_OUTPUT": 7,
    "NO_OUTPUT": 7,
}


def parse_faults(text):
    if text.strip().lower() == "all":
        return list(FaultType)
    names = [part.strip().upper() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in FAULT_NAME_TO_TYPE]
    if unknown:
        raise ValueError(f"Unknown fault names: {unknown}")
    return [FAULT_NAME_TO_TYPE[name] for name in names]


def parse_times(text):
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one fault time is required")
    return values


def safe_float(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def mapped_prediction(log, key):
    try:
        return merge_fault_for_monitoring_and_ftc(int(log.get(key, 0)))
    except (TypeError, ValueError):
        return 0


def confirmed_prediction(log):
    return diagnosis_name_to_id(log.get("ftc_diagnosis", "NO_FAULT"))


def first_time(logs, start_time, predicate):
    for log in logs:
        time_value = float(log.get("time", 0.0))
        if start_time is not None and time_value < start_time:
            continue
        if predicate(log):
            return time_value
    return None


def count_alarm_episodes(logs, end_time=None):
    episodes = 0
    active = False
    for log in logs:
        time_value = float(log.get("time", 0.0))
        if end_time is not None and time_value >= end_time:
            break
        alarm = confirmed_prediction(log) not in (0, -1)
        if alarm and not active:
            episodes += 1
        active = alarm
    return episodes


def effective_fault_start(logs, raw_fault_id, configured_time):
    if raw_fault_id == 0:
        return None
    return first_time(
        logs,
        configured_time,
        lambda log: int(log.get("fault_label", 0)) == raw_fault_id,
    )


def count_fault_event_episodes(logs, raw_fault_id):
    episodes = 0
    active = False
    for log in logs:
        current = int(log.get("fault_label", 0)) == raw_fault_id
        if current and not active:
            episodes += 1
        active = current
    return episodes


def first_sustained_time(logs, start_time, predicate, samples=20):
    count = 0
    first = None
    for log in logs:
        time_value = float(log.get("time", 0.0))
        if start_time is not None and time_value < start_time:
            continue
        if predicate(log):
            if count == 0:
                first = time_value
            count += 1
            if count >= samples:
                return first
        else:
            count = 0
            first = None
    return None


def diagnosis_name_to_id(name):
    return FTC_NAME_TO_ID.get(str(name).strip().upper(), -1)


def summarize_trial(details, fault_type, configured_time, repeat_index):
    logs = details["sensor_logs"]
    raw_fault_id = int(fault_type.value)
    target_id = merge_fault_for_monitoring_and_ftc(raw_fault_id)
    event_start = effective_fault_start(logs, raw_fault_id, configured_time)
    event_observed = raw_fault_id == 0 or event_start is not None

    if raw_fault_id == 0:
        analysis_start = None
        ai_time = None
        rule_time = None
        final_time = None
    else:
        analysis_start = event_start
        ai_time = first_time(
            logs,
            analysis_start,
            lambda log: mapped_prediction(log, "ai_pred") == target_id,
        )
        rule_time = first_time(
            logs,
            analysis_start,
            lambda log: mapped_prediction(log, "rule_pred") == target_id,
        )
        final_time = first_time(
            logs,
            analysis_start,
            lambda log: confirmed_prediction(log) == target_id,
        )

    expected_action = get_recovery_action(target_id)
    first_action_time = first_time(
        logs,
        analysis_start,
        lambda log: str(log.get("ftc_action", "Normal Cruising")) == expected_action,
    )
    observed_actions = {
        str(log.get("ftc_action", "Normal Cruising")) for log in logs
    }

    if raw_fault_id == 0:
        first_action_time = None
        false_alarm_count = count_alarm_episodes(logs)
        diagnosis_success = false_alarm_count == 0
        action_correct = observed_actions <= {"Normal Cruising"}
        safe_state_time = None
        safe_state_reached = diagnosis_success and action_correct
    else:
        false_alarm_count = count_alarm_episodes(logs, end_time=analysis_start)
        diagnosis_success = final_time is not None
        action_correct = expected_action in observed_actions

        if not action_correct or first_action_time is None:
            safe_state_time = None
            safe_state_reached = False
        elif "Ascent" in expected_action:
            safe_state_time = first_sustained_time(
                logs,
                first_action_time,
                lambda log: float(log.get("true_depth", np.inf)) <= 1.0,
                samples=10,
            )
            safe_state_reached = safe_state_time is not None
        elif "Safe Hover" in expected_action or "Depth-Hold" in expected_action:
            safe_state_time = first_sustained_time(
                logs,
                first_action_time,
                lambda log: abs(float(np.asarray(log.get("velocity", [0, 0, 99]))[2])) <= 0.15,
                samples=20,
            )
            safe_state_reached = safe_state_time is not None
        else:
            safe_state_time = first_action_time
            safe_state_reached = action_correct

    post_fault_logs = [
        log for log in logs
        if "ftc_action" in log
        and (analysis_start is None or float(log.get("time", 0.0)) >= analysis_start)
    ]
    tracking_errors = [
        abs(float(log.get("target_z", 0.0)) - float(log.get("true_depth", 0.0)))
        for log in post_fault_logs
    ]
    sensor_errors = [
        abs(float(log.get("depth", 0.0)) - float(log.get("true_depth", 0.0)))
        for log in post_fault_logs
    ]
    max_tracking_error = max(tracking_errors) if tracking_errors else None
    max_sensor_error = max(sensor_errors) if sensor_errors else None
    final_depth = float(logs[-1].get("true_depth", 0.0)) if logs else None

    mission_success = bool(diagnosis_success and action_correct and safe_state_reached)
    final_diagnosis_id = diagnosis_name_to_id(details["final_diagnosis"])

    return {
        "fault_type": fault_type.name,
        "raw_fault_id": raw_fault_id,
        "ftc_fault_id": target_id,
        "route_profile": details["route_profile"],
        "repeat": repeat_index,
        "random_seed": details["random_seed"],
        "mission_duration_s": details["duration"],
        "configured_fault_start_s": None if raw_fault_id == 0 else configured_time,
        "effective_fault_start_s": safe_float(event_start),
        "fault_event_observed": int(event_observed),
        "fault_event_count": count_fault_event_episodes(logs, raw_fault_id) if raw_fault_id != 0 else 0,
        "first_ai_detection_s": safe_float(ai_time),
        "first_rule_detection_s": safe_float(rule_time),
        "final_confirmation_s": safe_float(final_time),
        "first_recovery_action_s": safe_float(first_action_time),
        "ai_detection_delay_s": safe_float(
            None if ai_time is None or event_start is None else ai_time - event_start
        ),
        "rule_detection_delay_s": safe_float(
            None if rule_time is None or event_start is None else rule_time - event_start
        ),
        "diagnosis_delay_s": safe_float(
            None if final_time is None or event_start is None else final_time - event_start
        ),
        "safe_state_time_s": safe_float(safe_state_time),
        "safe_state_delay_s": safe_float(
            None if safe_state_time is None or event_start is None else safe_state_time - event_start
        ),
        "expected_recovery_action": expected_action,
        "reported_final_diagnosis": details["final_diagnosis"],
        "reported_final_diagnosis_id": final_diagnosis_id,
        "reported_final_action": details["final_action"],
        "diagnosis_success": int(diagnosis_success),
        "recovery_action_correct": int(action_correct),
        "safe_state_reached": int(safe_state_reached),
        "mission_success": int(mission_success),
        "false_alarm_count": false_alarm_count,
        "maximum_tracking_error_m": safe_float(max_tracking_error),
        "maximum_sensor_error_m": safe_float(max_sensor_error),
        "final_depth_m": safe_float(final_depth),
        "filtered_spike_event_count": len(details.get("spike_times", [])),
    }


def mean_std(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def build_summary(trials, fault_order):
    summary = []
    for fault in fault_order:
        planned_rows = [row for row in trials if row["fault_type"] == fault.name]
        if not planned_rows:
            continue
        rows = [row for row in planned_rows if row["fault_event_observed"] == 1]
        if not rows:
            summary.append({
                "fault_type": fault.name,
                "planned_trials": len(planned_rows),
                "valid_trials": 0,
                "fault_event_observation_rate_pct": 0.0,
                "diagnosis_success_rate_pct": None,
                "recovery_action_accuracy_pct": None,
                "safe_state_rate_pct": None,
                "mission_success_rate_pct": None,
                "mean_false_alarm_count": None,
                "ai_delay_mean_s": None,
                "ai_delay_std_s": None,
                "rule_delay_mean_s": None,
                "rule_delay_std_s": None,
                "diagnosis_delay_mean_s": None,
                "diagnosis_delay_std_s": None,
                "safe_state_delay_mean_s": None,
                "safe_state_delay_std_s": None,
                "maximum_tracking_error_mean_m": None,
                "maximum_sensor_error_mean_m": None,
            })
            continue
        ai_mean, ai_std = mean_std(rows, "ai_detection_delay_s")
        rule_mean, rule_std = mean_std(rows, "rule_detection_delay_s")
        final_mean, final_std = mean_std(rows, "diagnosis_delay_s")
        safe_mean, safe_std = mean_std(rows, "safe_state_delay_s")
        count = len(rows)
        summary.append({
            "fault_type": fault.name,
            "planned_trials": len(planned_rows),
            "valid_trials": count,
            "fault_event_observation_rate_pct": 100.0 * count / len(planned_rows),
            "diagnosis_success_rate_pct": 100.0 * sum(r["diagnosis_success"] for r in rows) / count,
            "recovery_action_accuracy_pct": 100.0 * sum(r["recovery_action_correct"] for r in rows) / count,
            "safe_state_rate_pct": 100.0 * sum(r["safe_state_reached"] for r in rows) / count,
            "mission_success_rate_pct": 100.0 * sum(r["mission_success"] for r in rows) / count,
            "mean_false_alarm_count": float(np.mean([r["false_alarm_count"] for r in rows])),
            "ai_delay_mean_s": ai_mean,
            "ai_delay_std_s": ai_std,
            "rule_delay_mean_s": rule_mean,
            "rule_delay_std_s": rule_std,
            "diagnosis_delay_mean_s": final_mean,
            "diagnosis_delay_std_s": final_std,
            "safe_state_delay_mean_s": safe_mean,
            "safe_state_delay_std_s": safe_std,
            "maximum_tracking_error_mean_m": float(np.mean([
                r["maximum_tracking_error_m"] for r in rows
                if r["maximum_tracking_error_m"] is not None
            ])),
            "maximum_sensor_error_mean_m": float(np.mean([
                r["maximum_sensor_error_m"] for r in rows
                if r["maximum_sensor_error_m"] is not None
            ])),
        })
    return summary


def write_dict_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


INTEGER_TRIAL_FIELDS = {
    "raw_fault_id", "ftc_fault_id", "repeat", "random_seed",
    "fault_event_observed", "fault_event_count", "reported_final_diagnosis_id",
    "diagnosis_success", "recovery_action_correct", "safe_state_reached",
    "mission_success", "false_alarm_count", "filtered_spike_event_count",
}
FLOAT_TRIAL_FIELDS = {
    "mission_duration_s", "configured_fault_start_s", "effective_fault_start_s",
    "first_ai_detection_s", "first_rule_detection_s", "final_confirmation_s",
    "first_recovery_action_s", "ai_detection_delay_s", "rule_detection_delay_s",
    "diagnosis_delay_s", "safe_state_time_s", "safe_state_delay_s",
    "maximum_tracking_error_m", "maximum_sensor_error_m", "final_depth_m",
}


def load_trials_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in INTEGER_TRIAL_FIELDS:
            value = row.get(field, "")
            row[field] = None if value in (None, "") else int(float(value))
        for field in FLOAT_TRIAL_FIELDS:
            value = row.get(field, "")
            row[field] = None if value in (None, "") else float(value)
    return rows


def rebuild_existing_outputs(output_dir, fault_order):
    output_dir = Path(output_dir).resolve()
    trials_path = output_dir / "chapter6_trials.csv"
    trials = load_trials_csv(trials_path)

    # Repair only derived safety fields. Raw detection/action times remain unchanged.
    for row in trials:
        if row["fault_type"] == "NO_FAULT":
            row["first_recovery_action_s"] = None
            row["safe_state_time_s"] = None
            row["safe_state_delay_s"] = None
            row["safe_state_reached"] = int(
                row["diagnosis_success"] == 1 and row["recovery_action_correct"] == 1
            )
        elif row["recovery_action_correct"] != 1 or row["first_recovery_action_s"] is None:
            row["safe_state_time_s"] = None
            row["safe_state_delay_s"] = None
            row["safe_state_reached"] = 0
        row["mission_success"] = int(
            row["diagnosis_success"] == 1
            and row["recovery_action_correct"] == 1
            and row["safe_state_reached"] == 1
        )

    summary = build_summary(trials, fault_order)
    write_dict_csv(trials_path, trials)
    write_dict_csv(output_dir / "chapter6_summary_by_fault.csv", summary)
    plot_detection_timing(summary, output_dir / "fig_ch6_detection_timing.png")
    plot_success_rates(summary, output_dir / "fig_ch6_recovery_success.png")
    print(f"Rebuilt Chapter 6 summaries and figures from: {trials_path}")
    return trials, summary


def export_representative_timeseries(path, details):
    rows = []
    for log in details["sensor_logs"]:
        thruster = log.get("thruster", {}) or {}
        current = log.get("current_sensor", {}) or {}
        rows.append({
            "time_s": log.get("time"),
            "true_depth_m": log.get("true_depth"),
            "measured_depth_m": log.get("depth"),
            "target_depth_m": log.get("target_z"),
            "cmd_vz_mps": thruster.get("cmd_vz"),
            "actual_vz_mps": thruster.get("actual_vz"),
            "measured_current_a": current.get("measured_current", thruster.get("current")),
            "expected_current_a": current.get("expected_current", thruster.get("expected_current")),
            "ai_pred": log.get("ai_pred", 0),
            "rule_pred": log.get("rule_pred", 0),
            "final_pred": log.get("final_pred", 0),
            "ftc_locked": int(bool(log.get("ftc_is_locked", False))),
            "ftc_diagnosis": log.get("ftc_diagnosis", "NO_FAULT"),
            "ftc_action": log.get("ftc_action", "Normal Cruising"),
        })
    write_dict_csv(path, rows)


def plot_detection_timing(summary, path):
    rows = [row for row in summary if row["fault_type"] != "NO_FAULT"]
    labels = [row["fault_type"].replace("THRUSTER_", "").replace("NOISE_INCREASE", "NOISE") for row in rows]
    x = np.arange(len(rows))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 5.8))
    series = [
        ("AI evidence", "ai_delay_mean_s", "ai_delay_std_s", "#2878B5"),
        ("Rule evidence", "rule_delay_mean_s", "rule_delay_std_s", "#D95F02"),
        ("Final confirmation", "diagnosis_delay_mean_s", "diagnosis_delay_std_s", "#3A923A"),
    ]
    for index, (label, mean_key, std_key, color) in enumerate(series):
        means = np.array([np.nan if row[mean_key] is None else row[mean_key] for row in rows])
        errors = np.array([0.0 if row[std_key] is None else row[std_key] for row in rows])
        ax.bar(x + (index - 1) * width, means, width, yerr=errors, capsize=3, label=label, color=color)
    ax.set_ylabel("Delay from effective fault onset (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Closed-loop diagnosis timing by fault type")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_success_rates(summary, path):
    labels = [row["fault_type"].replace("THRUSTER_", "").replace("NOISE_INCREASE", "NOISE") for row in summary]
    x = np.arange(len(summary))
    width = 0.22
    fig, ax = plt.subplots(figsize=(12, 5.8))
    series = [
        ("Correct diagnosis", "diagnosis_success_rate_pct", "#2878B5"),
        ("Correct recovery", "recovery_action_accuracy_pct", "#D95F02"),
        ("Safe state", "safe_state_rate_pct", "#3A923A"),
        ("Overall success", "mission_success_rate_pct", "#6F4E9C"),
    ]
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    for offset, (label, key, color) in zip(offsets, series):
        values = [np.nan if row[key] is None else row[key] for row in summary]
        ax.bar(x + offset, values, width, label=label, color=color)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Diagnosis and fault-tolerant recovery outcomes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, loc="lower center")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_representative_trajectory(details, trial, path):
    trajectory = np.asarray(details["trajectory"], dtype=float)
    waypoints = np.asarray(details["planned_waypoints"], dtype=float)
    logs = details["sensor_logs"]
    if trajectory.size == 0:
        return

    def index_at(time_value):
        if time_value is None:
            return None
        for index, log in enumerate(logs):
            if float(log.get("time", 0.0)) >= float(time_value):
                return min(index, len(trajectory) - 1)
        return len(trajectory) - 1

    fault_index = index_at(trial["effective_fault_start_s"])
    confirm_index = index_at(trial["final_confirmation_s"])
    fig = plt.figure(figsize=(10.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
        color="#777777", linestyle="--", linewidth=1.3, label="Planned route",
    )
    ax.scatter(
        waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
        color="#C9A800", marker="x", s=35, label="Waypoints",
    )
    ax.plot(
        trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
        color="#1F5FBF", linewidth=2.2, label="AUV trajectory",
    )
    ax.scatter(*trajectory[0], color="#2B8C3E", s=70, label="Start")
    ax.scatter(*trajectory[-1], color="#8E44AD", s=70, label="End")
    if fault_index is not None:
        ax.scatter(*trajectory[fault_index], color="#E67E22", marker="X", s=120, label="Fault onset")
    if confirm_index is not None:
        ax.scatter(*trajectory[confirm_index], color="#C62828", marker="P", s=130, label="FTC intervention")

    delay = trial["diagnosis_delay_s"]
    delay_text = "not confirmed" if delay is None else f"{delay:.1f} s"
    ax.text2D(
        0.02,
        0.96,
        f"Fault: {trial['fault_type']}\nConfirmation delay: {delay_text}\n"
        f"Recovery: {trial['expected_recovery_action']}",
        transform=ax.transAxes,
        va="top",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "#777777", "alpha": 0.9},
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")
    ax.set_title("Representative closed-loop AUV trajectory and FTC intervention")
    ax.invert_zaxis()
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run_evaluation(args):
    faults = parse_faults(args.faults)
    fault_times = parse_times(args.fault_times)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    representative_fault = args.representative_fault.upper()
    if representative_fault not in FAULT_NAME_TO_TYPE:
        raise ValueError(f"Unknown representative fault: {representative_fault}")
    representative_time = (
        float(args.representative_time)
        if args.representative_time is not None
        else fault_times[len(fault_times) // 2]
    )

    if args.repeats < 1:
        raise ValueError("Repeats must be at least 1")
    if FAULT_NAME_TO_TYPE[representative_fault] not in faults:
        raise ValueError("Representative fault must also be included in --faults")
    if not any(abs(time_value - representative_time) < 1e-9 for time_value in fault_times):
        raise ValueError("Representative time must also be included in --fault-times")
    if max(fault_times) >= args.duration:
        raise ValueError("Mission duration must be greater than every fault start time")

    trials = []
    representative_details = None
    representative_trial = None
    total = len(faults) * len(fault_times) * args.repeats
    counter = 0
    for fault in faults:
        for fault_time in fault_times:
            for repeat in range(args.repeats):
                counter += 1
                seed = args.seed + counter - 1
                is_representative = (
                    fault.name == representative_fault
                    and abs(fault_time - representative_time) < 1e-9
                    and repeat == 0
                )
                print(
                    f"[{counter}/{total}] fault={fault.name}, start={fault_time:g}s, "
                    f"repeat={repeat + 1}, seed={seed}"
                )
                stem = (
                    f"fig_ch6_representative_{fault.name.lower()}_T{int(fault_time)}"
                    if is_representative
                    else f"trial_{fault.name.lower()}_T{int(fault_time)}_R{repeat + 1}"
                )
                if args.verbose:
                    details = execute_mission(
                        fault_type=fault,
                        duration_override=args.duration,
                        fault_start_time=fault_time,
                        route_profile=args.route,
                        random_seed=seed,
                        output_dir=output_dir,
                        output_stem=stem,
                        save_response_plots=is_representative,
                        save_trajectory_plot=False,
                        return_details=True,
                        force_spike_at_start=True,
                    )
                else:
                    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
                        details = execute_mission(
                            fault_type=fault,
                            duration_override=args.duration,
                            fault_start_time=fault_time,
                            route_profile=args.route,
                            random_seed=seed,
                            output_dir=output_dir,
                            output_stem=stem,
                            save_response_plots=is_representative,
                            save_trajectory_plot=False,
                            return_details=True,
                            force_spike_at_start=True,
                        )
                trial = summarize_trial(details, fault, fault_time, repeat + 1)
                trials.append(trial)
                if is_representative:
                    representative_details = details
                    representative_trial = trial

    summary = build_summary(trials, faults)
    write_dict_csv(output_dir / "chapter6_trials.csv", trials)
    write_dict_csv(output_dir / "chapter6_summary_by_fault.csv", summary)
    plot_detection_timing(summary, output_dir / "fig_ch6_detection_timing.png")
    plot_success_rates(summary, output_dir / "fig_ch6_recovery_success.png")

    if representative_details is not None:
        export_representative_timeseries(
            output_dir / "chapter6_representative_timeseries.csv",
            representative_details,
        )
        plot_representative_trajectory(
            representative_details,
            representative_trial,
            output_dir / "fig_ch6_representative_trajectory.png",
        )

    config = {
        "faults": [fault.name for fault in faults],
        "fault_times_s": fault_times,
        "repeats": args.repeats,
        "duration_s": args.duration,
        "route_profile": args.route,
        "base_seed": args.seed,
        "representative_fault": representative_fault,
        "representative_time_s": representative_time,
        "force_spike_at_configured_time": True,
        "output_dir": str(output_dir),
        "definitions": {
            "diagnosis_delay": "final confirmation time minus effective fault onset",
            "false_alarm_count": "number of non-zero final-diagnosis episodes before fault onset, or during the complete no-fault mission",
            "safe_state": "near-surface state for ascent actions, sustained low vertical speed for hover, and correct activation for filtering actions",
        },
    }
    with (output_dir / "chapter6_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"Chapter 6 evaluation outputs saved to: {output_dir}")
    return trials, summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run repeatable closed-loop FTC experiments for thesis Chapter 6."
    )
    parser.add_argument("--faults", default="all", help="Comma-separated FaultType names or 'all'.")
    parser.add_argument("--fault-times", default="80,300,600,900", help="Comma-separated injection times in seconds.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeated runs per fault and injection time.")
    parser.add_argument("--duration", type=float, default=1200.0, help="Mission duration in seconds.")
    parser.add_argument("--route", default="comprehensive", help="Route profile used by execute_mission.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Base random seed.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Artifact output directory.")
    parser.add_argument("--representative-fault", default="DRIFT", help="Fault used for response and trajectory figures.")
    parser.add_argument("--representative-time", type=float, default=None, help="Injection time for the representative trial; defaults to the middle configured time.")
    parser.add_argument("--rebuild-only", action="store_true", help="Rebuild summaries and aggregate figures from an existing chapter6_trials.csv without rerunning simulations.")
    parser.add_argument("--verbose", action="store_true", help="Show controller diagnostic messages for every trial.")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.rebuild_only:
        rebuild_existing_outputs(arguments.output_dir, parse_faults(arguments.faults))
    else:
        run_evaluation(arguments)
