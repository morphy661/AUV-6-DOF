"""Run a hash-locked development batch of randomized unified demonstrations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_ROOT = (
    REPOSITORY_ROOT
    / "depth-sensor-fault-detection"
    / "depth_fault_detection"
)
for path in (SRC_ROOT, MODEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from demo_six_dof_unified_diagnostics import (
    DEFAULT_CHECKPOINT,
    DEFAULT_TEMPORAL_CONFIG,
    run_demo,
)
from model_six_dof_multitask import AUVSixDOFMultiTaskDetector
from presentation.six_dof_model_bridge import SixDOFModelBridge


DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT
    / "docs"
    / "six_dof_unified_random_batch_protocol_v1.json"
)
TIER_RANK = {"normal": 0, "log_only": 1, "possible": 2, "confirmed": 3}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def validate_protocol(protocol, protocol_path):
    if protocol.get("protocol_id") != "six_dof_unified_random_batch_v1":
        raise ValueError("unexpected protocol_id")
    if not protocol.get("locked_before_execution", False):
        raise ValueError("protocol is not locked")
    if protocol.get("evaluation_type") != "development_random_batch":
        raise ValueError("this script is only for the declared development batch")
    for relative, expected in protocol.get("code_sha256", {}).items():
        actual = sha256_file(REPOSITORY_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"code hash mismatch: {relative}")
    for relative, expected in protocol.get("artifact_sha256", {}).items():
        actual = sha256_file(REPOSITORY_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"artifact hash mismatch: {relative}")
    configuration = protocol["configuration"]
    count = int(configuration["mission_count"])
    base_seed = int(configuration["base_seed"])
    if count < 20:
        raise ValueError("development batch must contain at least 20 missions")
    seeds = list(range(base_seed, base_seed + count))
    output_dir = REPOSITORY_ROOT / protocol["output_directory"]
    if output_dir.exists():
        raise FileExistsError("locked batch output already exists")
    return configuration, seeds, output_dir, sha256_file(protocol_path)


def _frames_in_interval(frames, start, end, grace=0.0):
    return [
        frame for frame in frames
        if float(start) <= frame["time_s"] <= float(end) + float(grace)
    ]


def _sensor_tier(frame, sensor):
    return frame["sensors"][sensor]["tier"]


def _expected_sensor_action(sensor):
    return "safe_hold_or_abort" if sensor == "imu" else "degraded_operation"


def _first_time(frames, predicate):
    for frame in frames:
        if predicate(frame):
            return float(frame["time_s"])
    return None


def evaluate_mission(seed, frames, manifest):
    sensor_events = manifest["sensor_events"]
    weak = next(event for event in sensor_events if "weak_spike" in event["event_id"])
    ambiguous = next(
        event for event in sensor_events
        if event["mode"] in ("bias", "drift")
    )
    intermittent = [
        event for event in sensor_events if event["mode"] == "unavailable"
    ]

    weak_frames = _frames_in_interval(
        frames, weak["start_time_s"], weak["end_time_s"], grace=0.75
    )
    weak_peak = max(
        (TIER_RANK[_sensor_tier(frame, weak["sensor"])] for frame in weak_frames),
        default=0,
    )
    ambiguous_frames = _frames_in_interval(
        frames,
        ambiguous["start_time_s"],
        ambiguous["end_time_s"],
        grace=4.0,
    )
    ambiguous_tiers = [
        _sensor_tier(frame, ambiguous["sensor"])
        for frame in ambiguous_frames
    ]

    confirmed_intervals = 0
    matched_actions = 0
    expected_action = _expected_sensor_action(intermittent[0]["sensor"])
    for event in intermittent:
        interval = _frames_in_interval(
            frames, event["start_time_s"], event["end_time_s"], grace=0.10
        )
        confirmed_intervals += int(any(
            _sensor_tier(frame, event["sensor"]) == "confirmed"
            for frame in interval
        ))
        matched_actions += int(any(
            frame["ftc"]["action"] == expected_action for frame in interval
        ))
    intermittent_window = _frames_in_interval(
        frames,
        intermittent[0]["start_time_s"],
        intermittent[-1]["end_time_s"],
        grace=2.0,
    )
    intermittent_root_possible = any(
        _sensor_tier(frame, intermittent[0]["sensor"]) == "possible"
        for frame in intermittent_window
    )

    truth = manifest["thruster_fault"]
    thruster_name = truth["thruster_name"]
    thruster_mode = truth["mode"]
    fault_start = float(truth["start_time_s"])
    post_fault = [frame for frame in frames if frame["time_s"] >= fault_start]
    target_time = _first_time(
        post_fault,
        lambda frame: frame["ftc"]["target_thruster"] == thruster_name,
    )
    any_targets = [
        frame["ftc"]["target_thruster"]
        for frame in frames
        if frame["ftc"]["target_thruster"] is not None
    ]
    wrong_target = any(target != thruster_name for target in any_targets)
    evidence_time = _first_time(
        post_fault,
        lambda frame: next(
            card["no_output_score"]
            for card in frame["thrusters"]
            if card["name"] == thruster_name
        ) >= 0.80,
    )

    model_frames = [
        frame for frame in post_fault
        if frame["maintenance"]["available"]
        and frame["maintenance"]["updated"]
    ]
    peak_model = (
        max(
            model_frames,
            key=lambda frame: frame["maintenance"]["fault_probability"],
        )
        if model_frames else None
    )
    candidates = (
        [] if peak_model is None else peak_model["maintenance"]["candidates"]
    )
    expected_group = "horizontal" if thruster_name.startswith("H") else "vertical"
    pre_thruster_model_advisory = any(
        frame["maintenance"]["tier"] in ("log_only", "possible")
        for frame in frames
        if frame["time_s"] < fault_start
    )
    model_advisory = any(
        frame["maintenance"]["tier"] in ("log_only", "possible")
        for frame in post_fault
    )

    return {
        "seed": int(seed),
        "weak_sensor": weak["sensor"],
        "weak_spike_overpromoted": weak_peak >= TIER_RANK["possible"],
        "ambiguous_sensor": ambiguous["sensor"],
        "ambiguous_mode": ambiguous["mode"],
        "ambiguous_recorded": any(
            tier in ("log_only", "possible", "confirmed")
            for tier in ambiguous_tiers
        ),
        "ambiguous_operator_possible": "possible" in ambiguous_tiers,
        "intermittent_sensor": intermittent[0]["sensor"],
        "intermittent_confirmed_intervals": confirmed_intervals,
        "intermittent_interval_count": len(intermittent),
        "intermittent_root_possible": intermittent_root_possible,
        "sensor_ftc_action_matches": matched_actions,
        "sensor_ftc_action_count": len(intermittent),
        "thruster_name": thruster_name,
        "thruster_mode": thruster_mode,
        "thruster_fault_start_s": fault_start,
        "no_output_evidence_observed": evidence_time is not None,
        "no_output_evidence_delay_s": (
            None if evidence_time is None else evidence_time - fault_start
        ),
        "ftc_correct_target_observed": target_time is not None,
        "ftc_target_delay_s": (
            None if target_time is None else target_time - fault_start
        ),
        "wrong_thruster_target": wrong_target,
        "any_thruster_target": bool(any_targets),
        "safe_ftc_behavior": (
            target_time is not None and not wrong_target
            if thruster_mode == "no_output"
            else not any_targets
        ),
        "model_advisory_observed": model_advisory,
        "pre_thruster_model_advisory": pre_thruster_model_advisory,
        "model_peak_fault_probability": (
            None if peak_model is None
            else peak_model["maintenance"]["fault_probability"]
        ),
        "model_peak_mode": (
            "none" if peak_model is None
            else peak_model["maintenance"]["probable_mode"]
        ),
        "model_mode_correct": bool(
            peak_model is not None
            and peak_model["maintenance"]["probable_mode"] == thruster_mode
        ),
        "model_peak_group": (
            "none" if peak_model is None
            else peak_model["maintenance"]["suspected_group"]
        ),
        "model_group_correct": bool(
            peak_model is not None
            and peak_model["maintenance"]["suspected_group"] == expected_group
        ),
        "model_top1_correct": bool(
            candidates and candidates[0]["name"] == thruster_name
        ),
        "model_top2_correct": bool(
            any(candidate["name"] == thruster_name for candidate in candidates[:2])
        ),
        "model_top1_name": None if not candidates else candidates[0]["name"],
        "model_top1_probability": (
            None if not candidates else candidates[0]["probability"]
        ),
    }


def _rate(rows, key, predicate=lambda row: True):
    values = [bool(row[key]) for row in rows if predicate(row)]
    return None if not values else mean(values)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _delays(rows, key, predicate=lambda row: True):
    return [
        float(row[key]) for row in rows
        if predicate(row) and row[key] is not None
    ]


def summarize(rows, thresholds):
    no_output = [row for row in rows if row["thruster_mode"] == "no_output"]
    thrust_loss = [row for row in rows if row["thruster_mode"] == "thrust_loss"]
    confirmed = sum(row["intermittent_confirmed_intervals"] for row in rows)
    intervals = sum(row["intermittent_interval_count"] for row in rows)
    action_matches = sum(row["sensor_ftc_action_matches"] for row in rows)
    action_count = sum(row["sensor_ftc_action_count"] for row in rows)
    evidence_delays = _delays(no_output, "no_output_evidence_delay_s")
    target_delays = _delays(no_output, "ftc_target_delay_s")
    metrics = {
        "mission_count": len(rows),
        "no_output_mission_count": len(no_output),
        "thrust_loss_mission_count": len(thrust_loss),
        "sensor": {
            "weak_spike_overpromotion_rate": _rate(
                rows, "weak_spike_overpromoted"
            ),
            "ambiguous_record_rate": _rate(rows, "ambiguous_recorded"),
            "ambiguous_operator_possible_rate": _rate(
                rows, "ambiguous_operator_possible"
            ),
            "intermittent_sample_confirmation_rate": _ratio(
                confirmed, intervals
            ),
            "intermittent_root_possible_rate": _rate(
                rows, "intermittent_root_possible"
            ),
            "sensor_ftc_action_match_rate": _ratio(
                action_matches, action_count
            ),
        },
        "thruster_ftc": {
            "no_output_evidence_recall": _rate(
                no_output, "no_output_evidence_observed"
            ),
            "no_output_target_recall": _rate(
                no_output, "ftc_correct_target_observed"
            ),
            "thrust_loss_wrong_isolation_rate": _rate(
                thrust_loss, "any_thruster_target"
            ),
            "wrong_thruster_target_mission_rate": _rate(
                rows, "wrong_thruster_target"
            ),
            "safe_ftc_behavior_rate": _rate(rows, "safe_ftc_behavior"),
            "median_no_output_evidence_delay_s": (
                None if not evidence_delays else median(evidence_delays)
            ),
            "median_no_output_target_delay_s": (
                None if not target_delays else median(target_delays)
            ),
        },
        "model_advice": {
            "post_fault_advisory_rate": _rate(
                rows, "model_advisory_observed"
            ),
            "pre_thruster_advisory_mission_rate": _rate(
                rows, "pre_thruster_model_advisory"
            ),
            "peak_mode_accuracy": _rate(rows, "model_mode_correct"),
            "peak_group_accuracy": _rate(rows, "model_group_correct"),
            "peak_top1_location_accuracy": _rate(
                rows, "model_top1_correct"
            ),
            "peak_top2_location_accuracy": _rate(
                rows, "model_top2_correct"
            ),
            "no_output_peak_mode_accuracy": _rate(
                no_output, "model_mode_correct"
            ),
            "thrust_loss_peak_mode_accuracy": _rate(
                thrust_loss, "model_mode_correct"
            ),
        },
    }
    checks = {
        "weak_spike_overpromotion_at_most": (
            metrics["sensor"]["weak_spike_overpromotion_rate"]
            <= thresholds["weak_spike_overpromotion_rate_max"]
        ),
        "ambiguous_record_rate_at_least": (
            metrics["sensor"]["ambiguous_record_rate"]
            >= thresholds["ambiguous_record_rate_min"]
        ),
        "intermittent_confirmation_at_least": (
            metrics["sensor"]["intermittent_sample_confirmation_rate"]
            >= thresholds["intermittent_sample_confirmation_rate_min"]
        ),
        "sensor_ftc_match_at_least": (
            metrics["sensor"]["sensor_ftc_action_match_rate"]
            >= thresholds["sensor_ftc_action_match_rate_min"]
        ),
        "no_output_target_recall_at_least": (
            metrics["thruster_ftc"]["no_output_target_recall"]
            >= thresholds["no_output_target_recall_min"]
        ),
        "wrong_thruster_target_rate_at_most": (
            metrics["thruster_ftc"]["wrong_thruster_target_mission_rate"]
            <= thresholds["wrong_thruster_target_mission_rate_max"]
        ),
        "thrust_loss_wrong_isolation_at_most": (
            metrics["thruster_ftc"]["thrust_loss_wrong_isolation_rate"]
            <= thresholds["thrust_loss_wrong_isolation_rate_max"]
        ),
        "model_top2_accuracy_at_least": (
            metrics["model_advice"]["peak_top2_location_accuracy"]
            >= thresholds["model_peak_top2_accuracy_min"]
        ),
    }
    return {
        **metrics,
        "acceptance_thresholds": thresholds,
        "acceptance_checks": checks,
        "all_acceptance_checks_passed": all(checks.values()),
    }


def save_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(summary, path):
    panels = (
        (
            "Sensor display and guard",
            summary["sensor"],
            (
                ("Ambiguous recorded", "ambiguous_record_rate"),
                ("Intermittent confirmed", "intermittent_sample_confirmation_rate"),
                ("Sensor FTC match", "sensor_ftc_action_match_rate"),
                ("Intermittent root possible", "intermittent_root_possible_rate"),
            ),
        ),
        (
            "Thruster FTC safety",
            summary["thruster_ftc"],
            (
                ("No-output evidence", "no_output_evidence_recall"),
                ("No-output target", "no_output_target_recall"),
                ("Safe FTC behavior", "safe_ftc_behavior_rate"),
                ("No wrong target", "wrong_thruster_target_mission_rate"),
            ),
        ),
        (
            "Learned maintenance advice",
            summary["model_advice"],
            (
                ("Fault-mode accuracy", "peak_mode_accuracy"),
                ("Group accuracy", "peak_group_accuracy"),
                ("Top-1 location", "peak_top1_location_accuracy"),
                ("Top-2 location", "peak_top2_location_accuracy"),
            ),
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True)
    figure.patch.set_facecolor("#07131f")
    for axis, (title, values, items) in zip(axes, panels):
        axis.set_facecolor("#0d2031")
        labels = [item[0] for item in items]
        rates = []
        for _, key in items:
            value = values[key]
            if key == "wrong_thruster_target_mission_rate":
                value = 1.0 - value
            rates.append(value)
        colors = ["#31c7b5" if value >= 0.80 else "#f1b94b" for value in rates]
        y = np.arange(len(items))
        axis.barh(y, rates, color=colors, alpha=0.9)
        axis.set_yticks(y, labels, color="#eef6fb")
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.0)
        axis.set_title(title, color="#eef6fb", loc="left")
        axis.grid(axis="x", color="#294256", alpha=0.7)
        axis.tick_params(colors="#91a8ba")
        for spine in axis.spines.values():
            spine.set_color("#294256")
        for row, value in enumerate(rates):
            axis.text(
                min(value + 0.02, 0.98), row, f"{100 * value:.1f}%",
                color="#eef6fb", va="center", ha=("right" if value > 0.92 else "left"),
            )
    axes[0].set_xlabel("Mission/event rate", color="#91a8ba")
    figure.suptitle(
        f"Unified random batch development evaluation | {summary['mission_count']} missions",
        color="#eef6fb", x=0.04, ha="left", fontsize=14,
    )
    figure.text(
        0.04, 0.01,
        "Simulation development evidence; learned output remains advisory and is not an FTC command.",
        color="#91a8ba", fontsize=9,
    )
    figure.tight_layout(rect=(0.03, 0.05, 0.99, 0.92))
    figure.savefig(path, dpi=170, facecolor=figure.get_facecolor())
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--model-device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    configuration, seeds, output_dir, protocol_hash = validate_protocol(
        protocol, protocol_path
    )
    bridge = SixDOFModelBridge(
        REPOSITORY_ROOT / protocol["model_checkpoint_path"],
        REPOSITORY_ROOT / protocol["temporal_config_path"],
        AUVSixDOFMultiTaskDetector,
        device=(None if args.model_device == "auto" else args.model_device),
    )
    rows = []
    for mission_index, seed in enumerate(seeds, start=1):
        _, frames, _, manifest = run_demo(
            float(configuration["duration_s"]),
            float(configuration["dt_s"]),
            seed,
            injection_mode="random",
            model_bridge=bridge,
        )
        row = evaluate_mission(seed, frames, manifest)
        rows.append(row)
        print(
            f"{mission_index:02d}/{len(seeds)} seed={seed} "
            f"thruster={row['thruster_name']}:{row['thruster_mode']} "
            f"ftc_safe={row['safe_ftc_behavior']} "
            f"top2={row['model_top2_correct']}"
        )
    summary = summarize(rows, protocol["acceptance_thresholds"])
    payload = {
        "benchmark": "six_dof_unified_random_batch_v1",
        "evaluation_type": "development_random_batch",
        "real_sea_trial_claim": False,
        "independent_blind_test_claim": False,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "configuration": configuration,
        "model_device": str(bridge.device),
        "summary": summary,
        "missions": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "unified_random_batch_summary.json"
    csv_path = output_dir / "unified_random_batch_missions.csv"
    plot_path = output_dir / "unified_random_batch_summary.png"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_csv(rows, csv_path)
    save_plot(summary, plot_path)
    print(json.dumps(summary, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
