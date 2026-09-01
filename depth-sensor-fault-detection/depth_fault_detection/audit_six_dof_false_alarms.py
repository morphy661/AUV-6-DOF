"""Descriptive false-alarm audit for the frozen 0-300 m Focus V3 model.

This runner deliberately does not search thresholds or modify the model.  It
reconstructs the already-selected temporal and maintenance policy, then audits
normal-labelled windows by mission, depth band, guidance context, excitation,
duration, and operator escalation.  Validation and test results are reported
separately so the test split is never used as a calibration source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
MATH_MODEL_SRC = REPO_ROOT / "math-model" / "src"
if str(MATH_MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MATH_MODEL_SRC))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from diagnosis.maintenance_health_decision import (  # noqa: E402
    apply_maintenance_decision_layer,
)
from diagnosis.maintenance_log_policy import (  # noqa: E402
    MaintenanceLogConfig,
    apply_maintenance_log_policy,
    maintenance_log_metrics,
)
from diagnosis.maintenance_ticket_policy import (  # noqa: E402
    MaintenanceTicketConfig,
    apply_maintenance_ticket_policy,
    extract_maintenance_ticket_evidence,
    maintenance_ticket_metrics,
)
from diagnosis.temporal_fault_decision import (  # noqa: E402
    TemporalDecisionConfig,
    apply_temporal_decision_layer,
)
from model_six_dof_multitask import AUVSixDOFMultiTaskDetector  # noqa: E402
from train_six_dof_multitask import (  # noqa: E402
    class_weights,
    evaluate,
    load_and_validate_dataset,
    make_loader,
    set_seed,
)
from utils.six_dof_feature_extractor import (  # noqa: E402
    FAULT_MODE_NAMES,
    THRUSTER_NAMES,
)


LAYER_NAMES = ("raw", "temporal", "operator", "ticket")
LOG_LEVEL_NAMES = {
    0: "normal",
    1: "background_trace",
    2: "observation",
    3: "maintenance_advisory",
    4: "safety_alert",
}
EXCITATION_BINS = (
    (0.00, 0.05, "0.00-0.05"),
    (0.05, 0.15, "0.05-0.15"),
    (0.15, 0.30, "0.15-0.30"),
    (0.30, float("inf"), ">=0.30"),
)


def _as_numpy(values):
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_stats(values):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p90": None,
                "maximum": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "maximum": float(np.max(array)),
    }


def _mission_metadata(dataset, mission_id):
    metadata = dataset["mission_metadata"]
    return metadata.get(int(mission_id), metadata.get(str(int(mission_id)), {}))


def _mission_fields(dataset, mission_id):
    metadata = _mission_metadata(dataset, mission_id)
    scenario = str(metadata.get("scenario", "unknown"))
    parameters = metadata.get("parameters", {})
    mission = parameters.get("mission", {})
    band = mission.get("depth_band_m", [None, None])
    band_label = (
        f"{float(band[0]):g}-{float(band[1]):g} m"
        if len(band) == 2 and band[0] is not None and band[1] is not None
        else "unknown"
    )
    fault_start = parameters.get("fault_start_time_s")
    return {
        "scenario": scenario,
        "mission_kind": (
            "normal_mission" if scenario.strip().lower() == "normal"
            else "pre_fault_segment"
        ),
        "depth_band": band_label,
        "fault_start_time_s": (
            None if fault_start is None else float(fault_start)
        ),
    }


def _context_transition_flags(mission_ids, times, contexts, stable):
    near = np.zeros(len(times), dtype=bool)
    changed = np.zeros(len(times), dtype=bool)
    for mission_id in np.unique(mission_ids):
        positions = np.flatnonzero(mission_ids == mission_id)
        positions = positions[np.argsort(times[positions], kind="stable")]
        last_change_time = -float("inf")
        previous = None
        for position in positions:
            context = int(contexts[position])
            if previous is not None and context != previous:
                changed[position] = True
                last_change_time = float(times[position])
            near[position] = (
                float(times[position]) - last_change_time <= 2.5 + 1e-9
            )
            previous = context
    return changed, near | ~stable


def _bin_label(value, bins):
    value = float(value)
    for lower, upper, label in bins:
        if lower <= value < upper:
            return label
    return bins[-1][2]


def _group_rates(rows, group_key, false_key):
    grouped = {}
    for row in rows:
        key = str(row[group_key])
        record = grouped.setdefault(key, {"normal_windows": 0,
                                          "false_windows": 0})
        record["normal_windows"] += 1
        record["false_windows"] += int(bool(row[false_key]))
    for record in grouped.values():
        record["false_alarm_rate"] = (
            record["false_windows"] / max(record["normal_windows"], 1)
        )
    return dict(sorted(grouped.items()))


def _continuous_comparison(rows, false_key):
    names = (
        "raw_fault_probability",
        "maximum_excitation_ratio",
        "candidate_excitation_ratio",
        "maximum_local_anomaly_score",
        "candidate_local_anomaly_score",
        "maximum_motion_loss_evidence",
        "candidate_motion_loss_evidence",
        "maximum_saturation_fraction",
        "absolute_depth_tracking_error_m",
        "attitude_error_norm_rad",
    )
    false_rows = [row for row in rows if row[false_key]]
    correct_rows = [row for row in rows if not row[false_key]]
    return {
        name: {
            "false": _finite_stats([row[name] for row in false_rows]),
            "correct": _finite_stats([row[name] for row in correct_rows]),
        }
        for name in names
    }


def _mission_summary(split, rows):
    """Summarise normal-labelled exposure and false alarms per mission."""

    by_mission = {}
    for row in rows:
        by_mission.setdefault(int(row["mission_id"]), []).append(row)

    records = []
    for mission_id, mission_rows in sorted(by_mission.items()):
        record = {
            "split": split,
            "mission_id": mission_id,
            "scenario": mission_rows[0]["scenario"],
            "mission_kind": mission_rows[0]["mission_kind"],
            "depth_band": mission_rows[0]["depth_band"],
            "normal_windows": len(mission_rows),
            "maximum_raw_fault_probability": float(max(
                row["raw_fault_probability"] for row in mission_rows
            )),
            "median_raw_fault_probability": float(np.median([
                row["raw_fault_probability"] for row in mission_rows
            ])),
        }
        for layer in LAYER_NAMES:
            false_windows = int(sum(
                row[f"{layer}_false"] for row in mission_rows
            ))
            record[f"{layer}_false_windows"] = false_windows
            record[f"{layer}_false_alarm_rate"] = (
                false_windows / len(mission_rows)
            )
        records.append(record)
    return records


def _episode_records(split, rows, false_key, layer):
    by_mission = {}
    for row in rows:
        by_mission.setdefault(int(row["mission_id"]), []).append(row)
    episodes = []
    for mission_id, mission_rows in sorted(by_mission.items()):
        mission_rows.sort(key=lambda item: item["window_end_time_s"])
        times = np.asarray(
            [row["window_end_time_s"] for row in mission_rows], dtype=float
        )
        if len(times) > 1:
            step = float(np.median(np.diff(times)))
        else:
            step = 1.25
        active = []

        def finish():
            if not active:
                return
            modes = Counter(
                row[f"{layer}_predicted_mode"] for row in active
                if row.get(f"{layer}_predicted_mode") not in (None, "normal")
            )
            locations = Counter(
                row[f"{layer}_predicted_location"] for row in active
                if row.get(f"{layer}_predicted_location") not in (None, "none")
            )
            episodes.append({
                "split": split,
                "layer": layer,
                "mission_id": mission_id,
                "scenario": active[0]["scenario"],
                "mission_kind": active[0]["mission_kind"],
                "depth_band": active[0]["depth_band"],
                "start_time_s": float(active[0]["window_end_time_s"]),
                "end_time_s": float(active[-1]["window_end_time_s"]),
                "active_duration_s": float(len(active) * step),
                "window_count": int(len(active)),
                "dominant_predicted_mode": (
                    modes.most_common(1)[0][0] if modes else "n/a"
                ),
                "dominant_predicted_location": (
                    locations.most_common(1)[0][0] if locations else "n/a"
                ),
                "maximum_fault_probability": float(max(
                    row["raw_fault_probability"] for row in active
                )),
                "maximum_excitation_ratio": float(max(
                    row["maximum_excitation_ratio"] for row in active
                )),
                "maximum_independent_motion_evidence": float(max(
                    row["maximum_motion_loss_evidence"] for row in active
                )),
                "contains_unstable_context": bool(any(
                    not row["guidance_context_stable"] for row in active
                )),
                "near_context_transition": bool(any(
                    row["near_context_transition"] for row in active
                )),
            })

        for row in mission_rows:
            if not row[false_key]:
                finish()
                active = []
                continue
            if active and (
                row["window_end_time_s"] - active[-1]["window_end_time_s"]
                > 1.5 * step + 1e-9
            ):
                finish()
                active = []
            active.append(row)
        finish()
    return episodes


def _layer_summary(rows, episodes, false_key, layer):
    false_rows = [row for row in rows if row[false_key]]
    predicted_modes = Counter(
        row[f"{layer}_predicted_mode"] for row in false_rows
        if row.get(f"{layer}_predicted_mode") is not None
    )
    predicted_locations = Counter(
        row[f"{layer}_predicted_location"] for row in false_rows
        if row.get(f"{layer}_predicted_location") is not None
    )
    durations = [episode["active_duration_s"] for episode in episodes]
    return {
        "normal_windows": int(len(rows)),
        "false_windows": int(len(false_rows)),
        "false_alarm_rate": len(false_rows) / max(len(rows), 1),
        "missions_with_false_windows": int(len({
            row["mission_id"] for row in false_rows
        })),
        "normal_missions_with_false_windows": int(len({
            row["mission_id"] for row in false_rows
            if row["mission_kind"] == "normal_mission"
        })),
        "predicted_mode_counts": dict(sorted(predicted_modes.items())),
        "predicted_location_counts": dict(sorted(predicted_locations.items())),
        "episode_count": int(len(episodes)),
        "episode_duration_s": _finite_stats(durations),
        "episodes_at_least_2_5_s": int(sum(value >= 2.5 for value in durations)),
        "episodes_at_least_5_s": int(sum(value >= 5.0 for value in durations)),
        "episodes_at_least_8_s": int(sum(value >= 8.0 for value in durations)),
        "episodes_near_context_transition": int(sum(
            episode["near_context_transition"] for episode in episodes
        )),
        "by_depth_band": _group_rates(rows, "depth_band", false_key),
        "by_mission_kind": _group_rates(rows, "mission_kind", false_key),
        "by_context_stability": _group_rates(
            rows, "context_stability", false_key
        ),
        "by_context_transition_proximity": _group_rates(
            rows, "context_transition_proximity", false_key
        ),
        "by_excitation": _group_rates(rows, "excitation_bin", false_key),
        "continuous_comparison": _continuous_comparison(rows, false_key),
    }


def _audit_tickets(dataset, split, decisions):
    records = []
    for ticket in decisions["maintenance_tickets"]:
        mission_id = int(ticket["mission_id"])
        mission = _mission_fields(dataset, mission_id)
        fault_start = mission["fault_start_time_s"]
        start = float(ticket["start_time_s"])
        end = float(ticket["end_time_s"])
        expected_mode = None
        metadata = _mission_metadata(dataset, mission_id)
        scenario = str(metadata.get("scenario", "")).lower()
        if "no output" in scenario or "no_output" in scenario:
            expected_mode = "no_output"
        elif "thrust loss" in scenario or "thrust_loss" in scenario:
            expected_mode = "thrust_loss"
        entirely_before_fault = fault_start is not None and end < fault_start
        starts_before_fault = fault_start is not None and start < fault_start
        false_ticket = fault_start is None or entirely_before_fault
        records.append({
            "split": split,
            "mission_id": mission_id,
            "scenario": mission["scenario"],
            "depth_band": mission["depth_band"],
            "ticket_mode": str(ticket["fault_mode"]),
            "expected_mode": expected_mode,
            "start_time_s": start,
            "end_time_s": end,
            "fault_start_time_s": fault_start,
            "detection_delay_s": (
                None if fault_start is None else start - fault_start
            ),
            "false_ticket": bool(false_ticket),
            "starts_before_fault": bool(starts_before_fault),
            "mode_matches": bool(
                expected_mode is None or ticket["fault_mode"] == expected_mode
            ),
            "trigger": str(ticket.get("trigger", "")),
            "maximum_excitation_ratio": float(
                ticket.get("maximum_excitation_ratio", 0.0)
            ),
            "maximum_independent_evidence": float(
                ticket.get("maximum_independent_evidence", 0.0)
            ),
        })
    return records


def _build_split_audit(
    dataset,
    split,
    indices,
    predictions,
    temporal_config,
    ticket_config,
):
    indices = np.asarray(indices, dtype=np.int64)
    temporal = apply_temporal_decision_layer(
        dataset, indices, predictions, temporal_config
    )
    maintenance = apply_maintenance_decision_layer(
        dataset, indices, predictions, temporal_config
    )
    evidence = extract_maintenance_ticket_evidence(
        dataset, indices, ticket_config
    )
    maintenance = apply_maintenance_ticket_policy(
        dataset,
        indices,
        maintenance,
        ticket_config,
        ticket_evidence=evidence,
    )
    maintenance = apply_maintenance_log_policy(
        dataset, indices, maintenance, MaintenanceLogConfig()
    )

    mission_ids = _as_numpy(dataset["mission_ids"])[indices].astype(int)
    times = _as_numpy(dataset["window_end_times"])[indices].astype(float)
    true_modes = np.asarray(predictions["mode_true"], dtype=int)
    raw_modes = np.asarray(predictions["mode_pred"], dtype=int)
    raw_locations = np.asarray(predictions["location_pred"], dtype=int)
    temporal_modes = np.asarray(temporal["mode_pred"], dtype=int)
    temporal_locations = np.asarray(temporal["location_pred"], dtype=int)
    probabilities = np.asarray(predictions["mode_probabilities"], dtype=float)
    contexts = _as_numpy(dataset["guidance_context_ids"])[indices].astype(int)
    stable = _as_numpy(dataset["guidance_context_stable"])[indices].astype(bool)
    context_changed, near_transition = _context_transition_flags(
        mission_ids, times, contexts, stable
    )
    log_levels = np.asarray(
        maintenance["maintenance_log_level_pred"], dtype=int
    )
    ticket_active = np.asarray(
        maintenance["maintenance_ticket_active"], dtype=bool
    )
    ticket_modes = np.asarray(
        maintenance["maintenance_ticket_mode"], dtype=int
    )

    windows = _as_numpy(dataset["X"])[indices]
    feature_names = list(dataset["feature_names"])
    tail = windows[:, -20:]
    depth_index = feature_names.index("depth_m")
    depth_error_index = feature_names.index("depth_tracking_error_m")
    attitude_indices = [
        feature_names.index(f"attitude_error_{axis}_rad")
        for axis in ("roll", "pitch", "yaw")
    ]
    current_depth = tail[:, -1, depth_index]
    absolute_depth_error = np.mean(
        np.abs(tail[:, :, depth_error_index]), axis=1
    )
    attitude_error_norm = np.mean(
        np.linalg.norm(tail[:, :, attitude_indices], axis=2), axis=1
    )
    maximum_excitation = np.max(evidence["excitation_ratios"], axis=1)
    maximum_local = np.max(evidence["local_anomaly_scores"], axis=1)
    maximum_motion = np.max(evidence["motion_loss_evidence"], axis=1)
    maximum_saturation = np.max(evidence["saturation_fraction"], axis=1)

    rows = []
    for position in range(len(indices)):
        if true_modes[position] != 0:
            continue
        mission_id = int(mission_ids[position])
        mission = _mission_fields(dataset, mission_id)
        raw_location = int(raw_locations[position])
        temporal_location = int(temporal_locations[position])
        candidate = (
            raw_location - 1
            if raw_location > 0
            else int(np.argmax(predictions["location_probabilities"][position]))
        )
        row = {
            "split": split,
            "dataset_index": int(indices[position]),
            "mission_id": mission_id,
            "scenario": mission["scenario"],
            "mission_kind": mission["mission_kind"],
            "depth_band": mission["depth_band"],
            "window_end_time_s": float(times[position]),
            "current_depth_m": float(current_depth[position]),
            "guidance_context_id": int(contexts[position]),
            "guidance_context_stable": bool(stable[position]),
            "context_changed_here": bool(context_changed[position]),
            "near_context_transition": bool(near_transition[position]),
            "context_stability": (
                "stable" if stable[position] else "unstable"
            ),
            "context_transition_proximity": (
                "within_2.5_s" if near_transition[position]
                else "outside_2.5_s"
            ),
            "raw_predicted_mode": FAULT_MODE_NAMES[int(raw_modes[position])],
            "raw_predicted_location": (
                "none" if raw_location == 0
                else THRUSTER_NAMES[raw_location - 1]
            ),
            "temporal_predicted_mode": (
                FAULT_MODE_NAMES[int(temporal_modes[position])]
            ),
            "temporal_predicted_location": (
                "none" if temporal_location == 0
                else THRUSTER_NAMES[temporal_location - 1]
            ),
            "operator_predicted_mode": (
                FAULT_MODE_NAMES[int(maintenance["mode_pred"][position])]
                if log_levels[position] >= 3 else "normal"
            ),
            "operator_predicted_location": (
                "none" if log_levels[position] < 3
                else THRUSTER_NAMES[int(
                    np.argmax(maintenance[
                        "smoothed_location_probabilities"
                    ][position])
                )]
            ),
            "ticket_predicted_mode": (
                FAULT_MODE_NAMES[int(ticket_modes[position])]
                if ticket_active[position] else "normal"
            ),
            "ticket_predicted_location": (
                "none" if not ticket_active[position]
                else THRUSTER_NAMES[int(np.argmax(
                    maintenance["smoothed_location_probabilities"][position]
                ))]
            ),
            "raw_fault_probability": float(1.0 - probabilities[position, 0]),
            "raw_no_output_probability": float(probabilities[position, 1]),
            "raw_thrust_loss_probability": float(probabilities[position, 2]),
            "raw_false": bool(raw_modes[position] != 0),
            "temporal_false": bool(temporal_modes[position] != 0),
            "operator_false": bool(log_levels[position] >= 3),
            "ticket_false": bool(ticket_active[position]),
            "maintenance_health_level": int(
                maintenance["health_level_pred"][position]
            ),
            "maintenance_log_level": LOG_LEVEL_NAMES[int(log_levels[position])],
            "maximum_excitation_ratio": float(maximum_excitation[position]),
            "candidate_excitation_ratio": float(
                evidence["excitation_ratios"][position, candidate]
            ),
            "excitation_bin": _bin_label(
                maximum_excitation[position], EXCITATION_BINS
            ),
            "maximum_local_anomaly_score": float(maximum_local[position]),
            "candidate_local_anomaly_score": float(
                evidence["local_anomaly_scores"][position, candidate]
            ),
            "maximum_motion_loss_evidence": float(maximum_motion[position]),
            "candidate_motion_loss_evidence": float(
                evidence["motion_loss_evidence"][position, candidate]
            ),
            "maximum_saturation_fraction": float(
                maximum_saturation[position]
            ),
            "absolute_depth_tracking_error_m": float(
                absolute_depth_error[position]
            ),
            "attitude_error_norm_rad": float(attitude_error_norm[position]),
        }
        rows.append(row)

    episodes = []
    summaries = {}
    false_keys = {
        "raw": "raw_false",
        "temporal": "temporal_false",
        "operator": "operator_false",
        "ticket": "ticket_false",
    }
    for layer, false_key in false_keys.items():
        layer_episodes = _episode_records(split, rows, false_key, layer)
        episodes.extend(layer_episodes)
        summaries[layer] = _layer_summary(
            rows, layer_episodes, false_key, layer
        )

    ticket_records = _audit_tickets(dataset, split, maintenance)
    summaries["formal_ticket_audit"] = {
        "ticket_count": len(ticket_records),
        "false_ticket_count": int(sum(
            record["false_ticket"] for record in ticket_records
        )),
        "tickets_starting_before_fault": int(sum(
            record["starts_before_fault"] for record in ticket_records
        )),
        "mode_mismatch_count": int(sum(
            not record["mode_matches"] for record in ticket_records
        )),
        "ticket_metrics": maintenance_ticket_metrics(
            dataset, indices, maintenance
        ),
        "graded_log_metrics": maintenance_log_metrics(
            dataset, indices, maintenance
        ),
    }
    return {
        "summary": summaries,
        "normal_window_rows": rows,
        "mission_summary": _mission_summary(split, rows),
        "episodes": episodes,
        "tickets": ticket_records,
    }


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            THIS_DIR / "data" /
            "simulation_dataset_six_dof_hybrid_telemetry_0_300m_focus_v3.pth"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            THIS_DIR / "results" /
            "six_dof_hybrid_telemetry_0_300m_focus_v3" / "best_model.pth"
        ),
    )
    parser.add_argument(
        "--temporal-config",
        type=Path,
        default=(
            THIS_DIR / "results" /
            "six_dof_hybrid_telemetry_0_300m_focus_v3_temporal" /
            "temporal_decision_config.json"
        ),
    )
    parser.add_argument(
        "--ticket-config",
        type=Path,
        default=(
            THIS_DIR / "results" /
            "six_dof_hybrid_telemetry_0_300m_focus_v3_temporal" /
            "maintenance_ticket_config.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "math-model" / "results" /
            "six_dof_false_alarm_audit_v1_20260821"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, split_indices, split_missions = load_and_validate_dataset(
        args.dataset
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model = AUVSixDOFMultiTaskDetector(
        input_dim=int(checkpoint["input_dim"]),
        structured_fusion=bool(checkpoint.get("structured_fusion", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_indices = split_indices["train"]
    train_modes = dataset["y_mode"][train_indices]
    train_locations = dataset["y_location"][train_indices]
    mode_weights, _ = class_weights(train_modes, len(FAULT_MODE_NAMES))
    location_weights, _ = class_weights(
        train_locations[train_modes != 0] - 1, len(THRUSTER_NAMES)
    )
    mode_loss = torch.nn.CrossEntropyLoss(weight=mode_weights.to(device))
    location_loss = torch.nn.CrossEntropyLoss(
        weight=location_weights.to(device)
    )
    temporal_config = TemporalDecisionConfig(**_json(args.temporal_config))
    ticket_config = MaintenanceTicketConfig(**_json(args.ticket_config))

    audits = {}
    for split in ("validation", "test"):
        loader = make_loader(
            dataset,
            split_indices[split],
            checkpoint["mean"],
            checkpoint["std"],
            args.batch_size,
            shuffle=False,
            seed=args.seed,
        )
        predictions = evaluate(
            model, loader, device, mode_loss, location_loss
        )
        audits[split] = _build_split_audit(
            dataset,
            split,
            split_indices[split].cpu().numpy(),
            predictions,
            temporal_config,
            ticket_config,
        )

    expected = {
        "validation": {"raw": 0.08702408702408702,
                       "temporal": 0.07692307692307693},
        "test": {"raw": 0.1261127596439169,
                 "temporal": 0.12388724035608309},
    }
    for split, rates in expected.items():
        for layer, expected_rate in rates.items():
            observed = audits[split]["summary"][layer]["false_alarm_rate"]
            if not np.isclose(observed, expected_rate, atol=1e-12):
                raise RuntimeError(
                    f"{split} {layer} FAR mismatch: {observed} != "
                    f"{expected_rate}"
                )

    result = {
        "audit_id": "six_dof_false_alarm_audit_v1_20260821",
        "analysis_policy": (
            "descriptive audit only; no threshold, model, or test-set tuning"
        ),
        "device": str(device),
        "dataset_version": dataset["dataset_version"],
        "dataset_sha256": _sha256(args.dataset),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "temporal_config": asdict(temporal_config),
        "ticket_config": asdict(ticket_config),
        "split_missions": {
            name: len(values) for name, values in split_missions.items()
        },
        "validation": audits["validation"]["summary"],
        "test": audits["test"]["summary"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "false_alarm_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    false_rows = [
        row
        for split in ("validation", "test")
        for row in audits[split]["normal_window_rows"]
        if any(row[f"{layer}_false"] for layer in LAYER_NAMES)
    ]
    _write_csv(args.output_dir / "false_alarm_windows.csv", false_rows)
    _write_csv(
        args.output_dir / "false_alarm_mission_summary.csv",
        [
            row
            for split in ("validation", "test")
            for row in audits[split]["mission_summary"]
        ],
    )
    _write_csv(
        args.output_dir / "false_alarm_episodes.csv",
        [
            episode
            for split in ("validation", "test")
            for episode in audits[split]["episodes"]
        ],
    )
    _write_csv(
        args.output_dir / "formal_ticket_audit.csv",
        [
            ticket
            for split in ("validation", "test")
            for ticket in audits[split]["tickets"]
        ],
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "validation": {
            layer: audits["validation"]["summary"][layer][
                "false_alarm_rate"
            ] for layer in LAYER_NAMES
        },
        "test": {
            layer: audits["test"]["summary"][layer]["false_alarm_rate"]
            for layer in LAYER_NAMES
        },
        "test_false_tickets": audits["test"]["summary"][
            "formal_ticket_audit"
        ]["false_ticket_count"],
        "test_tickets_starting_before_fault": audits["test"]["summary"][
            "formal_ticket_audit"
        ]["tickets_starting_before_fault"],
    }, indent=2))


if __name__ == "__main__":
    main()
