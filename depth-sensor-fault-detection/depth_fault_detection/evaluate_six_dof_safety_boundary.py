"""Find the no-intervention boundary for recovered disturbances and DVL loss."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
MATH_MODEL_SRC = REPO_ROOT / "math-model" / "src"
if str(MATH_MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MATH_MODEL_SRC))

from diagnosis.maintenance_health_decision import (
    apply_maintenance_decision_layer,
)
from diagnosis.maintenance_ticket_policy import (
    MaintenanceTicketConfig,
    apply_maintenance_ticket_policy,
)
from diagnosis.temporal_fault_decision import TemporalDecisionConfig
from evaluation.transient_recovery import (
    TimedDVLDropoutScenario,
    boundary_transient_scenarios,
    dvl_dropout_boundary_scenarios,
    summarize_transient_recovery,
)

from evaluate_six_dof_transient_recovery import (
    _build_dataset,
    _load_json_config,
    _model_predictions,
    run_transient_mission,
)
from train_six_dof_multitask import set_seed


def _intensity_label(name):
    for label in ("weak", "medium", "strong"):
        if f"_{label}_" in name:
            return label
    return None


def _dvl_recovery(logs, scenario, final_window_s=5.0):
    times = np.asarray([log["time"] for log in logs], dtype=float)
    valid = np.asarray([bool(log["dvl"]["valid"]) for log in logs])
    reasons = np.asarray([
        log["dvl"].get("dropout_reason") for log in logs
    ], dtype=object)
    event = (
        (times >= scenario.start_time_s)
        & (times < scenario.end_time_s)
    )
    final = times >= times[-1] - float(final_window_s)
    scheduled_samples = int(np.sum(event & (reasons == "scheduled")))
    final_valid_rate = float(np.mean(valid[final]))
    return {
        "response_observed": scheduled_samples > 0,
        "recovered": bool(
            scheduled_samples > 0
            and not np.any((reasons[final] == "scheduled"))
            and final_valid_rate >= 0.90
        ),
        "scheduled_dropout_samples": scheduled_samples,
        "final_dvl_valid_rate": final_valid_rate,
        "peak_position_error_m": float(max(
            np.linalg.norm(log["position_error_ned"]) for log in logs
        )),
        "final_position_error_m": float(np.median([
            np.linalg.norm(logs[index]["position_error_ned"])
            for index in np.flatnonzero(final)
        ])),
    }


def _mission_rows(missions, dataset, decisions):
    mission_ids = dataset["mission_ids"].cpu().numpy()
    health = np.asarray(decisions["health_level_pred"], dtype=np.int64)
    tickets_by_mission = Counter(
        int(ticket["mission_id"])
        for ticket in decisions["maintenance_tickets"]
    )
    events_by_mission = Counter(
        int(event["mission_id"])
        for event in decisions["maintenance_events"]
    )
    rows = []
    for mission_id, item in missions.items():
        scenario = item["scenario"]
        if isinstance(scenario, TimedDVLDropoutScenario):
            recovery = _dvl_recovery(item["logs"], scenario)
        else:
            recovery = summarize_transient_recovery(
                item["logs"], scenario
            )
        logs = item["logs"]
        actions = Counter(log["ftc_action"] for log in logs)
        intervention_steps = int(sum(
            bool(log["ftc_intervention_requested"]) for log in logs
        ))
        degraded_steps = int(actions["degraded_operation"])
        abort_or_ascent_steps = int(
            actions["safe_hold_or_abort"]
            + actions["controlled_ascent"]
        )
        targeted_reallocation_steps = int(
            actions["targeted_reallocation"]
        )
        dangerous_ftc_steps = (
            abort_or_ascent_steps + targeted_reallocation_steps
        )
        isolated = sorted({
            log["ftc_targeted_thruster_name"]
            for log in logs
            if log["ftc_targeted_thruster_name"] is not None
        })
        ticket_count = int(tickets_by_mission[mission_id])
        event_count = int(events_by_mission[mission_id])
        positions = mission_ids == mission_id
        recovered = bool(recovery["recovered"])
        if not recovered:
            outcome = "unrecovered"
        elif abort_or_ascent_steps:
            outcome = "ftc_abort_or_ascent"
        elif targeted_reallocation_steps or isolated:
            outcome = "targeted_reallocation"
        elif degraded_steps:
            outcome = "degraded_observation"
        elif ticket_count:
            outcome = "maintenance_ticket"
        elif event_count:
            outcome = "log_only_recovered"
        else:
            outcome = "normal_recovered"
        safe = bool(
            recovered and ticket_count == 0
            and intervention_steps == 0 and not isolated
        )
        observation_safe = bool(
            recovered
            and ticket_count == 0
            and dangerous_ftc_steps == 0
            and not isolated
        )
        rows.append({
            "mission_id": int(mission_id),
            "scenario": scenario.name,
            "kind": item["kind"],
            "duration_s": scenario.duration_s,
            "intensity": item["intensity"],
            "seed": int(item["seed"]),
            **recovery,
            "maximum_health_level": int(np.max(health[positions])),
            "raw_health_event_count": event_count,
            "formal_maintenance_ticket_count": ticket_count,
            "ftc_action_counts": dict(sorted(actions.items())),
            "ftc_intervention_steps": intervention_steps,
            "ftc_degraded_steps": degraded_steps,
            "ftc_abort_or_ascent_steps": abort_or_ascent_steps,
            "ftc_targeted_reallocation_steps": (
                targeted_reallocation_steps
            ),
            "ftc_targeted_thrusters": isolated,
            "outcome": outcome,
            "safe_no_intervention": safe,
            "safe_observation": observation_safe,
        })
    return rows


def _setting_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(
            row["kind"], row["intensity"], row["duration_s"]
        )].append(row)
    summaries = []
    for (kind, intensity, duration), group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            "" if item[0][1] is None else item[0][1],
            item[0][2],
        ),
    ):
        count = len(group)
        summaries.append({
            "kind": kind,
            "intensity": intensity,
            "duration_s": float(duration),
            "missions": count,
            "safe_no_intervention_rate": float(np.mean([
                row["safe_no_intervention"] for row in group
            ])),
            "safe_observation_rate": float(np.mean([
                row["safe_observation"] for row in group
            ])),
            "recovery_rate": float(np.mean([
                row["recovered"] for row in group
            ])),
            "raw_log_mission_rate": float(np.mean([
                row["raw_health_event_count"] > 0 for row in group
            ])),
            "maintenance_ticket_mission_rate": float(np.mean([
                row["formal_maintenance_ticket_count"] > 0
                for row in group
            ])),
            "ftc_intervention_mission_rate": float(np.mean([
                row["ftc_intervention_steps"] > 0 for row in group
            ])),
            "degraded_observation_mission_rate": float(np.mean([
                row["ftc_degraded_steps"] > 0 for row in group
            ])),
            "abort_or_ascent_mission_rate": float(np.mean([
                row["ftc_abort_or_ascent_steps"] > 0 for row in group
            ])),
            "targeted_reallocation_mission_rate": float(np.mean([
                row["ftc_targeted_reallocation_steps"] > 0
                for row in group
            ])),
            "maximum_peak_position_error_m": float(max(
                row["peak_position_error_m"] for row in group
            )),
            "outcome_counts": dict(sorted(Counter(
                row["outcome"] for row in group
            ).items())),
        })
    return summaries


def _derived_boundaries(settings):
    intensity_order = ("weak", "medium", "strong")
    disturbance = [
        row for row in settings if row["kind"] == "body_wrench"
    ]
    by_intensity = {}
    for intensity in intensity_order:
        candidates = sorted(
            (row for row in disturbance if row["intensity"] == intensity),
            key=lambda row: row["duration_s"],
        )
        safe = [
            row["duration_s"] for row in candidates
            if row["safe_no_intervention_rate"] == 1.0
        ]
        observation_safe = [
            row["duration_s"] for row in candidates
            if row["safe_observation_rate"] == 1.0
        ]
        action = [
            row for row in candidates
            if row["safe_no_intervention_rate"] < 1.0
        ]
        by_intensity[intensity] = {
            "maximum_fully_safe_tested_duration_s": (
                max(safe) if safe else None
            ),
            "maximum_safe_observation_tested_duration_s": (
                max(observation_safe) if observation_safe else None
            ),
            "first_non_safe_tested_duration_s": (
                action[0]["duration_s"] if action else None
            ),
            "first_non_safe_outcomes": (
                action[0]["outcome_counts"] if action else {}
            ),
        }
    dvl = sorted(
        (row for row in settings if row["kind"] == "dvl_dropout"),
        key=lambda row: row["duration_s"],
    )
    dvl_safe = [
        row["duration_s"] for row in dvl
        if row["safe_no_intervention_rate"] == 1.0
    ]
    dvl_observation_safe = [
        row["duration_s"] for row in dvl
        if row["safe_observation_rate"] == 1.0
    ]
    return {
        "body_wrench_by_intensity": by_intensity,
        "maximum_fully_safe_tested_dvl_dropout_s": (
            max(dvl_safe) if dvl_safe else None
        ),
        "maximum_safe_observation_tested_dvl_dropout_s": (
            max(dvl_observation_safe) if dvl_observation_safe else None
        ),
        "interpretation": (
            "fully safe means recovered with no formal ticket or FTC action; "
            "safe observation additionally permits degraded operation but "
            "still forbids tickets, abort/ascent, and thruster isolation; "
            "boundaries are discrete tested durations, not exact continuous "
            "thresholds"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--duration", type=float, default=50.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=18421)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            THIS_DIR / "results" / "six_dof_hybrid_telemetry"
            / "best_model.pth"
        ),
    )
    parser.add_argument(
        "--temporal-config",
        type=Path,
        default=(
            THIS_DIR / "results" / "six_dof_hybrid_telemetry_temporal"
            / "temporal_decision_config.json"
        ),
    )
    parser.add_argument(
        "--ticket-config",
        type=Path,
        default=(
            THIS_DIR / "results" / "six_dof_hybrid_telemetry_temporal"
            / "maintenance_ticket_config.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_DIR / "results" / "six_dof_safety_boundary",
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.duration <= 30.0 or args.dt <= 0.0:
        parser.error("--duration must exceed 30 s and --dt must be positive")

    set_seed(args.seed)
    definitions = [
        (scenario, "body_wrench", _intensity_label(scenario.name))
        for scenario in boundary_transient_scenarios()
    ] + [
        (scenario, "dvl_dropout", None)
        for scenario in dvl_dropout_boundary_scenarios()
    ]
    missions = {}
    mission_id = 0
    for scenario_index, (scenario, kind, intensity) in enumerate(definitions):
        for repeat in range(args.repeats):
            seed = args.seed + scenario_index * 1000 + repeat
            if kind == "dvl_dropout":
                logs, ambient = run_transient_mission(
                    None,
                    args.duration,
                    args.dt,
                    seed,
                    dvl_dropout_windows=(scenario.dropout_window,),
                )
            else:
                logs, ambient = run_transient_mission(
                    scenario, args.duration, args.dt, seed
                )
            missions[mission_id] = {
                "scenario": scenario,
                "kind": kind,
                "intensity": intensity,
                "seed": seed,
                "logs": logs,
                "ambient": ambient,
            }
            print(
                f"Mission {mission_id + 1:02d}/"
                f"{len(definitions) * args.repeats}: "
                f"{scenario.name}, repeat={repeat + 1}"
            )
            mission_id += 1

    dataset = _build_dataset(missions, args.seq_len, args.stride)
    if np.any(dataset["y_mode"].cpu().numpy() != 0):
        raise RuntimeError("safety boundary cases must remain fault-free")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices, predictions = _model_predictions(
        dataset, args.checkpoint, args.batch_size, device
    )
    temporal_config = _load_json_config(
        args.temporal_config, TemporalDecisionConfig
    )
    ticket_config = _load_json_config(
        args.ticket_config, MaintenanceTicketConfig
    )
    decisions = apply_maintenance_decision_layer(
        dataset, indices, predictions, temporal_config
    )
    decisions = apply_maintenance_ticket_policy(
        dataset, indices, decisions, ticket_config
    )
    mission_rows = _mission_rows(missions, dataset, decisions)
    settings = _setting_summary(mission_rows)
    summary = {
        "benchmark": "six_dof_safety_boundary_v1",
        "device": str(device),
        "matrix": {
            "body_wrench_durations_s": [1.0, 2.0, 4.0],
            "body_wrench_intensities": ["weak", "medium", "strong"],
            "dvl_dropout_durations_s": [1.0, 2.0, 4.0],
            "repeats_per_setting": args.repeats,
        },
        "mission_count": len(mission_rows),
        "window_count": int(len(dataset["X"])),
        "setting_summary": settings,
        "derived_boundaries": _derived_boundaries(settings),
        "outcome_counts": dict(sorted(Counter(
            row["outcome"] for row in mission_rows
        ).items())),
        "mission_results": mission_rows,
        "raw_health_events": decisions["maintenance_events"],
        "formal_maintenance_tickets": decisions["maintenance_tickets"],
        "pending_maintenance_observations": decisions[
            "maintenance_pending_observations"
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "safety_boundary_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    concise = {
        key: summary[key]
        for key in (
            "benchmark",
            "matrix",
            "mission_count",
            "window_count",
            "setting_summary",
            "derived_boundaries",
            "outcome_counts",
        )
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
