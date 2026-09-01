"""Evaluate paired physical-factor counterfactuals for normal missions.

The runner supports lateral disturbance, vertical-thruster force margin, and
vehicle mass.  Every baseline/candidate pair shares the same simulator random
streams and all non-target parameters.  The frozen V3 model and rules are used
without calibration or retraining.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
MATH_MODEL_ROOT = REPO_ROOT / "math-model"
MATH_MODEL_SRC = MATH_MODEL_ROOT / "src"
MATH_MODEL_EXAMPLES = MATH_MODEL_ROOT / "examples"
for path in (THIS_DIR, MATH_MODEL_SRC, MATH_MODEL_EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_six_dof_false_alarms import _build_split_audit  # noqa: E402
from diagnosis.maintenance_ticket_policy import (  # noqa: E402
    MaintenanceTicketConfig,
)
from diagnosis.temporal_fault_decision import (  # noqa: E402
    TemporalDecisionConfig,
)
from evaluate_six_dof_rpm_noise_counterfactual import (  # noqa: E402
    _concatenate,
    _condition_indices,
    _condition_result,
    _read_json,
    _sha256,
    _write_csv,
)
from generate_six_dof_fault_dataset import run_mission  # noqa: E402
from model_six_dof_multitask import (  # noqa: E402
    AUVSixDOFMultiTaskDetector,
)
from train_six_dof_multitask import (  # noqa: E402
    evaluate,
    make_loader,
    set_seed,
)
from utils.six_dof_dataset_builder import (  # noqa: E402
    build_six_dof_sequence_dataset,
)


FACTOR_SPECS = {
    "lateral_disturbance": {
        "baseline_condition": "low_lateral_disturbance",
        "candidate_condition": "high_lateral_disturbance",
        "baseline_value": 0.25,
        "candidate_value": 1.40,
        "value_name": "lateral_force_amplitude_n",
        "base_seed": 940_000,
    },
    "vertical_force": {
        "baseline_condition": "high_vertical_force_margin",
        "candidate_condition": "low_vertical_force_margin",
        "baseline_value": 45.0,
        "candidate_value": 34.5,
        "value_name": "vertical_force_limit_n",
        "base_seed": 950_000,
    },
    "mass": {
        "baseline_condition": "low_vehicle_mass",
        "candidate_condition": "high_vehicle_mass",
        "baseline_value": 45.0,
        "candidate_value": 58.0,
        "value_name": "vehicle_mass_kg",
        "base_seed": 960_000,
    },
}


def _comparable_metadata(metadata, factor):
    comparable = json.loads(json.dumps(metadata, sort_keys=True))
    if factor == "lateral_disturbance":
        amplitudes = comparable["disturbance"]["amplitudes"]
        comparable["disturbance"]["amplitudes"] = amplitudes[2:]
    elif factor == "vertical_force":
        comparable["thrusters"].pop("vertical_force_limit_n")
    elif factor == "mass":
        comparable["dynamics"].pop("mass_kg")
    else:
        raise ValueError(f"unsupported factor {factor}")
    return comparable


def _mission_overrides(factor, value):
    overrides = {
        "rpm_noise_std_override": 40.0,
        "lateral_force_amplitudes_override": (0.25, 0.25),
        "vertical_force_limit_override": 45.0,
    }
    if factor == "lateral_disturbance":
        overrides["lateral_force_amplitudes_override"] = (value, value)
    elif factor == "vertical_force":
        overrides["vertical_force_limit_override"] = value
    elif factor == "mass":
        overrides["mass_kg_override"] = value
    return overrides


def _generate_dataset(args, spec):
    chunks = []
    mission_metadata = {}
    pair_integrity = []
    mission_id = 0
    conditions = (
        (spec["baseline_condition"], spec["baseline_value"]),
        (spec["candidate_condition"], spec["candidate_value"]),
    )
    for band_index in range(3):
        for repetition in range(args.repeats_per_depth_band):
            pair_id = f"band{band_index}_rep{repetition}"
            environment_seed = (
                args.base_seed + band_index * 10_000 + repetition
            )
            fault_seed = environment_seed + 5_000_000
            pair_parameters = {}
            for condition, value in conditions:
                logs, parameters = run_mission(
                    thruster_name=None,
                    fault_mode=None,
                    duration=args.duration,
                    dt=args.dt,
                    seed=environment_seed,
                    split="counterfactual_test",
                    depth_band_index=band_index,
                    fault_seed=fault_seed,
                    **_mission_overrides(args.factor, value),
                )
                chunk = build_six_dof_sequence_dataset(
                    {mission_id: logs},
                    seq_len=args.seq_len,
                    stride=args.stride,
                )
                chunks.append(chunk)
                pair_parameters[condition] = parameters
                mission_metadata[mission_id] = {
                    "scenario": "Normal",
                    "seed": environment_seed,
                    "split": "counterfactual_test",
                    "condition": condition,
                    "pair_id": pair_id,
                    "parameters": parameters,
                }
                print(
                    f"Mission {mission_id + 1:03d}: {pair_id}, "
                    f"{condition}, windows={len(chunk['X'])}"
                )
                mission_id += 1

            baseline = pair_parameters[spec["baseline_condition"]]
            candidate = pair_parameters[spec["candidate_condition"]]
            integrity_ok = _comparable_metadata(
                baseline, args.factor
            ) == _comparable_metadata(candidate, args.factor)
            if not integrity_ok:
                raise RuntimeError(
                    f"counterfactual pair {pair_id} differs outside "
                    f"{args.factor}"
                )
            pair_integrity.append({
                "pair_id": pair_id,
                "environment_seed": environment_seed,
                "depth_band_index": band_index,
                "parameters_equal_except_target_factor": True,
            })

    dataset = _concatenate(chunks, mission_metadata)
    dataset["dataset_version"] = (
        f"six_dof_{args.factor}_counterfactual_v1"
    )
    return dataset, pair_integrity


def _paired_rows(dataset, condition_results, spec):
    rates = {}
    for condition, result in condition_results.items():
        for mission in result["missions"]:
            metadata = dataset["mission_metadata"][mission["mission_id"]]
            rates[(metadata["pair_id"], condition)] = mission

    rows = []
    for pair_id in sorted({key[0] for key in rates}):
        baseline = rates[(pair_id, spec["baseline_condition"])]
        candidate = rates[(pair_id, spec["candidate_condition"])]
        metadata = next(
            item for item in dataset["mission_metadata"].values()
            if item["pair_id"] == pair_id
        )
        mission = metadata["parameters"]["mission"]
        rows.append({
            "pair_id": pair_id,
            "environment_seed": int(metadata["seed"]),
            "depth_band_index": int(mission["depth_band_index"]),
            "depth_band_m": "-".join(
                f"{value:g}" for value in mission["depth_band_m"]
            ),
            "baseline_raw_false_alarm_rate": baseline[
                "raw_false_alarm_rate"
            ],
            "candidate_raw_false_alarm_rate": candidate[
                "raw_false_alarm_rate"
            ],
            "delta_raw_false_alarm_rate": (
                candidate["raw_false_alarm_rate"]
                - baseline["raw_false_alarm_rate"]
            ),
            "baseline_temporal_false_alarm_rate": baseline[
                "temporal_false_alarm_rate"
            ],
            "candidate_temporal_false_alarm_rate": candidate[
                "temporal_false_alarm_rate"
            ],
            "delta_temporal_false_alarm_rate": (
                candidate["temporal_false_alarm_rate"]
                - baseline["temporal_false_alarm_rate"]
            ),
        })
    return rows


def _paired_summary(rows, minimum_mean_delta):
    raw_deltas = [row["delta_raw_false_alarm_rate"] for row in rows]
    temporal_deltas = [
        row["delta_temporal_false_alarm_rate"] for row in rows
    ]
    required_positive = math.ceil(2.0 * len(rows) / 3.0)
    positive_raw = sum(value > 0.0 for value in raw_deltas)
    mean_raw = statistics.mean(raw_deltas)
    return {
        "pair_count": len(rows),
        "raw_delta_mean": mean_raw,
        "raw_delta_median": statistics.median(raw_deltas),
        "raw_delta_sample_standard_deviation": (
            statistics.stdev(raw_deltas) if len(raw_deltas) > 1 else 0.0
        ),
        "raw_positive_pair_count": positive_raw,
        "temporal_delta_mean": statistics.mean(temporal_deltas),
        "temporal_delta_median": statistics.median(temporal_deltas),
        "temporal_positive_pair_count": sum(
            value > 0.0 for value in temporal_deltas
        ),
        "predeclared_support_rule": {
            "minimum_mean_raw_far_increase": minimum_mean_delta,
            "minimum_positive_pair_count": required_positive,
        },
        "supports_factor_as_causal_false_alarm_factor": bool(
            mean_raw >= minimum_mean_delta
            and positive_raw >= required_positive
        ),
    }


def _depth_band_summary(rows):
    summary = {}
    for band_index in range(3):
        selected = [
            row for row in rows
            if row["depth_band_index"] == band_index
        ]
        summary[str(band_index)] = {
            "depth_band_m": selected[0]["depth_band_m"],
            "pair_count": len(selected),
            "baseline_raw_false_alarm_rate_mean": statistics.mean(
                row["baseline_raw_false_alarm_rate"] for row in selected
            ),
            "candidate_raw_false_alarm_rate_mean": statistics.mean(
                row["candidate_raw_false_alarm_rate"] for row in selected
            ),
            "raw_delta_mean": statistics.mean(
                row["delta_raw_false_alarm_rate"] for row in selected
            ),
            "temporal_delta_mean": statistics.mean(
                row["delta_temporal_false_alarm_rate"] for row in selected
            ),
            "raw_positive_pair_count": sum(
                row["delta_raw_false_alarm_rate"] > 0.0
                for row in selected
            ),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", choices=sorted(FACTOR_SPECS), required=True)
    parser.add_argument("--repeats-per-depth-band", type=int, default=4)
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--minimum-mean-delta", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    spec = FACTOR_SPECS[args.factor]
    if args.base_seed is None:
        args.base_seed = spec["base_seed"]
    if args.output_dir is None:
        args.output_dir = (
            REPO_ROOT / "math-model" / "results" /
            f"six_dof_{args.factor}_counterfactual_v1_20260901"
        )
    if args.repeats_per_depth_band <= 0:
        parser.error("--repeats-per-depth-band must be positive")
    if args.duration <= 0.0 or args.dt <= 0.0:
        parser.error("--duration and --dt must be positive")

    set_seed(args.seed)
    dataset, pair_integrity = _generate_dataset(args, spec)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model = AUVSixDOFMultiTaskDetector(
        input_dim=int(checkpoint["input_dim"]),
        structured_fusion=bool(checkpoint.get("structured_fusion", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    mode_loss = torch.nn.CrossEntropyLoss()
    location_loss = torch.nn.CrossEntropyLoss()
    temporal_config = TemporalDecisionConfig(**_read_json(
        args.temporal_config
    ))
    ticket_config = MaintenanceTicketConfig(**_read_json(
        args.ticket_config
    ))

    condition_results = {}
    conditions = (
        spec["baseline_condition"],
        spec["candidate_condition"],
    )
    for condition in conditions:
        indices = _condition_indices(dataset, condition)
        predictions = evaluate(
            model,
            make_loader(
                dataset,
                indices,
                checkpoint["mean"],
                checkpoint["std"],
                args.batch_size,
                shuffle=False,
                seed=args.seed,
            ),
            device,
            mode_loss,
            location_loss,
        )
        audit = _build_split_audit(
            dataset,
            condition,
            indices,
            predictions,
            temporal_config,
            ticket_config,
        )
        condition_results[condition] = _condition_result(audit)

    paired_rows = _paired_rows(dataset, condition_results, spec)
    paired_summary = _paired_summary(
        paired_rows, args.minimum_mean_delta
    )
    result = {
        "experiment_id": (
            f"six_dof_{args.factor}_counterfactual_v1_20260901"
        ),
        "experiment_type": "paired_frozen_model_counterfactual",
        "factor": args.factor,
        "interpretation_policy": (
            "Simulation-only causal evidence. The frozen V3 model and rules "
            "are evaluated without calibration or retraining."
        ),
        "device": str(device),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "temporal_config_sha256": _sha256(args.temporal_config),
        "ticket_config_sha256": _sha256(args.ticket_config),
        "parameters": {
            "repeats_per_depth_band": args.repeats_per_depth_band,
            "depth_band_indices": [0, 1, 2],
            "duration_s": args.duration,
            "dt_s": args.dt,
            "sequence_length": args.seq_len,
            "stride": args.stride,
            "base_seed": args.base_seed,
            "baseline_condition": spec["baseline_condition"],
            "candidate_condition": spec["candidate_condition"],
            "factor_value_name": spec["value_name"],
            "baseline_factor_value": spec["baseline_value"],
            "candidate_factor_value": spec["candidate_value"],
            "fixed_rpm_noise_std": 40.0,
            "fixed_lateral_force_amplitude_n": 0.25,
            "fixed_vertical_force_limit_n": 45.0,
        },
        "pair_integrity": {
            "all_pairs_equal_except_target_factor": all(
                item["parameters_equal_except_target_factor"]
                for item in pair_integrity
            ),
            "pairs": pair_integrity,
        },
        "conditions": condition_results,
        "paired_summary": paired_summary,
        "depth_band_summary": _depth_band_summary(paired_rows),
        "paired_rows": paired_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_stem = f"{args.factor}_counterfactual"
    (args.output_dir / f"{result_stem}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        args.output_dir / f"{result_stem}_pairs.csv",
        paired_rows,
    )
    print(json.dumps({
        "factor": args.factor,
        "condition_false_alarm_rates": {
            condition: {
                "raw": values["raw_false_alarm_rate"],
                "temporal": values["temporal_false_alarm_rate"],
                "false_formal_tickets": values[
                    "false_formal_ticket_count"
                ],
            }
            for condition, values in condition_results.items()
        },
        "paired_summary": paired_summary,
        "depth_band_summary": result["depth_band_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
