"""Evaluate RPM-noise causality with paired normal six-DOF missions.

For every environment seed, the nominal- and high-RPM-noise missions share
the exact same dynamics, thruster geometry, sensor parameters, disturbance,
trajectory, and simulator random streams.  Only ESC RPM telemetry noise is
changed.  The frozen V3 model and its temporal/maintenance policies are used
without calibration or retraining.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from audit_six_dof_false_alarms import (  # noqa: E402
    _build_split_audit,
)
from diagnosis.maintenance_ticket_policy import (  # noqa: E402
    MaintenanceTicketConfig,
)
from diagnosis.temporal_fault_decision import (  # noqa: E402
    TemporalDecisionConfig,
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


CONDITIONS = ("nominal_rpm_noise", "high_rpm_noise")
ARRAY_KEYS = (
    "X",
    "y_mode",
    "y_location",
    "y_joint",
    "mission_ids",
    "window_end_times",
    "guidance_context_ids",
    "guidance_context_stable",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _without_rpm_noise(metadata):
    comparable = json.loads(json.dumps(metadata, sort_keys=True))
    comparable["sensors"].pop("rpm_noise_std")
    return comparable


def _concatenate(chunks, mission_metadata):
    dataset = {
        key: torch.from_numpy(np.concatenate(
            [chunk[key] for chunk in chunks], axis=0
        ))
        for key in ARRAY_KEYS
    }
    dataset.update({
        key: chunks[0][key]
        for key in (
            "feature_names",
            "raw_feature_dim",
            "model_input_dim",
            "sequence_length",
        )
    })
    dataset.update({
        "dataset_version": "six_dof_rpm_noise_counterfactual_v1",
        "mission_metadata": mission_metadata,
    })
    return dataset


def _generate_dataset(args):
    chunks = []
    mission_metadata = {}
    pair_integrity = []
    mission_id = 0
    for band_index in range(3):
        for repetition in range(args.repeats_per_depth_band):
            pair_id = f"band{band_index}_rep{repetition}"
            environment_seed = (
                args.base_seed + band_index * 10_000 + repetition
            )
            fault_seed = environment_seed + 5_000_000
            pair_parameters = {}
            for condition, rpm_noise in (
                ("nominal_rpm_noise", args.nominal_rpm_noise),
                ("high_rpm_noise", args.high_rpm_noise),
            ):
                logs, parameters = run_mission(
                    thruster_name=None,
                    fault_mode=None,
                    duration=args.duration,
                    dt=args.dt,
                    seed=environment_seed,
                    split="counterfactual_test",
                    depth_band_index=band_index,
                    fault_seed=fault_seed,
                    rpm_noise_std_override=rpm_noise,
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

            nominal = pair_parameters["nominal_rpm_noise"]
            high = pair_parameters["high_rpm_noise"]
            integrity_ok = _without_rpm_noise(nominal) == _without_rpm_noise(
                high
            )
            if not integrity_ok:
                raise RuntimeError(
                    f"counterfactual pair {pair_id} differs outside RPM noise"
                )
            pair_integrity.append({
                "pair_id": pair_id,
                "environment_seed": environment_seed,
                "depth_band_index": band_index,
                "parameters_equal_except_rpm_noise": True,
            })
    return _concatenate(chunks, mission_metadata), pair_integrity


def _condition_indices(dataset, condition):
    missions = [
        int(mission_id)
        for mission_id, metadata in dataset["mission_metadata"].items()
        if metadata["condition"] == condition
    ]
    mission_ids = dataset["mission_ids"].cpu().numpy()
    return np.flatnonzero(np.isin(mission_ids, missions))


def _condition_result(audit):
    summaries = audit["summary"]
    formal = summaries["formal_ticket_audit"]
    result = {
        "normal_windows": int(summaries["raw"]["normal_windows"]),
        "raw_false_alarm_rate": float(
            summaries["raw"]["false_alarm_rate"]
        ),
        "temporal_false_alarm_rate": float(
            summaries["temporal"]["false_alarm_rate"]
        ),
        "operator_attention_window_coverage": float(
            summaries["operator"]["false_alarm_rate"]
        ),
        "formal_ticket_window_coverage": float(
            summaries["ticket"]["false_alarm_rate"]
        ),
        "raw_episode_count": int(summaries["raw"]["episode_count"]),
        "raw_episodes_at_least_8_s": int(
            summaries["raw"]["episodes_at_least_8_s"]
        ),
        "false_formal_ticket_count": int(
            formal["false_ticket_count"]
        ),
        "missions": [],
    }
    for mission in audit["mission_summary"]:
        result["missions"].append({
            "mission_id": int(mission["mission_id"]),
            "depth_band": mission["depth_band"],
            "normal_windows": int(mission["normal_windows"]),
            "raw_false_windows": int(mission["raw_false_windows"]),
            "raw_false_alarm_rate": float(
                mission["raw_false_alarm_rate"]
            ),
            "temporal_false_alarm_rate": float(
                mission["temporal_false_alarm_rate"]
            ),
            "operator_false_windows": int(
                mission["operator_false_windows"]
            ),
            "ticket_false_windows": int(
                mission["ticket_false_windows"]
            ),
        })
    return result


def _paired_rows(dataset, condition_results):
    rates = {}
    for condition, result in condition_results.items():
        for mission in result["missions"]:
            metadata = dataset["mission_metadata"][mission["mission_id"]]
            rates[(metadata["pair_id"], condition)] = mission

    rows = []
    for pair_id in sorted({key[0] for key in rates}):
        nominal = rates[(pair_id, "nominal_rpm_noise")]
        high = rates[(pair_id, "high_rpm_noise")]
        metadata = next(
            item for item in dataset["mission_metadata"].values()
            if item["pair_id"] == pair_id
        )
        rows.append({
            "pair_id": pair_id,
            "environment_seed": int(metadata["seed"]),
            "depth_band_index": int(
                metadata["parameters"]["mission"]["depth_band_index"]
            ),
            "depth_band_m": "-".join(
                f"{value:g}"
                for value in metadata["parameters"]["mission"][
                    "depth_band_m"
                ]
            ),
            "nominal_raw_false_alarm_rate": nominal[
                "raw_false_alarm_rate"
            ],
            "high_raw_false_alarm_rate": high["raw_false_alarm_rate"],
            "delta_raw_false_alarm_rate": (
                high["raw_false_alarm_rate"]
                - nominal["raw_false_alarm_rate"]
            ),
            "nominal_temporal_false_alarm_rate": nominal[
                "temporal_false_alarm_rate"
            ],
            "high_temporal_false_alarm_rate": high[
                "temporal_false_alarm_rate"
            ],
            "delta_temporal_false_alarm_rate": (
                high["temporal_false_alarm_rate"]
                - nominal["temporal_false_alarm_rate"]
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
        "supports_rpm_noise_as_causal_false_alarm_factor": bool(
            mean_raw >= minimum_mean_delta
            and positive_raw >= required_positive
        ),
    }


def _write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats-per-depth-band", type=int, default=4)
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--base-seed", type=int, default=930_000)
    parser.add_argument("--nominal-rpm-noise", type=float, default=40.0)
    parser.add_argument("--high-rpm-noise", type=float, default=70.0)
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "math-model" / "results" /
            "six_dof_rpm_noise_counterfactual_v1_20260901"
        ),
    )
    args = parser.parse_args()
    if args.repeats_per_depth_band <= 0:
        parser.error("--repeats-per-depth-band must be positive")
    if args.duration <= 0.0 or args.dt <= 0.0:
        parser.error("--duration and --dt must be positive")
    if args.nominal_rpm_noise >= args.high_rpm_noise:
        parser.error("nominal RPM noise must be below high RPM noise")

    set_seed(args.seed)
    dataset, pair_integrity = _generate_dataset(args)
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
    for condition in CONDITIONS:
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

    paired_rows = _paired_rows(dataset, condition_results)
    paired_summary = _paired_summary(
        paired_rows, args.minimum_mean_delta
    )
    result = {
        "experiment_id": "six_dof_rpm_noise_counterfactual_v1_20260901",
        "experiment_type": "paired_frozen_model_counterfactual",
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
            "nominal_rpm_noise_std": args.nominal_rpm_noise,
            "high_rpm_noise_std": args.high_rpm_noise,
        },
        "pair_integrity": {
            "all_pairs_equal_except_rpm_noise": all(
                item["parameters_equal_except_rpm_noise"]
                for item in pair_integrity
            ),
            "pairs": pair_integrity,
        },
        "conditions": condition_results,
        "paired_summary": paired_summary,
        "paired_rows": paired_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rpm_noise_counterfactual.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        args.output_dir / "rpm_noise_counterfactual_pairs.csv",
        paired_rows,
    )
    print(json.dumps({
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
    }, indent=2))


if __name__ == "__main__":
    main()
