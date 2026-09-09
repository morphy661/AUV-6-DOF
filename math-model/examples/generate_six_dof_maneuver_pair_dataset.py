"""Augment a six-DOF dataset with paired high-yaw-turn missions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from actuators.six_dof_thruster_faults import (  # noqa: E402
    SixDOFThrusterFaultMode,
)
from generate_six_dof_fault_dataset import run_mission  # noqa: E402
from utils.six_dof_dataset_builder import (  # noqa: E402
    build_six_dof_sequence_dataset,
)
from utils.six_dof_feature_extractor import THRUSTER_NAMES  # noqa: E402


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
TURN_RANGE_RAD = (0.55, 1.05)
DEPTH_BAND_INDEX = 2
DATASET_VERSION = "six_dof_hybrid_telemetry_0_300m_focus_v3_2_maneuver_pairs"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _split_for_repetition(repetition, repetitions_per_thruster):
    if repetitions_per_thruster < 4:
        raise ValueError("at least four repetitions per thruster are required")
    validation_count = max(1, int(round(0.15 * repetitions_per_thruster)))
    test_count = max(1, int(round(0.25 * repetitions_per_thruster)))
    train_count = repetitions_per_thruster - validation_count - test_count
    if train_count < 2:
        raise ValueError("augmentation requires at least two train repetitions")
    if repetition < train_count:
        return "train"
    if repetition < train_count + validation_count:
        return "validation"
    return "test"


def _numeric_stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(len(values)),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
    }


def _combine(base, chunks, mission_metadata):
    payload = dict(base)
    for key in ARRAY_KEYS:
        addition = torch.from_numpy(np.concatenate([
            chunk[key] for chunk in chunks
        ], axis=0))
        payload[key] = torch.cat((base[key], addition), dim=0)

    payload["mission_metadata"] = mission_metadata
    mission_ids = payload["mission_ids"].long()
    split_indices = {}
    for split in ("train", "validation", "test"):
        selected = torch.as_tensor([
            mission_id
            for mission_id, metadata in mission_metadata.items()
            if metadata["split"] == split
        ], dtype=torch.long)
        split_indices[split] = torch.nonzero(
            torch.isin(mission_ids, selected), as_tuple=False
        ).flatten()
    payload["split_indices"] = split_indices
    return payload


def generate_augmentation(args, base):
    mission_metadata = {
        int(key): value for key, value in base["mission_metadata"].items()
    }
    next_mission_id = int(base["mission_ids"].max()) + 1
    chunks = []
    pair_rows = []
    split_namespaces = {
        "train": 0,
        "validation": 1_000_000,
        "test": 2_000_000,
    }

    for thruster_index, thruster_name in enumerate(THRUSTER_NAMES):
        for repetition in range(args.repetitions_per_thruster):
            split = _split_for_repetition(
                repetition, args.repetitions_per_thruster
            )
            environment_seed = (
                args.base_seed
                + split_namespaces[split]
                + thruster_index * 10_000
                + repetition
            )
            fault_seed = environment_seed + 5_000_000
            pair_id = f"{thruster_name}_rep{repetition:02d}"
            pair_parameters = {}
            pair_mission_ids = {}
            conditions = (
                ("normal", None, None, "Normal"),
                (
                    "thrust_loss",
                    thruster_name,
                    SixDOFThrusterFaultMode.THRUST_LOSS,
                    f"{thruster_name} Thrust Loss",
                ),
            )
            for condition, fault_thruster, fault_mode, scenario in conditions:
                logs, parameters = run_mission(
                    thruster_name=fault_thruster,
                    fault_mode=fault_mode,
                    duration=args.duration,
                    dt=args.dt,
                    seed=environment_seed,
                    split=split,
                    depth_band_index=DEPTH_BAND_INDEX,
                    fault_seed=fault_seed,
                    mass_relative_error=args.mass_relative_error,
                    heading_delta_abs_range_rad=TURN_RANGE_RAD,
                )
                mission_id = next_mission_id
                next_mission_id += 1
                chunk = build_six_dof_sequence_dataset(
                    {mission_id: logs},
                    seq_len=args.seq_len,
                    stride=args.stride,
                    diagnostic_generalized_mass_diagonal=(
                        parameters["dynamics"]
                        ["configured_generalized_mass_diagonal"]
                    ),
                )
                chunks.append(chunk)
                pair_parameters[condition] = parameters
                pair_mission_ids[condition] = mission_id
                mission_metadata[mission_id] = {
                    "scenario": scenario,
                    "seed": environment_seed,
                    "split": split,
                    "augmentation": args.augmentation_tag,
                    "pair_id": pair_id,
                    "pair_condition": condition,
                    "paired_thruster": thruster_name,
                    "parameters": parameters,
                }
                print(
                    f"Pair {pair_id}, {split}, {condition}, "
                    f"mission={mission_id}, windows={len(chunk['X'])}"
                )

            normal_parameters = pair_parameters["normal"]
            fault_parameters = pair_parameters["thrust_loss"]
            for key in (
                "dynamics",
                "controller",
                "thrusters",
                "sensors",
                "disturbance",
                "mission",
            ):
                if normal_parameters[key] != fault_parameters[key]:
                    raise RuntimeError(
                        f"pair {pair_id} differs in environment field {key}"
                    )
            pair_rows.append({
                "pair_id": pair_id,
                "split": split,
                "thruster": thruster_name,
                "environment_seed": environment_seed,
                "normal_mission_id": pair_mission_ids["normal"],
                "fault_mission_id": pair_mission_ids["thrust_loss"],
                "initial_depth_m": normal_parameters["mission"]
                ["initial_position_ned_m"][2],
                "cruise_speed_mps": normal_parameters["mission"]
                ["cruise_speed_mps"],
                "vertical_speed_mps": normal_parameters["mission"]
                ["vertical_speed_mps"],
                "heading_deltas_rad": normal_parameters["mission"]
                ["heading_deltas_rad"],
                "fault_start_time_s": fault_parameters["fault_start_time_s"],
                "thrust_efficiency": fault_parameters["thrust_efficiency"],
            })
    return _combine(base, chunks, mission_metadata), pair_rows


def _summary(args, base_path, payload, pair_rows, output):
    unique_turns = [
        abs(delta)
        for row in pair_rows
        for delta in row["heading_deltas_rad"]
    ]
    split_pair_counts = {
        split: sum(row["split"] == split for row in pair_rows)
        for split in ("train", "validation", "test")
    }
    split_mission_counts = {
        split: len(torch.unique(
            payload["mission_ids"][payload["split_indices"][split]]
        ))
        for split in ("train", "validation", "test")
    }
    return {
        "dataset_version": args.dataset_version,
        "parent_dataset": str(base_path),
        "parent_dataset_sha256": _sha256(base_path),
        "output": str(output),
        "total_missions": len(payload["mission_metadata"]),
        "total_windows": len(payload["X"]),
        "split_missions": split_mission_counts,
        "augmentation": {
            "paired_environment_policy": (
                "Each normal/thrust-loss pair shares every simulated "
                "environment parameter and remains in one mission split."
            ),
            "depth_band_m": [150.0, 300.0],
            "turns_per_mission": 3,
            "requested_heading_delta_abs_range_rad": list(TURN_RANGE_RAD),
            "requested_heading_delta_abs_range_deg": [
                float(np.degrees(value)) for value in TURN_RANGE_RAD
            ],
            "realized_heading_delta_abs_rad": _numeric_stats(unique_turns),
            "realized_heading_delta_abs_deg": _numeric_stats(
                np.degrees(unique_turns)
            ),
            "initial_depth_m": _numeric_stats([
                row["initial_depth_m"] for row in pair_rows
            ]),
            "cruise_speed_mps": _numeric_stats([
                row["cruise_speed_mps"] for row in pair_rows
            ]),
            "vertical_speed_mps": _numeric_stats([
                row["vertical_speed_mps"] for row in pair_rows
            ]),
            "fault_start_time_s": _numeric_stats([
                row["fault_start_time_s"] for row in pair_rows
            ]),
            "thrust_efficiency": _numeric_stats([
                row["thrust_efficiency"] for row in pair_rows
            ]),
            "pairs": len(pair_rows),
            "missions": 2 * len(pair_rows),
            "pairs_by_split": split_pair_counts,
            "pairs_by_thruster": {
                name: sum(row["thruster"] == name for row in pair_rows)
                for name in THRUSTER_NAMES
            },
            "base_seed": args.base_seed,
        },
        "pair_rows": pair_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=(
            REPO_ROOT / "depth-sensor-fault-detection"
            / "depth_fault_detection" / "data"
            / "simulation_dataset_six_dof_hybrid_telemetry_0_300m_focus_v3_1.pth"
        ),
    )
    parser.add_argument("--repetitions-per-thruster", type=int, default=8)
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--base-seed", type=int, default=12_000_000)
    parser.add_argument("--mass-relative-error", type=float, default=0.02)
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument(
        "--augmentation-tag", default="v3_2_high_yaw_turn_pair"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT / "depth-sensor-fault-detection"
            / "depth_fault_detection" / "data"
            / "simulation_dataset_six_dof_hybrid_telemetry_0_300m_focus_v3_2.pth"
        ),
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.repetitions_per_thruster < 4:
        parser.error("--repetitions-per-thruster must be at least 4")
    if args.duration <= 0.0 or args.dt <= 0.0:
        parser.error("--duration and --dt must be positive")
    if not 0.0 <= args.mass_relative_error <= 0.10:
        parser.error("--mass-relative-error must be within [0, 0.10]")

    base = torch.load(args.base_dataset, map_location="cpu", weights_only=False)
    payload, pair_rows = generate_augmentation(args, base)
    payload.update({
        "dataset_version": args.dataset_version,
        "parent_dataset_version": base.get("dataset_version"),
        "augmentation_profile": args.augmentation_tag,
        "augmentation_turn_range_rad": list(TURN_RANGE_RAD),
        "augmentation_base_seed": args.base_seed,
        "split_policy": (
            "Parent mission splits are preserved; new paired normal and "
            "thrust-loss missions use disjoint train/validation/test seeds, "
            "and both members of each pair remain in the same split."
        ),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary_path = args.summary or args.output.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = _summary(
        args, args.base_dataset, payload, pair_rows, args.output
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: value for key, value in summary.items() if key != "pair_rows"
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
