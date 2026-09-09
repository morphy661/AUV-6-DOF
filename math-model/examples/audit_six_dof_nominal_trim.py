"""Audit nominal high-yaw missions after CG/CB trim randomization changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_six_dof_fault_dataset import run_mission
from utils.six_dof_feature_extractor import THRUSTER_NAMES


TURN_RANGE_RAD = (0.55, 1.05)
TEST_SEED_NAMESPACE = 2_000_000


def _stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(len(values)),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95.0)),
        "maximum": float(np.max(values)),
    }


def _wrap_angle(values):
    values = np.asarray(values, dtype=float)
    return np.arctan2(np.sin(values), np.cos(values))


def _mission_metrics(pair_name, repetition, seed, logs, parameters):
    times = np.asarray([log["time"] for log in logs], dtype=float)
    attitude = np.asarray([log["euler_rpy"] for log in logs], dtype=float)
    target_attitude = np.asarray([
        log["target_euler_rpy"] for log in logs
    ], dtype=float)
    body_rates = np.asarray([
        log["body_velocity"][3:] for log in logs
    ], dtype=float)
    depth_error = np.asarray([
        log["target_position_ned"][2] - log["position_ned"][2]
        for log in logs
    ], dtype=float)
    saturated = np.asarray([
        log["thruster_saturated"] for log in logs
    ], dtype=bool)
    yaw = np.unwrap(attitude[:, 2])
    yaw_error = _wrap_angle(target_attitude[:, 2] - attitude[:, 2])

    turns = []
    requested = parameters["mission"]["heading_deltas_rad"]
    for turn_number, (start, end, delta) in enumerate(zip(
        (18.75, 42.50, 66.25),
        (23.75, 47.50, 71.25),
        requested,
    ), start=1):
        indices = [
            int(np.argmin(np.abs(times - value)))
            for value in (start, end, end + 5.0)
        ]
        start_index, end_index, settle_index = indices
        turns.append({
            "turn": turn_number,
            "requested_deg": float(np.degrees(delta)),
            "measured_change_at_command_end_deg": float(np.degrees(
                yaw[end_index] - yaw[start_index]
            )),
            "measured_change_after_5s_deg": float(np.degrees(
                yaw[settle_index] - yaw[start_index]
            )),
            "yaw_error_after_5s_deg": float(np.degrees(
                yaw_error[settle_index]
            )),
        })

    dynamics = parameters["dynamics"]
    return {
        "pair_name": pair_name,
        "repetition": repetition,
        "environment_seed": seed,
        "initial_depth_m": parameters["mission"]
        ["initial_position_ned_m"][2],
        "cruise_speed_mps": parameters["mission"]["cruise_speed_mps"],
        "vertical_speed_mps": parameters["mission"]["vertical_speed_mps"],
        "mass_kg": dynamics["mass_kg"],
        "center_relative_xy_m": dynamics["center_relative_xy_m"],
        "roll_abs_deg": _stats(np.abs(np.degrees(attitude[:, 0]))),
        "pitch_abs_deg": _stats(np.abs(np.degrees(attitude[:, 1]))),
        "yaw_error_abs_deg": _stats(np.abs(np.degrees(yaw_error))),
        "body_rate_abs_degps": {
            axis: _stats(np.abs(np.degrees(body_rates[:, index])))
            for index, axis in enumerate(("p", "q", "r"))
        },
        "depth_error_abs_m": _stats(np.abs(depth_error)),
        "thruster_saturation_fraction": float(np.mean(saturated)),
        "turns": turns,
    }


def _aggregate(rows, sequence_length=100, stride=25):
    turns = [turn for row in rows for turn in row["turns"]]
    scalar_paths = {
        "roll_abs_deg": ("roll_abs_deg",),
        "pitch_abs_deg": ("pitch_abs_deg",),
        "yaw_error_abs_deg": ("yaw_error_abs_deg",),
        "depth_error_abs_m": ("depth_error_abs_m",),
    }
    aggregate = {
        "mission_count": len(rows),
        "mass_kg": _stats([row["mass_kg"] for row in rows]),
        "center_relative_x_abs_mm": _stats([
            1000.0 * abs(row["center_relative_xy_m"][0])
            for row in rows
        ]),
        "center_relative_y_abs_mm": _stats([
            1000.0 * abs(row["center_relative_xy_m"][1])
            for row in rows
        ]),
        "thruster_saturation_fraction": _stats([
            row["thruster_saturation_fraction"] for row in rows
        ]),
        "turn_response": {
            "requested_abs_deg": _stats([
                abs(turn["requested_deg"]) for turn in turns
            ]),
            "measured_abs_at_command_end_deg": _stats([
                abs(turn["measured_change_at_command_end_deg"])
                for turn in turns
            ]),
            "measured_abs_after_5s_deg": _stats([
                abs(turn["measured_change_after_5s_deg"])
                for turn in turns
            ]),
            "yaw_error_abs_after_5s_deg": _stats([
                abs(turn["yaw_error_after_5s_deg"]) for turn in turns
            ]),
        },
    }
    # Aggregate full-rate distributions from the per-mission summaries only
    # where the requested statistics can be combined without raw duplication.
    for output_name, path in scalar_paths.items():
        aggregate[f"mission_{output_name}_median"] = _stats([
            row[path[0]]["median"] for row in rows
        ])
        aggregate[f"mission_{output_name}_maximum"] = _stats([
            row[path[0]]["maximum"] for row in rows
        ])
    aggregate["window_endpoint_sampling"] = {
        "sequence_length": sequence_length,
        "stride": stride,
        "note": (
            "The main audit uses every 20 Hz simulator sample; the previous "
            "dataset report used overlapping-window endpoints."
        ),
    }
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-seed", type=int, default=22_000_000)
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--mass-relative-error", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for thruster_index, pair_name in enumerate(THRUSTER_NAMES):
        for repetition in (6, 7):
            seed = (
                args.base_seed
                + TEST_SEED_NAMESPACE
                + thruster_index * 10_000
                + repetition
            )
            logs, parameters = run_mission(
                thruster_name=None,
                fault_mode=None,
                duration=args.duration,
                dt=args.dt,
                seed=seed,
                split="test",
                depth_band_index=2,
                fault_seed=seed + 5_000_000,
                mass_relative_error=args.mass_relative_error,
                heading_delta_abs_range_rad=TURN_RANGE_RAD,
            )
            rows.append(_mission_metrics(
                pair_name, repetition, seed, logs, parameters
            ))
            print(
                f"Audited {pair_name} rep {repetition}: "
                f"roll max={rows[-1]['roll_abs_deg']['maximum']:.2f} deg, "
                f"pitch max={rows[-1]['pitch_abs_deg']['maximum']:.2f} deg"
            )

    payload = {
        "audit_id": "six_dof_nominal_trim_same_seed_20260902",
        "policy": (
            "Normal missions only; same 12 independent-test environment "
            "seeds and maneuver range used before the CG/CB trim change."
        ),
        "base_seed": args.base_seed,
        "duration_s": args.duration,
        "dt_s": args.dt,
        "heading_delta_abs_range_rad": list(TURN_RANGE_RAD),
        "aggregate": _aggregate(rows),
        "missions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
