"""Compare no-FTC baselines with oracle effectiveness-aware allocation."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from demo_six_dof_fault_coverage import (
    FAULT_TIME,
    angle_difference,
    scenario_metadata,
    scenarios,
    target_provider,
)
from environment.six_dof_simulator import SixDOFSimulator


def run_comparison(duration, dt):
    normal_logs = SixDOFSimulator().run(duration, dt, target_provider)
    baseline = {}
    ideal_ftc = {}
    for name, fault in scenarios().items():
        if fault is None:
            continue
        baseline[name] = SixDOFSimulator(fault=fault).run(
            duration, dt, target_provider
        )
        ideal_ftc[name] = SixDOFSimulator(
            fault=fault,
            ideal_fault_tolerant_allocation=True,
        ).run(duration, dt, target_provider)
    return normal_logs, baseline, ideal_ftc


def _mission_metrics(logs, normal_positions, normal_attitudes):
    times = np.array([log["time"] for log in logs])
    mask = times >= FAULT_TIME
    positions = np.array([log["position_ned"] for log in logs])
    attitudes = np.array([log["euler_rpy"] for log in logs])
    deviations = np.linalg.norm(positions - normal_positions, axis=1)[mask]
    attitude_delta = np.rad2deg(
        np.abs(angle_difference(attitudes, normal_attitudes)[mask])
    )
    final_target = logs[-1]["target_position_ned"]
    saturated = np.array([
        np.any(log["thruster_saturated"]) for log in logs
    ])[mask]
    actuation_residuals = np.array([
        np.linalg.norm(log["actuation_residual_body"]) for log in logs
    ])[mask]
    return {
        "max_trajectory_deviation_m": float(np.max(deviations)),
        "rms_trajectory_deviation_m": float(np.sqrt(np.mean(deviations ** 2))),
        "max_roll_deviation_deg": float(np.max(attitude_delta[:, 0])),
        "max_pitch_deviation_deg": float(np.max(attitude_delta[:, 1])),
        "max_yaw_deviation_deg": float(np.max(attitude_delta[:, 2])),
        "final_position_error_m": float(
            np.linalg.norm(final_target - positions[-1])
        ),
        "saturation_fraction": float(np.mean(saturated)),
        "max_actuation_residual": float(np.max(actuation_residuals)),
    }


def _improvement_percent(baseline_value, ideal_value):
    if abs(baseline_value) < 1e-12:
        return 0.0
    return 100.0 * (baseline_value - ideal_value) / baseline_value


def extract_comparison(normal_logs, baseline, ideal_ftc):
    normal_positions = np.array([
        log["position_ned"] for log in normal_logs
    ])
    normal_attitudes = np.array([
        log["euler_rpy"] for log in normal_logs
    ])
    rows = []
    for name in baseline:
        thruster_name, fault_mode = scenario_metadata(name)
        baseline_metrics = _mission_metrics(
            baseline[name], normal_positions, normal_attitudes
        )
        ideal_metrics = _mission_metrics(
            ideal_ftc[name], normal_positions, normal_attitudes
        )
        row = {
            "scenario": name,
            "faulted_thruster": thruster_name,
            "fault_mode": fault_mode,
        }
        for metric, value in baseline_metrics.items():
            row[f"baseline_{metric}"] = value
            row[f"ideal_ftc_{metric}"] = ideal_metrics[metric]
        for metric in (
            "max_trajectory_deviation_m",
            "max_roll_deviation_deg",
            "max_pitch_deviation_deg",
            "max_yaw_deviation_deg",
            "final_position_error_m",
        ):
            row[f"{metric}_improvement_percent"] = _improvement_percent(
                baseline_metrics[metric], ideal_metrics[metric]
            )
        rows.append(row)
    return rows


def validate_oracle_ftc(ideal_ftc):
    for name, logs in ideal_ftc.items():
        active_logs = [
            log for log in logs
            if log["time"] > FAULT_TIME and log["thruster_fault_active"]
        ]
        if not active_logs:
            raise AssertionError(f"{name} never activated")
        if not all(log["ftc_active"] for log in active_logs):
            raise AssertionError(f"{name} did not activate ideal FTC")
        max_residual = max(
            np.linalg.norm(log["actuation_residual_body"])
            for log in active_logs
        )
        if max_residual > 1e-8:
            raise AssertionError(
                f"{name} oracle model mismatch: residual={max_residual}"
            )


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _short_label(name):
    return name.replace(" No Output", "-NO").replace(" Thrust Loss", "-TL")


def save_plot(rows, path):
    labels = [_short_label(row["scenario"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.38
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    panels = (
        ("max_trajectory_deviation_m", "Maximum trajectory deviation (m)"),
        ("final_position_error_m", "Final position error (m)"),
        ("max_yaw_deviation_deg", "Maximum yaw deviation (deg)"),
        ("max_pitch_deviation_deg", "Maximum pitch deviation (deg)"),
    )
    for axis, (metric, ylabel) in zip(axes.flat, panels):
        axis.bar(
            x - width / 2,
            [row[f"baseline_{metric}"] for row in rows],
            width,
            label="No FTC",
            color="#C44E52",
        )
        axis.bar(
            x + width / 2,
            [row[f"ideal_ftc_{metric}"] for row in rows],
            width,
            label="Ideal FTC",
            color="#4C72B0",
        )
        axis.set_xticks(x, labels, rotation=60, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend()
    figure.suptitle("No FTC versus ideal effectiveness-aware allocation")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "six_dof_ideal_ftc",
    )
    args = parser.parse_args()
    if args.duration <= FAULT_TIME:
        parser.error(f"--duration must be greater than {FAULT_TIME:g} s")

    normal_logs, baseline, ideal_ftc = run_comparison(args.duration, args.dt)
    validate_oracle_ftc(ideal_ftc)
    rows = extract_comparison(normal_logs, baseline, ideal_ftc)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ideal_ftc_comparison.csv"
    plot_path = args.output_dir / "ideal_ftc_comparison.png"
    save_csv(rows, csv_path)
    save_plot(rows, plot_path)

    for row in rows:
        print(row)
    print("Oracle FTC consistency checks: PASS")
    print(f"Summary: {csv_path}")
    print(f"Figure: {plot_path}")


if __name__ == "__main__":
    main()
