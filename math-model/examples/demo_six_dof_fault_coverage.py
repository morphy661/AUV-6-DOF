"""Validate fault signatures of the six-thruster KYUBIC-style layout."""

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

from actuators.six_dof_thruster_faults import (
    SingleThrusterFault,
    SixDOFThrusterFaultMode,
)
from environment.six_dof_simulator import SixDOFSimulator
from simple_control.six_dof_controller import PoseTarget


FAULT_TIME = 55.0
AXIS_NAMES = ("X", "Y", "Z", "K", "M", "N")
MISSION_TARGETS = [
    (0.0, np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.0])),
    (25.0, np.array([6.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.0])),
    (55.0, np.array([6.0, 4.0, 3.0]), np.array([0.0, 0.0, np.pi / 2.0])),
    (85.0, np.array([1.0, 4.0, 2.0]), np.array([0.0, 0.0, np.pi])),
]


def target_provider(time_s, _state):
    selected = MISSION_TARGETS[0]
    for candidate in MISSION_TARGETS:
        if time_s >= candidate[0]:
            selected = candidate
        else:
            break
    return PoseTarget(selected[1], selected[2])


def scenarios():
    return {
        "Normal": None,
        "H1 No Output": SingleThrusterFault(
            "H1", SixDOFThrusterFaultMode.NO_OUTPUT, FAULT_TIME
        ),
        "H1 Thrust Loss": SingleThrusterFault(
            "H1", SixDOFThrusterFaultMode.THRUST_LOSS, FAULT_TIME, 0.45
        ),
        "V1 No Output": SingleThrusterFault(
            "V1", SixDOFThrusterFaultMode.NO_OUTPUT, FAULT_TIME
        ),
        "V1 Thrust Loss": SingleThrusterFault(
            "V1", SixDOFThrusterFaultMode.THRUST_LOSS, FAULT_TIME, 0.45
        ),
    }


def run_scenarios(duration, dt):
    return {
        name: SixDOFSimulator(fault=fault).run(duration, dt, target_provider)
        for name, fault in scenarios().items()
    }


def angle_difference(values, reference):
    return np.arctan2(np.sin(values - reference), np.cos(values - reference))


def extract_metrics(results):
    normal_logs = results["Normal"]
    nominal_positions = np.array([log["position_ned"] for log in normal_logs])
    nominal_attitudes = np.array([log["euler_rpy"] for log in normal_logs])
    rows = []
    peak_residuals = {}

    for name, logs in results.items():
        times = np.array([log["time"] for log in logs])
        mask = times >= FAULT_TIME
        positions = np.array([log["position_ned"] for log in logs])
        attitudes = np.array([log["euler_rpy"] for log in logs])
        deviations = np.linalg.norm(positions - nominal_positions, axis=1)[mask]
        attitude_delta = angle_difference(attitudes, nominal_attitudes)[mask]
        residual_matrix = np.array([
            log["actuation_residual_body"] for log in logs
        ])[mask]
        peak_residual = np.max(np.abs(residual_matrix), axis=0)
        peak_residuals[name] = peak_residual

        if name == "Normal":
            current_residual = 0.0
        else:
            fault_index = 0 if name.startswith("H1") else 4
            current_residual = max(
                abs(
                    log["thruster_measured_currents"][fault_index]
                    - log["thruster_expected_currents"][fault_index]
                )
                for log, keep in zip(logs, mask)
                if keep
            )

        direct_axes = "/".join(
            axis for axis, value in zip(AXIS_NAMES, peak_residual)
            if value > 1e-8
        ) or "none"
        final_target = logs[-1]["target_position_ned"]
        rows.append({
            "scenario": name,
            "direct_wrench_axes": direct_axes,
            "max_trajectory_deviation_m": float(np.max(deviations)),
            "rms_trajectory_deviation_m": float(np.sqrt(np.mean(deviations ** 2))),
            "max_roll_deviation_deg": float(np.rad2deg(np.max(np.abs(attitude_delta[:, 0])))),
            "max_pitch_deviation_deg": float(np.rad2deg(np.max(np.abs(attitude_delta[:, 1])))),
            "max_yaw_deviation_deg": float(np.rad2deg(np.max(np.abs(attitude_delta[:, 2])))),
            "final_position_error_m": float(
                np.linalg.norm(final_target - positions[-1])
            ),
            "max_current_residual_a": float(current_residual),
        })

    return rows, peak_residuals


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(results, peak_residuals, path):
    colors = {
        "Normal": "#2468A2",
        "H1 No Output": "#C73E1D",
        "H1 Thrust Loss": "#E68A00",
        "V1 No Output": "#7A5195",
        "V1 Thrust Loss": "#3A923A",
    }
    normal_logs = results["Normal"]
    nominal_positions = np.array([log["position_ned"] for log in normal_logs])
    nominal_attitudes = np.array([log["euler_rpy"] for log in normal_logs])

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    deviation_axis, yaw_axis = axes[0]
    tilt_axis, residual_axis = axes[1]

    for name, logs in results.items():
        times = np.array([log["time"] for log in logs])
        positions = np.array([log["position_ned"] for log in logs])
        attitudes = np.array([log["euler_rpy"] for log in logs])
        attitude_delta = angle_difference(attitudes, nominal_attitudes)
        deviation_axis.plot(
            times,
            np.linalg.norm(positions - nominal_positions, axis=1),
            color=colors[name],
            label=name,
        )
        yaw_axis.plot(
            times,
            np.rad2deg(attitude_delta[:, 2]),
            color=colors[name],
            label=name,
        )
        tilt_axis.plot(
            times,
            np.rad2deg(np.linalg.norm(attitude_delta[:, :2], axis=1)),
            color=colors[name],
            label=name,
        )

    fault_names = [name for name in results if name != "Normal"]
    x = np.arange(len(AXIS_NAMES))
    width = 0.18
    for index, name in enumerate(fault_names):
        residual_axis.bar(
            x + (index - 1.5) * width,
            peak_residuals[name],
            width,
            color=colors[name],
            label=name,
        )
    residual_axis.set_xticks(x, AXIS_NAMES)
    residual_axis.set_ylabel("Peak absolute actuation residual")
    residual_axis.set_xlabel("Body wrench axis")
    residual_axis.legend(fontsize=8)
    residual_axis.grid(True, axis="y", alpha=0.3)

    for axis in (deviation_axis, yaw_axis, tilt_axis):
        axis.axvline(FAULT_TIME, color="black", linestyle="--", alpha=0.7)
        axis.set_xlabel("Time (s)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    deviation_axis.set_ylabel("Trajectory deviation (m)")
    yaw_axis.set_ylabel("Yaw deviation from nominal (deg)")
    tilt_axis.set_ylabel("Roll/pitch deviation norm (deg)")

    figure.suptitle("Six-thruster fault signatures on the 6-DOF vehicle")
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
        default=PROJECT_ROOT / "results" / "six_dof_fault_coverage",
    )
    args = parser.parse_args()

    results = run_scenarios(args.duration, args.dt)
    rows, peak_residuals = extract_metrics(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "six_dof_fault_coverage_summary.csv"
    plot_path = args.output_dir / "six_dof_fault_coverage.png"
    save_csv(rows, csv_path)
    save_plot(results, peak_residuals, plot_path)

    for row in rows:
        print(row)
    print(f"Summary: {csv_path}")
    print(f"Figure: {plot_path}")


if __name__ == "__main__":
    main()
