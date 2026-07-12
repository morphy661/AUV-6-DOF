"""Compare nominal, no-output, and thrust-loss six-DOF missions."""

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
FAULTED_THRUSTER = "V1"
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


def build_scenarios():
    return {
        "Normal": None,
        "No Output": SingleThrusterFault(
            FAULTED_THRUSTER,
            SixDOFThrusterFaultMode.NO_OUTPUT,
            start_time=FAULT_TIME,
        ),
        "Thrust Loss": SingleThrusterFault(
            FAULTED_THRUSTER,
            SixDOFThrusterFaultMode.THRUST_LOSS,
            start_time=FAULT_TIME,
            thrust_efficiency=0.45,
        ),
    }


def run_scenarios(duration, dt):
    results = {}
    for name, fault in build_scenarios().items():
        simulator = SixDOFSimulator(fault=fault)
        logs = simulator.run(duration, dt, target_provider)
        results[name] = logs
    return results


def summarize(results):
    rows = []
    nominal_positions = np.array([
        log["position_ned"] for log in results["Normal"]
    ])
    for name, logs in results.items():
        after_mask = np.array([log["time"] >= FAULT_TIME for log in logs])
        after = [log for log, keep in zip(logs, after_mask) if keep]
        positions = np.array([log["position_ned"] for log in logs])
        trajectory_deviation = np.linalg.norm(
            positions - nominal_positions,
            axis=1,
        )[after_mask]
        position_errors = [
            np.linalg.norm(log["target_position_ned"] - log["position_ned"])
            for log in after
        ]
        tilt_deg = [
            np.rad2deg(np.linalg.norm(log["euler_rpy"][:2])) for log in after
        ]
        residuals = [
            np.linalg.norm(log["actuation_residual_body"]) for log in after
        ]
        rows.append({
            "scenario": name,
            "max_position_error_m": max(position_errors),
            "final_position_error_m": position_errors[-1],
            "max_trajectory_deviation_m": max(trajectory_deviation),
            "rms_trajectory_deviation_m": float(
                np.sqrt(np.mean(trajectory_deviation ** 2))
            ),
            "max_tilt_deg": max(tilt_deg),
            "max_actuation_residual": max(residuals),
        })
    return rows


def save_summary(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(results, path):
    colors = {"Normal": "#2468A2", "No Output": "#C73E1D", "Thrust Loss": "#7A5195"}
    nominal_positions = np.array([
        log["position_ned"] for log in results["Normal"]
    ])
    figure = plt.figure(figsize=(13, 9))
    trajectory_axis = figure.add_subplot(2, 2, 1, projection="3d")
    error_axis = figure.add_subplot(2, 2, 2)
    tilt_axis = figure.add_subplot(2, 2, 3)
    force_axis = figure.add_subplot(2, 2, 4)

    for name, logs in results.items():
        times = np.array([log["time"] for log in logs])
        positions = np.array([log["position_ned"] for log in logs])
        targets = np.array([log["target_position_ned"] for log in logs])
        attitudes = np.array([log["euler_rpy"] for log in logs])
        trajectory_deviation = np.linalg.norm(
            positions - nominal_positions,
            axis=1,
        )
        color = colors[name]

        trajectory_axis.plot(
            positions[:, 0], positions[:, 1], positions[:, 2],
            color=color, label=name,
        )
        error_axis.plot(times, trajectory_deviation, color=color, label=name)
        tilt_axis.plot(
            times, np.rad2deg(attitudes[:, 0]), color=color,
            label=f"{name} roll",
        )
        tilt_axis.plot(
            times, np.rad2deg(attitudes[:, 1]), color=color,
            linestyle="--", label=f"{name} pitch",
        )

        if name != "Normal":
            commanded = np.array([
                log["commanded_thruster_forces"][4] for log in logs
            ])
            actual = np.array([
                log["actual_thruster_forces"][4] for log in logs
            ])
            force_axis.plot(times, commanded, color=color, linestyle=":")
            force_axis.plot(times, actual, color=color, label=f"{name} actual")

    target_trace = np.array([
        target_provider(time_s, None).position_ned
        for time_s in np.linspace(0.0, max(log["time"] for log in next(iter(results.values()))), 500)
    ])
    trajectory_axis.plot(
        target_trace[:, 0], target_trace[:, 1], target_trace[:, 2],
        color="black", linestyle="--", label="Target",
    )
    trajectory_axis.set_xlabel("North (m)")
    trajectory_axis.set_ylabel("East (m)")
    trajectory_axis.set_zlabel("Depth (m)")
    trajectory_axis.invert_zaxis()
    trajectory_axis.legend()

    for axis in (error_axis, tilt_axis, force_axis):
        axis.axvline(FAULT_TIME, color="black", linestyle="--", alpha=0.7)
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("Time (s)")
        axis.legend(fontsize=8)
    error_axis.set_ylabel("Deviation from nominal trajectory (m)")
    tilt_axis.set_ylabel("Attitude (deg)")
    force_axis.set_ylabel(f"{FAULTED_THRUSTER} force (N)")
    force_axis.set_title("Dotted: commanded; solid: actual")

    figure.suptitle(
        f"Single-thruster fault response ({FAULTED_THRUSTER}, fault at {FAULT_TIME:.0f} s)"
    )
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
        default=PROJECT_ROOT / "results" / "six_dof_thruster_faults",
    )
    args = parser.parse_args()

    results = run_scenarios(args.duration, args.dt)
    rows = summarize(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "six_dof_thruster_fault_summary.csv"
    plot_path = args.output_dir / "six_dof_thruster_fault_comparison.png"
    save_summary(rows, summary_path)
    save_plot(results, plot_path)

    for row in rows:
        print(row)
    print(f"Summary: {summary_path}")
    print(f"Figure: {plot_path}")


if __name__ == "__main__":
    main()
