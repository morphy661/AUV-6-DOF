"""Run and plot a deterministic nominal six-degree-of-freedom mission."""

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

from environment.six_dof_simulator import SixDOFNominalSimulator
from simple_control.six_dof_controller import PoseTarget


MISSION_TARGETS = [
    (0.0, np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.0])),
    (25.0, np.array([6.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.0])),
    (55.0, np.array([6.0, 4.0, 3.0]), np.array([0.0, 0.0, np.pi / 2.0])),
    (85.0, np.array([1.0, 4.0, 2.0]), np.array([0.0, 0.0, np.pi])),
    (115.0, np.array([1.0, 0.0, 1.0]), np.array([0.0, 0.0, -np.pi / 2.0])),
]


def target_provider(time_s, _state):
    selected = MISSION_TARGETS[0]
    for candidate in MISSION_TARGETS:
        if time_s >= candidate[0]:
            selected = candidate
        else:
            break
    return PoseTarget(position_ned=selected[1], euler_rpy=selected[2])


def save_csv(logs, path):
    headers = [
        "time_s", "north_m", "east_m", "depth_m",
        "roll_deg", "pitch_deg", "yaw_deg",
        "target_north_m", "target_east_m", "target_depth_m",
        "allocation_residual_norm", "saturated_thruster_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for log in logs:
            writer.writerow([
                log["time"],
                *log["position_ned"],
                *np.rad2deg(log["euler_rpy"]),
                *log["target_position_ned"],
                np.linalg.norm(log["allocation_residual_body"]),
                int(np.sum(log["thruster_saturated"])),
            ])


def save_plot(logs, path):
    times = np.array([log["time"] for log in logs])
    positions = np.array([log["position_ned"] for log in logs])
    targets = np.array([log["target_position_ned"] for log in logs])
    attitudes = np.array([log["euler_rpy"] for log in logs])
    target_attitudes = np.array([log["target_euler_rpy"] for log in logs])
    attitude_deg = np.rad2deg(attitudes)
    target_attitude_deg = np.rad2deg(target_attitudes)
    attitude_deg[:, 2] = np.rad2deg(np.unwrap(attitudes[:, 2]))
    target_attitude_deg[:, 2] = np.rad2deg(np.unwrap(target_attitudes[:, 2]))

    figure = plt.figure(figsize=(12, 9))
    axis_3d = figure.add_subplot(2, 2, 1, projection="3d")
    axis_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="AUV")
    axis_3d.plot(
        targets[:, 0], targets[:, 1], targets[:, 2],
        linestyle="--", label="Target",
    )
    axis_3d.set_xlabel("North (m)")
    axis_3d.set_ylabel("East (m)")
    axis_3d.set_zlabel("Depth (m)")
    axis_3d.invert_zaxis()
    axis_3d.legend()
    axis_3d.set_title("Nominal 6-DOF trajectory")

    axis_position = figure.add_subplot(2, 2, 2)
    labels = ("North", "East", "Depth")
    for index, label in enumerate(labels):
        axis_position.plot(times, positions[:, index], label=label)
        axis_position.plot(times, targets[:, index], linestyle="--", alpha=0.6)
    axis_position.set_xlabel("Time (s)")
    axis_position.set_ylabel("Position (m)")
    axis_position.grid(True, alpha=0.3)
    axis_position.legend()

    axis_attitude = figure.add_subplot(2, 1, 2)
    labels = ("Roll", "Pitch", "Yaw")
    for index, label in enumerate(labels):
        axis_attitude.plot(times, attitude_deg[:, index], label=label)
        axis_attitude.plot(
            times, target_attitude_deg[:, index], linestyle="--", alpha=0.6
        )
    axis_attitude.set_xlabel("Time (s)")
    axis_attitude.set_ylabel("Attitude (deg)")
    axis_attitude.grid(True, alpha=0.3)
    axis_attitude.legend(ncol=3)

    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=145.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "six_dof_nominal",
    )
    args = parser.parse_args()

    simulator = SixDOFNominalSimulator()
    logs = simulator.run(args.duration, args.dt, target_provider)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "nominal_six_dof_timeseries.csv"
    plot_path = args.output_dir / "nominal_six_dof_validation.png"
    save_csv(logs, csv_path)
    save_plot(logs, plot_path)

    final_target = target_provider(simulator.state.time, simulator.state)
    final_position_error = np.linalg.norm(
        final_target.position_ned - simulator.state.position_ned
    )
    print(f"Final position NED (m): {simulator.state.position_ned}")
    print(f"Final attitude RPY (deg): {np.rad2deg(simulator.state.euler_rpy)}")
    print(f"Final position error (m): {final_position_error:.4f}")
    print(f"CSV: {csv_path}")
    print(f"Figure: {plot_path}")


if __name__ == "__main__":
    main()
