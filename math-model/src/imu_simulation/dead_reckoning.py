from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt


def load_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    time = raw[:, 0]
    positions = raw[:, 1:4]
    accelerations = raw[:, 4:7]
    return time, positions, accelerations


def integrate_midpoint(
    time: np.ndarray,
    accelerations: np.ndarray,
    initial_position: np.ndarray | None = None,
) -> np.ndarray:
    n = len(time)
    velocities = np.zeros((n, 3))
    for idx in range(1, n):
        dt = time[idx] - time[idx - 1]
        if dt <= 0:
            continue
        acc_avg = 0.5 * (accelerations[idx - 1] + accelerations[idx])
        velocities[idx] = velocities[idx - 1] + acc_avg * dt

    positions = np.zeros((n, 3))
    for idx in range(1, n):
        dt = time[idx] - time[idx - 1]
        if dt <= 0:
            continue
        vel_avg = 0.5 * (velocities[idx - 1] + velocities[idx])
        positions[idx] = positions[idx - 1] + vel_avg * dt

    if initial_position is not None:
        positions += initial_position
    return positions


def plot_comparison(time: np.ndarray, truth: np.ndarray, estimate: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["x", "y", "z"]

    for axis in range(3):
        axes[0].plot(time, truth[:, axis], label=f"truth {labels[axis]}", color=colors[axis], linewidth=1.5)
        axes[0].plot(
            time,
            estimate[:, axis],
            label=f"estimate {labels[axis]}",
            color=colors[axis],
            linestyle="--",
        )
    axes[0].set_title("Time series comparison")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("position (m)")
    axes[0].legend(fontsize="small")
    axes[0].grid(True)

    axes[1].plot(truth[:, 0], truth[:, 1], label="truth XY", color="tab:red")
    axes[1].plot(estimate[:, 0], estimate[:, 1], label="estimate XY", color="tab:purple", linestyle="--")
    axes[1].set_title("XY trajectory")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].legend(fontsize="small")
    axes[1].axis("equal")
    axes[1].grid(True)

    fig.suptitle("IMU-derived dead-reckoning vs reference")
    fig.savefig(output_path)
    print(f"Saved comparison plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dead-reckoned path with ground truth data.")
    parser.add_argument("--imu", type=Path, default=Path("IMU_C.csv"), help="Path to the center IMU CSV file")
    parser.add_argument("--imu-noised", type=Path, default=None, help="Optional path to a noised center IMU CSV file to use for acceleration")
    parser.add_argument("--output", type=Path, default=Path("trajectory_comparison.png"), help="Path to write the comparison figure")
    args = parser.parse_args()

    truth_time, positions, truth_acc = load_imu_csv(args.imu)
    acc_time = truth_time
    accelerations = truth_acc

    if args.imu_noised is not None:
        noised_time, _, noised_acc = load_imu_csv(args.imu_noised)
        if len(noised_time) != len(truth_time) or not np.allclose(noised_time, truth_time):
            raise ValueError("Noised IMU file must share the same time base as the truth file")
        accelerations = noised_acc
        acc_time = noised_time

    estimate = integrate_midpoint(
        acc_time,
        accelerations,
        initial_position=positions[0],
    )
    plot_comparison(acc_time, positions, estimate, args.output)


if __name__ == "__main__":
    main()
