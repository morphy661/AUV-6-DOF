import argparse
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from .imu_pair import IMUPairSimulator
from .trajectory import TrajectoryConfig, WobbleTrajectory


def _plot_trajectory(center_positions: np.ndarray, imu_positions: np.ndarray) -> None:
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(center_positions[:, 0], center_positions[:, 1], center_positions[:, 2], c="tab:blue", label="Center")
    ax.plot(imu_positions[:, 0, 0], imu_positions[:, 0, 1], imu_positions[:, 0, 2], c="tab:orange", label="IMU A")
    ax.plot(imu_positions[:, 1, 0], imu_positions[:, 1, 1], imu_positions[:, 1, 2], c="tab:green", label="IMU B")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")  # type: ignore[attr-defined]
    ax.set_title("Center and IMU trajectories")
    ax.legend()
    ax.grid(True)

    ax2 = fig.add_subplot(122)
    ax2.plot(center_positions[:, 0], center_positions[:, 2], c="tab:blue")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_title("Downward profile")
    ax2.grid(True)


def _plot_time_series(time: np.ndarray, data: np.ndarray, labels: Sequence[str], title: str, ylabel: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for idx, label in enumerate(labels):
        ax.plot(time, data[:, idx], label=label)
    ax.set_xlabel("Time (s)")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


def run_demo(config: TrajectoryConfig, imu_offset: float) -> None:
    trajectory = WobbleTrajectory(config).generate()
    simulator = IMUPairSimulator(imu_offset=imu_offset)
    results = simulator.simulate(trajectory)

    imu_positions = np.stack([results.imus[0].positions, results.imus[1].positions], axis=1)
    _plot_trajectory(results.center.positions, imu_positions)

    euler = R.from_quat(results.center.orientations).as_euler("xyz", degrees=True)
    _plot_time_series(results.time, euler, ["roll", "pitch", "yaw"], "Center orientation", ylabel="deg")
    _plot_time_series(
        results.time,
        results.center.angular_velocity,
        ["ωx", "ωy", "ωz"],
        "Angular velocity",
        ylabel="rad/s",
    )
    _plot_time_series(
        results.time,
        results.center.angular_acceleration,
        ["αx", "αy", "αz"],
        "Angular acceleration",
        ylabel="rad/s²",
    )
    _plot_time_series(
        results.time,
        results.center.velocities,
        ["vx", "vy", "vz"],
        "Center linear velocity",
        ylabel="m/s",
    )
    _plot_time_series(
        results.time,
        results.center.accelerations,
        ["ax", "ay", "az"],
        "Center linear acceleration",
        ylabel="m/s²",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual IMU spiral trajectory demo")
    parser.add_argument("--total-time", type=float, default=20.0, help="Trajectory duration")
    parser.add_argument("--dt", type=float, default=0.01, help="Integration step")
    parser.add_argument("--imu-offset", type=float, default=0.5, help="Offset from center to each IMU")
    parser.add_argument("--nominal-acc", type=float, default=0.5, help="Path parameter acceleration")
    parser.add_argument("--spiral-radius", type=float, default=5.0, help="Spiral radius")
    parser.add_argument("--spiral-rate", type=float, default=0.2, help="Spiral angular rate")
    parser.add_argument("--spiral-vertical-gain", type=float, default=0.2, help="Spiral climb rate")
    parser.add_argument("--fade-in", type=float, default=2.0, help="Wobble fade-in duration")
    parser.add_argument("--wobble-roll", type=float, default=15.0, help="Roll wobble amplitude (deg)")
    parser.add_argument("--wobble-pitch", type=float, default=5.0, help="Pitch wobble amplitude (deg)")
    parser.add_argument("--wobble-yaw", type=float, default=0.0, help="Yaw wobble amplitude (deg)")
    parser.add_argument("--descent-angle", type=float, default=10.0, help="Descent angle (deg)")
    parser.add_argument("--path-length", type=float, default=100.0, help="Path length")
    args = parser.parse_args()

    config = TrajectoryConfig(
        total_time=args.total_time,
        dt=args.dt,
        nominal_acceleration=args.nominal_acc,
        spiral_radius=args.spiral_radius,
        spiral_rate=args.spiral_rate,
        spiral_vertical_gain=args.spiral_vertical_gain,
        fade_in_duration=args.fade_in,
        wobble_roll_deg=args.wobble_roll,
        wobble_pitch_deg=args.wobble_pitch,
        wobble_yaw_deg=args.wobble_yaw,
        descent_angle_deg=args.descent_angle,
        path_length=args.path_length,
    )
    run_demo(config=config, imu_offset=args.imu_offset)
