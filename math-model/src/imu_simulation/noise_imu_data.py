from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np


def load_imu_csv(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    return raw


def add_noise(
    accelerations: np.ndarray,
    angular_velocity: np.ndarray,
    gaussian_std_acc: float,
    gaussian_std_gyro: float,
    walk_std_acc: float,
    walk_std_gyro: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    acc_gauss = rng.normal(0.0, gaussian_std_acc, size=accelerations.shape)
    gyro_gauss = rng.normal(0.0, gaussian_std_gyro, size=angular_velocity.shape)

    acc_walk = np.cumsum(rng.normal(0.0, walk_std_acc, size=accelerations.shape), axis=0)
    gyro_walk = np.cumsum(rng.normal(0.0, walk_std_gyro, size=angular_velocity.shape), axis=0)

    return accelerations + acc_gauss + acc_walk, angular_velocity + gyro_gauss + gyro_walk


def write_noised_csv(path: Path, data: np.ndarray) -> None:
    header = "time,x,y,z,ax,ay,az,wx,wy,wz"
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.6f")


def process_files(
    input_dir: Path,
    output_dir: Path,
    gaussian_std_acc: float,
    gaussian_std_gyro: float,
    walk_std_acc: float,
    walk_std_gyro: float,
    seed: int | None,
) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(input_dir.glob("IMU_*.csv")):
        raw = load_imu_csv(csv_path)
        if raw.ndim != 2 or raw.shape[1] < 10:
            raise ValueError(f"Unexpected CSV shape for {csv_path}")
        accelerations = raw[:, 4:7]
        angular_velocity = raw[:, 7:10]
        acc_noised, gyro_noised = add_noise(
            accelerations,
            angular_velocity,
            gaussian_std_acc,
            gaussian_std_gyro,
            walk_std_acc,
            walk_std_gyro,
            rng,
        )
        noised = raw.copy()
        noised[:, 4:7] = acc_noised
        noised[:, 7:10] = gyro_noised

        target = output_dir / f"{csv_path.stem}_Noised{csv_path.suffix}"
        write_noised_csv(target, noised)
        print(f"Wrote {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrupt IMU CSV data with configurable noise.")
    parser.add_argument("--input-dir", type=Path, default=Path(""), help="Directory holding IMU_*.csv files")
    parser.add_argument("--output-dir", type=Path, default=Path("./noised"), help="Directory to write noised files")
    parser.add_argument("--gaussian-std-acc", type=float, default=0.01, help="Std dev for Gaussian noise on acceleration (m/s^2)")
    parser.add_argument("--gaussian-std-gyro", type=float, default=0.005, help="Std dev for Gaussian noise on angular velocity (rad/s)")
    parser.add_argument("--walk-std-acc", type=float, default=0.001, help="Std dev for random walk increments on acceleration")
    parser.add_argument("--walk-std-gyro", type=float, default=0.0005, help="Std dev for random walk increments on angular velocity")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    process_files(
        args.input_dir,
        args.output_dir,
        args.gaussian_std_acc,
        args.gaussian_std_gyro,
        args.walk_std_acc,
        args.walk_std_gyro,
        args.seed,
    )


if __name__ == "__main__":
    main()
