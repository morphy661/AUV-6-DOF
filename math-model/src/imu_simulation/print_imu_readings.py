from pathlib import Path

import numpy as np

from .imu_pair import IMUPairSimulator
from .trajectory import TrajectoryConfig, WobbleTrajectory


def _write_sensor_csv(
    path: Path,
    time: np.ndarray,
    positions: np.ndarray,
    accelerations: np.ndarray,
    angular_velocity: np.ndarray,
    angular_acceleration: np.ndarray | None = None,
) -> None:
    """Write sensor data to CSV file.
    
    If angular_acceleration is provided, it will be included as additional columns.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if angular_acceleration is not None:
        data = np.column_stack(
            (time, positions, accelerations, angular_velocity, angular_acceleration)
        )
        header = "time,x,y,z,ax,ay,az,wx,wy,wz,alphax,alphay,alphaz"
    else:
        data = np.column_stack(
            (time, positions, accelerations, angular_velocity)
        )
        header = "time,x,y,z,ax,ay,az,wx,wy,wz"
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.6f")

# add buffer of zeros at the beginning of the data, default 1 second
def _prepend_buffer(
    time: np.ndarray,
    positions: np.ndarray,
    accelerations: np.ndarray,
    angular_velocity: np.ndarray,
    angular_acceleration: np.ndarray | None,
    buffer_duration: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if buffer_duration <= 0 or dt <= 0:
        return time, positions, accelerations, angular_velocity, angular_acceleration
    pad_samples = int(np.ceil(buffer_duration / dt))
    if pad_samples == 0:
        return time, positions, accelerations, angular_velocity, angular_acceleration

    pad_time = np.arange(-pad_samples, 0) * dt
    pad_data = np.zeros((pad_samples, 3))
    initial_position = positions[0, :]
    pad_positions = np.tile(initial_position, (pad_samples, 1))
    padded_time = np.concatenate((pad_time, time))
    padded_positions = np.vstack((pad_positions, positions))
    padded_accelerations = np.vstack((pad_data, accelerations))
    padded_angular_velocity = np.vstack((pad_data, angular_velocity))
    
    if angular_acceleration is not None:
        padded_angular_acceleration = np.vstack((pad_data, angular_acceleration))
    else:
        padded_angular_acceleration = None
    
    return padded_time, padded_positions, padded_accelerations, padded_angular_velocity, padded_angular_acceleration


def print_simulated_imu_data(
    config: TrajectoryConfig,
    imu_offset: float,
    output_dir: Path | str = ".",
    buffer_duration: float = 1.0,
    include_angular_acceleration: bool = True,
) -> None:
    """Generate simulated IMU data and save to CSV files.
    
    Args:
        config: Trajectory configuration
        imu_offset: Distance from center to each IMU (m)
        output_dir: Output directory for CSV files
        buffer_duration: Duration of zero-motion buffer at start (s)
        include_angular_acceleration: If True, include angular acceleration columns
    """
    trajectory = WobbleTrajectory(config).generate()
    simulator = IMUPairSimulator(imu_offset=imu_offset)
    results = simulator.simulate(trajectory)

    # Get angular acceleration if requested
    ang_acc = trajectory.angular_acceleration if include_angular_acceleration else None

    center_time, center_positions, center_acc, center_angv, center_ang_acc = _prepend_buffer(
        results.time,
        results.center.positions,
        results.center.accelerations,
        results.center.angular_velocity,
        ang_acc,
        buffer_duration,
        config.dt,
    )
    imu_l_time, imu_l_positions, imu_l_acc, imu_l_angv, imu_l_ang_acc = _prepend_buffer(
        results.time,
        results.imus[0].positions,
        results.imus[0].accelerations,
        results.imus[0].angular_velocity,
        ang_acc,  # Same angular acceleration for all rigid body points
        buffer_duration,
        config.dt,
    )
    imu_r_time, imu_r_positions, imu_r_acc, imu_r_angv, imu_r_ang_acc = _prepend_buffer(
        results.time,
        results.imus[1].positions,
        results.imus[1].accelerations,
        results.imus[1].angular_velocity,
        ang_acc,  # Same angular acceleration for all rigid body points
        buffer_duration,
        config.dt,
    )

    base_dir = Path(output_dir)
    _write_sensor_csv(base_dir / "IMU_C.csv", center_time, center_positions, center_acc, center_angv, center_ang_acc)
    _write_sensor_csv(base_dir / "IMU_L.csv", imu_l_time, imu_l_positions, imu_l_acc, imu_l_angv, imu_l_ang_acc)
    _write_sensor_csv(base_dir / "IMU_R.csv", imu_r_time, imu_r_positions, imu_r_acc, imu_r_angv, imu_r_ang_acc)

# Using TrajectoryConfig and IMUPairSimulator from the other files
if __name__ == "__main__":
    config = TrajectoryConfig(
        total_time=20.0,
        dt=0.01,
        nominal_acceleration=0.5,
        spiral_radius=5.0,
        spiral_rate=0.2,
        spiral_vertical_gain=0.2,
        fade_in_duration=2.0,
        wobble_roll_deg=15.0,
        wobble_pitch_deg=5.0,
        wobble_yaw_deg=0.0,
        descent_angle_deg=-10.0,
        path_length=50.0,
    )
    print_simulated_imu_data(config=config, imu_offset=0.5)