"""
Error-State Kalman Filter (ESKF) for single IMU navigation.

This implementation uses the error-state formulation where:
- Nominal state: position, velocity, orientation (quaternion)
- Error state: δp, δv, δθ, δba, δbg (15-dimensional)

The ESKF is well-suited for IMU integration because:
1. Error states remain small, improving linearization accuracy
2. Quaternion representation avoids gimbal lock
3. Natural handling of IMU biases
"""
from __future__ import annotations

from pathlib import Path
import argparse
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


@dataclass
class ESKFConfig:
    """Configuration parameters for ESKF."""
    # Process noise standard deviations
    sigma_acc: float = 0.1          # Accelerometer noise (m/s^2)
    sigma_gyro: float = 0.01        # Gyroscope noise (rad/s)
    sigma_acc_bias: float = 0.001   # Accelerometer bias random walk
    sigma_gyro_bias: float = 0.0001 # Gyroscope bias random walk
    
    # Measurement noise standard deviations (for position updates if available)
    sigma_pos: float = 0.1          # Position measurement noise (m)
    
    # ----- GPS Position Update (Surface Mode) -----
    enable_gps_update: bool = False         # Enable GPS position updates when available
    sigma_gps_pos: float = 0.01             # GPS position measurement noise (m)
    gps_cutoff_time: float = 0.0            # Time at which GPS becomes unavailable (e.g., submersion)
    gps_underwater_duration: float = 0.0    # GPS still available for this duration after submersion
    
    # ----- Depth Sensor (Underwater Mode) -----
    enable_depth_update: bool = False       # Enable depth sensor updates when underwater
    sigma_depth: float = 0.1                # Depth sensor measurement noise (m)
    depth_start_time: float = 0.0           # Time at which depth sensor becomes available (submersion)
    
    # ----- Heading Alignment (Hard Reset at Submersion) -----
    enable_heading_alignment: bool = False  # Enable hard reset of orientation at alignment time
    heading_alignment_time: float = 0.0     # Time at which to perform heading alignment
    initial_heading_deg: float = 90.0       # Initial heading in degrees (0=+X, 90=+Y, 180=-X, 270=-Y)
    
    # Initial uncertainty standard deviations
    init_sigma_pos: float = 0.01    # Initial position uncertainty (m)
    init_sigma_vel: float = 0.01    # Initial velocity uncertainty (m/s)
    init_sigma_theta: float = 0.01  # Initial orientation uncertainty (rad)
    init_sigma_acc_bias: float = 0.1   # Initial accel bias uncertainty
    init_sigma_gyro_bias: float = 0.01 # Initial gyro bias uncertainty


@dataclass 
class NominalState:
    """Nominal state for ESKF."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))  # [w, x, y, z]
    acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def copy(self) -> "NominalState":
        return NominalState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            acc_bias=self.acc_bias.copy(),
            gyro_bias=self.gyro_bias.copy(),
        )


class ESKF:
    """Error-State Kalman Filter for IMU navigation."""
    
    # State indices
    POS_IDX = slice(0, 3)    # Position error
    VEL_IDX = slice(3, 6)    # Velocity error
    THETA_IDX = slice(6, 9)  # Orientation error (rotation vector)
    BA_IDX = slice(9, 12)    # Accelerometer bias error
    BG_IDX = slice(12, 15)   # Gyroscope bias error
    STATE_DIM = 15
    
    def __init__(self, config: ESKFConfig | None = None):
        self.config = config or ESKFConfig()
        self.nominal = NominalState()
        
        # Error state covariance (15x15)
        self.P = np.diag([
            self.config.init_sigma_pos**2,
            self.config.init_sigma_pos**2,
            self.config.init_sigma_pos**2,
            self.config.init_sigma_vel**2,
            self.config.init_sigma_vel**2,
            self.config.init_sigma_vel**2,
            self.config.init_sigma_theta**2,
            self.config.init_sigma_theta**2,
            self.config.init_sigma_theta**2,
            self.config.init_sigma_acc_bias**2,
            self.config.init_sigma_acc_bias**2,
            self.config.init_sigma_acc_bias**2,
            self.config.init_sigma_gyro_bias**2,
            self.config.init_sigma_gyro_bias**2,
            self.config.init_sigma_gyro_bias**2,
        ])
        
        # Process noise covariance
        self._build_process_noise()
    
    def _build_process_noise(self) -> None:
        """Build the continuous-time process noise matrix."""
        self.Q_c = np.diag([
            0, 0, 0,  # position (driven by velocity)
            self.config.sigma_acc**2,
            self.config.sigma_acc**2,
            self.config.sigma_acc**2,
            self.config.sigma_gyro**2,
            self.config.sigma_gyro**2,
            self.config.sigma_gyro**2,
            self.config.sigma_acc_bias**2,
            self.config.sigma_acc_bias**2,
            self.config.sigma_acc_bias**2,
            self.config.sigma_gyro_bias**2,
            self.config.sigma_gyro_bias**2,
            self.config.sigma_gyro_bias**2,
        ])
    
    def initialize(
        self,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
        quaternion: np.ndarray | None = None,
    ) -> None:
        """Initialize the filter with known initial state."""
        self.nominal.position = position.copy()
        if velocity is not None:
            self.nominal.velocity = velocity.copy()
        if quaternion is not None:
            self.nominal.quaternion = quaternion.copy()
    
    @staticmethod
    def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)],
        ])
    
    @staticmethod
    def skew_symmetric(v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix from vector."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
    
    @staticmethod
    def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
    
    @staticmethod
    def rotation_vector_to_quaternion(rv: np.ndarray) -> np.ndarray:
        """Convert small rotation vector to quaternion."""
        angle = np.linalg.norm(rv)
        if angle < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = rv / angle
        half_angle = angle / 2
        return np.array([
            np.cos(half_angle),
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle),
        ])
    
    def predict(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> None:
        """
        Prediction step: propagate nominal state and error covariance.
        
        Args:
            acc: Measured acceleration in body frame (m/s^2)
            gyro: Measured angular velocity in body frame (rad/s)
            dt: Time step (s)
        """
        if dt <= 0:
            return
        
        # Correct measurements with current bias estimates
        acc_corrected = acc - self.nominal.acc_bias
        gyro_corrected = gyro - self.nominal.gyro_bias
        
        # Get rotation matrix from body to world
        R = self.quaternion_to_rotation_matrix(self.nominal.quaternion)
        
        # Propagate nominal state
        # Position: p_k+1 = p_k + v_k * dt + 0.5 * R * a * dt^2
        acc_world = R @ acc_corrected
        self.nominal.position = (
            self.nominal.position 
            + self.nominal.velocity * dt 
            + 0.5 * acc_world * dt**2
        )
        
        # Velocity: v_k+1 = v_k + R * a * dt
        self.nominal.velocity = self.nominal.velocity + acc_world * dt
        
        # Orientation: q_k+1 = q_k ⊗ δq(ω * dt)
        delta_theta = gyro_corrected * dt
        delta_q = self.rotation_vector_to_quaternion(delta_theta)
        self.nominal.quaternion = self.quaternion_multiply(self.nominal.quaternion, delta_q)
        self.nominal.quaternion /= np.linalg.norm(self.nominal.quaternion)  # Normalize
        
        # Biases remain constant in prediction
        # (random walk is modeled in process noise)
        
        # Build state transition matrix F (Jacobian of error dynamics)
        F = np.eye(self.STATE_DIM)
        
        # ∂δp/∂δv
        F[self.POS_IDX, self.VEL_IDX] = np.eye(3) * dt
        
        # ∂δv/∂δθ (cross product with acceleration)
        F[self.VEL_IDX, self.THETA_IDX] = -R @ self.skew_symmetric(acc_corrected) * dt
        
        # ∂δv/∂δba
        F[self.VEL_IDX, self.BA_IDX] = -R * dt
        
        # ∂δθ/∂δbg
        F[self.THETA_IDX, self.BG_IDX] = -np.eye(3) * dt
        
        # Process noise covariance for this time step
        Q = self.Q_c * dt
        
        # Propagate error covariance
        self.P = F @ self.P @ F.T + Q
        
        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)
    
    def update_position(self, measured_position: np.ndarray) -> None:
        """
        Measurement update with position observation.
        
        Args:
            measured_position: Measured position in world frame (m)
        """
        # Measurement model: z = p + δp
        # Innovation
        y = measured_position - self.nominal.position
        
        # Measurement Jacobian H (only position part)
        H = np.zeros((3, self.STATE_DIM))
        H[:, self.POS_IDX] = np.eye(3)
        
        # Measurement noise covariance
        R = np.eye(3) * self.config.sigma_pos**2
        
        # Kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Error state update
        delta_x = K @ y
        
        # Inject error into nominal state
        self._inject_error(delta_x)
        
        # Update covariance
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form for numerical stability
        self.P = 0.5 * (self.P + self.P.T)
    
    def _inject_error(self, delta_x: np.ndarray) -> None:
        """Inject error state into nominal state and reset error."""
        # Position and velocity are additive
        self.nominal.position += delta_x[self.POS_IDX]
        self.nominal.velocity += delta_x[self.VEL_IDX]
        
        # Orientation: q ← q ⊗ δq(δθ)
        delta_theta = delta_x[self.THETA_IDX]
        delta_q = self.rotation_vector_to_quaternion(delta_theta)
        self.nominal.quaternion = self.quaternion_multiply(self.nominal.quaternion, delta_q)
        self.nominal.quaternion /= np.linalg.norm(self.nominal.quaternion)
        
        # Biases are additive
        self.nominal.acc_bias += delta_x[self.BA_IDX]
        self.nominal.gyro_bias += delta_x[self.BG_IDX]
    
    def get_position(self) -> np.ndarray:
        """Get current estimated position."""
        return self.nominal.position.copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current estimated velocity."""
        return self.nominal.velocity.copy()
    
    def get_euler_angles(self) -> np.ndarray:
        """Get current estimated orientation as Euler angles (roll, pitch, yaw)."""
        q = self.nominal.quaternion
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x, y, z, w]
        return r.as_euler('xyz')
    
    def update_with_gps(self, gps_position: np.ndarray) -> dict:
        """
        Update with GPS position measurement.
        
        Args:
            gps_position: GPS measured position [x, y, z] in world frame
            
        Returns:
            Diagnostic information
        """
        # Innovation (measurement residual)
        y = gps_position - self.nominal.position
        
        # Measurement Jacobian (3x15): H = [I_3, 0, 0, 0, 0]
        H = np.zeros((3, self.STATE_DIM))
        H[:, self.POS_IDX] = np.eye(3)
        
        # Measurement noise covariance
        R = np.eye(3) * self.config.sigma_gps_pos**2
        
        # Kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Error state update
        delta_x = K @ y
        
        # Inject error into nominal state
        self._inject_error(delta_x)
        
        # Update covariance (Joseph form)
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        return {"gps_residual": y, "gps_innovation_norm": np.linalg.norm(y)}
    
    def update_with_depth(self, depth: float) -> dict:
        """
        Update with depth sensor measurement.
        
        The depth sensor measures only the Z coordinate (depth).
        
        Args:
            depth: Measured depth (Z coordinate) in world frame
            
        Returns:
            Diagnostic information
        """
        # Innovation (measurement residual) - only Z component
        y = np.array([depth - self.nominal.position[2]])
        
        # Measurement Jacobian (1x15): H = [0, 0, 1, 0, ..., 0]
        H = np.zeros((1, self.STATE_DIM))
        H[0, 2] = 1.0  # Only observes Z position
        
        # Measurement noise covariance
        R = np.array([[self.config.sigma_depth**2]])
        
        # Kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Error state update
        delta_x = K @ y
        
        # Inject error into nominal state
        self._inject_error(delta_x)
        
        # Update covariance (Joseph form)
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        return {"depth_residual": y[0]}
    
    def reset_orientation(self, true_quaternion: np.ndarray) -> None:
        """
        Hard reset orientation to ground truth (heading alignment).
        
        Args:
            true_quaternion: Ground truth quaternion [w, x, y, z]
        """
        # Normalize quaternion
        q = true_quaternion / np.linalg.norm(true_quaternion)
        
        # Hard reset nominal quaternion
        self.nominal.quaternion = q.copy()
        
        # Reset orientation error covariance to small value
        small_sigma = 0.001
        self.P[self.THETA_IDX, self.THETA_IDX] = np.eye(3) * small_sigma**2
        
        # Zero out cross-correlations with orientation
        for idx in [self.POS_IDX, self.VEL_IDX, self.BA_IDX, self.BG_IDX]:
            self.P[self.THETA_IDX, idx] = 0
            self.P[idx, self.THETA_IDX] = 0


def simulate_depth_sensor(
    true_positions: np.ndarray,
    sigma_depth: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate depth sensor measurements with Gaussian noise.
    
    Args:
        true_positions: Ground truth positions (n, 3)
        sigma_depth: Depth measurement noise standard deviation (m)
        seed: Random seed for reproducibility
    
    Returns:
        depth_measurements: Simulated depth measurements (n,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    true_depth = true_positions[:, 2]  # Z coordinate
    noise = np.random.normal(0, sigma_depth, size=true_depth.shape)
    
    return true_depth + noise


def load_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load IMU data from CSV file.
    
    Returns:
        time: Timestamps
        positions: Ground truth positions (x, y, z)
        accelerations: Linear accelerations (ax, ay, az)
        angular_velocities: Angular velocities (wx, wy, wz)
    """
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    time = raw[:, 0]
    positions = raw[:, 1:4]
    accelerations = raw[:, 4:7]
    angular_velocities = raw[:, 7:10]
    return time, positions, accelerations, angular_velocities


def run_eskf(
    time: np.ndarray,
    accelerations: np.ndarray,
    angular_velocities: np.ndarray,
    initial_position: np.ndarray,
    config: ESKFConfig | None = None,
    position_updates: np.ndarray | None = None,
    update_interval: int = 100,
    gps_positions: np.ndarray | None = None,
    depth_measurements: np.ndarray | None = None,
) -> np.ndarray:
    """
    Run ESKF on IMU data.
    
    Args:
        time: Timestamps
        accelerations: Accelerometer measurements
        angular_velocities: Gyroscope measurements
        initial_position: Initial position
        config: ESKF configuration
        position_updates: Optional position measurements for updates
        update_interval: How often to apply position updates (in samples)
        gps_positions: GPS position data for surface updates
        depth_measurements: Depth sensor data for underwater updates
    
    Returns:
        Estimated positions over time
    """
    eskf = ESKF(config)
    eskf.initialize(position=initial_position)
    
    n = len(time)
    positions = np.zeros((n, 3))
    positions[0] = initial_position.copy()
    
    # GPS configuration
    gps_cutoff = (config.gps_cutoff_time + config.gps_underwater_duration) if config else 0.0
    use_gps = config.enable_gps_update if config else False
    
    # Depth sensor configuration
    depth_start = config.depth_start_time if config else 0.0
    use_depth = config.enable_depth_update if config else False
    
    # Heading alignment configuration
    heading_alignment_time = config.heading_alignment_time if config else 0.0
    use_heading_alignment = config.enable_heading_alignment if config else False
    heading_aligned = False
    
    gps_update_count = 0
    depth_update_count = 0
    
    for idx in range(1, n):
        dt = time[idx] - time[idx - 1]
        current_time = time[idx]
        
        # Prediction step
        eskf.predict(
            acc=accelerations[idx],
            gyro=angular_velocities[idx],
            dt=dt,
        )
        
        # Heading alignment at specified time (hard reset, done only once)
        if use_heading_alignment and not heading_aligned:
            prev_time = time[idx - 1]
            if prev_time < heading_alignment_time <= current_time:
                heading_rad = np.deg2rad(config.initial_heading_deg)
                heading_q = np.array([
                    np.cos(heading_rad / 2),
                    0.0,
                    0.0,
                    np.sin(heading_rad / 2)
                ])
                eskf.reset_orientation(heading_q)
                heading_aligned = True
        
        # GPS update (surface mode)
        gps_available = (
            use_gps and 
            gps_positions is not None and 
            current_time < gps_cutoff
        )
        if gps_available:
            eskf.update_with_gps(gps_positions[idx])
            gps_update_count += 1
        
        # Depth sensor update (underwater mode)
        depth_available = (
            use_depth and
            depth_measurements is not None and
            current_time >= depth_start
        )
        if depth_available:
            eskf.update_with_depth(depth_measurements[idx])
            depth_update_count += 1
        
        # Legacy position updates (if using old interface)
        if position_updates is not None and idx % update_interval == 0:
            eskf.update_position(position_updates[idx])
        
        positions[idx] = eskf.get_position()
    
    if use_gps:
        print(f"  GPS updates applied: {gps_update_count} (t < {gps_cutoff}s)")
    if use_depth:
        print(f"  Depth updates applied: {depth_update_count} (t >= {depth_start}s)")
    
    return positions


def plot_comparison(
    time: np.ndarray,
    truth: np.ndarray,
    estimate: np.ndarray,
    output_path: Path,
    title: str = "ESKF-derived trajectory vs reference",
) -> None:
    """Plot comparison between ground truth and estimated trajectory."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["x", "y", "z"]

    # Time series comparison
    for axis in range(3):
        axes[0].plot(
            time, truth[:, axis],
            label=f"truth {labels[axis]}",
            color=colors[axis],
            linewidth=1.5,
        )
        axes[0].plot(
            time, estimate[:, axis],
            label=f"estimate {labels[axis]}",
            color=colors[axis],
            linestyle="--",
        )
    axes[0].set_title("Position time series comparison")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("position (m)")
    axes[0].legend(fontsize="small", ncol=2)
    axes[0].grid(True)

    # XY trajectory
    axes[1].plot(truth[:, 0], truth[:, 1], label="truth XY", color="tab:red", linewidth=2)
    axes[1].plot(
        estimate[:, 0], estimate[:, 1],
        label="estimate XY",
        color="tab:purple",
        linestyle="--",
        linewidth=2,
    )
    axes[1].scatter([truth[0, 0]], [truth[0, 1]], color="green", s=100, zorder=5, label="start")
    axes[1].scatter([truth[-1, 0]], [truth[-1, 1]], color="red", s=100, zorder=5, label="end")
    axes[1].set_title("XY trajectory")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].legend(fontsize="small")
    axes[1].axis("equal")
    axes[1].grid(True)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    print(f"Saved comparison plot to {output_path}")


def compute_rmse(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """Compute RMSE for position estimates."""
    error = estimate - truth
    rmse_xyz = np.sqrt(np.mean(error**2, axis=0))
    rmse_total = np.sqrt(np.mean(np.sum(error**2, axis=1)))
    return {
        "x": rmse_xyz[0],
        "y": rmse_xyz[1],
        "z": rmse_xyz[2],
        "total": rmse_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ESKF-based single IMU navigation."
    )
    parser.add_argument(
        "--imu",
        type=Path,
        default=Path("IMU_C.csv"),
        help="Path to the center IMU CSV file (ground truth)",
    )
    parser.add_argument(
        "--imu-noised",
        type=Path,
        default=None,
        help="Optional path to a noised center IMU CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eskf_trajectory_comparison.png"),
        help="Path to write the comparison figure",
    )
    parser.add_argument(
        "--sigma-acc",
        type=float,
        default=0.1,
        help="Accelerometer noise standard deviation (m/s^2)",
    )
    parser.add_argument(
        "--sigma-gyro",
        type=float,
        default=0.01,
        help="Gyroscope noise standard deviation (rad/s)",
    )
    parser.add_argument(
        "--with-updates",
        action="store_true",
        help="Enable periodic position updates (simulating external measurements)",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=100,
        help="Position update interval (samples)",
    )
    
    # ----- GPS Position Update Options -----
    parser.add_argument(
        "--enable-gps", action="store_true",
        help="Enable GPS position updates when on surface (t < gps-cutoff-time)",
    )
    parser.add_argument(
        "--gps-cutoff-time", type=float, default=0.0,
        help="Time at which GPS becomes unavailable (submersion time)",
    )
    parser.add_argument(
        "--gps-underwater-duration", type=float, default=0.0,
        help="GPS still available for this duration after submersion",
    )
    parser.add_argument(
        "--sigma-gps-pos", type=float, default=0.01,
        help="GPS position measurement noise (m)",
    )
    
    # ----- Depth Sensor Options -----
    parser.add_argument(
        "--enable-depth", action="store_true",
        help="Enable depth sensor updates when underwater (t >= depth-start-time)",
    )
    parser.add_argument(
        "--depth-start-time", type=float, default=0.0,
        help="Time at which depth sensor becomes available",
    )
    parser.add_argument(
        "--sigma-depth", type=float, default=0.1,
        help="Depth sensor measurement noise (m)",
    )
    parser.add_argument(
        "--depth-seed", type=int, default=None,
        help="Random seed for depth sensor noise simulation",
    )
    
    # ----- Heading Alignment Options -----
    parser.add_argument(
        "--enable-heading-alignment", action="store_true",
        help="Enable hard reset of orientation at alignment time",
    )
    parser.add_argument(
        "--heading-alignment-time", type=float, default=0.0,
        help="Time at which to perform heading alignment",
    )
    parser.add_argument(
        "--initial-heading-deg", type=float, default=90.0,
        help="Initial heading in degrees (0=+X, 90=+Y, 180=-X, 270=-Y)",
    )
    
    args = parser.parse_args()

    # Load ground truth data
    truth_time, positions, truth_acc, truth_gyro = load_imu_csv(args.imu)
    accelerations = truth_acc
    angular_velocities = truth_gyro

    # Load noised data if provided
    if args.imu_noised is not None:
        noised_time, _, noised_acc, noised_gyro = load_imu_csv(args.imu_noised)
        if len(noised_time) != len(truth_time) or not np.allclose(noised_time, truth_time):
            raise ValueError("Noised IMU file must share the same time base as the truth file")
        accelerations = noised_acc
        angular_velocities = noised_gyro

    # Simulate depth sensor if enabled
    depth_measurements = None
    if args.enable_depth:
        print(f"Simulating depth sensor with sigma = {args.sigma_depth} m...")
        depth_measurements = simulate_depth_sensor(
            positions,
            sigma_depth=args.sigma_depth,
            seed=args.depth_seed,
        )

    # Configure ESKF
    config = ESKFConfig(
        sigma_acc=args.sigma_acc,
        sigma_gyro=args.sigma_gyro,
        # GPS settings
        enable_gps_update=args.enable_gps,
        sigma_gps_pos=args.sigma_gps_pos,
        gps_cutoff_time=args.gps_cutoff_time,
        gps_underwater_duration=args.gps_underwater_duration,
        # Depth sensor settings
        enable_depth_update=args.enable_depth,
        sigma_depth=args.sigma_depth,
        depth_start_time=args.depth_start_time,
        # Heading alignment settings
        enable_heading_alignment=args.enable_heading_alignment,
        heading_alignment_time=args.heading_alignment_time,
        initial_heading_deg=args.initial_heading_deg,
    )

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  IMU: {args.imu}")
    if args.enable_gps:
        gps_end_time = args.gps_cutoff_time + args.gps_underwater_duration
        if args.gps_underwater_duration > 0:
            print(f"  GPS updates ENABLED for t < {gps_end_time}s (cutoff={args.gps_cutoff_time}s + underwater={args.gps_underwater_duration}s, sigma = {args.sigma_gps_pos} m)")
        else:
            print(f"  GPS updates ENABLED for t < {args.gps_cutoff_time}s (sigma = {args.sigma_gps_pos} m)")
    else:
        print(f"  GPS updates DISABLED")
    if args.enable_depth:
        print(f"  Depth sensor ENABLED for t >= {args.depth_start_time}s (sigma = {args.sigma_depth} m)")
    else:
        print(f"  Depth sensor DISABLED")
    if args.enable_heading_alignment:
        print(f"  Heading alignment ENABLED at t = {args.heading_alignment_time}s (heading = {args.initial_heading_deg}°)")

    # Run ESKF
    print("\nRunning Single IMU ESKF...")
    position_updates = positions if args.with_updates else None
    estimate = run_eskf(
        time=truth_time,
        accelerations=accelerations,
        angular_velocities=angular_velocities,
        initial_position=positions[0],
        config=config,
        position_updates=position_updates,
        update_interval=args.update_interval,
        gps_positions=positions if args.enable_gps else None,  # Use ground truth as GPS
        depth_measurements=depth_measurements,
    )

    # Compute and print RMSE
    rmse = compute_rmse(positions, estimate)
    print(f"\nPosition RMSE:")
    print(f"  x: {rmse['x']:.4f} m")
    print(f"  y: {rmse['y']:.4f} m")
    print(f"  z: {rmse['z']:.4f} m")
    print(f"  total: {rmse['total']:.4f} m")

    # Plot results
    plot_comparison(
        truth_time,
        positions,
        estimate,
        args.output,
        title="ESKF Single IMU Navigation",
    )


if __name__ == "__main__":
    main()
