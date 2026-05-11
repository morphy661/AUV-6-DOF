"""
Error-State Kalman Filter (ESKF) for dual IMU navigation with rigid body constraints
and online lever arm (extrinsic) calibration.

This implementation uses:
- Master IMU (e.g., IMU_L) for state prediction
- Slave IMU (e.g., IMU_R) for observation updates via rigid body constraints
- Online estimation of lever arm (extrinsic parameters) between IMUs

State Vector (24-dimensional):
- δp (3): Position error
- δv (3): Velocity error
- δθ (3): Orientation error (rotation vector)
- δba_m (3): Master accelerometer bias error
- δbg_m (3): Master gyroscope bias error
- δba_s (3): Slave accelerometer bias error
- δbg_s (3): Slave gyroscope bias error
- δr (3): Lever arm error (extrinsic calibration)

Rigid Body Constraints:
1. Gyroscope constraint: ω_slave = ω_master (same angular velocity)
2. Accelerometer constraint: a_slave = a_master + α × r + ω × (ω × r)
   where r is the lever arm from master to slave, α is angular acceleration

Observation Model:
- The observation residual is the difference between:
  * Actual slave IMU readings
  * Theoretical slave IMU readings computed from master IMU + rigid body kinematics
  
Jacobian for Lever Arm Estimation:
- H[acc, lever] = ∂h_acc/∂r = [α]× + [ω]×[ω]×
  where [·]× denotes the skew-symmetric matrix

Key Features:
- Online lever arm calibration (no need to know exact IMU placement)
- Observability-aware updates (disables lever arm updates when |ω| is small)
- Outlier rejection using Mahalanobis distance to handle model mismatch
- Numerical stability via Joseph form covariance update and bounds

Note on Data:
- IMU data should be in BODY frame (as measured by real IMUs)
- Angular velocity and accelerations must be consistent with the body frame
- If data is generated in world frame, coordinate transformation is needed
"""
from __future__ import annotations

from pathlib import Path
import argparse
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


# =============================================================================
# Configuration
# =============================================================================

class MasterSlave(Enum):
    """Enum to specify which IMU is master."""
    LEFT_MASTER = "left"   # IMU_L is master, IMU_R is slave
    RIGHT_MASTER = "right" # IMU_R is master, IMU_L is slave


@dataclass
class DualESKFConfig:
    """Configuration parameters for Dual IMU ESKF with online extrinsic calibration."""
    
    # ----- Process Noise -----
    sigma_acc: float = 0.1              # Accelerometer noise (m/s^2)
    sigma_gyro: float = 0.01            # Gyroscope noise (rad/s)
    sigma_acc_bias: float = 0.001       # Accelerometer bias random walk
    sigma_gyro_bias: float = 0.0001     # Gyroscope bias random walk
    sigma_lever_arm: float = 0.0001     # Lever arm random walk (should be very small for rigid body)
    
    # ----- Measurement Noise -----
    sigma_gyro_constraint: float = 0.02     # Gyroscope constraint noise (rad/s)
    sigma_acc_constraint: float = 0.1       # Accelerometer constraint noise (m/s^2)
    
    # ----- Numerical Stability -----
    min_omega_for_lever_obs: float = 0.1    # Minimum |ω| for lever arm observability (rad/s)
    max_lever_arm_variance: float = 10.0    # Maximum lever arm variance (m^2) to prevent explosion
    
    # ----- Outlier Rejection -----
    enable_outlier_rejection: bool = True   # Enable Mahalanobis distance based outlier rejection
    outlier_threshold: float = 5.0          # Chi-squared threshold for outlier detection (5 sigma)
    
    # ----- GPS Position Update (Surface Mode) -----
    enable_gps_update: bool = True          # Enable GPS position updates when available
    sigma_gps_pos: float = 0.01             # GPS position measurement noise (m)
    gps_cutoff_time: float = 0.0            # Time at which GPS becomes unavailable (e.g., submersion)
    gps_underwater_duration: float = 0.0    # GPS still available for this duration after submersion
    
    # ----- Depth Sensor (Underwater Mode) -----
    enable_depth_update: bool = False       # Enable depth sensor updates when underwater
    sigma_depth: float = 0.1                # Depth sensor measurement noise (m)
    depth_start_time: float = 0.0           # Time at which depth sensor becomes available (submersion)
    
    # ----- Heading Alignment (Hard Reset at Submersion) -----
    enable_heading_alignment: bool = False  # Enable hard reset of orientation at alignment time
    heading_alignment_time: float = 0.0     # Time at which to perform heading alignment (typically submersion time)
    initial_heading_deg: float = 90.0       # Initial heading in degrees (0=+X, 90=+Y, 180=-X, 270=-Y)
    
    # ----- Initial Uncertainties -----
    init_sigma_pos: float = 0.01            # Initial position uncertainty (m)
    init_sigma_vel: float = 0.01            # Initial velocity uncertainty (m/s)
    init_sigma_theta: float = 0.01          # Initial orientation uncertainty (rad)
    init_sigma_acc_bias: float = 0.1        # Initial accel bias uncertainty
    init_sigma_gyro_bias: float = 0.01      # Initial gyro bias uncertainty
    init_sigma_lever_arm: float = 1.0       # Initial lever arm uncertainty (m) - LARGE for online estimation
    
    # ----- Lever Arm Initial Guess -----
    # Initial guess for lever arm from master to slave IMU (in body frame)
    # Set to zero if completely unknown, or to approximate value if known
    lever_arm_init: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    
    # Known lever arm magnitude (distance between IMUs)
    # Set to None if unknown, otherwise enforces HARD constraint on magnitude
    # Only the direction is estimated, magnitude is forced to this value
    lever_arm_magnitude: float | None = None
    
    # Enable magnitude constraint enforcement
    enforce_magnitude_constraint: bool = True
    
    # ----- Master/Slave Configuration -----
    master_slave: MasterSlave = MasterSlave.LEFT_MASTER
    
    # ----- Angular Acceleration -----
    use_gyro_diff_for_alpha: bool = True  # Use gyro differentiation for angular acceleration


# =============================================================================
# Nominal State
# =============================================================================

@dataclass 
class NominalState:
    """Nominal state for ESKF with extrinsic calibration."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))  # [w, x, y, z]
    acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))       # Master IMU
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))      # Master IMU
    slave_acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3)) # Slave IMU
    slave_gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))# Slave IMU
    lever_arm: np.ndarray = field(default_factory=lambda: np.zeros(3))      # Extrinsic: master to slave
    
    def copy(self) -> "NominalState":
        return NominalState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            acc_bias=self.acc_bias.copy(),
            gyro_bias=self.gyro_bias.copy(),
            slave_acc_bias=self.slave_acc_bias.copy(),
            slave_gyro_bias=self.slave_gyro_bias.copy(),
            lever_arm=self.lever_arm.copy(),
        )


# =============================================================================
# Dual IMU ESKF with Online Extrinsic Calibration
# =============================================================================

class DualIMU_ESKF:
    """
    Error-State Kalman Filter for Dual IMU navigation with:
    - Rigid body constraints
    - Online lever arm (extrinsic) calibration
    
    State vector (24-dimensional):
    - δp (3): Position error
    - δv (3): Velocity error  
    - δθ (3): Orientation error (rotation vector)
    - δba_m (3): Master accelerometer bias error
    - δbg_m (3): Master gyroscope bias error
    - δba_s (3): Slave accelerometer bias error
    - δbg_s (3): Slave gyroscope bias error
    - δr (3): Lever arm error
    """
    
    # State indices
    POS_IDX = slice(0, 3)       # Position error
    VEL_IDX = slice(3, 6)       # Velocity error
    THETA_IDX = slice(6, 9)     # Orientation error (rotation vector)
    BA_M_IDX = slice(9, 12)     # Master accelerometer bias error
    BG_M_IDX = slice(12, 15)    # Master gyroscope bias error
    BA_S_IDX = slice(15, 18)    # Slave accelerometer bias error
    BG_S_IDX = slice(18, 21)    # Slave gyroscope bias error
    LEVER_IDX = slice(21, 24)   # Lever arm error (extrinsic)
    STATE_DIM = 24
    
    def __init__(self, config: DualESKFConfig | None = None):
        self.config = config or DualESKFConfig()
        self.nominal = NominalState()
        
        # Initialize lever arm from config
        self.nominal.lever_arm = self.config.lever_arm_init.copy()
        
        # Previous gyro for angular acceleration estimation
        self.prev_gyro_master: np.ndarray | None = None
        self.prev_gyro_slave: np.ndarray | None = None
        
        # Initialize error state covariance (24x24)
        self._init_covariance()
        
        # Build process noise matrix
        self._build_process_noise()
    
    def _init_covariance(self) -> None:
        """Initialize error state covariance matrix."""
        self.P = np.diag([
            # Position (3)
            self.config.init_sigma_pos**2,
            self.config.init_sigma_pos**2,
            self.config.init_sigma_pos**2,
            # Velocity (3)
            self.config.init_sigma_vel**2,
            self.config.init_sigma_vel**2,
            self.config.init_sigma_vel**2,
            # Orientation (3)
            self.config.init_sigma_theta**2,
            self.config.init_sigma_theta**2,
            self.config.init_sigma_theta**2,
            # Master acc bias (3)
            self.config.init_sigma_acc_bias**2,
            self.config.init_sigma_acc_bias**2,
            self.config.init_sigma_acc_bias**2,
            # Master gyro bias (3)
            self.config.init_sigma_gyro_bias**2,
            self.config.init_sigma_gyro_bias**2,
            self.config.init_sigma_gyro_bias**2,
            # Slave acc bias (3)
            self.config.init_sigma_acc_bias**2,
            self.config.init_sigma_acc_bias**2,
            self.config.init_sigma_acc_bias**2,
            # Slave gyro bias (3)
            self.config.init_sigma_gyro_bias**2,
            self.config.init_sigma_gyro_bias**2,
            self.config.init_sigma_gyro_bias**2,
            # Lever arm (3) - LARGE initial uncertainty for online estimation
            self.config.init_sigma_lever_arm**2,
            self.config.init_sigma_lever_arm**2,
            self.config.init_sigma_lever_arm**2,
        ])
    
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
            self.config.sigma_acc_bias**2,
            self.config.sigma_acc_bias**2,
            self.config.sigma_acc_bias**2,
            self.config.sigma_gyro_bias**2,
            self.config.sigma_gyro_bias**2,
            self.config.sigma_gyro_bias**2,
            # Lever arm: very small random walk (rigid body assumption)
            self.config.sigma_lever_arm**2,
            self.config.sigma_lever_arm**2,
            self.config.sigma_lever_arm**2,
        ])
    
    def reset_orientation(self, true_quaternion: np.ndarray) -> None:
        """
        Hard reset orientation to ground truth (heading alignment).
        This is a hard constraint that directly sets the quaternion and
        resets the orientation uncertainty to a small value.
        
        Args:
            true_quaternion: Ground truth quaternion [w, x, y, z]
        """
        # Normalize quaternion
        q = true_quaternion / np.linalg.norm(true_quaternion)
        
        # Hard reset nominal quaternion
        self.nominal.quaternion = q.copy()
        
        # Reset orientation error covariance to small value
        small_sigma = 0.001  # Very small orientation uncertainty after alignment
        self.P[self.THETA_IDX, self.THETA_IDX] = np.eye(3) * small_sigma**2
        
        # Also zero out cross-correlations with orientation
        for idx in [self.POS_IDX, self.VEL_IDX, self.BA_M_IDX, self.BG_M_IDX,
                    self.BA_S_IDX, self.BG_S_IDX, self.LEVER_IDX]:
            self.P[self.THETA_IDX, idx] = 0
            self.P[idx, self.THETA_IDX] = 0

    # -------------------------------------------------------------------------
    # Configuration Methods
    # -------------------------------------------------------------------------
    
    def set_lever_arm_init(self, lever_arm: np.ndarray) -> None:
        """Set initial guess for lever arm."""
        self.nominal.lever_arm = lever_arm.copy()
    
    def set_master_slave(self, master_slave: MasterSlave) -> None:
        """Set which IMU is the master."""
        self.config.master_slave = master_slave
    
    def initialize(
        self,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
        quaternion: np.ndarray | None = None,
        lever_arm: np.ndarray | None = None,
    ) -> None:
        """Initialize the filter with known initial state."""
        self.nominal.position = position.copy()
        if velocity is not None:
            self.nominal.velocity = velocity.copy()
        if quaternion is not None:
            self.nominal.quaternion = quaternion.copy()
        if lever_arm is not None:
            self.nominal.lever_arm = lever_arm.copy()
        
        self.prev_gyro_master = None
        self.prev_gyro_slave = None
    
    # -------------------------------------------------------------------------
    # Math Utilities
    # -------------------------------------------------------------------------
    
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
        """Create skew-symmetric matrix from vector: [v]× """
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
    
    # -------------------------------------------------------------------------
    # Angular Acceleration Estimation
    # -------------------------------------------------------------------------
    
    def compute_angular_acceleration(
        self,
        gyro_current: np.ndarray,
        gyro_prev: np.ndarray | None,
        dt: float,
    ) -> np.ndarray:
        """Compute angular acceleration from gyroscope differentiation."""
        if gyro_prev is None or dt <= 0:
            return np.zeros(3)
        return (gyro_current - gyro_prev) / dt
    
    # -------------------------------------------------------------------------
    # Rigid Body Kinematics
    # -------------------------------------------------------------------------
    
    def compute_theoretical_slave_readings(
        self,
        acc_master: np.ndarray,
        gyro_master: np.ndarray,
        angular_acc: np.ndarray,
        lever_arm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute theoretical slave IMU readings based on rigid body kinematics.
        
        For a rigid body:
        - ω_slave = ω_master (same angular velocity)
        - a_slave = a_master + α × r + ω × (ω × r)
        
        where r is the lever arm from master to slave.
        
        Args:
            acc_master: Master accelerometer reading (bias-corrected)
            gyro_master: Master gyroscope reading (bias-corrected)
            angular_acc: Angular acceleration estimate (α = ω̇)
            lever_arm: Current lever arm estimate
        
        Returns:
            (theoretical_acc_slave, theoretical_gyro_slave)
        """
        omega = gyro_master
        alpha = angular_acc
        r = lever_arm
        
        # Gyroscope: same angular velocity for rigid body
        theoretical_gyro = gyro_master.copy()
        
        # Accelerometer: a_slave = a_master + α × r + ω × (ω × r)
        tangential = np.cross(alpha, r)      # α × r: tangential acceleration
        centripetal = np.cross(omega, np.cross(omega, r))  # ω × (ω × r): centripetal
        
        theoretical_acc = acc_master + tangential + centripetal
        
        return theoretical_acc, theoretical_gyro
    
    # -------------------------------------------------------------------------
    # ESKF Prediction Step
    # -------------------------------------------------------------------------
    
    def predict(self, acc_master: np.ndarray, gyro_master: np.ndarray, dt: float) -> None:
        """
        Prediction step using master IMU.
        
        The lever arm is constant in the body frame (rigid body),
        so its prediction is simply: r_{k+1} = r_k (identity transition).
        
        Args:
            acc_master: Master accelerometer reading (m/s^2)
            gyro_master: Master gyroscope reading (rad/s)
            dt: Time step (s)
        """
        if dt <= 0:
            return
        
        # Correct measurements with current bias estimates
        acc_corrected = acc_master - self.nominal.acc_bias
        gyro_corrected = gyro_master - self.nominal.gyro_bias
        
        # Get rotation matrix from body to world
        R = self.quaternion_to_rotation_matrix(self.nominal.quaternion)
        
        # Propagate nominal state
        acc_world = R @ acc_corrected
        self.nominal.position = (
            self.nominal.position 
            + self.nominal.velocity * dt 
            + 0.5 * acc_world * dt**2
        )
        
        self.nominal.velocity = self.nominal.velocity + acc_world * dt
        
        delta_theta = gyro_corrected * dt
        delta_q = self.rotation_vector_to_quaternion(delta_theta)
        self.nominal.quaternion = self.quaternion_multiply(self.nominal.quaternion, delta_q)
        self.nominal.quaternion /= np.linalg.norm(self.nominal.quaternion)
        
        # Lever arm: constant in body frame (no change in nominal state)
        # self.nominal.lever_arm stays the same
        
        # Build state transition matrix F (24x24)
        F = np.eye(self.STATE_DIM)
        
        # ∂δp/∂δv
        F[self.POS_IDX, self.VEL_IDX] = np.eye(3) * dt
        
        # ∂δv/∂δθ
        F[self.VEL_IDX, self.THETA_IDX] = -R @ self.skew_symmetric(acc_corrected) * dt
        
        # ∂δv/∂δba_m
        F[self.VEL_IDX, self.BA_M_IDX] = -R * dt
        
        # ∂δθ/∂δbg_m
        F[self.THETA_IDX, self.BG_M_IDX] = -np.eye(3) * dt
        
        # Lever arm: identity transition (F[LEVER_IDX, LEVER_IDX] = I, already set)
        
        # Process noise
        Q = self.Q_c * dt
        
        # Propagate covariance
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        
        # ----- Covariance Bounds in Predict Step -----
        # Limit all diagonal elements to prevent explosion
        max_var = self.config.max_lever_arm_variance * 100  # More relaxed for general states
        diag_P = np.diag(self.P)
        if np.any(diag_P > max_var) or np.any(np.isnan(diag_P)):
            # Reset to initial covariance if things go wrong
            self._init_covariance()
    
    # -------------------------------------------------------------------------
    # ESKF Measurement Update with Slave IMU
    # -------------------------------------------------------------------------
    
    def update_with_slave_imu(
        self,
        acc_master: np.ndarray,
        gyro_master: np.ndarray,
        acc_slave: np.ndarray,
        gyro_slave: np.ndarray,
        dt: float,
        true_angular_acc: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | bool]:
        """
        Measurement update using slave IMU with rigid body constraints.
        
        The observation model includes the lever arm as part of the state,
        allowing online estimation of the extrinsic parameters.
        
        Observation equation:
        z = [z_gyro; z_acc] where:
        - z_gyro = (gyro_s - bg_s) - (gyro_m - bg_m) ≈ 0
        - z_acc = (acc_s - ba_s) - (acc_m - ba_m + α×r + ω×(ω×r)) ≈ 0
        
        Args:
            acc_master: Master accelerometer reading
            gyro_master: Master gyroscope reading
            acc_slave: Slave accelerometer reading
            gyro_slave: Slave gyroscope reading
            dt: Time step
        
        Returns:
            Dictionary with diagnostic information; values may be numpy arrays or boolean flags
        """
        # Correct for biases
        acc_m_corrected = acc_master - self.nominal.acc_bias
        gyro_m_corrected = gyro_master - self.nominal.gyro_bias
        acc_s_corrected = acc_slave - self.nominal.slave_acc_bias
        gyro_s_corrected = gyro_slave - self.nominal.slave_gyro_bias
        
        # Current lever arm estimate
        r = self.nominal.lever_arm
        omega = gyro_m_corrected
        
        # Compute angular acceleration
        if true_angular_acc is not None:
            # Use provided ground truth angular acceleration
            angular_acc = true_angular_acc
        elif self.config.use_gyro_diff_for_alpha:
            # Estimate by differentiating gyro (noisy!)
            angular_acc = self.compute_angular_acceleration(
                gyro_m_corrected, self.prev_gyro_master, dt
            )
        else:
            angular_acc = np.zeros(3)
        
        # Compute theoretical slave readings using current lever arm estimate
        theoretical_acc, theoretical_gyro = self.compute_theoretical_slave_readings(
            acc_m_corrected, gyro_m_corrected, angular_acc, r
        )
        
        # Observation residuals (should be close to zero for correct lever arm)
        residual_gyro = gyro_s_corrected - theoretical_gyro
        residual_acc = acc_s_corrected - theoretical_acc
        
        # Combined observation vector (6D: 3 gyro + 3 acc)
        z = np.concatenate([residual_gyro, residual_acc])
        
        # Build observation Jacobian H (6 x 24)
        H = np.zeros((6, self.STATE_DIM))
        
        # ----- Gyroscope Constraint Jacobian -----
        # z_gyro = (gyro_s - bg_s) - (gyro_m - bg_m)
        # ∂z_gyro/∂δbg_m = +I
        # ∂z_gyro/∂δbg_s = -I
        # ∂z_gyro/∂δr = 0 (gyro doesn't depend on lever arm)
        H[0:3, self.BG_M_IDX] = np.eye(3)
        H[0:3, self.BG_S_IDX] = -np.eye(3)
        
        # ----- Accelerometer Constraint Jacobian -----
        # z_acc = (acc_s - ba_s) - (acc_m - ba_m + α×r + ω×(ω×r))
        #
        # ∂z_acc/∂δba_m = +I
        # ∂z_acc/∂δba_s = -I
        # ∂z_acc/∂δbg_m: from ω×(ω×r) term
        # ∂z_acc/∂δr: from α×r + ω×(ω×r) terms  <-- KEY for lever arm estimation
        
        # ∂z_acc/∂δba_m
        H[3:6, self.BA_M_IDX] = np.eye(3)
        
        # ∂z_acc/∂δba_s
        H[3:6, self.BA_S_IDX] = -np.eye(3)
        
        # ∂z_acc/∂δbg_m: comes from ω×(ω×r) term
        # ∂(ω×(ω×r))/∂ω = [ω]×[r]× + [(ω×r)]×
        omega_cross = self.skew_symmetric(omega)
        r_cross = self.skew_symmetric(r)
        omega_cross_r = np.cross(omega, r)
        omega_cross_r_cross = self.skew_symmetric(omega_cross_r)
        
        d_centripetal_d_omega = omega_cross @ r_cross + omega_cross_r_cross
        # Since gyro_corrected = gyro - bias, ∂ω/∂bg = -I
        H[3:6, self.BG_M_IDX] = -d_centripetal_d_omega
        
        # ∂z_acc/∂δr: This is the KEY Jacobian for lever arm observability
        #
        # The observation model is:
        #   h_acc(x) = α×r + ω×(ω×r)      (predicted acc difference)
        # 
        # H = ∂h/∂x, so:
        #   ∂h_acc/∂r = ∂(α×r)/∂r + ∂(ω×(ω×r))/∂r
        #
        # For cross product: ∂(a×b)/∂b = [a]× (skew symmetric matrix of a)
        # So: ∂(α×r)/∂r = [α]×
        #
        # For centripetal: let u = ω×r, then ω×u
        #   ∂(ω×(ω×r))/∂r = [ω]× ∂(ω×r)/∂r = [ω]× [ω]× = [ω]×[ω]×
        #
        alpha_cross = self.skew_symmetric(angular_acc)
        
        # ∂(α×r)/∂r = [α]×
        d_tangential_d_r = alpha_cross
        
        # ∂(ω×(ω×r))/∂r = [ω]×[ω]×
        d_centripetal_d_r = omega_cross @ omega_cross
        
        # H[3:6, lever] = ∂h_acc/∂r = [α]× + [ω]×[ω]×
        H[3:6, self.LEVER_IDX] = d_tangential_d_r + d_centripetal_d_r
        
        # ----- Lever Arm Observability Check -----
        # When |ω| is too small, lever arm is not observable
        # Set H[acc, lever] to zero to prevent ill-conditioned updates
        omega_norm = np.linalg.norm(omega)
        if omega_norm < self.config.min_omega_for_lever_obs:
            H[3:6, self.LEVER_IDX] = np.zeros((3, 3))
        
        # Measurement noise covariance
        R_meas = np.diag([
            self.config.sigma_gyro_constraint**2,
            self.config.sigma_gyro_constraint**2,
            self.config.sigma_gyro_constraint**2,
            self.config.sigma_acc_constraint**2,
            self.config.sigma_acc_constraint**2,
            self.config.sigma_acc_constraint**2,
        ])
        
        # Kalman gain with numerical stability
        S = H @ self.P @ H.T + R_meas
        
        # ----- Outlier Rejection -----
        # Use Mahalanobis distance to detect outliers
        if self.config.enable_outlier_rejection:
            try:
                S_inv = np.linalg.inv(S)
                mahalanobis_sq = z.T @ S_inv @ z
                # Chi-squared threshold for 6 DOF at given sigma level
                threshold = self.config.outlier_threshold**2 * 6  # Approximate chi-squared
                if mahalanobis_sq > threshold:
                    # Skip this update - measurement is an outlier
                    self.prev_gyro_master = gyro_m_corrected.copy()
                    self.prev_gyro_slave = gyro_s_corrected.copy()
                    return {
                        "residual_gyro": residual_gyro,
                        "residual_acc": residual_acc,
                        "theoretical_acc": theoretical_acc,
                        "theoretical_gyro": theoretical_gyro,
                        "angular_acc": angular_acc,
                        "lever_arm_estimate": self.nominal.lever_arm.copy(),
                        "lever_arm_std": np.sqrt(np.diag(self.P)[21:24]),
                        "outlier_rejected": True,
                    }
            except np.linalg.LinAlgError:
                pass  # Fallback to normal update if S is singular
        
        # Use pseudo-inverse for numerical stability
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = self.P @ H.T @ S_inv
        
        # Error state update
        delta_x = K @ z
        
        # Inject error into nominal state
        self._inject_error(delta_x)
        
        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        # ----- Covariance Bounds -----
        # Prevent lever arm covariance from exploding during unobservable periods
        lever_var = np.diag(self.P)[21:24]
        if np.any(lever_var > self.config.max_lever_arm_variance):
            scale = np.sqrt(self.config.max_lever_arm_variance / np.maximum(lever_var, 1e-10))
            for i in range(3):
                if lever_var[i] > self.config.max_lever_arm_variance:
                    self.P[21+i, :] *= scale[i]
                    self.P[:, 21+i] *= scale[i]
        
        # Store gyro for next angular acceleration computation
        self.prev_gyro_master = gyro_m_corrected.copy()
        self.prev_gyro_slave = gyro_s_corrected.copy()
        
        return {
            "residual_gyro": residual_gyro,
            "residual_acc": residual_acc,
            "theoretical_acc": theoretical_acc,
            "theoretical_gyro": theoretical_gyro,
            "angular_acc": angular_acc,
            "lever_arm_estimate": self.nominal.lever_arm.copy(),
            "lever_arm_std": np.sqrt(np.diag(self.P)[21:24]),
        }
    
    # -------------------------------------------------------------------------
    # GPS Position Update (Surface Mode)
    # -------------------------------------------------------------------------
    
    def update_with_gps(
        self,
        gps_position: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Update state using GPS position measurement.
        
        This is used when the vehicle is on the surface and GPS is available.
        The GPS provides a direct observation of position, which helps:
        1. Correct position drift
        2. Indirectly improve velocity estimates
        3. Provide a stable initial state before submersion
        
        Args:
            gps_position: GPS-measured position (3,) in world frame
        
        Returns:
            Dictionary with diagnostic information
        """
        # Observation residual: z = gps_pos - estimated_pos
        z = gps_position - self.nominal.position
        
        # Observation Jacobian: H_gps observes position directly
        # z = H * δx, where δp is at indices 0:3
        H = np.zeros((3, self.STATE_DIM))
        H[0:3, self.POS_IDX] = np.eye(3)
        
        # Measurement noise
        R_gps = np.eye(3) * self.config.sigma_gps_pos**2
        
        # Kalman gain
        S = H @ self.P @ H.T + R_gps
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)
        
        # Error state update
        delta_x = K @ z
        
        # Inject error into nominal state
        self._inject_error(delta_x)
        
        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_gps @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        return {
            "gps_residual": z,
            "position_after_gps": self.nominal.position.copy(),
        }
    
    # -------------------------------------------------------------------------
    # Depth Sensor Update (Underwater Mode)
    # -------------------------------------------------------------------------
    
    def update_with_depth(
        self,
        depth_measurement: float,
    ) -> dict[str, float | np.ndarray]:
        """
        Update state using depth sensor measurement.
        
        The depth sensor provides a direct observation of the Z coordinate,
        which helps constrain vertical drift during underwater navigation.
        
        Args:
            depth_measurement: Measured depth (Z coordinate) from depth sensor
        
        Returns:
            Dictionary with diagnostic information
        """
        # Observation residual: z = depth_meas - estimated_z
        z = np.array([depth_measurement - self.nominal.position[2]])
        
        # Observation Jacobian: H_depth only observes position Z
        # z = H * δx, where δp_z is at index 2
        H = np.zeros((1, self.STATE_DIM))
        H[0, 2] = 1.0  # Observe only Z component of position
        
        # Measurement noise
        R_depth = np.array([[self.config.sigma_depth**2]])
        
        # Kalman gain
        S = H @ self.P @ H.T + R_depth
        K = self.P @ H.T / S[0, 0]  # Scalar division for 1D observation
        
        # Error state update
        delta_x = K.flatten() * z[0]
        
        # Inject error into nominal state
        self._inject_error(delta_x)
        
        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.STATE_DIM) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * R_depth[0, 0]
        self.P = 0.5 * (self.P + self.P.T)
        
        return {
            "depth_residual": z[0],
            "depth_measurement": depth_measurement,
            "estimated_depth": self.nominal.position[2],
        }
    
    # -------------------------------------------------------------------------
    # Error Injection
    # -------------------------------------------------------------------------
    
    def _inject_error(self, delta_x: np.ndarray) -> None:
        """Inject error state into nominal state and reset error."""
        # Position and velocity: additive
        self.nominal.position += delta_x[self.POS_IDX]
        self.nominal.velocity += delta_x[self.VEL_IDX]
        
        # Orientation: quaternion update
        delta_theta = delta_x[self.THETA_IDX]
        delta_q = self.rotation_vector_to_quaternion(delta_theta)
        self.nominal.quaternion = self.quaternion_multiply(self.nominal.quaternion, delta_q)
        self.nominal.quaternion /= np.linalg.norm(self.nominal.quaternion)
        
        # Biases: additive
        self.nominal.acc_bias += delta_x[self.BA_M_IDX]
        self.nominal.gyro_bias += delta_x[self.BG_M_IDX]
        self.nominal.slave_acc_bias += delta_x[self.BA_S_IDX]
        self.nominal.slave_gyro_bias += delta_x[self.BG_S_IDX]
        
        # Lever arm: additive
        self.nominal.lever_arm += delta_x[self.LEVER_IDX]
        
        # Enforce lever arm magnitude constraint if configured
        if self.config.lever_arm_magnitude is not None and self.config.enforce_magnitude_constraint:
            self._enforce_lever_arm_magnitude()
    
    def _enforce_lever_arm_magnitude(self) -> None:
        """
        Enforce hard constraint on lever arm magnitude.
        
        This projects the lever arm estimate onto a sphere of known radius,
        preserving only the direction while forcing the magnitude to be exact.
        
        Also updates the covariance matrix to be consistent with the constraint:
        - The covariance in the radial direction is set to near-zero
        - Only tangential uncertainty is preserved
        """
        if self.config.lever_arm_magnitude is None:
            return
        
        r = self.nominal.lever_arm
        r_norm = np.linalg.norm(r)
        target_norm = self.config.lever_arm_magnitude
        
        if r_norm < 1e-10:
            # If lever arm is near zero, cannot determine direction
            # Initialize along x-axis with target magnitude
            self.nominal.lever_arm = np.array([target_norm, 0.0, 0.0])
            return
        
        # Project onto sphere: r_new = r * (target_norm / |r|)
        scale = target_norm / r_norm
        self.nominal.lever_arm = r * scale
        
        # Update covariance to remove radial uncertainty
        # The radial direction unit vector
        r_hat = r / r_norm
        
        # Projection matrix onto tangent plane: P_tan = I - r_hat @ r_hat.T
        # This removes the component in the radial direction
        P_tangent = np.eye(3) - np.outer(r_hat, r_hat)
        
        # Extract lever arm covariance block (3x3)
        P_lever = self.P[self.LEVER_IDX, self.LEVER_IDX].copy()
        
        # Project covariance onto tangent plane
        # This sets radial variance to near-zero while preserving tangential variance
        P_lever_constrained = P_tangent @ P_lever @ P_tangent.T
        
        # Add small radial variance for numerical stability
        radial_var = 1e-8
        P_lever_constrained += radial_var * np.outer(r_hat, r_hat)
        
        # Update the covariance matrix
        self.P[self.LEVER_IDX, self.LEVER_IDX] = P_lever_constrained
        
        # Also scale cross-covariances to be consistent
        # P[lever, other] = P_tangent @ P[lever, other]
        for idx in [self.POS_IDX, self.VEL_IDX, self.THETA_IDX, 
                    self.BA_M_IDX, self.BG_M_IDX, self.BA_S_IDX, self.BG_S_IDX]:
            self.P[self.LEVER_IDX, idx] = P_tangent @ self.P[self.LEVER_IDX, idx]
            self.P[idx, self.LEVER_IDX] = self.P[self.LEVER_IDX, idx].T
    
    # -------------------------------------------------------------------------
    # State Getters
    # -------------------------------------------------------------------------
    
    def get_position(self) -> np.ndarray:
        """Get current estimated position."""
        return self.nominal.position.copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current estimated velocity."""
        return self.nominal.velocity.copy()
    
    def get_lever_arm(self) -> np.ndarray:
        """Get current estimated lever arm (extrinsic)."""
        return self.nominal.lever_arm.copy()
    
    def get_lever_arm_uncertainty(self) -> np.ndarray:
        """Get current lever arm standard deviation."""
        return np.sqrt(np.diag(self.P)[21:24])
    
    def get_biases(self) -> dict[str, np.ndarray]:
        """Get all bias estimates."""
        return {
            "master_acc_bias": self.nominal.acc_bias.copy(),
            "master_gyro_bias": self.nominal.gyro_bias.copy(),
            "slave_acc_bias": self.nominal.slave_acc_bias.copy(),
            "slave_gyro_bias": self.nominal.slave_gyro_bias.copy(),
        }
    
    def get_euler_angles(self) -> np.ndarray:
        """Get current estimated orientation as Euler angles (roll, pitch, yaw)."""
        q = self.nominal.quaternion
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x, y, z, w]
        return r.as_euler('xyz')


# =============================================================================
# Data Loading
# =============================================================================

def load_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Load IMU data from CSV file.
    
    Returns:
        time: Timestamps
        positions: Ground truth positions (x, y, z)
        accelerations: Linear accelerations (ax, ay, az)
        angular_velocities: Angular velocities (wx, wy, wz)
        angular_accelerations: Angular accelerations (alphax, alphay, alphaz) or None if not present
    """
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    time = raw[:, 0]
    positions = raw[:, 1:4]
    accelerations = raw[:, 4:7]
    angular_velocities = raw[:, 7:10]
    
    # Check if angular acceleration columns exist (13 columns total)
    if raw.shape[1] >= 13:
        angular_accelerations = raw[:, 10:13]
    else:
        angular_accelerations = None
    
    return time, positions, accelerations, angular_velocities, angular_accelerations


# =============================================================================
# Depth Sensor Simulation
# =============================================================================

def simulate_depth_sensor(
    true_positions: np.ndarray,
    sigma_depth: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate depth sensor measurements with Gaussian noise.
    
    The depth sensor measures the Z coordinate (depth) of the vehicle.
    In practice, depth sensors measure pressure and convert to depth.
    
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


# =============================================================================
# Main ESKF Runner
# =============================================================================

def run_dual_eskf(
    time: np.ndarray,
    acc_master: np.ndarray,
    gyro_master: np.ndarray,
    acc_slave: np.ndarray,
    gyro_slave: np.ndarray,
    initial_position: np.ndarray,
    config: DualESKFConfig | None = None,
    true_angular_acc: np.ndarray | None = None,
    gps_positions: np.ndarray | None = None,
    depth_measurements: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Run Dual IMU ESKF with online extrinsic calibration.
    
    Args:
        time: Timestamps
        acc_master: Master accelerometer data (n, 3)
        gyro_master: Master gyroscope data (n, 3)
        acc_slave: Slave accelerometer data (n, 3)
        gyro_slave: Slave gyroscope data (n, 3)
        initial_position: Initial position (3,)
        config: ESKF configuration
        true_angular_acc: Ground truth angular acceleration (n, 3) for validation.
                         If provided, uses this instead of gyro differentiation.
        gps_positions: GPS position data (n, 3) for surface updates. Used when
                       time < gps_cutoff_time and enable_gps_update is True.
        depth_measurements: Depth sensor data (n,) for underwater updates. Used when
                           time >= depth_start_time and enable_depth_update is True.
    
    Returns:
        positions: Estimated positions over time (n, 3)
        lever_arms: Estimated lever arms over time (n, 3)
        diagnostics: List of diagnostic info per timestep
    """
    eskf = DualIMU_ESKF(config)
    eskf.initialize(
        position=initial_position,
        lever_arm=config.lever_arm_init if config else None,
    )
    
    n = len(time)
    positions = np.zeros((n, 3))
    lever_arms = np.zeros((n, 3))
    lever_arm_stds = np.zeros((n, 3))
    
    positions[0] = initial_position.copy()
    lever_arms[0] = eskf.get_lever_arm()
    lever_arm_stds[0] = eskf.get_lever_arm_uncertainty()
    
    diagnostics = []
    gps_update_count = 0
    depth_update_count = 0
    
    # Determine GPS cutoff time (includes underwater duration if set)
    gps_cutoff_base = config.gps_cutoff_time if config else 0.0
    gps_underwater_dur = config.gps_underwater_duration if config else 0.0
    gps_cutoff = gps_cutoff_base + gps_underwater_dur  # GPS available until this time
    use_gps = config.enable_gps_update if config else False
    
    # Determine depth sensor availability
    depth_start = config.depth_start_time if config else 0.0
    use_depth = config.enable_depth_update if config else False
    
    # Heading alignment configuration
    heading_alignment_time = config.heading_alignment_time if config else 0.0
    use_heading_alignment = config.enable_heading_alignment if config else False
    heading_aligned = False  # Track if alignment has been performed
    
    for idx in range(1, n):
        dt = time[idx] - time[idx - 1]
        current_time = time[idx]
        
        # Prediction with master IMU
        eskf.predict(
            acc_master=acc_master[idx],
            gyro_master=gyro_master[idx],
            dt=dt,
        )
        
        # Heading alignment at specified time (hard reset, done only once)
        if use_heading_alignment and not heading_aligned:
            prev_time = time[idx - 1]
            # Check if we just crossed the alignment time
            if prev_time < heading_alignment_time <= current_time:
                # Reset orientation to correct heading from config
                # heading_deg: 0=+X, 90=+Y, 180=-X, 270=-Y (rotation around Z axis)
                heading_rad = np.deg2rad(config.initial_heading_deg)
                heading_q = np.array([
                    np.cos(heading_rad / 2),  # w
                    0.0,                       # x
                    0.0,                       # y  
                    np.sin(heading_rad / 2)   # z (rotation around Z)
                ])
                eskf.reset_orientation(heading_q)
                heading_aligned = True
        
        # Check if GPS is available (surface mode: time < cutoff)
        gps_available = (
            use_gps and 
            gps_positions is not None and 
            current_time < gps_cutoff
        )
        
        # Check if depth sensor is available (underwater mode: time >= start)
        depth_available = (
            use_depth and
            depth_measurements is not None and
            current_time >= depth_start
        )
        
        if gps_available and gps_positions is not None:
            # Surface mode: use GPS position update FIRST
            gps_diag = eskf.update_with_gps(gps_positions[idx])
            gps_update_count += 1
            
            # ALSO apply dual IMU constraints to estimate lever arm
            # (even on surface, the rigid body constraint is valid)
            alpha = true_angular_acc[idx] if true_angular_acc is not None else None
            imu_diag = eskf.update_with_slave_imu(
                acc_master=acc_master[idx],
                gyro_master=gyro_master[idx],
                acc_slave=acc_slave[idx],
                gyro_slave=gyro_slave[idx],
                dt=dt,
                true_angular_acc=alpha,
            )
            
            diag = {
                "residual_gyro": imu_diag["residual_gyro"],
                "residual_acc": imu_diag["residual_acc"],
                "theoretical_acc": imu_diag["theoretical_acc"],
                "theoretical_gyro": imu_diag["theoretical_gyro"],
                "angular_acc": imu_diag["angular_acc"],
                "lever_arm_estimate": eskf.get_lever_arm(),
                "lever_arm_std": eskf.get_lever_arm_uncertainty(),
                "gps_update": True,
                "gps_residual": gps_diag["gps_residual"],
                "depth_update": False,
            }
        else:
            # Underwater mode: use dual IMU rigid body constraints
            # Get true angular acceleration for this timestep if available
            alpha = true_angular_acc[idx] if true_angular_acc is not None else None
            
            # Update with slave IMU constraints
            diag = eskf.update_with_slave_imu(
                acc_master=acc_master[idx],
                gyro_master=gyro_master[idx],
                acc_slave=acc_slave[idx],
                gyro_slave=gyro_slave[idx],
                dt=dt,
                true_angular_acc=alpha,
            )
            diag["gps_update"] = False
            
            # Apply depth sensor update if available (underwater)
            if depth_available and depth_measurements is not None:
                depth_diag = eskf.update_with_depth(depth_measurements[idx])
                depth_update_count += 1
                diag["depth_update"] = True
                diag["depth_residual"] = depth_diag["depth_residual"]
            else:
                diag["depth_update"] = False
        
        diagnostics.append(diag)
        
        positions[idx] = eskf.get_position()
        lever_arms[idx] = eskf.get_lever_arm()
        lever_arm_stds[idx] = eskf.get_lever_arm_uncertainty()
    
    # Add lever arm history to diagnostics summary
    for i, diag in enumerate(diagnostics):
        diag["lever_arm_std"] = lever_arm_stds[i + 1]
    
    # Print update statistics
    if use_gps:
        print(f"  GPS updates applied: {gps_update_count} (t < {gps_cutoff}s)")
    if use_depth:
        print(f"  Depth updates applied: {depth_update_count} (t >= {depth_start}s)")
    
    return positions, lever_arms, diagnostics


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_comparison(
    time: np.ndarray,
    truth: np.ndarray,
    estimate: np.ndarray,
    output_path: Path,
    title: str = "Dual IMU ESKF trajectory",
) -> None:
    """Plot comparison between ground truth and estimated trajectory."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["x", "y", "z"]

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


def plot_residuals(
    time: np.ndarray,
    diagnostics: list[dict],
    output_path: Path,
) -> None:
    """Plot constraint residuals over time."""
    residual_gyro = np.array([d["residual_gyro"] for d in diagnostics])
    residual_acc = np.array([d["residual_acc"] for d in diagnostics])
    angular_acc = np.array([d["angular_acc"] for d in diagnostics])
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    
    labels = ["x", "y", "z"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    
    # Gyro residuals
    for i in range(3):
        axes[0].plot(time[1:], residual_gyro[:, i], label=labels[i], color=colors[i])
    axes[0].set_title("Gyroscope constraint residuals")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("residual (rad/s)")
    axes[0].legend()
    axes[0].grid(True)
    
    # Acc residuals
    for i in range(3):
        axes[1].plot(time[1:], residual_acc[:, i], label=labels[i], color=colors[i])
    axes[1].set_title("Accelerometer constraint residuals")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("residual (m/s²)")
    axes[1].legend()
    axes[1].grid(True)
    
    # Angular acceleration
    for i in range(3):
        axes[2].plot(time[1:], angular_acc[:, i], label=labels[i], color=colors[i])
    axes[2].set_title("Estimated angular acceleration")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("α (rad/s²)")
    axes[2].legend()
    axes[2].grid(True)
    
    fig.suptitle("Rigid Body Constraint Diagnostics")
    fig.savefig(output_path, dpi=150)
    print(f"Saved residuals plot to {output_path}")


def plot_lever_arm_estimation(
    time: np.ndarray,
    lever_arms: np.ndarray,
    true_lever_arm: np.ndarray | None,
    output_path: Path,
) -> None:
    """Plot lever arm estimation over time."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    
    labels = ["x", "y", "z"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    
    # Lever arm components
    for i in range(3):
        axes[0].plot(time, lever_arms[:, i], label=f"estimate {labels[i]}", color=colors[i])
        if true_lever_arm is not None:
            axes[0].axhline(
                y=true_lever_arm[i], 
                color=colors[i], 
                linestyle="--", 
                alpha=0.7,
                label=f"true {labels[i]}"
            )
    axes[0].set_title("Lever arm estimation over time")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("lever arm (m)")
    axes[0].legend(fontsize="small", ncol=2)
    axes[0].grid(True)
    
    # Lever arm magnitude
    estimated_magnitude = np.linalg.norm(lever_arms, axis=1)
    axes[1].plot(time, estimated_magnitude, label="estimated |r|", color="tab:purple")
    if true_lever_arm is not None:
        true_magnitude = np.linalg.norm(true_lever_arm)
        axes[1].axhline(
            y=true_magnitude, 
            color="tab:red", 
            linestyle="--",
            label=f"true |r| = {true_magnitude:.3f} m"
        )
    axes[1].set_title("Lever arm magnitude")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("|r| (m)")
    axes[1].legend()
    axes[1].grid(True)
    
    fig.suptitle("Online Lever Arm (Extrinsic) Calibration")
    fig.savefig(output_path, dpi=150)
    print(f"Saved lever arm plot to {output_path}")


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


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dual IMU ESKF navigation with rigid body constraints and online extrinsic calibration."
    )
    
    # ----- Input Files -----
    parser.add_argument(
        "--imu-l", type=Path, default=Path("IMU_L.csv"),
        help="Path to the left IMU CSV file",
    )
    parser.add_argument(
        "--imu-r", type=Path, default=Path("IMU_R.csv"),
        help="Path to the right IMU CSV file",
    )
    parser.add_argument(
        "--imu-l-noised", type=Path, default=None,
        help="Optional path to noised left IMU CSV file",
    )
    parser.add_argument(
        "--imu-r-noised", type=Path, default=None,
        help="Optional path to noised right IMU CSV file",
    )
    parser.add_argument(
        "--imu-c", type=Path, default=None,
        help="Path to center IMU CSV file (for ground truth comparison)",
    )
    
    # ----- Output Files -----
    parser.add_argument(
        "--output", type=Path, default=Path("dual_eskf_trajectory.png"),
        help="Path to write the trajectory figure",
    )
    parser.add_argument(
        "--output-residuals", type=Path, default=Path("dual_eskf_residuals.png"),
        help="Path to write the residuals figure",
    )
    parser.add_argument(
        "--output-lever-arm", type=Path, default=Path("dual_eskf_lever_arm.png"),
        help="Path to write the lever arm estimation figure",
    )
    
    # ----- Master/Slave Configuration -----
    parser.add_argument(
        "--master", type=str, choices=["left", "right"], default="left",
        help="Which IMU to use as master (left or right)",
    )
    
    # ----- Lever Arm (Extrinsic) Configuration -----
    parser.add_argument(
        "--lever-arm-init-x", type=float, default=0.0,
        help="Initial guess for lever arm X component (m)",
    )
    parser.add_argument(
        "--lever-arm-init-y", type=float, default=0.0,
        help="Initial guess for lever arm Y component (m)",
    )
    parser.add_argument(
        "--lever-arm-init-z", type=float, default=0.0,
        help="Initial guess for lever arm Z component (m)",
    )
    parser.add_argument(
        "--lever-arm-true-x", type=float, default=None,
        help="True lever arm X (for comparison, if known)",
    )
    parser.add_argument(
        "--lever-arm-true-y", type=float, default=None,
        help="True lever arm Y (for comparison, if known)",
    )
    parser.add_argument(
        "--lever-arm-true-z", type=float, default=None,
        help="True lever arm Z (for comparison, if known)",
    )
    parser.add_argument(
        "--init-sigma-lever-arm", type=float, default=0.5,
        help="Initial lever arm uncertainty std (m) - larger = more uncertain",
    )
    
    # ----- Noise Parameters -----
    parser.add_argument(
        "--sigma-acc", type=float, default=0.1,
        help="Accelerometer noise standard deviation (m/s^2)",
    )
    parser.add_argument(
        "--sigma-gyro", type=float, default=0.01,
        help="Gyroscope noise standard deviation (rad/s)",
    )
    parser.add_argument(
        "--sigma-acc-constraint", type=float, default=0.2,
        help="Accelerometer constraint noise (m/s^2)",
    )
    parser.add_argument(
        "--sigma-gyro-constraint", type=float, default=0.02,
        help="Gyroscope constraint noise (rad/s)",
    )
    parser.add_argument(
        "--sigma-lever-arm", type=float, default=0.0001,
        help="Lever arm random walk std (should be very small for rigid body)",
    )
    
    # ----- Algorithm Options -----
    parser.add_argument(
        "--no-angular-acc", action="store_true",
        help="Disable angular acceleration estimation (ignore α×r term)",
    )
    parser.add_argument(
        "--use-true-angular-acc", action="store_true",
        help="Use ground truth angular acceleration from CSV (requires 13-column CSV with alphax,alphay,alphaz)",
    )
    parser.add_argument(
        "--lever-arm-magnitude", type=float, default=None,
        help="Known lever arm magnitude (m). If specified, enforces hard constraint on |r| while estimating direction only.",
    )
    parser.add_argument(
        "--no-magnitude-constraint", action="store_true",
        help="Disable magnitude constraint enforcement even if lever-arm-magnitude is specified",
    )
    
    # ----- GPS Position Update Options -----
    parser.add_argument(
        "--enable-gps", action="store_true",
        help="Enable GPS position updates when on surface (t < gps-cutoff-time)",
    )
    parser.add_argument(
        "--gps-cutoff-time", type=float, default=0.0,
        help="Time at which GPS becomes unavailable (submersion time). GPS updates are applied for t < this value.",
    )
    parser.add_argument(
        "--gps-underwater-duration", type=float, default=0.0,
        help="GPS still available for this duration after submersion (e.g., 0.2 means GPS works until t=cutoff+0.2s)",
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
        help="Time at which depth sensor becomes available (submersion time). Depth updates are applied for t >= this value.",
    )
    parser.add_argument(
        "--sigma-depth", type=float, default=0.1,
        help="Depth sensor measurement noise (m). Typical values: 0.01-0.1m for pressure sensors.",
    )
    parser.add_argument(
        "--depth-seed", type=int, default=None,
        help="Random seed for depth sensor noise simulation (for reproducibility)",
    )
    
    # ----- Heading Alignment Options -----
    parser.add_argument(
        "--enable-heading-alignment", action="store_true",
        help="Enable hard reset of orientation at alignment time (e.g., at submersion)",
    )
    parser.add_argument(
        "--heading-alignment-time", type=float, default=0.0,
        help="Time at which to perform heading alignment (hard reset to known orientation)",
    )
    parser.add_argument(
        "--initial-heading-deg", type=float, default=90.0,
        help="Initial heading in degrees (0=+X, 90=+Y, 180=-X, 270=-Y). Default: 90 (facing +Y)",
    )
    
    args = parser.parse_args()

    # Load IMU data
    print("Loading IMU data...")
    time_l, pos_l, acc_l, gyro_l, ang_acc_l = load_imu_csv(args.imu_l)
    time_r, pos_r, acc_r, gyro_r, ang_acc_r = load_imu_csv(args.imu_r)
    
    # Check if true angular acceleration is available
    if args.use_true_angular_acc:
        if ang_acc_l is None:
            print("WARNING: --use-true-angular-acc specified but CSV does not contain angular acceleration columns!")
            print("         Falling back to gyro differentiation.")
            true_angular_acc = None
        else:
            print("Using ground truth angular acceleration from CSV.")
            true_angular_acc = ang_acc_l  # Same for all rigid body points
    else:
        true_angular_acc = None
    
    # Verify time alignment
    if not np.allclose(time_l, time_r):
        raise ValueError("Left and right IMU files must have the same timestamps")
    
    # Use noised data if provided
    if args.imu_l_noised is not None:
        print(f"Using noised left IMU data from {args.imu_l_noised}")
        _, _, acc_l, gyro_l, _ = load_imu_csv(args.imu_l_noised)
    if args.imu_r_noised is not None:
        print(f"Using noised right IMU data from {args.imu_r_noised}")
        _, _, acc_r, gyro_r, _ = load_imu_csv(args.imu_r_noised)
    
    # Determine master/slave and set initial lever arm guess
    lever_arm_init = np.array([
        args.lever_arm_init_x,
        args.lever_arm_init_y,
        args.lever_arm_init_z,
    ])
    
    if args.master == "left":
        master_slave = MasterSlave.LEFT_MASTER
        acc_master, gyro_master = acc_l, gyro_l
        acc_slave, gyro_slave = acc_r, gyro_r
        pos_master = pos_l
        # L -> R direction
    else:
        master_slave = MasterSlave.RIGHT_MASTER
        acc_master, gyro_master = acc_r, gyro_r
        acc_slave, gyro_slave = acc_l, gyro_l
        pos_master = pos_r
        # R -> L direction: invert initial guess
        lever_arm_init = -lever_arm_init
    
    # True lever arm for comparison (if provided)
    true_lever_arm = None
    if args.lever_arm_true_x is not None:
        true_lever_arm = np.array([
            args.lever_arm_true_x,
            args.lever_arm_true_y or 0.0,
            args.lever_arm_true_z or 0.0,
        ])
        if args.master == "right":
            true_lever_arm = -true_lever_arm
    
    # Configure ESKF
    config = DualESKFConfig(
        sigma_acc=args.sigma_acc,
        sigma_gyro=args.sigma_gyro,
        sigma_acc_constraint=args.sigma_acc_constraint,
        sigma_gyro_constraint=args.sigma_gyro_constraint,
        sigma_lever_arm=args.sigma_lever_arm,
        init_sigma_lever_arm=args.init_sigma_lever_arm,
        lever_arm_init=lever_arm_init,
        master_slave=master_slave,
        use_gyro_diff_for_alpha=not args.no_angular_acc,
        lever_arm_magnitude=args.lever_arm_magnitude,
        enforce_magnitude_constraint=not args.no_magnitude_constraint,
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
    
    # Simulate depth sensor measurements if enabled
    depth_measurements = None
    if args.enable_depth:
        print(f"Simulating depth sensor with sigma = {args.sigma_depth} m...")
        depth_measurements = simulate_depth_sensor(
            pos_master, 
            sigma_depth=args.sigma_depth,
            seed=args.depth_seed,
        )
    
    print(f"\nConfiguration:")
    print(f"  Master IMU: {args.master.upper()}")
    print(f"  Initial lever arm guess: {lever_arm_init}")
    print(f"  Initial lever arm uncertainty: ±{args.init_sigma_lever_arm} m")
    if args.lever_arm_magnitude is not None:
        print(f"  Lever arm magnitude constraint: |r| = {args.lever_arm_magnitude} m (ENFORCED)")
    if true_lever_arm is not None:
        print(f"  True lever arm: {true_lever_arm}")
    if true_angular_acc is not None:
        print(f"  Using TRUE angular acceleration")
    elif not args.no_angular_acc:
        print(f"  Using gyro differentiation for angular acceleration (noisy!)")
    else:
        print(f"  Angular acceleration DISABLED")
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
    
    # Run Dual ESKF
    print("\nRunning Dual IMU ESKF with online extrinsic calibration...")
    estimate, lever_arms, diagnostics = run_dual_eskf(
        time=time_l,
        acc_master=acc_master,
        gyro_master=gyro_master,
        acc_slave=acc_slave,
        gyro_slave=gyro_slave,
        initial_position=pos_master[0],
        config=config,
        true_angular_acc=true_angular_acc,
        gps_positions=pos_master if args.enable_gps else None,  # Use ground truth as GPS
        depth_measurements=depth_measurements,
    )
    
    # Use center IMU for ground truth if provided, otherwise use master position
    if args.imu_c is not None:
        # load_imu_csv returns 5 items: time, positions, acc, gyro, ang_acc (ang_acc may be None)
        _, truth_positions, _, _, _ = load_imu_csv(args.imu_c)
    else:
        truth_positions = pos_master
    
    # Compute and print RMSE
    rmse = compute_rmse(truth_positions, estimate)
    print(f"\nPosition RMSE:")
    print(f"  x: {rmse['x']:.4f} m")
    print(f"  y: {rmse['y']:.4f} m")
    print(f"  z: {rmse['z']:.4f} m")
    print(f"  total: {rmse['total']:.4f} m")
    
    # Final lever arm estimate
    final_lever_arm = lever_arms[-1]
    final_lever_arm_std = diagnostics[-1]["lever_arm_std"]
    print(f"\nFinal lever arm estimate:")
    print(f"  x: {final_lever_arm[0]:.4f} ± {final_lever_arm_std[0]:.4f} m")
    print(f"  y: {final_lever_arm[1]:.4f} ± {final_lever_arm_std[1]:.4f} m")
    print(f"  z: {final_lever_arm[2]:.4f} ± {final_lever_arm_std[2]:.4f} m")
    print(f"  |r|: {np.linalg.norm(final_lever_arm):.4f} m")
    
    if true_lever_arm is not None:
        lever_arm_error = final_lever_arm - true_lever_arm
        print(f"\nLever arm estimation error:")
        print(f"  x: {lever_arm_error[0]:.4f} m")
        print(f"  y: {lever_arm_error[1]:.4f} m")
        print(f"  z: {lever_arm_error[2]:.4f} m")
        print(f"  |error|: {np.linalg.norm(lever_arm_error):.4f} m")
    
    # Plot results
    print("\nGenerating plots...")
    plot_comparison(
        time_l,
        truth_positions,
        estimate,
        args.output,
        title=f"Dual IMU ESKF with Online Calibration (Master: {args.master.upper()})",
    )
    
    plot_residuals(time_l, diagnostics, args.output_residuals)
    
    plot_lever_arm_estimation(
        time_l,
        lever_arms,
        true_lever_arm,
        args.output_lever_arm,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
