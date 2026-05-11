"""
Error-State Kalman Filter (ESKF) for Triple IMU navigation with rigid body constraints
and online lever arm (extrinsic) calibration.

This implementation uses:
- Center IMU (IMU_C) as master for state prediction
- Left IMU (IMU_L) and Right IMU (IMU_R) as slaves for observation updates
- Online estimation of lever arms (extrinsic parameters) between IMUs
- Rigid body dynamics for improved reliability through redundancy

State Vector (33-dimensional):
- δp (3): Position error
- δv (3): Velocity error
- δθ (3): Orientation error (rotation vector)
- δba_c (3): Center accelerometer bias error (master)
- δbg_c (3): Center gyroscope bias error (master)
- δba_l (3): Left accelerometer bias error (slave)
- δbg_l (3): Left gyroscope bias error (slave)
- δba_r (3): Right accelerometer bias error (slave)
- δbg_r (3): Right gyroscope bias error (slave)
- δr_l (3): Lever arm error (center to left)
- δr_r (3): Lever arm error (center to right)

Rigid Body Constraints:
1. Gyroscope constraint: ω_slave = ω_master (same angular velocity)
2. Accelerometer constraint: a_slave = a_master + α × r + ω × (ω × r)
   where r is the lever arm from master to slave, α is angular acceleration

Observation Model (12D for two slaves):
- z_gyro_l (3): Gyro constraint residual for left IMU
- z_acc_l (3): Accelerometer constraint residual for left IMU
- z_gyro_r (3): Gyro constraint residual for right IMU  
- z_acc_r (3): Accelerometer constraint residual for right IMU

Key Features:
- Triple IMU redundancy for increased reliability
- Online lever arm calibration for both slaves
- Cross-validation between slaves for outlier detection
- Configurable angular acceleration source (estimated vs ground truth)
- Observability-aware updates when |ω| is small
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

class AngularAccelerationSource(Enum):
    """Source for angular acceleration estimation."""
    GYRO_DIFFERENTIATION = "gyro_diff"  # Estimate from gyro differentiation (noisy)
    GROUND_TRUTH = "ground_truth"        # Use ground truth from CSV
    ZERO = "zero"                         # Assume zero (ignore tangential term)


@dataclass
class TripleESKFConfig:
    """Configuration parameters for Triple IMU ESKF with online extrinsic calibration."""
    
    # ----- Process Noise -----
    sigma_acc: float = 0.1              # Accelerometer noise (m/s^2)
    sigma_gyro: float = 0.01            # Gyroscope noise (rad/s)
    sigma_acc_bias: float = 0.001       # Accelerometer bias random walk
    sigma_gyro_bias: float = 0.0001     # Gyroscope bias random walk
    sigma_lever_arm: float = 0.0001     # Lever arm random walk (very small for rigid body)
    
    # ----- Measurement Noise -----
    sigma_gyro_constraint: float = 0.02     # Gyroscope constraint noise (rad/s)
    sigma_acc_constraint: float = 0.1       # Accelerometer constraint noise (m/s^2)
    
    # ----- Numerical Stability -----
    min_omega_for_lever_obs: float = 0.1    # Minimum |ω| for lever arm observability (rad/s)
    max_lever_arm_variance: float = 10.0    # Maximum lever arm variance (m^2)
    
    # ----- Outlier Rejection -----
    enable_outlier_rejection: bool = True
    outlier_threshold: float = 5.0          # Chi-squared threshold (sigma level)
    
    # ----- Initial Uncertainties -----
    init_sigma_pos: float = 0.01
    init_sigma_vel: float = 0.01
    init_sigma_theta: float = 0.01
    init_sigma_acc_bias: float = 0.1
    init_sigma_gyro_bias: float = 0.01
    init_sigma_lever_arm: float = 1.0       # Large for online estimation
    
    # ----- Lever Arm Initial Guesses -----
    # Lever arm from center to left IMU (in body frame)
    lever_arm_left_init: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.0, 0.0]))
    # Lever arm from center to right IMU (in body frame)
    lever_arm_right_init: np.ndarray = field(default_factory=lambda: np.array([-0.5, 0.0, 0.0]))
    
    # Known lever arm magnitudes (if known)
    lever_arm_left_magnitude: float | None = None
    lever_arm_right_magnitude: float | None = None
    enforce_magnitude_constraint: bool = True
    
    # ----- Angular Acceleration Source -----
    alpha_source: AngularAccelerationSource = AngularAccelerationSource.GYRO_DIFFERENTIATION
    
    # ----- Gyro Differentiation Smoothing -----
    alpha_smoothing_factor: float = 0.3     # Exponential smoothing for estimated alpha
    
    # ----- GPS Position Update (Surface Mode) -----
    enable_gps_update: bool = False         # Enable GPS position updates when available
    sigma_gps_pos: float = 0.01             # GPS position measurement noise (m)
    gps_cutoff_time: float = 0.0            # Time at which GPS becomes unavailable
    gps_underwater_duration: float = 0.0    # GPS still available for this duration after submersion
    
    # ----- Depth Sensor (Underwater Mode) -----
    enable_depth_update: bool = False       # Enable depth sensor updates when underwater
    sigma_depth: float = 0.1                # Depth sensor measurement noise (m)
    depth_start_time: float = 0.0           # Time at which depth sensor becomes available
    
    # ----- Heading Alignment (Hard Reset at Submersion) -----
    enable_heading_alignment: bool = False  # Enable hard reset of orientation at alignment time
    heading_alignment_time: float = 0.0     # Time at which to perform heading alignment
    initial_heading_deg: float = 90.0       # Initial heading in degrees (0=+X, 90=+Y)


# =============================================================================
# Nominal State
# =============================================================================

@dataclass 
class NominalState:
    """Nominal state for Triple IMU ESKF."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))  # [w, x, y, z]
    
    # Center (master) IMU biases
    acc_bias_c: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias_c: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Left (slave) IMU biases
    acc_bias_l: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias_l: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Right (slave) IMU biases
    acc_bias_r: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias_r: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Lever arms (center to slave)
    lever_arm_l: np.ndarray = field(default_factory=lambda: np.zeros(3))
    lever_arm_r: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def copy(self) -> "NominalState":
        return NominalState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            acc_bias_c=self.acc_bias_c.copy(),
            gyro_bias_c=self.gyro_bias_c.copy(),
            acc_bias_l=self.acc_bias_l.copy(),
            gyro_bias_l=self.gyro_bias_l.copy(),
            acc_bias_r=self.acc_bias_r.copy(),
            gyro_bias_r=self.gyro_bias_r.copy(),
            lever_arm_l=self.lever_arm_l.copy(),
            lever_arm_r=self.lever_arm_r.copy(),
        )


# =============================================================================
# Triple IMU ESKF with Online Extrinsic Calibration
# =============================================================================

class TripleIMU_ESKF:
    """
    Error-State Kalman Filter for Triple IMU navigation with:
    - Rigid body constraints from two slave IMUs
    - Online lever arm (extrinsic) calibration
    - Redundancy for improved reliability
    
    State vector (33-dimensional):
    - δp (3): Position error
    - δv (3): Velocity error  
    - δθ (3): Orientation error (rotation vector)
    - δba_c (3): Center accelerometer bias error
    - δbg_c (3): Center gyroscope bias error
    - δba_l (3): Left accelerometer bias error
    - δbg_l (3): Left gyroscope bias error
    - δba_r (3): Right accelerometer bias error
    - δbg_r (3): Right gyroscope bias error
    - δr_l (3): Lever arm error (center to left)
    - δr_r (3): Lever arm error (center to right)
    """
    
    # State indices
    POS_IDX = slice(0, 3)       # Position error
    VEL_IDX = slice(3, 6)       # Velocity error
    THETA_IDX = slice(6, 9)     # Orientation error
    BA_C_IDX = slice(9, 12)     # Center accelerometer bias
    BG_C_IDX = slice(12, 15)    # Center gyroscope bias
    BA_L_IDX = slice(15, 18)    # Left accelerometer bias
    BG_L_IDX = slice(18, 21)    # Left gyroscope bias
    BA_R_IDX = slice(21, 24)    # Right accelerometer bias
    BG_R_IDX = slice(24, 27)    # Right gyroscope bias
    LEVER_L_IDX = slice(27, 30) # Lever arm (center to left)
    LEVER_R_IDX = slice(30, 33) # Lever arm (center to right)
    STATE_DIM = 33
    
    def __init__(self, config: TripleESKFConfig | None = None):
        self.config = config or TripleESKFConfig()
        self.nominal = NominalState()
        
        # Initialize lever arms from config
        self.nominal.lever_arm_l = self.config.lever_arm_left_init.copy()
        self.nominal.lever_arm_r = self.config.lever_arm_right_init.copy()
        
        # Previous gyro for angular acceleration estimation
        self.prev_gyro: np.ndarray | None = None
        self.smoothed_alpha: np.ndarray = np.zeros(3)
        
        # Initialize error state covariance (33x33)
        self._init_covariance()
        
        # Build process noise matrix
        self._build_process_noise()
    
    def _init_covariance(self) -> None:
        """Initialize error state covariance matrix."""
        diag = np.concatenate([
            # Position (3)
            np.full(3, self.config.init_sigma_pos**2),
            # Velocity (3)
            np.full(3, self.config.init_sigma_vel**2),
            # Orientation (3)
            np.full(3, self.config.init_sigma_theta**2),
            # Center acc bias (3)
            np.full(3, self.config.init_sigma_acc_bias**2),
            # Center gyro bias (3)
            np.full(3, self.config.init_sigma_gyro_bias**2),
            # Left acc bias (3)
            np.full(3, self.config.init_sigma_acc_bias**2),
            # Left gyro bias (3)
            np.full(3, self.config.init_sigma_gyro_bias**2),
            # Right acc bias (3)
            np.full(3, self.config.init_sigma_acc_bias**2),
            # Right gyro bias (3)
            np.full(3, self.config.init_sigma_gyro_bias**2),
            # Lever arm left (3)
            np.full(3, self.config.init_sigma_lever_arm**2),
            # Lever arm right (3)
            np.full(3, self.config.init_sigma_lever_arm**2),
        ])
        self.P = np.diag(diag)
    
    def _build_process_noise(self) -> None:
        """Build the continuous-time process noise matrix."""
        self.Q_c = np.diag(np.concatenate([
            # Position (driven by velocity)
            np.zeros(3),
            # Velocity
            np.full(3, self.config.sigma_acc**2),
            # Orientation
            np.full(3, self.config.sigma_gyro**2),
            # Center acc bias
            np.full(3, self.config.sigma_acc_bias**2),
            # Center gyro bias
            np.full(3, self.config.sigma_gyro_bias**2),
            # Left acc bias
            np.full(3, self.config.sigma_acc_bias**2),
            # Left gyro bias
            np.full(3, self.config.sigma_gyro_bias**2),
            # Right acc bias
            np.full(3, self.config.sigma_acc_bias**2),
            # Right gyro bias
            np.full(3, self.config.sigma_gyro_bias**2),
            # Lever arm left (very small - rigid body)
            np.full(3, self.config.sigma_lever_arm**2),
            # Lever arm right (very small - rigid body)
            np.full(3, self.config.sigma_lever_arm**2),
        ]))
    
    # -------------------------------------------------------------------------
    # Configuration Methods
    # -------------------------------------------------------------------------
    
    def set_lever_arms_init(
        self, 
        lever_arm_l: np.ndarray, 
        lever_arm_r: np.ndarray
    ) -> None:
        """Set initial guesses for both lever arms."""
        self.nominal.lever_arm_l = lever_arm_l.copy()
        self.nominal.lever_arm_r = lever_arm_r.copy()
    
    def initialize(
        self,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
        quaternion: np.ndarray | None = None,
        lever_arm_l: np.ndarray | None = None,
        lever_arm_r: np.ndarray | None = None,
    ) -> None:
        """Initialize the filter with known initial state."""
        self.nominal.position = position.copy()
        if velocity is not None:
            self.nominal.velocity = velocity.copy()
        if quaternion is not None:
            self.nominal.quaternion = quaternion.copy()
        if lever_arm_l is not None:
            self.nominal.lever_arm_l = lever_arm_l.copy()
        if lever_arm_r is not None:
            self.nominal.lever_arm_r = lever_arm_r.copy()
        
        self.prev_gyro = None
        self.smoothed_alpha = np.zeros(3)
    
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
        dt: float,
        true_alpha: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute angular acceleration based on configured source.
        
        Args:
            gyro_current: Current gyroscope reading (bias-corrected)
            dt: Time step
            true_alpha: Ground truth angular acceleration (if available)
        
        Returns:
            Angular acceleration estimate
        """
        if self.config.alpha_source == AngularAccelerationSource.GROUND_TRUTH:
            if true_alpha is not None:
                return true_alpha.copy()
            # Fallback to differentiation if ground truth not available
        
        if self.config.alpha_source == AngularAccelerationSource.ZERO:
            return np.zeros(3)
        
        # GYRO_DIFFERENTIATION (default)
        if self.prev_gyro is None or dt <= 0:
            alpha_raw = np.zeros(3)
        else:
            alpha_raw = (gyro_current - self.prev_gyro) / dt
        
        # Apply exponential smoothing to reduce noise
        alpha_factor = self.config.alpha_smoothing_factor
        self.smoothed_alpha = (
            alpha_factor * alpha_raw + 
            (1 - alpha_factor) * self.smoothed_alpha
        )
        
        return self.smoothed_alpha.copy()
    
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
        
        Args:
            acc_master: Master accelerometer reading (bias-corrected)
            gyro_master: Master gyroscope reading (bias-corrected)
            angular_acc: Angular acceleration estimate (α = ω̇)
            lever_arm: Lever arm from master to slave
        
        Returns:
            (theoretical_acc_slave, theoretical_gyro_slave)
        """
        omega = gyro_master
        alpha = angular_acc
        r = lever_arm
        
        # Gyroscope: same angular velocity for rigid body
        theoretical_gyro = gyro_master.copy()
        
        # Accelerometer: a_slave = a_master + α × r + ω × (ω × r)
        tangential = np.cross(alpha, r)           # α × r
        centripetal = np.cross(omega, np.cross(omega, r))  # ω × (ω × r)
        
        theoretical_acc = acc_master + tangential + centripetal
        
        return theoretical_acc, theoretical_gyro
    
    # -------------------------------------------------------------------------
    # ESKF Prediction Step
    # -------------------------------------------------------------------------
    
    def predict(
        self, 
        acc_c: np.ndarray, 
        gyro_c: np.ndarray, 
        dt: float
    ) -> None:
        """
        Prediction step using center (master) IMU.
        
        Args:
            acc_c: Center accelerometer reading (m/s^2)
            gyro_c: Center gyroscope reading (rad/s)
            dt: Time step (s)
        """
        if dt <= 0:
            return
        
        # Correct measurements with current bias estimates
        acc_corrected = acc_c - self.nominal.acc_bias_c
        gyro_corrected = gyro_c - self.nominal.gyro_bias_c
        
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
        self.nominal.quaternion = self.quaternion_multiply(
            self.nominal.quaternion, delta_q
        )
        self.nominal.quaternion /= np.linalg.norm(self.nominal.quaternion)
        
        # Build state transition matrix F (33x33)
        F = np.eye(self.STATE_DIM)
        
        # ∂δp/∂δv
        F[self.POS_IDX, self.VEL_IDX] = np.eye(3) * dt
        
        # ∂δv/∂δθ
        F[self.VEL_IDX, self.THETA_IDX] = -R @ self.skew_symmetric(acc_corrected) * dt
        
        # ∂δv/∂δba_c
        F[self.VEL_IDX, self.BA_C_IDX] = -R * dt
        
        # ∂δθ/∂δbg_c
        F[self.THETA_IDX, self.BG_C_IDX] = -np.eye(3) * dt
        
        # Lever arms: identity transition (constant in body frame)
        
        # Process noise
        Q = self.Q_c * dt
        
        # Propagate covariance
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        
        # Covariance bounds
        max_var = self.config.max_lever_arm_variance * 100
        diag_P = np.diag(self.P)
        if np.any(diag_P > max_var) or np.any(np.isnan(diag_P)):
            self._init_covariance()
    
    # -------------------------------------------------------------------------
    # ESKF Measurement Update with Both Slave IMUs
    # -------------------------------------------------------------------------
    
    def update_with_slave_imus(
        self,
        acc_c: np.ndarray,
        gyro_c: np.ndarray,
        acc_l: np.ndarray,
        gyro_l: np.ndarray,
        acc_r: np.ndarray,
        gyro_r: np.ndarray,
        dt: float,
        true_angular_acc: np.ndarray | None = None,
    ) -> dict:
        """
        Measurement update using both slave IMUs with rigid body constraints.
        
        Observation equation (12D):
        z = [z_gyro_l; z_acc_l; z_gyro_r; z_acc_r]
        
        Args:
            acc_c: Center accelerometer reading
            gyro_c: Center gyroscope reading
            acc_l: Left accelerometer reading
            gyro_l: Left gyroscope reading
            acc_r: Right accelerometer reading
            gyro_r: Right gyroscope reading
            dt: Time step
            true_angular_acc: Ground truth angular acceleration (optional)
        
        Returns:
            Diagnostic information dictionary
        """
        # Correct for biases
        acc_c_corr = acc_c - self.nominal.acc_bias_c
        gyro_c_corr = gyro_c - self.nominal.gyro_bias_c
        acc_l_corr = acc_l - self.nominal.acc_bias_l
        gyro_l_corr = gyro_l - self.nominal.gyro_bias_l
        acc_r_corr = acc_r - self.nominal.acc_bias_r
        gyro_r_corr = gyro_r - self.nominal.gyro_bias_r
        
        # Current lever arm estimates
        r_l = self.nominal.lever_arm_l
        r_r = self.nominal.lever_arm_r
        omega = gyro_c_corr
        
        # Compute angular acceleration
        angular_acc = self.compute_angular_acceleration(
            gyro_c_corr, dt, true_angular_acc
        )
        
        # Update previous gyro for next iteration
        self.prev_gyro = gyro_c_corr.copy()
        
        # Compute theoretical slave readings
        theoretical_acc_l, theoretical_gyro_l = self.compute_theoretical_slave_readings(
            acc_c_corr, gyro_c_corr, angular_acc, r_l
        )
        theoretical_acc_r, theoretical_gyro_r = self.compute_theoretical_slave_readings(
            acc_c_corr, gyro_c_corr, angular_acc, r_r
        )
        
        # Observation residuals
        residual_gyro_l = gyro_l_corr - theoretical_gyro_l
        residual_acc_l = acc_l_corr - theoretical_acc_l
        residual_gyro_r = gyro_r_corr - theoretical_gyro_r
        residual_acc_r = acc_r_corr - theoretical_acc_r
        
        # Combined observation vector (12D)
        z = np.concatenate([
            residual_gyro_l, residual_acc_l,
            residual_gyro_r, residual_acc_r
        ])
        
        # Build observation Jacobian H (12 x 33)
        H = np.zeros((12, self.STATE_DIM))
        
        # Helper matrices
        omega_cross = self.skew_symmetric(omega)
        alpha_cross = self.skew_symmetric(angular_acc)
        
        # ----- Left IMU Constraints (rows 0-5) -----
        # Gyro constraint: z_gyro_l = (gyro_l - bg_l) - (gyro_c - bg_c)
        H[0:3, self.BG_C_IDX] = np.eye(3)
        H[0:3, self.BG_L_IDX] = -np.eye(3)
        
        # Acc constraint: z_acc_l = (acc_l - ba_l) - (acc_c - ba_c + α×r_l + ω×(ω×r_l))
        H[3:6, self.BA_C_IDX] = np.eye(3)
        H[3:6, self.BA_L_IDX] = -np.eye(3)
        
        # ∂z_acc_l/∂δbg_c (from centripetal term)
        r_l_cross = self.skew_symmetric(r_l)
        omega_cross_r_l = np.cross(omega, r_l)
        omega_cross_r_l_cross = self.skew_symmetric(omega_cross_r_l)
        d_centripetal_l_d_omega = omega_cross @ r_l_cross + omega_cross_r_l_cross
        H[3:6, self.BG_C_IDX] = -d_centripetal_l_d_omega
        
        # ∂z_acc_l/∂δr_l (lever arm observability)
        # H = [α]× + [ω]×[ω]×
        d_tangential_l_d_r = alpha_cross
        d_centripetal_l_d_r = omega_cross @ omega_cross
        H_lever_l = d_tangential_l_d_r + d_centripetal_l_d_r
        
        # Check observability for left lever arm
        omega_norm = np.linalg.norm(omega)
        if omega_norm >= self.config.min_omega_for_lever_obs:
            H[3:6, self.LEVER_L_IDX] = H_lever_l
        # else: H[3:6, self.LEVER_L_IDX] remains zero
        
        # ----- Right IMU Constraints (rows 6-11) -----
        # Gyro constraint
        H[6:9, self.BG_C_IDX] = np.eye(3)
        H[6:9, self.BG_R_IDX] = -np.eye(3)
        
        # Acc constraint
        H[9:12, self.BA_C_IDX] = np.eye(3)
        H[9:12, self.BA_R_IDX] = -np.eye(3)
        
        # ∂z_acc_r/∂δbg_c
        r_r_cross = self.skew_symmetric(r_r)
        omega_cross_r_r = np.cross(omega, r_r)
        omega_cross_r_r_cross = self.skew_symmetric(omega_cross_r_r)
        d_centripetal_r_d_omega = omega_cross @ r_r_cross + omega_cross_r_r_cross
        H[9:12, self.BG_C_IDX] = -d_centripetal_r_d_omega
        
        # ∂z_acc_r/∂δr_r
        d_tangential_r_d_r = alpha_cross
        d_centripetal_r_d_r = omega_cross @ omega_cross
        H_lever_r = d_tangential_r_d_r + d_centripetal_r_d_r
        
        if omega_norm >= self.config.min_omega_for_lever_obs:
            H[9:12, self.LEVER_R_IDX] = H_lever_r
        
        # Measurement noise covariance (12x12)
        R_meas = np.diag(np.concatenate([
            np.full(3, self.config.sigma_gyro_constraint**2),  # left gyro
            np.full(3, self.config.sigma_acc_constraint**2),   # left acc
            np.full(3, self.config.sigma_gyro_constraint**2),  # right gyro
            np.full(3, self.config.sigma_acc_constraint**2),   # right acc
        ]))
        
        # Innovation covariance
        S = H @ self.P @ H.T + R_meas
        
        # Outlier rejection
        outlier_rejected = False
        if self.config.enable_outlier_rejection:
            try:
                S_inv = np.linalg.inv(S)
                mahalanobis_sq = z.T @ S_inv @ z
                threshold = self.config.outlier_threshold**2 * 12
                if mahalanobis_sq > threshold:
                    outlier_rejected = True
                    return {
                        "residual_gyro_l": residual_gyro_l,
                        "residual_acc_l": residual_acc_l,
                        "residual_gyro_r": residual_gyro_r,
                        "residual_acc_r": residual_acc_r,
                        "angular_acc": angular_acc,
                        "lever_arm_l_estimate": r_l.copy(),
                        "lever_arm_r_estimate": r_r.copy(),
                        "lever_arm_l_std": np.sqrt(np.diag(self.P)[self.LEVER_L_IDX]),
                        "lever_arm_r_std": np.sqrt(np.diag(self.P)[self.LEVER_R_IDX]),
                        "outlier_rejected": True,
                    }
            except np.linalg.LinAlgError:
                pass
        
        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = self.P @ H.T @ S_inv
        
        # Error state update
        delta_x = K @ z
        
        # Inject error into nominal state
        self._inject_error(delta_x)
        
        # Update covariance (Joseph form)
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        # Covariance bounds for lever arms
        self._apply_covariance_bounds()
        
        return {
            "residual_gyro_l": residual_gyro_l,
            "residual_acc_l": residual_acc_l,
            "residual_gyro_r": residual_gyro_r,
            "residual_acc_r": residual_acc_r,
            "angular_acc": angular_acc,
            "lever_arm_l_estimate": self.nominal.lever_arm_l.copy(),
            "lever_arm_r_estimate": self.nominal.lever_arm_r.copy(),
            "lever_arm_l_std": np.sqrt(np.diag(self.P)[self.LEVER_L_IDX]),
            "lever_arm_r_std": np.sqrt(np.diag(self.P)[self.LEVER_R_IDX]),
            "outlier_rejected": outlier_rejected,
        }
    
    def _apply_covariance_bounds(self) -> None:
        """Apply bounds to prevent covariance explosion."""
        max_var = self.config.max_lever_arm_variance
        
        for lever_idx in [self.LEVER_L_IDX, self.LEVER_R_IDX]:
            lever_var = np.diag(self.P)[lever_idx]
            if np.any(lever_var > max_var):
                scale = np.sqrt(max_var / np.maximum(lever_var, 1e-10))
                for i, idx in enumerate(range(lever_idx.start, lever_idx.stop)):
                    if lever_var[i] > max_var:
                        self.P[idx, :] *= scale[i]
                        self.P[:, idx] *= scale[i]
    
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
        self.nominal.quaternion = self.quaternion_multiply(
            self.nominal.quaternion, delta_q
        )
        self.nominal.quaternion /= np.linalg.norm(self.nominal.quaternion)
        
        # Biases: additive
        self.nominal.acc_bias_c += delta_x[self.BA_C_IDX]
        self.nominal.gyro_bias_c += delta_x[self.BG_C_IDX]
        self.nominal.acc_bias_l += delta_x[self.BA_L_IDX]
        self.nominal.gyro_bias_l += delta_x[self.BG_L_IDX]
        self.nominal.acc_bias_r += delta_x[self.BA_R_IDX]
        self.nominal.gyro_bias_r += delta_x[self.BG_R_IDX]
        
        # Lever arms: additive
        self.nominal.lever_arm_l += delta_x[self.LEVER_L_IDX]
        self.nominal.lever_arm_r += delta_x[self.LEVER_R_IDX]
        
        # Enforce magnitude constraints if configured
        if self.config.enforce_magnitude_constraint:
            if self.config.lever_arm_left_magnitude is not None:
                self._enforce_lever_arm_magnitude(
                    "lever_arm_l", 
                    self.config.lever_arm_left_magnitude,
                    self.LEVER_L_IDX
                )
            if self.config.lever_arm_right_magnitude is not None:
                self._enforce_lever_arm_magnitude(
                    "lever_arm_r",
                    self.config.lever_arm_right_magnitude,
                    self.LEVER_R_IDX
                )
    
    def _enforce_lever_arm_magnitude(
        self, 
        attr_name: str, 
        target_magnitude: float,
        state_idx: slice
    ) -> None:
        """Enforce hard constraint on lever arm magnitude."""
        r = getattr(self.nominal, attr_name)
        r_norm = np.linalg.norm(r)
        
        if r_norm < 1e-10:
            # Cannot determine direction, set along x-axis
            setattr(self.nominal, attr_name, np.array([target_magnitude, 0.0, 0.0]))
            return
        
        # Project onto sphere
        scale = target_magnitude / r_norm
        setattr(self.nominal, attr_name, r * scale)
        
        # Update covariance to remove radial uncertainty
        r_hat = r / r_norm
        P_tangent = np.eye(3) - np.outer(r_hat, r_hat)
        
        P_lever = self.P[state_idx, state_idx].copy()
        P_lever_constrained = P_tangent @ P_lever @ P_tangent.T
        P_lever_constrained += 1e-8 * np.outer(r_hat, r_hat)
        
        self.P[state_idx, state_idx] = P_lever_constrained
    
    # -------------------------------------------------------------------------
    # State Getters
    # -------------------------------------------------------------------------
    
    def get_position(self) -> np.ndarray:
        return self.nominal.position.copy()
    
    def get_velocity(self) -> np.ndarray:
        return self.nominal.velocity.copy()
    
    def get_lever_arms(self) -> tuple[np.ndarray, np.ndarray]:
        """Get estimated lever arms (center to left, center to right)."""
        return self.nominal.lever_arm_l.copy(), self.nominal.lever_arm_r.copy()
    
    def get_lever_arm_uncertainties(self) -> tuple[np.ndarray, np.ndarray]:
        """Get lever arm standard deviations."""
        return (
            np.sqrt(np.diag(self.P)[self.LEVER_L_IDX]),
            np.sqrt(np.diag(self.P)[self.LEVER_R_IDX]),
        )
    
    def get_biases(self) -> dict[str, np.ndarray]:
        """Get all bias estimates."""
        return {
            "center_acc_bias": self.nominal.acc_bias_c.copy(),
            "center_gyro_bias": self.nominal.gyro_bias_c.copy(),
            "left_acc_bias": self.nominal.acc_bias_l.copy(),
            "left_gyro_bias": self.nominal.gyro_bias_l.copy(),
            "right_acc_bias": self.nominal.acc_bias_r.copy(),
            "right_gyro_bias": self.nominal.gyro_bias_r.copy(),
        }
    
    def get_euler_angles(self) -> np.ndarray:
        """Get current orientation as Euler angles (roll, pitch, yaw)."""
        q = self.nominal.quaternion
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        return r.as_euler('xyz')
    
    def update_with_gps(self, gps_position: np.ndarray) -> dict:
        """
        Update with GPS position measurement.
        
        Args:
            gps_position: GPS measured position [x, y, z] in world frame
            
        Returns:
            Diagnostic information
        """
        y = gps_position - self.nominal.position
        
        H = np.zeros((3, self.STATE_DIM))
        H[:, self.POS_IDX] = np.eye(3)
        
        R = np.eye(3) * self.config.sigma_gps_pos**2
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        delta_x = K @ y
        self._inject_error(delta_x)
        
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        return {"gps_residual": y, "gps_innovation_norm": np.linalg.norm(y)}
    
    def update_with_depth(self, depth: float) -> dict:
        """
        Update with depth sensor measurement.
        
        Args:
            depth: Measured depth (Z coordinate) in world frame
            
        Returns:
            Diagnostic information
        """
        y = np.array([depth - self.nominal.position[2]])
        
        H = np.zeros((1, self.STATE_DIM))
        H[0, 2] = 1.0
        
        R = np.array([[self.config.sigma_depth**2]])
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        delta_x = K @ y
        self._inject_error(delta_x)
        
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
        q = true_quaternion / np.linalg.norm(true_quaternion)
        self.nominal.quaternion = q.copy()
        
        small_sigma = 0.001
        self.P[self.THETA_IDX, self.THETA_IDX] = np.eye(3) * small_sigma**2
        
        for idx in [self.POS_IDX, self.VEL_IDX, self.BA_C_IDX, self.BG_C_IDX,
                    self.BA_L_IDX, self.BG_L_IDX, self.BA_R_IDX, self.BG_R_IDX,
                    self.LEVER_L_IDX, self.LEVER_R_IDX]:
            self.P[self.THETA_IDX, idx] = 0
            self.P[idx, self.THETA_IDX] = 0


# =============================================================================
# Data Loading
# =============================================================================

def load_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Load IMU data from CSV file.
    
    Returns:
        time, positions, accelerations, angular_velocities, angular_accelerations
    """
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    time = raw[:, 0]
    positions = raw[:, 1:4]
    accelerations = raw[:, 4:7]
    angular_velocities = raw[:, 7:10]
    
    if raw.shape[1] >= 13:
        angular_accelerations = raw[:, 10:13]
    else:
        angular_accelerations = None
    
    return time, positions, accelerations, angular_velocities, angular_accelerations


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
    
    true_depth = true_positions[:, 2]
    noise = np.random.normal(0, sigma_depth, size=true_depth.shape)
    
    return true_depth + noise


# =============================================================================
# Main ESKF Runner
# =============================================================================

def run_triple_eskf(
    time: np.ndarray,
    acc_c: np.ndarray,
    gyro_c: np.ndarray,
    acc_l: np.ndarray,
    gyro_l: np.ndarray,
    acc_r: np.ndarray,
    gyro_r: np.ndarray,
    initial_position: np.ndarray,
    config: TripleESKFConfig | None = None,
    true_angular_acc: np.ndarray | None = None,
    gps_positions: np.ndarray | None = None,
    depth_measurements: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    Run Triple IMU ESKF with online extrinsic calibration.
    
    Args:
        time: Timestamps
        acc_c: Center accelerometer data (n, 3)
        gyro_c: Center gyroscope data (n, 3)
        acc_l: Left accelerometer data (n, 3)
        gyro_l: Left gyroscope data (n, 3)
        acc_r: Right accelerometer data (n, 3)
        gyro_r: Right gyroscope data (n, 3)
        initial_position: Initial position (3,)
        config: ESKF configuration
        true_angular_acc: Ground truth angular acceleration (n, 3)
        gps_positions: GPS position data for surface updates
        depth_measurements: Depth sensor data for underwater updates
    
    Returns:
        positions: Estimated positions (n, 3)
        lever_arms_l: Estimated left lever arms (n, 3)
        lever_arms_r: Estimated right lever arms (n, 3)
        diagnostics: List of diagnostic info per timestep
    """
    if config is None:
        config = TripleESKFConfig()
    
    eskf = TripleIMU_ESKF(config)
    eskf.initialize(
        position=initial_position,
        lever_arm_l=config.lever_arm_left_init,
        lever_arm_r=config.lever_arm_right_init,
    )
    
    n = len(time)
    positions = np.zeros((n, 3))
    lever_arms_l = np.zeros((n, 3))
    lever_arms_r = np.zeros((n, 3))
    
    positions[0] = initial_position.copy()
    lever_arms_l[0], lever_arms_r[0] = eskf.get_lever_arms()
    
    diagnostics = []
    
    # GPS configuration
    gps_cutoff = config.gps_cutoff_time + config.gps_underwater_duration
    use_gps = config.enable_gps_update
    
    # Depth sensor configuration
    depth_start = config.depth_start_time
    use_depth = config.enable_depth_update
    
    # Heading alignment configuration
    heading_alignment_time = config.heading_alignment_time
    use_heading_alignment = config.enable_heading_alignment
    heading_aligned = False
    
    gps_update_count = 0
    depth_update_count = 0
    
    for idx in range(1, n):
        dt = time[idx] - time[idx - 1]
        current_time = time[idx]
        
        # Prediction with center IMU
        eskf.predict(
            acc_c=acc_c[idx],
            gyro_c=gyro_c[idx],
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
        
        # Get true angular acceleration for this timestep if configured
        alpha = None
        if (config.alpha_source == AngularAccelerationSource.GROUND_TRUTH 
            and true_angular_acc is not None):
            alpha = true_angular_acc[idx]
        
        # Update with both slave IMUs
        diag = eskf.update_with_slave_imus(
            acc_c=acc_c[idx],
            gyro_c=gyro_c[idx],
            acc_l=acc_l[idx],
            gyro_l=gyro_l[idx],
            acc_r=acc_r[idx],
            gyro_r=gyro_r[idx],
            dt=dt,
            true_angular_acc=alpha,
        )
        diagnostics.append(diag)
        
        positions[idx] = eskf.get_position()
        lever_arms_l[idx], lever_arms_r[idx] = eskf.get_lever_arms()
    
    if use_gps:
        print(f"  GPS updates applied: {gps_update_count} (t < {gps_cutoff}s)")
    if use_depth:
        print(f"  Depth updates applied: {depth_update_count} (t >= {depth_start}s)")
    
    return positions, lever_arms_l, lever_arms_r, diagnostics


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_trajectory_comparison(
    time: np.ndarray,
    truth: np.ndarray,
    estimate: np.ndarray,
    output_path: Path,
    title: str = "Triple IMU ESKF Trajectory",
) -> None:
    """Plot comparison between ground truth and estimated trajectory."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

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

    axes[1].plot(
        truth[:, 0], truth[:, 1], 
        label="truth XY", color="tab:red", linewidth=2
    )
    axes[1].plot(
        estimate[:, 0], estimate[:, 1],
        label="estimate XY",
        color="tab:purple",
        linestyle="--",
        linewidth=2,
    )
    axes[1].scatter(
        [truth[0, 0]], [truth[0, 1]], 
        color="green", s=100, zorder=5, label="start"
    )
    axes[1].scatter(
        [truth[-1, 0]], [truth[-1, 1]], 
        color="red", s=100, zorder=5, label="end"
    )
    axes[1].set_title("XY trajectory")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].legend(fontsize="small")
    axes[1].axis("equal")
    axes[1].grid(True)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    print(f"Saved trajectory plot to {output_path}")


def plot_residuals(
    time: np.ndarray,
    diagnostics: list[dict],
    output_path: Path,
) -> None:
    """Plot constraint residuals for both slave IMUs."""
    residual_gyro_l = np.array([d["residual_gyro_l"] for d in diagnostics])
    residual_acc_l = np.array([d["residual_acc_l"] for d in diagnostics])
    residual_gyro_r = np.array([d["residual_gyro_r"] for d in diagnostics])
    residual_acc_r = np.array([d["residual_acc_r"] for d in diagnostics])
    angular_acc = np.array([d["angular_acc"] for d in diagnostics])
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), constrained_layout=True)
    
    labels = ["x", "y", "z"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    t = time[1:]
    
    # Left gyro residuals
    for i in range(3):
        axes[0].plot(t, residual_gyro_l[:, i], label=labels[i], color=colors[i])
    axes[0].set_title("Left IMU gyroscope residuals")
    axes[0].set_ylabel("residual (rad/s)")
    axes[0].legend()
    axes[0].grid(True)
    
    # Left acc residuals
    for i in range(3):
        axes[1].plot(t, residual_acc_l[:, i], label=labels[i], color=colors[i])
    axes[1].set_title("Left IMU accelerometer residuals")
    axes[1].set_ylabel("residual (m/s²)")
    axes[1].legend()
    axes[1].grid(True)
    
    # Right gyro residuals
    for i in range(3):
        axes[2].plot(t, residual_gyro_r[:, i], label=labels[i], color=colors[i])
    axes[2].set_title("Right IMU gyroscope residuals")
    axes[2].set_ylabel("residual (rad/s)")
    axes[2].legend()
    axes[2].grid(True)
    
    # Right acc residuals
    for i in range(3):
        axes[3].plot(t, residual_acc_r[:, i], label=labels[i], color=colors[i])
    axes[3].set_title("Right IMU accelerometer residuals")
    axes[3].set_ylabel("residual (m/s²)")
    axes[3].legend()
    axes[3].grid(True)
    
    # Angular acceleration
    for i in range(3):
        axes[4].plot(t, angular_acc[:, i], label=labels[i], color=colors[i])
    axes[4].set_title("Angular acceleration")
    axes[4].set_xlabel("time (s)")
    axes[4].set_ylabel("α (rad/s²)")
    axes[4].legend()
    axes[4].grid(True)
    
    fig.suptitle("Triple IMU Rigid Body Constraint Diagnostics")
    fig.savefig(output_path, dpi=150)
    print(f"Saved residuals plot to {output_path}")


def plot_lever_arm_estimation(
    time: np.ndarray,
    lever_arms_l: np.ndarray,
    lever_arms_r: np.ndarray,
    true_lever_arm_l: np.ndarray | None,
    true_lever_arm_r: np.ndarray | None,
    output_path: Path,
) -> None:
    """Plot lever arm estimation over time for both slaves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    
    labels = ["x", "y", "z"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    
    # Left lever arm components
    for i in range(3):
        axes[0, 0].plot(time, lever_arms_l[:, i], label=f"est {labels[i]}", color=colors[i])
        if true_lever_arm_l is not None:
            axes[0, 0].axhline(
                y=true_lever_arm_l[i], 
                color=colors[i], 
                linestyle="--", 
                alpha=0.7,
            )
    axes[0, 0].set_title("Left lever arm (C→L)")
    axes[0, 0].set_xlabel("time (s)")
    axes[0, 0].set_ylabel("r_l (m)")
    axes[0, 0].legend(fontsize="small")
    axes[0, 0].grid(True)
    
    # Right lever arm components
    for i in range(3):
        axes[0, 1].plot(time, lever_arms_r[:, i], label=f"est {labels[i]}", color=colors[i])
        if true_lever_arm_r is not None:
            axes[0, 1].axhline(
                y=true_lever_arm_r[i],
                color=colors[i],
                linestyle="--",
                alpha=0.7,
            )
    axes[0, 1].set_title("Right lever arm (C→R)")
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("r_r (m)")
    axes[0, 1].legend(fontsize="small")
    axes[0, 1].grid(True)
    
    # Left lever arm magnitude
    est_mag_l = np.linalg.norm(lever_arms_l, axis=1)
    axes[1, 0].plot(time, est_mag_l, label="estimated |r_l|", color="tab:purple")
    if true_lever_arm_l is not None:
        true_mag_l = np.linalg.norm(true_lever_arm_l)
        axes[1, 0].axhline(
            y=true_mag_l, color="tab:red", linestyle="--",
            label=f"true |r_l| = {true_mag_l:.3f} m"
        )
    axes[1, 0].set_title("Left lever arm magnitude")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("|r_l| (m)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Right lever arm magnitude
    est_mag_r = np.linalg.norm(lever_arms_r, axis=1)
    axes[1, 1].plot(time, est_mag_r, label="estimated |r_r|", color="tab:purple")
    if true_lever_arm_r is not None:
        true_mag_r = np.linalg.norm(true_lever_arm_r)
        axes[1, 1].axhline(
            y=true_mag_r, color="tab:red", linestyle="--",
            label=f"true |r_r| = {true_mag_r:.3f} m"
        )
    axes[1, 1].set_title("Right lever arm magnitude")
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 1].set_ylabel("|r_r| (m)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    fig.suptitle("Online Lever Arm (Extrinsic) Calibration - Triple IMU")
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
        description="Triple IMU ESKF navigation with rigid body constraints and online extrinsic calibration."
    )
    
    # ----- Input Files -----
    parser.add_argument(
        "--imu-c", type=Path, default=Path("IMU_C.csv"),
        help="Path to center (master) IMU CSV file",
    )
    parser.add_argument(
        "--imu-l", type=Path, default=Path("IMU_L.csv"),
        help="Path to left (slave) IMU CSV file",
    )
    parser.add_argument(
        "--imu-r", type=Path, default=Path("IMU_R.csv"),
        help="Path to right (slave) IMU CSV file",
    )
    parser.add_argument(
        "--imu-c-noised", type=Path, default=None,
        help="Optional noised center IMU CSV file",
    )
    parser.add_argument(
        "--imu-l-noised", type=Path, default=None,
        help="Optional noised left IMU CSV file",
    )
    parser.add_argument(
        "--imu-r-noised", type=Path, default=None,
        help="Optional noised right IMU CSV file",
    )
    
    # ----- Output Files -----
    parser.add_argument(
        "--output", type=Path, default=Path("triple_eskf_trajectory.png"),
        help="Path for trajectory figure",
    )
    parser.add_argument(
        "--output-residuals", type=Path, default=Path("triple_eskf_residuals.png"),
        help="Path for residuals figure",
    )
    parser.add_argument(
        "--output-lever-arm", type=Path, default=Path("triple_eskf_lever_arm.png"),
        help="Path for lever arm estimation figure",
    )
    
    # ----- Lever Arm Configuration -----
    parser.add_argument(
        "--lever-arm-l-init-x", type=float, default=0.5,
        help="Initial guess for left lever arm X (m)",
    )
    parser.add_argument(
        "--lever-arm-l-init-y", type=float, default=0.0,
        help="Initial guess for left lever arm Y (m)",
    )
    parser.add_argument(
        "--lever-arm-l-init-z", type=float, default=0.0,
        help="Initial guess for left lever arm Z (m)",
    )
    parser.add_argument(
        "--lever-arm-r-init-x", type=float, default=-0.5,
        help="Initial guess for right lever arm X (m)",
    )
    parser.add_argument(
        "--lever-arm-r-init-y", type=float, default=0.0,
        help="Initial guess for right lever arm Y (m)",
    )
    parser.add_argument(
        "--lever-arm-r-init-z", type=float, default=0.0,
        help="Initial guess for right lever arm Z (m)",
    )
    
    # True lever arms for comparison
    parser.add_argument(
        "--lever-arm-l-true-x", type=float, default=None,
        help="True left lever arm X (for comparison)",
    )
    parser.add_argument(
        "--lever-arm-r-true-x", type=float, default=None,
        help="True right lever arm X (for comparison)",
    )
    
    # ----- Angular Acceleration Source -----
    parser.add_argument(
        "--alpha-source", type=str, 
        choices=["gyro_diff", "ground_truth", "zero"],
        default="gyro_diff",
        help="Source for angular acceleration: gyro_diff (estimate from gyro), ground_truth (use CSV), zero (ignore)",
    )
    parser.add_argument(
        "--alpha-smoothing", type=float, default=0.3,
        help="Smoothing factor for gyro differentiation (0-1, higher = more responsive)",
    )
    
    # ----- Noise Parameters -----
    parser.add_argument(
        "--sigma-acc", type=float, default=0.1,
        help="Accelerometer noise std (m/s^2)",
    )
    parser.add_argument(
        "--sigma-gyro", type=float, default=0.01,
        help="Gyroscope noise std (rad/s)",
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
        "--init-sigma-lever-arm", type=float, default=0.5,
        help="Initial lever arm uncertainty std (m)",
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

    # Load IMU data
    print("Loading IMU data...")
    time_c, pos_c, acc_c, gyro_c, ang_acc_c = load_imu_csv(args.imu_c)
    time_l, pos_l, acc_l, gyro_l, _ = load_imu_csv(args.imu_l)
    time_r, pos_r, acc_r, gyro_r, _ = load_imu_csv(args.imu_r)
    
    # Verify time alignment
    if not (np.allclose(time_c, time_l) and np.allclose(time_c, time_r)):
        raise ValueError("All IMU files must have the same timestamps")
    
    # Use noised data if provided
    if args.imu_c_noised is not None:
        print(f"Using noised center IMU data from {args.imu_c_noised}")
        _, _, acc_c, gyro_c, _ = load_imu_csv(args.imu_c_noised)
    if args.imu_l_noised is not None:
        print(f"Using noised left IMU data from {args.imu_l_noised}")
        _, _, acc_l, gyro_l, _ = load_imu_csv(args.imu_l_noised)
    if args.imu_r_noised is not None:
        print(f"Using noised right IMU data from {args.imu_r_noised}")
        _, _, acc_r, gyro_r, _ = load_imu_csv(args.imu_r_noised)
    
    # Configure lever arms
    lever_arm_l_init = np.array([
        args.lever_arm_l_init_x,
        args.lever_arm_l_init_y,
        args.lever_arm_l_init_z,
    ])
    lever_arm_r_init = np.array([
        args.lever_arm_r_init_x,
        args.lever_arm_r_init_y,
        args.lever_arm_r_init_z,
    ])
    
    # True lever arms for comparison
    true_lever_arm_l = None
    true_lever_arm_r = None
    if args.lever_arm_l_true_x is not None:
        true_lever_arm_l = np.array([args.lever_arm_l_true_x, 0.0, 0.0])
    if args.lever_arm_r_true_x is not None:
        true_lever_arm_r = np.array([args.lever_arm_r_true_x, 0.0, 0.0])
    
    # Angular acceleration source
    alpha_source_map = {
        "gyro_diff": AngularAccelerationSource.GYRO_DIFFERENTIATION,
        "ground_truth": AngularAccelerationSource.GROUND_TRUTH,
        "zero": AngularAccelerationSource.ZERO,
    }
    alpha_source = alpha_source_map[args.alpha_source]
    
    # Check if ground truth alpha is available
    true_angular_acc = None
    if alpha_source == AngularAccelerationSource.GROUND_TRUTH:
        if ang_acc_c is None:
            print("WARNING: --alpha-source ground_truth specified but CSV lacks angular acceleration!")
            print("         Falling back to gyro differentiation.")
            alpha_source = AngularAccelerationSource.GYRO_DIFFERENTIATION
        else:
            print("Using ground truth angular acceleration from CSV.")
            true_angular_acc = ang_acc_c
    
    # Simulate depth sensor if enabled
    depth_measurements = None
    if args.enable_depth:
        print(f"Simulating depth sensor with sigma = {args.sigma_depth} m...")
        depth_measurements = simulate_depth_sensor(
            pos_c,
            sigma_depth=args.sigma_depth,
            seed=args.depth_seed,
        )
    
    # Configure ESKF
    config = TripleESKFConfig(
        sigma_acc=args.sigma_acc,
        sigma_gyro=args.sigma_gyro,
        sigma_acc_constraint=args.sigma_acc_constraint,
        sigma_gyro_constraint=args.sigma_gyro_constraint,
        init_sigma_lever_arm=args.init_sigma_lever_arm,
        lever_arm_left_init=lever_arm_l_init,
        lever_arm_right_init=lever_arm_r_init,
        alpha_source=alpha_source,
        alpha_smoothing_factor=args.alpha_smoothing,
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
    
    print(f"\nConfiguration:")
    print(f"  Master IMU: CENTER")
    print(f"  Slave IMUs: LEFT, RIGHT")
    print(f"  Initial left lever arm (C→L): {lever_arm_l_init}")
    print(f"  Initial right lever arm (C→R): {lever_arm_r_init}")
    print(f"  Angular acceleration source: {alpha_source.value}")
    if alpha_source == AngularAccelerationSource.GYRO_DIFFERENTIATION:
        print(f"  Alpha smoothing factor: {args.alpha_smoothing}")
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
    
    # Run Triple ESKF
    print("\nRunning Triple IMU ESKF...")
    estimate, lever_arms_l, lever_arms_r, diagnostics = run_triple_eskf(
        time=time_c,
        acc_c=acc_c,
        gyro_c=gyro_c,
        acc_l=acc_l,
        gyro_l=gyro_l,
        acc_r=acc_r,
        gyro_r=gyro_r,
        initial_position=pos_c[0],
        config=config,
        true_angular_acc=true_angular_acc,
        gps_positions=pos_c if args.enable_gps else None,
        depth_measurements=depth_measurements,
    )
    
    # Use center IMU position as ground truth
    truth_positions = pos_c
    
    # Compute and print RMSE
    rmse = compute_rmse(truth_positions, estimate)
    print(f"\nPosition RMSE:")
    print(f"  x: {rmse['x']:.4f} m")
    print(f"  y: {rmse['y']:.4f} m")
    print(f"  z: {rmse['z']:.4f} m")
    print(f"  total: {rmse['total']:.4f} m")
    
    # Final lever arm estimates
    final_lever_l = lever_arms_l[-1]
    final_lever_r = lever_arms_r[-1]
    final_std_l = diagnostics[-1]["lever_arm_l_std"]
    final_std_r = diagnostics[-1]["lever_arm_r_std"]
    
    print(f"\nFinal left lever arm (C→L):")
    print(f"  x: {final_lever_l[0]:.4f} ± {final_std_l[0]:.4f} m")
    print(f"  y: {final_lever_l[1]:.4f} ± {final_std_l[1]:.4f} m")
    print(f"  z: {final_lever_l[2]:.4f} ± {final_std_l[2]:.4f} m")
    print(f"  |r_l|: {np.linalg.norm(final_lever_l):.4f} m")
    
    print(f"\nFinal right lever arm (C→R):")
    print(f"  x: {final_lever_r[0]:.4f} ± {final_std_r[0]:.4f} m")
    print(f"  y: {final_lever_r[1]:.4f} ± {final_std_r[1]:.4f} m")
    print(f"  z: {final_lever_r[2]:.4f} ± {final_std_r[2]:.4f} m")
    print(f"  |r_r|: {np.linalg.norm(final_lever_r):.4f} m")
    
    # Compare with true lever arms if provided
    if true_lever_arm_l is not None:
        error_l = final_lever_l - true_lever_arm_l
        print(f"\nLeft lever arm error: {np.linalg.norm(error_l):.4f} m")
    if true_lever_arm_r is not None:
        error_r = final_lever_r - true_lever_arm_r
        print(f"Right lever arm error: {np.linalg.norm(error_r):.4f} m")
    
    # Plot results
    print("\nGenerating plots...")
    plot_trajectory_comparison(
        time_c,
        truth_positions,
        estimate,
        args.output,
        title="Triple IMU ESKF with Online Calibration",
    )
    
    plot_residuals(time_c, diagnostics, args.output_residuals)
    
    plot_lever_arm_estimation(
        time_c,
        lever_arms_l,
        lever_arms_r,
        true_lever_arm_l,
        true_lever_arm_r,
        args.output_lever_arm,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

