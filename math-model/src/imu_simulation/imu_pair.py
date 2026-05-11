from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from .trajectory import TrajectorySamples


@dataclass(frozen=True)
class RigidBodyState:
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    orientations: np.ndarray
    angular_velocity: np.ndarray
    angular_acceleration: np.ndarray


@dataclass(frozen=True)
class IMUSimulationResult:
    time: np.ndarray
    center: RigidBodyState
    imus: tuple[RigidBodyState, RigidBodyState]


class IMUPairSimulator:
    """Compute the kinematics of two IMUs rigidly attached to the center body."""

    def __init__(self, imu_offset: float = 0.5):
        if imu_offset <= 0.0:
            raise ValueError("imu_offset must be positive")
        self.imu_offset = imu_offset

    def _build_offset_vectors(self, rotations: np.ndarray) -> np.ndarray:
        body_offsets = np.array(
            [
                [0.0, -self.imu_offset, 0.0],
                [0.0, self.imu_offset, 0.0],
            ]
        )
        return np.einsum("nij,kj->nki", rotations, body_offsets)

    def simulate(self, samples: TrajectorySamples) -> IMUSimulationResult:
        """Simulate IMU pair measurements.
        
        All accelerations and angular velocities are output in the BODY FRAME,
        which is what real IMUs measure.
        
        The body frame lever arms are:
        - IMU_L: [0, -imu_offset, 0]  (left of center)
        - IMU_R: [0, +imu_offset, 0]  (right of center)
        
        The lever arm from L to R is [0, 2*imu_offset, 0].
        """
        rotation_mats = R.from_quat(samples.orientations).as_matrix()
        offsets_world = self._build_offset_vectors(rotation_mats)

        center_positions = samples.positions       # World frame
        center_velocities = samples.velocities     # World frame
        center_accelerations = samples.accelerations  # World frame
        omega = samples.angular_velocity       # Body frame
        alpha = samples.angular_acceleration   # Body frame

        # Body frame offset vectors (constant, not rotated)
        body_offsets = np.array([
            [0.0, -self.imu_offset, 0.0],  # IMU_L
            [0.0, self.imu_offset, 0.0],   # IMU_R
        ])
        
        # Expand dimensions for broadcasting: (n_timesteps, n_imus, 3)
        omega_exp = omega[:, None, :]   # (N, 1, 3)
        alpha_exp = alpha[:, None, :]   # (N, 1, 3)
        body_offsets_exp = body_offsets[None, :, :]  # (1, 2, 3)
        
        # ---- Positions in World Frame ----
        imu_positions = center_positions[:, None, :] + offsets_world  # World frame positions
        
        # ---- Velocities in World Frame ----
        # Transform omega to world frame for velocity calculation
        omega_world = np.einsum("nij,nj->ni", rotation_mats, omega)
        omega_world_exp = omega_world[:, None, :]
        imu_velocities = center_velocities[:, None, :] + np.cross(omega_world_exp, offsets_world)
        
        # ---- Accelerations in BODY Frame ----
        # Transform center acceleration from world to body frame
        # a_body = R^T @ a_world
        rotation_mats_T = np.transpose(rotation_mats, (0, 2, 1))
        center_accelerations_body = np.einsum("nij,nj->ni", rotation_mats_T, center_accelerations)
        
        # Compute IMU accelerations in body frame using body frame lever arms
        # a_imu_body = a_center_body + α_body × r_body + ω_body × (ω_body × r_body)
        imu_accelerations_body = (
            center_accelerations_body[:, None, :]
            + np.cross(alpha_exp, body_offsets_exp)
            + np.cross(omega_exp, np.cross(omega_exp, body_offsets_exp))
        )

        center_state = RigidBodyState(
            positions=center_positions,
            velocities=center_velocities,
            accelerations=center_accelerations_body,  # Body frame!
            orientations=samples.orientations,
            angular_velocity=omega,
            angular_acceleration=alpha,
        )

        imu_states = (
            RigidBodyState(
                positions=imu_positions[:, 0, :],
                velocities=imu_velocities[:, 0, :],
                accelerations=imu_accelerations_body[:, 0, :],  # Body frame!
                orientations=samples.orientations,
                angular_velocity=omega,
                angular_acceleration=alpha,
            ),
            RigidBodyState(
                positions=imu_positions[:, 1, :],
                velocities=imu_velocities[:, 1, :],
                accelerations=imu_accelerations_body[:, 1, :],  # Body frame!
                orientations=samples.orientations,
                angular_velocity=omega,
                angular_acceleration=alpha,
            ),
        )

        return IMUSimulationResult(
            time=samples.time,
            center=center_state,
            imus=imu_states,
        )
