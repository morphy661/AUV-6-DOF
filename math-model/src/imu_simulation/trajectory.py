from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class TrajectoryConfig:
    total_time: float = 10.0
    dt: float = 0.01
    nominal_acceleration: float = 0.5
    spiral_radius: float = 5.0
    spiral_rate: float = 0.2
    spiral_vertical_gain: float = 0.2
    fade_in_duration: float = 2.0
    wobble_roll_deg: float = 15.0
    wobble_roll_freq: float = 2.0
    wobble_pitch_deg: float = 5.0
    wobble_pitch_freq: float = 1.0
    wobble_yaw_deg: float = 0.0
    wobble_yaw_freq: float = 0.5
    path_length: float | None = None
    # 新增: 下潜角(度) 与最终深度(正值表示向下)，二选一或只设其一
    descent_angle_deg: float | None = None
    depth_final: float | None = None


@dataclass(frozen=True)
class TrajectorySamples:
    time: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    orientations: np.ndarray
    angular_velocity: np.ndarray
    angular_acceleration: np.ndarray


def _build_spiral_functions(radius: float, rate: float, vertical_gain: float):
    def position(s: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                radius * np.cos(rate * s),
                radius * np.sin(rate * s),
                vertical_gain * s,
            )
        )

    def first_derivative(s: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                -radius * rate * np.sin(rate * s),
                radius * rate * np.cos(rate * s),
                np.full_like(s, vertical_gain),
            )
        )

    def second_derivative(s: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                -radius * rate * rate * np.cos(rate * s),
                -radius * rate * rate * np.sin(rate * s),
                np.zeros_like(s),
            )
        )

    return position, first_derivative, second_derivative


def _normalize_rows(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return data / norms


def _compute_orientation_frames(direction_vectors: np.ndarray, time: np.ndarray, cfg: TrajectoryConfig) -> np.ndarray:
    tangents = _normalize_rows(direction_vectors)
    world_up = np.array([0.0, 0.0, 1.0])
    quats: list[np.ndarray] = []
    for t_vec, t_val in zip(tangents, time):
        x_axis = t_vec
        y_axis = np.cross(world_up, x_axis)
        if np.linalg.norm(y_axis) < 1e-6:
            y_axis = np.array([0.0, 1.0, 0.0])
        else:
            y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        rot_nominal = R.from_matrix(np.column_stack((x_axis, y_axis, z_axis)))

        fade = 1.0
        if cfg.fade_in_duration > 0.0 and t_val < cfg.fade_in_duration:
            phase = (t_val / cfg.fade_in_duration) * np.pi
            fade = 0.5 * (1.0 - np.cos(phase))

        wobble = R.from_euler(
            "xyz",
            [
                fade * cfg.wobble_roll_deg * np.sin(2.0 * np.pi * cfg.wobble_roll_freq * t_val),
                fade * cfg.wobble_pitch_deg * np.cos(2.0 * np.pi * cfg.wobble_pitch_freq * t_val),
                fade * cfg.wobble_yaw_deg * np.sin(2.0 * np.pi * cfg.wobble_yaw_freq * t_val),
            ],
            degrees=True,
        )
        quats.append((rot_nominal * wobble).as_quat())

    return np.asarray(quats)


def _compute_angular_kinematics(quaternions: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    if len(quaternions) < 2:
        zeros = np.zeros((len(quaternions), 3))
        return zeros, zeros

    rotations = R.from_quat(quaternions)
    omegas = np.zeros_like(quaternions[:, :3])
    for idx in range(len(quaternions) - 1):
        r_diff = rotations[idx + 1] * rotations[idx].inv()
        omegas[idx] = r_diff.as_rotvec() / dt
    omegas[-1] = omegas[-2]
    angular_acc = np.gradient(omegas, dt, axis=0)
    return omegas, angular_acc


class WobbleTrajectory:
    """Generate the wobbling spiral trajectory described in trajectory_test.py."""

    def __init__(self, config: TrajectoryConfig | None = None):
        self.config = config or TrajectoryConfig()
        cfg = self.config
        if cfg.total_time <= 0.0:
            raise ValueError("total_time must be positive")
        if cfg.dt <= 0.0:
            raise ValueError("dt must be positive")

        # 计算 s_final (用于 depth_final 派生 vertical_gain)
        if cfg.path_length is not None:
            s_final = cfg.path_length
        else:
            s_final = 0.5 * cfg.nominal_acceleration * cfg.total_time**2

        # 根据优先级确定 vertical_gain
        vertical_gain = cfg.spiral_vertical_gain
        if cfg.depth_final is not None:
            vertical_gain = cfg.depth_final / s_final
        elif cfg.descent_angle_deg is not None:
            angle_rad = np.deg2rad(cfg.descent_angle_deg)
            vertical_gain = cfg.spiral_radius * cfg.spiral_rate * np.tan(angle_rad)

        # 保存实际使用值(便于外部查询)
        object.__setattr__(self.config, "spiral_vertical_gain", vertical_gain)

        self._position_fn, self._first_fn, self._second_fn = _build_spiral_functions(
            cfg.spiral_radius,
            cfg.spiral_rate,
            vertical_gain,
        )

    def generate(self) -> TrajectorySamples:
        cfg = self.config
        times = np.arange(0.0, cfg.total_time + 0.5 * cfg.dt, cfg.dt)
        tau = np.clip(times / cfg.total_time, 0.0, 1.0)
        # 如果用户提供了明确的路径长度就使用它，否则按匀加速公式
        if cfg.path_length is not None:
            s_final = cfg.path_length
        else:
            s_final = 0.5 * cfg.nominal_acceleration * cfg.total_time**2
        # 保持平滑的五次多项式时间重参数
        s_values = s_final * (10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5)
        ds_dt = (s_final / cfg.total_time) * (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4)
        d2s_dt2 = (s_final / (cfg.total_time**2)) * (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3)

        tangents_s = self._first_fn(s_values)
        curvature_s = self._second_fn(s_values)

        positions = self._position_fn(s_values)
        velocities = tangents_s * ds_dt[:, None]
        accelerations = curvature_s * (ds_dt[:, None] ** 2) + tangents_s * d2s_dt2[:, None]

        orientations = _compute_orientation_frames(tangents_s, times, cfg)
        angular_velocity, angular_acceleration = _compute_angular_kinematics(orientations, cfg.dt)

        return TrajectorySamples(
            time=times,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            orientations=orientations,
            angular_velocity=angular_velocity,
            angular_acceleration=angular_acceleration,
        )
