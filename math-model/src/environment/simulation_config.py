import numpy as np
from config.auv_config import AUVConfig
from typing import Dict, Tuple, List, Optional


class AUVModel:
    """升级版 AUV 数学与水动力学物理模型"""

    def __init__(self, config: AUVConfig):
        self.config = config

        # 状态变量
        self.position = np.zeros(3)  # [x, y, z] in meters
        self.orientation = np.zeros(3)  # [roll, pitch, yaw] in radians
        self.velocity = np.zeros(3)  # [vx, vy, vz] in m/s
        self.angular_velocity = np.zeros(3)  # [ω_roll, ω_pitch, ω_yaw] in rad/s

        # 电池状态
        self.battery_remaining = config.battery_capacity  # Wh
        self.battery_percentage = 100.0  # %

        # 时间追踪
        self.last_update_time = None
        self.mission_time = 0.0  # seconds

        # 目标位置
        self.destination = None
        self.waypoints = []
        self.visited_waypoints = []

        # ==========================================
        # 水动力学物理参数 (可根据实际 AUV 调整)
        # ==========================================
        self.mass = 50.0  # AUV 质量 (kg) - 决定惯性大小
        self.thrust_coeff = 300  # 推进器推力系数 - 将指令速度差转化为物理推力
        self.linear_drag = 15.0  # 线性水阻力系数
        self.quadratic_drag = 5.0  # 二次水阻力系数
        self.buoyancy = 0.5  # 净浮力 (N) - 模拟微弱的正浮力
        self.physical_max_speed = 2.0   # 物理上最大速度 (m/s)，超过这个速度会有强烈的阻力增加

    # ==========================================
    # 动态计算 Target Z
    # 这样外部读取 auv.target_z 时，就能永远拿到最准确的当前目标深度
    # ==========================================
    @property
    def target_z(self) -> float:
        """动态获取当前的目标深度"""
        # 如果还有未到达的航点，以当前正在前往的第一个航点深度为准
        if self.waypoints and len(self.waypoints) > 0:
            return float(self.waypoints[0][2])
        # 如果航点走完了，但有最终目的地，以目的地深度为准
        elif self.destination is not None:
            return float(self.destination[2])
        # 否则默认目标深度为 0 (水面)
        return 0.0

    def set_destination(self, x: float, y: float, z: float):
        self.destination = np.array([x, y, z])

    def set_waypoints(self, waypoints: list[Tuple[float, float, float]]):
        self.waypoints = [np.array(wp) for wp in waypoints]
        self.visited_waypoints = []

    def mark_waypoint_as_visited(self, waypoint: np.ndarray):
        self.visited_waypoints.append(waypoint.tolist())

    def update(self, dt: float, command_velocity: np.ndarray, command_yaw: float):
        """更新AUV状态 (引入基于力学积分的物理引擎)"""
        self.mission_time += dt

        # ==========================================
        # 1. 受力分析 (Force Analysis)
        # ==========================================
        thrust_force = self.thrust_coeff * (command_velocity - self.velocity)

        speed = np.linalg.norm(self.velocity)
        if speed > 0:
            drag_force = - (self.linear_drag + self.quadratic_drag * speed) * self.velocity
        else:
            drag_force = np.zeros(3)

        total_force = thrust_force + drag_force
        total_force[2] = thrust_force[2] + drag_force[2] - self.buoyancy # 注意这里用减号  # 浮力 0.5 是向上的(+Z)，正确

        # ==========================================
        # 2. 运动学积分 (Kinematic Integration)
        # ==========================================
        acceleration = total_force / self.mass
        self.velocity += acceleration * dt

        self.velocity = np.clip(
            self.velocity,
            [-self.config.max_velocity_x, -self.config.max_velocity_y, -self.config.max_velocity_z],
            [self.config.max_velocity_x, self.config.max_velocity_y, self.config.max_velocity_z]
        )
        # ==========================================
        # 🌟 新增：超频指令的核心保护！强制综合物理速度不超过极限
        # ==========================================
        current_speed = np.linalg.norm(self.velocity)
        if current_speed > self.physical_max_speed:
            # 如果算出来的物理速度超过 2.0，强行按比例缩放回 2.0
            self.velocity = (self.velocity / current_speed) * self.physical_max_speed

        self.position += self.velocity * dt

        # ==========================================
        # 3. 偏航角更新
        # ==========================================
        max_dyaw = self.config.max_angular_acceleration * dt
        dyaw = np.clip(command_yaw - self.orientation[2], -max_dyaw, max_dyaw)
        self.orientation[2] += dyaw

        # ==========================================
        # 4. 电池能耗升级
        # ==========================================
        thrust_magnitude = np.linalg.norm(thrust_force)
        power_consumption = (
                self.config.power_consumption_idle +
                (thrust_magnitude * 1.2)
        )
        energy_consumed = power_consumption * dt / 3600.0  # 转换为Wh

        self.battery_remaining = max(0.0, self.battery_remaining - energy_consumed)
        self.battery_percentage = (self.battery_remaining / self.config.battery_capacity) * 100.0

    def get_sensor_data(self) -> Dict:
        """获取传感器数据"""
        position_noise = np.random.normal(0, 0.1, 3)
        velocity_noise = np.random.normal(0, 0.05, 3)
        orientation_noise = np.random.normal(0, 0.01, 3)

        return {
            "imu": {
                "orientation": self.orientation + orientation_noise,
                "angular_velocity": self.angular_velocity
            },
            "dvl": {
                "velocity": self.velocity + velocity_noise
            },
            "position": self.position + position_noise,
            "battery_percentage": self.battery_percentage
        }