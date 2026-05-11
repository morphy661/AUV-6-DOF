from dataclasses import dataclass
# from environment.simulation_config import AUVModel
# from simple_control.simple_control import simple_controller
# from utils.visualization import plot_trajectory
# from src.environment.auv_simulator import Simulator
@dataclass
class AUVConfig:
    """AUV基础配置"""
    # 物理参数
    mass: float  # 质量 (kg)
    length: float  # 长度 (m)
    width: float  # 宽度 (m)
    height: float  # 高度 (m)
    # 最大速度限制 (m/s)
    max_velocity_x: float  
    max_velocity_y: float
    max_velocity_z: float
    max_angular_velocity: float  # 最大角速度 (rad/s)
    # 加速度限制 (m/s²)
    max_acceleration_x: float
    max_acceleration_y: float
    max_acceleration_z: float
    max_angular_acceleration: float  # 最大角加速度 (rad/s²)
    # 电池参数
    battery_capacity: float  # 电池容量 (Wh)
    # 能耗系数 (Wh/s 在各种状态下)
    power_consumption_idle: float
    power_consumption_per_velocity: float  # Wh/s per m/sa



    