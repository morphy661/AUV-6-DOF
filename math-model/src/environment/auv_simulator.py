import numpy as np
from environment.simulation_config import AUVModel
from typing import Dict, Tuple, List, Optional


class Simulator:
    """升级版 AUV + USV 系统仿真器"""

    def __init__(self, auv_model: AUVModel, depth_sensor=None, fault_injector=None):
        self.auv = auv_model
        self.depth_sensor = depth_sensor
        self.fault_injector = fault_injector
        self.time = 0.0
        self.trajectory = []
        self.sensor_logs = []

        # --- 新增：USV 初始状态 ---
        self.usv_position = np.array([0.0, 0.0, 0.0])

    def step(
            self,
            dt: float,
            command_velocity: np.ndarray,
            command_yaw: float
    ) -> Dict:
        # ===============================
        # 1. 更新真实物理状态（Ground Truth）
        # ===============================
        self.auv.update(dt, command_velocity, command_yaw)
        self.time += dt

        # ===============================
        # 2. 模拟 USV 随动逻辑 (简化版)
        # ===============================
        gps_noise = np.random.normal(0, 0.2, size=2)
        self.usv_position[0] = self.auv.position[0] + gps_noise[0]
        self.usv_position[1] = self.auv.position[1] + gps_noise[1]
        self.usv_position[2] = 0.0

        # ===============================
        # 3. 传感器测量与深度故障注入
        # ===============================
        true_depth = self.auv.position[2]

        if self.depth_sensor is not None:
            depth_measured = self.depth_sensor.measure(true_depth=true_depth, dt=dt)
        else:
            depth_measured = true_depth

        if self.fault_injector is not None:
            depth_measured = self.fault_injector.apply(
                depth_measured,
                self.auv.mission_time
            )
        # ===============================
        # 我们用垂直指令 (Vz) 来作为“指令转速(RPM)”的代理变量
        cmd_vz = command_velocity[2]

        # 正常情况下，实际速度会逐渐逼近指令速度。这里直接取 auv 的当前速度作为“实际反馈”
        actual_vz = self.auv.velocity[2]

        # 模拟电机电流 (Amps)：基础待机电流(2A) + 与指令绝对值成正比的负载电流 + 随机噪声
        base_current = 2.0
        current_noise = np.random.normal(0, 0.1)
        motor_current = base_current + (abs(cmd_vz) * 15.0) + current_noise

        # ---------------------------------------------------------
        # 核心修改：获取当前故障标签，并制造“物理特征扭曲”
        # ---------------------------------------------------------
        current_fault_label = 0
        if self.fault_injector is not None:
            current_fault_label = self.fault_injector.get_fault_label(self.time)

        if current_fault_label == 6:
            # ==========================================
            # 🚨 真正的卡死：强制剥夺垂直物理动力！
            # ==========================================
            # 1. 真实的物理速度强制阻断 (让 AUV 在水里完全停住或者自由落体，这里设为受阻力缓慢下沉的微小速度，比如 0.05)
            self.auv.velocity[2] *= 0.05

            # 2. 传感器读到的速度自然也变成了极小值
            actual_vz = self.auv.velocity[2]

            # 3. 堵转大电流必须极其夸张 (比如直接拉到 40A 以上)，因为电机转不动但指令还在满载
            motor_current = 45.0 + np.random.normal(0, 2.0)

        elif current_fault_label == 7:
            # ==========================================
            # 🚨 真正的断桨：电机空转，推力为零！
            # ==========================================
            # 1. 失去推力，速度掉到底
            self.auv.velocity[2] *= 0.05
            actual_vz = self.auv.velocity[2]

            # 2. 电流变成极小的空转电流
            motor_current = base_current + current_noise + 0.5
            # ---------------------------------------------------------

        # ===============================
        # 4. 核心：构建多维传感器数据包
        # ===============================

        #  安全获取 target_z 的逻辑：
        # 尝试从 AUVModel 获取 target_z，如果模型中没有这个属性，默认给 0.0，防止报错
        current_target_z = getattr(self.auv, 'target_z', 0.0)

        sensor_packet = {
            "time": self.auv.mission_time,

            "position": self.auv.position.copy(),
            "velocity": self.auv.velocity.copy(),

            "thruster": {
                "cmd_vz": cmd_vz,
                "actual_vz": actual_vz,
                "current": motor_current,
                "yaw_cmd": command_yaw
            },

            "imu": {
                "orientation": self.auv.orientation.copy(),
                "angular_velocity": self.auv.angular_velocity.copy(),
                "linear_acceleration": self.auv.velocity / dt
            },

            "depth": depth_measured,

            # 极其关键的新增：将 target_z 正式打包进日志！
            "target_z": current_target_z,

            "usv_gps": self.usv_position.copy(),
            "relative_pos": {
                "delta_x": self.auv.position[0] - self.usv_position[0],
                "delta_y": self.auv.position[1] - self.usv_position[1]
            },

            "true_depth": true_depth,
            "fault_label": current_fault_label
        }

        # ===============================
        # 5. 日志记录
        # ===============================
        self.trajectory.append(self.auv.position.copy())
        self.sensor_logs.append(sensor_packet)

        return sensor_packet

    def run_mission(
            self,
            duration: float,
            control_function,
            dt: float = 0.1
    ) -> List[Dict]:
        """运行一个完整的任务"""

        steps = int(duration / dt)
        sensor_data = None

        for _ in range(steps):
            if sensor_data is not None:
                command_velocity, command_yaw = control_function(sensor_data)
            else:
                command_velocity = np.zeros(3)
                command_yaw = 0.0

            sensor_data = self.step(dt, command_velocity, command_yaw)

            if self.auv.battery_percentage <= 0:
                print("Mission terminated: Battery depleted")
                break

        return self.sensor_logs