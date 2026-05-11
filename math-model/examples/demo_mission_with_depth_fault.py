# examples/demo_mission_with_depth_fault.py
from matplotlib.animation import FuncAnimation


def animate_trajectory(trajectory):

    x = trajectory[:,0]
    y = trajectory[:,1]
    z = trajectory[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    line, = ax.plot([], [], [], 'b', label="Trajectory")
    point, = ax.plot([], [], [], 'ro', label="AUV")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Depth")

    ax.set_xlim(min(x)-10, max(x)+10)
    ax.set_ylim(min(y)-10, max(y)+10)
    ax.set_zlim(min(z)-10, max(z)+10)

    def update(frame):

        line.set_data(x[:frame], y[:frame])
        line.set_3d_properties(z[:frame])

        point.set_data([x[frame]], [y[frame]])
        point.set_3d_properties([z[frame]])

        return line, point

    ani = FuncAnimation(fig, update, frames=len(x), interval=30)

    plt.legend()
    plt.show()
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))
import numpy as np
import matplotlib.pyplot as plt

from config.auv_config import AUVConfig
from environment.simulation_config import AUVModel
from environment.auv_simulator import Simulator

from sensors.depth_sensor import DepthSensor
from faults.system_faults import DepthFaultInjector, DepthFaultType

from simple_control.simple_control import simple_controller


def main():

    # ===============================
    # 1. 创建 AUV 配置
    # ===============================
    config = AUVConfig(
        mass=50.0,
        length=1.5,
        width=0.5,
        height=0.5,
        max_velocity_x=1.5,
        max_velocity_y=1.0,
        max_velocity_z=0.5,
        max_angular_velocity=0.5,
        max_acceleration_x=0.5,
        max_acceleration_y=0.3,
        max_acceleration_z=0.2,
        max_angular_acceleration=0.2,
        battery_capacity=500.0,
        power_consumption_idle=10.0,
        power_consumption_per_velocity=5.0
    )

    auv = AUVModel(config)

    # ===============================
    # 2. 设置 Mission
    # ===============================
    waypoints = [
        (0.0, 20.0, -5.0),
        (30.0, 40.0, -10.0),
        (60.0, 60.0, -15.0),
    ]

    auv.set_waypoints(waypoints)
    auv.set_destination(80.0, 80.0, -20.0)

    # ===============================
    # 3. 创建 Depth Sensor
    # ===============================
    depth_sensor = DepthSensor(
        noise_std=0.05,
        drift_std=0.002,
        seed=0
    )

    # ===============================
    # 4. 创建 Fault Injector
    # ===============================
    fault_injector = DepthFaultInjector(
        fault_type=DepthFaultType.DRIFT,
        start_time=60.0,     # 60s 后出现故障
        drift_rate=0.02      # 每秒 2cm 漂移
    )

    # ===============================
    # 5. 创建 Simulator
    # ===============================
    simulator = Simulator(
        auv_model=auv,
        depth_sensor=depth_sensor,
        fault_injector=fault_injector
    )

    # ===============================
    # 6. 控制器封装
    # ===============================
    def controller_wrapper(sensor_packet):
        return simple_controller(sensor_packet, auv)

    # ===============================
    # 7. 运行 Mission
    # ===============================
    sensor_logs = simulator.run_mission(
        duration=300.0,   # 5 分钟
        control_function=controller_wrapper,
        dt=0.5
    )

    # ===============================
    # 8. 提取数据
    # ===============================
    time = [s["time"] for s in sensor_logs]
    depth_measured = [s["depth"] for s in sensor_logs]
    fault_label = [s["fault_label"] for s in sensor_logs]

    # Ground Truth depth
    trajectory = np.array(simulator.trajectory)
    true_depth = -trajectory[:, 2]

    # ===============================
    # 9. 绘图
    # ===============================
    plt.figure(figsize=(12, 6))

    plt.plot(time, true_depth, label="True Depth", linestyle="--")
    plt.plot(time, depth_measured, label="Measured Depth")

    plt.axvline(
        x=60.0,
        color="red",
        linestyle="--",
        label="Fault Start"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Depth (m)")
    plt.title("Depth Sensor Output During Mission (with Fault)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# ===============================
# Script Entry
# ===============================
if __name__ == "__main__":
    main()