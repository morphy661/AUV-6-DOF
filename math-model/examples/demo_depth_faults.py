import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

import numpy as np
import matplotlib.pyplot as plt

from sensors.depth_sensor import DepthSensor
from faults.system_faults import DepthFaultInjector, DepthFaultType


def run_fault_demo(
    fault_type: DepthFaultType,
    title: str,
    fault_kwargs: dict
):
    """
    运行单种故障的深度传感器仿真并画图
    """

    # ========= 仿真参数 =========
    dt = 0.1
    total_time = 30.0
    steps = int(total_time / dt)

    time_axis = np.arange(steps) * dt
    true_depth = 10.0 + 0.05 * time_axis  # 缓慢下潜

    # ========= 传感器 =========
    sensor = DepthSensor(
        noise_std=0.05,
        drift_std=0.002,
        seed=42
    )

    # ========= 故障注入 =========
    fault = DepthFaultInjector(
        fault_type=fault_type,
        start_time=10.0,
        **fault_kwargs
    )

    sensor.reset()
    fault.reset()

    # ========= 仿真 =========
    measured_depth = []

    for t, d in zip(time_axis, true_depth):
        depth = sensor.measure(true_depth=-d, dt=dt)
        depth = fault.apply(depth, t)
        measured_depth.append(depth)

    measured_depth = np.array(measured_depth)

    # ========= 绘图 =========
    plt.plot(time_axis, measured_depth, label="Measured Depth")
    plt.axvline(10.0, linestyle="--", color="r", label="Fault Start")
    plt.xlabel("Time (s)")
    plt.ylabel("Depth (m)")
    plt.title(title)
    plt.grid(True)
    plt.legend()


if __name__ == "__main__":

    plt.figure(figsize=(14, 10))

    # ========= Normal =========
    plt.subplot(3, 2, 1)
    run_fault_demo(
        DepthFaultType.NO_FAULT,
        "Normal (No Fault)",
        {}
    )

    # ========= Bias =========
    plt.subplot(3, 2, 2)
    run_fault_demo(
        DepthFaultType.BIAS,
        "Bias Fault (+0.5 m)",
        {"bias": 0.5}
    )

    # ========= Drift =========
    plt.subplot(3, 2, 3)
    run_fault_demo(
        DepthFaultType.DRIFT,
        "Drift Fault (0.05 m/s)",
        {"drift_rate": 0.05}
    )

    # ========= Stuck =========
    plt.subplot(3, 2, 4)
    run_fault_demo(
        DepthFaultType.STUCK,
        "Stuck Fault",
        {}
    )

    # ========= Spike =========
    plt.subplot(3, 2, 5)
    run_fault_demo(
        DepthFaultType.SPIKE,
        "Spike Fault",
        {
            "spike_prob": 0.05,
            "spike_magnitude": 2.0
        }
    )

    # ========= Noise Increase =========
    plt.subplot(3, 2, 6)
    run_fault_demo(
        DepthFaultType.NOISE_INCREASE,
        "Noise Increase Fault",
        {"noise_std": 0.5}
    )

    plt.tight_layout()
    plt.show()
