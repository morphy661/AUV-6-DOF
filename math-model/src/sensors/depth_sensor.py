import numpy as np


class DepthSensor:
    """
    深度传感器模型（Pressure Sensor）

    depth = -z + bias + drift + noise
    """

    def __init__(
        self,
        noise_std: float = 0.05,
        bias: float = 0.0,
        drift_std: float = 0.001,
        seed: int | None = None
    ):
        """
        Args:
            noise_std: 测量噪声标准差 (m)
            bias: 固定偏置 (m)
            drift_std: 随机游走强度 (m / sqrt(s))
            seed: 随机种子（可复现实验）
        """
        self.noise_std = noise_std
        self.bias = bias
        self.drift_std = drift_std

        self.current_drift = 0.0

        if seed is not None:
            np.random.seed(seed)

    def reset(self):
        """重置随机游走（用于新一轮仿真）"""
        self.current_drift = 0.0

    def measure(self, true_depth: float, dt: float = 1.0) -> float:
        """
        测量深度

        Args:
            true_depth: 真实深度 (m, >=0)
            dt: 时间步长 (秒)

        Returns:
            depth: 测量深度 (m)
        """

        # 随机游走（漂移）
        drift_increment = np.random.normal(
            0.0,
            self.drift_std * np.sqrt(dt)
        )
        self.current_drift += drift_increment

        # 测量噪声
        noise = np.random.normal(0.0, self.noise_std)

        measured_depth = (
                true_depth +
                self.bias +
                self.current_drift +
                noise
        )
        return measured_depth
