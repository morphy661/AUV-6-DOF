# src/faults/system_faults.py

import numpy as np
from enum import Enum
from typing import Optional


# 1. 通用故障枚举
class FaultType(Enum):
    NO_FAULT = 0
    BIAS = 1
    DRIFT = 2
    STUCK = 3
    SPIKE = 4
    NOISE_INCREASE = 5
    THRUSTER_ENTANGLED = 6  #  海草缠绕 (转速降，电流升)
    THRUSTER_BROKEN = 7  #  桨叶断裂 (转速无力，电流极小)


# 2. 系统故障注入器
class SystemFaultInjector:

    def __init__(
            self,
            fault_type: FaultType = FaultType.NO_FAULT,
            start_time: float = 0.0,
            bias: float = 0.0,
            drift_rate: float = 0.0,
            noise_std: float = 0.0,
            spike_prob: float = 0.0,
            spike_magnitude: float = 0.0,
            random_seed: Optional[int] = None
    ):
        self.fault_type = fault_type
        self.start_time = start_time

        self.bias = bias
        self.drift_rate = drift_rate
        self.noise_std = noise_std

        self.spike_prob = spike_prob
        self.spike_magnitude = spike_magnitude

        self.rng = np.random.default_rng(random_seed)
        self._stuck_value = None

    def apply(self, depth_value, current_time):

        if current_time < self.start_time:
            return depth_value

        if self.fault_type == FaultType.NO_FAULT:
            return depth_value

        # 关键逻辑：如果是推进器故障，深度计本身是好的！所以直接返回真实深度
        if self.fault_type in [FaultType.THRUSTER_ENTANGLED, FaultType.THRUSTER_BROKEN]:
            return depth_value

        if self.fault_type == FaultType.BIAS:
            return depth_value + self.bias

        # 在 system_faults.py 的 apply 函数中：
        if self.fault_type == FaultType.DRIFT:
            t = current_time - self.start_time

            # 限制加速时间最大为 60 秒
            t_effective = min(t, 60.0)

            # 🌟 削弱二次项系数：从 0.02 降到 0.005，让漂移更加隐蔽真实
            drift = (self.drift_rate * t) + (0.005 * (t_effective ** 2)) * np.sign(self.drift_rate)

            return depth_value + drift

        if self.fault_type == FaultType.STUCK:
            if self._stuck_value is None:
                self._stuck_value = depth_value
            return self._stuck_value

        if self.fault_type == FaultType.SPIKE:
            if self.rng.random() < self.spike_prob:
                spike = self.spike_magnitude * self.rng.choice([-1, 1])
                return depth_value + spike
            return depth_value

        if self.fault_type == FaultType.NOISE_INCREASE:
            noise = self.rng.normal(0.0, self.noise_std)
            high_freq = 0.3 * np.sin(15 * current_time)
            burst = 0.0
            if self.rng.random() < 0.2:
                burst = self.rng.normal(0.0, self.noise_std * 3)
            return depth_value + noise + high_freq + burst

    def reset(self):
        self._stuck_value = None

    def get_fault_label(self, current_time):
        if current_time < self.start_time:
            return FaultType.NO_FAULT.value
        return self.fault_type.value