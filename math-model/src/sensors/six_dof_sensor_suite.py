"""Synchronized depth, IMU, and DVL measurements for six-DOF simulation."""

from typing import Optional

import numpy as np

from sensors.depth_sensor import DepthSensor
from sensors.dvl_sensor import DVLSensor
from sensors.imu_sensor import IMUSensor


class SixDOFSensorSuite:
    """Read synchronized observable signals from one six-DOF state."""

    def __init__(
        self,
        depth_sensor: Optional[DepthSensor] = None,
        imu_sensor: Optional[IMUSensor] = None,
        dvl_sensor: Optional[DVLSensor] = None,
        seed: Optional[int] = None,
    ):
        seed_sequence = np.random.SeedSequence(seed)
        child_seeds = [
            int(sequence.generate_state(1)[0])
            for sequence in seed_sequence.spawn(3)
        ]
        self.depth_sensor = depth_sensor or DepthSensor(seed=child_seeds[0])
        self.imu_sensor = imu_sensor or IMUSensor(seed=child_seeds[1])
        self.dvl_sensor = dvl_sensor or DVLSensor(seed=child_seeds[2])

    def reset(self):
        self.depth_sensor.reset()
        self.imu_sensor.reset()
        self.dvl_sensor.reset()

    @staticmethod
    def _true_depth(state):
        if hasattr(state, "position_ned"):
            position = np.asarray(state.position_ned, dtype=float)
        else:
            position = np.asarray(
                getattr(state, "position", np.zeros(3)),
                dtype=float,
            )
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position must be a finite vector with shape (3,)")
        return float(position[2])

    def read(self, state, dt, linear_acceleration_body=None):
        return {
            "depth": self.depth_sensor.measure(
                self._true_depth(state),
                dt=dt,
            ),
            "imu": self.imu_sensor.read(
                state,
                linear_acceleration_body=linear_acceleration_body,
                dt=dt,
            ),
            "dvl": self.dvl_sensor.read(state),
        }
