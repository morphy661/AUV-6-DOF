import numpy as np


class IMUSensor:
    """
    Simulated IMU sensor for AUV.

    Provides attitude, angular velocity, and linear acceleration.
    """

    def __init__(self, attitude_noise_std=0.002, gyro_noise_std=0.001, accel_noise_std=0.01):
        self.attitude_noise_std = attitude_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std

    def read(self, auv_state):
        """
        Read IMU data from AUV state.

        Expected auv_state fields:
            position
            velocity
            orientation or yaw
        """

        yaw = float(getattr(auv_state, "yaw", 0.0))
        pitch = float(getattr(auv_state, "pitch", 0.0)) if hasattr(auv_state, "pitch") else 0.0
        roll = float(getattr(auv_state, "roll", 0.0)) if hasattr(auv_state, "roll") else 0.0

        return {
            "roll": roll + np.random.normal(0.0, self.attitude_noise_std),
            "pitch": pitch + np.random.normal(0.0, self.attitude_noise_std),
            "yaw": yaw + np.random.normal(0.0, self.attitude_noise_std),

            "angular_velocity": np.array([
                np.random.normal(0.0, self.gyro_noise_std),
                np.random.normal(0.0, self.gyro_noise_std),
                np.random.normal(0.0, self.gyro_noise_std),
            ]),

            "linear_acceleration": np.array([
                np.random.normal(0.0, self.accel_noise_std),
                np.random.normal(0.0, self.accel_noise_std),
                np.random.normal(0.0, self.accel_noise_std),
            ])
        }