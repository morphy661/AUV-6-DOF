import numpy as np


class DVLSensor:
    """
    Simulated Doppler Velocity Log sensor.

    DVL measures AUV velocity relative to the seabed or water mass.
    """

    def __init__(self, velocity_noise_std=0.02, dropout_prob=0.0):
        self.velocity_noise_std = velocity_noise_std
        self.dropout_prob = dropout_prob

    def read(self, auv_state):
        """
        Read DVL velocity.

        Expected auv_state fields:
            velocity: np.array([vx, vy, vz])
        """

        if np.random.rand() < self.dropout_prob:
            return {
                "valid": False,
                "velocity": np.array([np.nan, np.nan, np.nan]),
                "speed": np.nan,
            }

        velocity = np.array(getattr(auv_state, "velocity", np.zeros(3)), dtype=float)

        measured_velocity = velocity + np.random.normal(
            0.0,
            self.velocity_noise_std,
            size=3
        )

        return {
            "valid": True,
            "velocity": measured_velocity,
            "vx": float(measured_velocity[0]),
            "vy": float(measured_velocity[1]),
            "vz": float(measured_velocity[2]),
            "speed": float(np.linalg.norm(measured_velocity)),
        }