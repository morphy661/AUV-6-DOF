import numpy as np


class BatterySensor:
    """
    Simulated AUV battery sensor.

    Provides voltage, state of charge, and estimated power.
    """

    def __init__(self, nominal_voltage=48.0, initial_soc=1.0, voltage_noise_std=0.1):
        self.nominal_voltage = nominal_voltage
        self.soc = initial_soc
        self.voltage_noise_std = voltage_noise_std

    def read(self, motor_current, dt=0.1):
        """
        Read battery state.

        motor_current:
            current drawn by thruster motor
        """

        current = abs(float(motor_current))

        # Simple SOC decay model
        self.soc -= current * dt * 1e-5
        self.soc = float(np.clip(self.soc, 0.0, 1.0))

        voltage_drop = (1.0 - self.soc) * 6.0
        voltage = self.nominal_voltage - voltage_drop + np.random.normal(
            0.0,
            self.voltage_noise_std
        )

        power = voltage * current

        return {
            "voltage": float(voltage),
            "soc": float(self.soc),
            "current_draw": float(current),
            "power": float(power),
        }