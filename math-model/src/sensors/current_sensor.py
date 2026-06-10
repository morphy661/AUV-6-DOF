import numpy as np


class CurrentSensor:
    """
    Simulated motor current sensor for thruster diagnosis.

    The expected current model is:
        I_expected = idle_current + current_gain * abs(cmd_vz)
    """

    def __init__(self, idle_current=2.0, current_gain=15.0, noise_std=0.3):
        self.idle_current = idle_current
        self.current_gain = current_gain
        self.noise_std = noise_std

    def expected_current(self, cmd_vz):
        return self.idle_current + self.current_gain * abs(float(cmd_vz))

    def read(self, cmd_vz, fault_mode=None):
        """
        Read motor current.

        fault_mode:
            None
            "entangled"
            "no_output"
        """

        expected = self.expected_current(cmd_vz)

        if fault_mode == "entangled":
            measured = expected + 20.0 + np.random.normal(0.0, 2.0)

        elif fault_mode == "no_output":
            measured = max(0.5, expected * 0.08 + np.random.normal(0.0, 0.2))

        else:
            measured = expected + np.random.normal(0.0, self.noise_std)

        return {
            "measured_current": float(measured),
            "expected_current": float(expected),
            "current_residual": float(measured - expected),
        }
