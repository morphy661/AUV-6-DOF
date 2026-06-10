from dataclasses import dataclass
from enum import Enum
import numpy as np


class ThrusterFaultMode(str, Enum):
    NORMAL = "normal"
    ENTANGLED = "entangled"
    NO_OUTPUT = "no_output"
    THRUST_LOSS = "thrust_loss"
    SHORT_CIRCUIT = "short_circuit"


@dataclass
class ThrusterMotorConfig:
    """
    Simplified motor-equation-based thruster model.

    cmd is treated as a vertical control command.
    The model generates expected current, motor speed, and thrust.
    """

    # command normalization
    max_cmd: float = 1.0

    # motor current model
    idle_current: float = 0.4
    current_gain: float = 5.0

    # motor speed model
    omega_gain: float = 1200.0

    # thrust model
    thrust_coeff: float = 1.0e-5

    # noise
    current_noise_std: float = 0.05
    omega_noise_std: float = 5.0
    thrust_noise_std: float = 0.02


@dataclass
class ThrusterMotorState:
    cmd: float

    expected_current: float
    measured_current: float
    current_residual: float

    expected_omega: float
    measured_omega: float
    omega_residual: float

    expected_thrust: float
    actual_thrust: float
    thrust_residual: float

    fault_mode: str


class ThrusterMotorModel:
    def __init__(self, config: ThrusterMotorConfig | None = None, seed: int = 42):
        self.config = config or ThrusterMotorConfig()
        self.rng = np.random.default_rng(seed)

    def _normalize_cmd(self, cmd: float) -> float:
        cfg = self.config
        if cfg.max_cmd <= 0:
            return float(np.clip(cmd, -1.0, 1.0))
        return float(np.clip(cmd / cfg.max_cmd, -1.0, 1.0))

    def expected_current(self, cmd: float) -> float:
        cfg = self.config
        cmd_n = abs(self._normalize_cmd(cmd))
        return cfg.idle_current + cfg.current_gain * cmd_n

    def expected_omega(self, cmd: float) -> float:
        cfg = self.config
        cmd_n = self._normalize_cmd(cmd)
        return cfg.omega_gain * cmd_n

    def expected_thrust(self, cmd: float) -> float:
        cfg = self.config
        omega = self.expected_omega(cmd)
        return cfg.thrust_coeff * omega * abs(omega)

    def simulate(self, cmd: float, fault_mode: str | ThrusterFaultMode = ThrusterFaultMode.NORMAL) -> ThrusterMotorState:
        """
        Generate expected and actual thruster behavior.

        Fault meanings:
        - normal: motor and propeller are healthy
        - entangled: high current, low speed, low thrust
        - no_output: current almost zero, speed zero, thrust zero
        - thrust_loss: current and speed nearly normal, but thrust is reduced
        - short_circuit: very high current, unstable/low speed, low thrust
        """

        if isinstance(fault_mode, ThrusterFaultMode):
            fault_mode = fault_mode.value

        cfg = self.config

        i_exp = self.expected_current(cmd)
        omega_exp = self.expected_omega(cmd)
        thrust_exp = self.expected_thrust(cmd)

        # default normal behavior
        i_meas = i_exp + self.rng.normal(0.0, cfg.current_noise_std)
        omega_meas = omega_exp + self.rng.normal(0.0, cfg.omega_noise_std)
        thrust_actual = thrust_exp + self.rng.normal(0.0, cfg.thrust_noise_std)

        if fault_mode == ThrusterFaultMode.ENTANGLED.value:
            # Motor works harder, but rotation and thrust are blocked.
            i_meas = i_exp * 1.8 + self.rng.normal(0.0, cfg.current_noise_std)
            omega_meas = omega_exp * 0.2 + self.rng.normal(0.0, cfg.omega_noise_std)
            thrust_actual = thrust_exp * 0.2 + self.rng.normal(0.0, cfg.thrust_noise_std)

        elif fault_mode == ThrusterFaultMode.NO_OUTPUT.value:
            # Open circuit / ESC no output / disconnected motor.
            i_meas = self.rng.normal(0.02, cfg.current_noise_std)
            omega_meas = self.rng.normal(0.0, cfg.omega_noise_std)
            thrust_actual = self.rng.normal(0.0, cfg.thrust_noise_std)

        elif fault_mode == ThrusterFaultMode.THRUST_LOSS.value:
            # Propeller damage or efficiency loss:
            # current and speed may look normal, but generated thrust is low.
            i_meas = i_exp + self.rng.normal(0.0, cfg.current_noise_std)
            omega_meas = omega_exp + self.rng.normal(0.0, cfg.omega_noise_std)
            thrust_actual = thrust_exp * 0.45 + self.rng.normal(0.0, cfg.thrust_noise_std)

        elif fault_mode == ThrusterFaultMode.SHORT_CIRCUIT.value:
            # Electrical abnormality: current becomes very high but useful thrust is low.
            i_meas = i_exp * 3.0 + self.rng.normal(0.0, cfg.current_noise_std)
            omega_meas = omega_exp * 0.1 + self.rng.normal(0.0, cfg.omega_noise_std)
            thrust_actual = thrust_exp * 0.1 + self.rng.normal(0.0, cfg.thrust_noise_std)

        return ThrusterMotorState(
            cmd=float(cmd),

            expected_current=float(i_exp),
            measured_current=float(i_meas),
            current_residual=float(i_meas - i_exp),

            expected_omega=float(omega_exp),
            measured_omega=float(omega_meas),
            omega_residual=float(omega_meas - omega_exp),

            expected_thrust=float(thrust_exp),
            actual_thrust=float(thrust_actual),
            thrust_residual=float(thrust_actual - thrust_exp),

            fault_mode=str(fault_mode),
        )