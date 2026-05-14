from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class ResidualObserverConfig:
    idle_current: float = 2.0
    current_gain: float = 15.0
    epsilon: float = 1e-8


class ResidualObserver:
    """
    Model-based residual observer for AUV fault diagnosis.

    This version supports the newly added sensor modules:
        - DVL Sensor: provides measured vertical velocity vz
        - Current Sensor: provides measured / expected motor current
        - Depth Sensor: provides measured depth
        - Battery Sensor: kept in sensor_data for future diagnosis extension

    Priority rule:
        actual_vz:
            1) dvl["vz"] if DVL is valid
            2) thruster["actual_vz"] fallback

        motor current:
            1) current_sensor["measured_current"] and ["expected_current"]
            2) thruster["current"] and model-based expected_current fallback
    """

    def __init__(self, config: Optional[ResidualObserverConfig] = None):
        self.config = config or ResidualObserverConfig()

    def expected_current(self, cmd_vz: float) -> float:
        """
        Expected motor current under normal operation.

        I_expected = I_idle + k_I * |cmd_vz|
        """
        return self.config.idle_current + self.config.current_gain * abs(float(cmd_vz))

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            value = float(value)
            if np.isfinite(value):
                return value
            return default
        except (TypeError, ValueError):
            return default

    def _get_actual_vz(self, sensor_data: Dict[str, Any], thruster: Dict[str, Any]) -> tuple[float, str]:
        """
        Get measured vertical velocity.

        Prefer DVL measured vz when available and valid.
        Fall back to thruster actual_vz for compatibility.
        """
        dvl = sensor_data.get("dvl", {})

        if isinstance(dvl, dict) and dvl.get("valid", False):
            dvl_vz = self._safe_float(dvl.get("vz", np.nan), default=np.nan)
            if np.isfinite(dvl_vz):
                return dvl_vz, "dvl"

        return self._safe_float(thruster.get("actual_vz", 0.0)), "thruster"

    def _get_current_data(self, sensor_data: Dict[str, Any], thruster: Dict[str, Any], cmd_vz: float) -> tuple[float, float, str]:
        """
        Get measured and expected motor current.

        Prefer independent CurrentSensor output.
        Fall back to thruster current and internal expected current model.
        """
        current_sensor = sensor_data.get("current_sensor", {})

        if isinstance(current_sensor, dict) and "measured_current" in current_sensor:
            measured_current = self._safe_float(
                current_sensor.get("measured_current", 0.0)
            )

            if "expected_current" in current_sensor:
                expected_current = self._safe_float(
                    current_sensor.get("expected_current", self.expected_current(cmd_vz))
                )
            else:
                expected_current = self.expected_current(cmd_vz)

            return measured_current, expected_current, "current_sensor"

        measured_current = self._safe_float(thruster.get("current", 0.0))
        expected_current = self._safe_float(
            thruster.get("expected_current", self.expected_current(cmd_vz))
        )

        return measured_current, expected_current, "thruster"

    def compute(self, sensor_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute residuals from one sensor packet.
        """

        # -------------------------------
        # Depth residual
        # -------------------------------
        depth = self._safe_float(sensor_data.get("depth", 0.0))
        target_z = self._safe_float(sensor_data.get("target_z", 0.0))

        # In simulation, true_depth can be used as the observer reference.
        # In a real system, this should be replaced by an estimated depth from
        # an observer / filter.
        estimated_depth = sensor_data.get(
            "estimated_depth",
            sensor_data.get("true_depth", None)
        )

        if estimated_depth is None:
            depth_residual = 0.0
        else:
            depth_residual = depth - self._safe_float(estimated_depth)

        tracking_error = target_z - depth

        # -------------------------------
        # Velocity residual
        # -------------------------------
        thruster = sensor_data.get("thruster", {})
        cmd_vz = self._safe_float(thruster.get("cmd_vz", 0.0))

        actual_vz, velocity_source = self._get_actual_vz(
            sensor_data=sensor_data,
            thruster=thruster,
        )

        velocity_residual = cmd_vz - actual_vz

        # -------------------------------
        # Current residual
        # -------------------------------
        measured_current, expected_current, current_source = self._get_current_data(
            sensor_data=sensor_data,
            thruster=thruster,
            cmd_vz=cmd_vz,
        )

        current_residual = measured_current - expected_current

        # -------------------------------
        # Optional battery values
        # -------------------------------
        battery = sensor_data.get("battery", {})
        battery_voltage = self._safe_float(battery.get("voltage", 0.0)) if isinstance(battery, dict) else 0.0
        battery_soc = self._safe_float(battery.get("soc", 0.0)) if isinstance(battery, dict) else 0.0

        return {
            "tracking_error": tracking_error,
            "depth_residual": depth_residual,
            "velocity_residual": velocity_residual,
            "current_residual": current_residual,

            "expected_current": expected_current,
            "measured_current": measured_current,

            "cmd_vz": cmd_vz,
            "actual_vz": actual_vz,

            "depth": depth,
            "target_z": target_z,

            # Extra diagnostic metadata.
            # Existing diagnosis code can ignore these fields safely.
            "velocity_source": velocity_source,
            "current_source": current_source,
            "battery_voltage": battery_voltage,
            "battery_soc": battery_soc,
        }
