import numpy as np
from typing import Dict, List


class Simulator:
    """AUV + USV system simulator with optional multi-sensor outputs."""

    def __init__(
            self,
            auv_model,
            depth_sensor=None,
            fault_injector=None,
            imu_sensor=None,
            dvl_sensor=None,
            current_sensor=None,
            battery_sensor=None
    ):
        self.auv = auv_model
        self.depth_sensor = depth_sensor
        self.fault_injector = fault_injector

        # Optional additional sensors
        self.imu_sensor = imu_sensor
        self.dvl_sensor = dvl_sensor
        self.current_sensor = current_sensor
        self.battery_sensor = battery_sensor

        self.time = 0.0
        self.trajectory = []
        self.sensor_logs = []

        # Simplified USV state
        self.usv_position = np.array([0.0, 0.0, 0.0])

    def step(
            self,
            dt: float,
            command_velocity: np.ndarray,
            command_yaw: float
    ) -> Dict:
        # ===============================
        # 1. Update ground-truth AUV state
        # ===============================
        self.auv.update(dt, command_velocity, command_yaw)
        self.time += dt

        # ===============================
        # 2. Simulated USV following logic
        # ===============================
        gps_noise = np.random.normal(0, 0.2, size=2)
        self.usv_position[0] = self.auv.position[0] + gps_noise[0]
        self.usv_position[1] = self.auv.position[1] + gps_noise[1]
        self.usv_position[2] = 0.0

        # ===============================
        # 3. Depth measurement and fault injection
        # ===============================
        true_depth = float(self.auv.position[2])

        if self.depth_sensor is not None:
            depth_measured = self.depth_sensor.measure(true_depth=true_depth, dt=dt)
        else:
            depth_measured = true_depth

        if self.fault_injector is not None:
            depth_measured = self.fault_injector.apply(
                depth_measured,
                self.auv.mission_time
            )

        # ===============================
        # 4. Thruster command / feedback signals
        # ===============================
        cmd_vz = float(command_velocity[2])
        actual_vz = float(self.auv.velocity[2])

        # Default current model, used when no independent CurrentSensor is provided.
        base_current = 2.0
        current_noise = np.random.normal(0.0, 0.1)
        expected_current = base_current + abs(cmd_vz) * 15.0
        motor_current = expected_current + current_noise

        # Current active fault label
        current_fault_label = 0
        if self.fault_injector is not None:
            current_fault_label = self.fault_injector.get_fault_label(self.time)

        # ---------------------------------------------------------
        # Physical distortion for actuator faults
        # ---------------------------------------------------------
        if current_fault_label == 6:
            # THRUSTER_ENTANGLED:
            # The propeller is blocked, so vertical speed becomes very small
            # while motor current becomes abnormally high.
            self.auv.velocity[2] *= 0.05
            actual_vz = float(self.auv.velocity[2])
            motor_current = 45.0 + np.random.normal(0.0, 2.0)

        elif current_fault_label == 7:
            # THRUSTER_BROKEN:
            # The propeller is broken or detached, so thrust is lost and
            # the motor current becomes abnormally low.
            self.auv.velocity[2] *= 0.05
            actual_vz = float(self.auv.velocity[2])
            motor_current = base_current + current_noise + 0.5

        # ===============================
        # 5. Build basic sensor packet
        # ===============================
        current_target_z = float(getattr(self.auv, "target_z", 0.0))

        sensor_packet = {
            "time": float(self.auv.mission_time),

            "position": self.auv.position.copy(),
            "velocity": self.auv.velocity.copy(),

            "thruster": {
                "cmd_vz": cmd_vz,
                "actual_vz": actual_vz,
                "current": float(motor_current),
                "expected_current": float(expected_current),
                "current_residual": float(motor_current - expected_current),
                "yaw_cmd": float(command_yaw),
            },

            "depth": float(depth_measured),
            "target_z": current_target_z,

            "usv_gps": self.usv_position.copy(),
            "relative_pos": {
                "delta_x": float(self.auv.position[0] - self.usv_position[0]),
                "delta_y": float(self.auv.position[1] - self.usv_position[1]),
            },

            "true_depth": true_depth,
            "fault_label": int(current_fault_label),
        }

        # ===============================
        # 6. Additional AUV sensors
        # ===============================

        # 6.1 IMU
        # Prefer the independent IMU sensor if available.
        # Otherwise keep the old built-in IMU-like packet for compatibility.
        if self.imu_sensor is not None:
            sensor_packet["imu"] = self.imu_sensor.read(self.auv)
        else:
            sensor_packet["imu"] = {
                "orientation": self.auv.orientation.copy(),
                "angular_velocity": self.auv.angular_velocity.copy(),
                "linear_acceleration": self.auv.velocity / max(dt, 1e-8),
            }

        # 6.2 DVL
        if self.dvl_sensor is not None:
            sensor_packet["dvl"] = self.dvl_sensor.read(self.auv)

        # 6.3 Current Sensor
        # The external CurrentSensor is used to create an explicit sensor module.
        # The measured current is also synchronized back to thruster["current"]
        # so the existing residual observer and diagnosis rules remain compatible.
        if self.current_sensor is not None:
            fault_mode = None

            if current_fault_label == 6:
                fault_mode = "entangled"
            elif current_fault_label == 7:
                fault_mode = "broken"

            current_data = self.current_sensor.read(
                cmd_vz=cmd_vz,
                fault_mode=fault_mode
            )

            sensor_packet["current_sensor"] = current_data

            sensor_packet["thruster"]["current"] = float(current_data["measured_current"])
            sensor_packet["thruster"]["expected_current"] = float(current_data["expected_current"])
            sensor_packet["thruster"]["current_residual"] = float(current_data["current_residual"])

        # 6.4 Battery Sensor
        if self.battery_sensor is not None:
            motor_current_for_battery = sensor_packet["thruster"].get("current", 0.0)
            sensor_packet["battery"] = self.battery_sensor.read(
                motor_current=motor_current_for_battery,
                dt=dt
            )

        # ===============================
        # 7. Logs
        # ===============================
        self.trajectory.append(self.auv.position.copy())
        self.sensor_logs.append(sensor_packet)

        return sensor_packet

    def run_mission(
            self,
            duration: float,
            control_function,
            dt: float = 0.1
    ) -> List[Dict]:
        """Run a complete mission."""

        steps = int(duration / dt)
        sensor_data = None

        for _ in range(steps):
            if sensor_data is not None:
                command_velocity, command_yaw = control_function(sensor_data)
            else:
                command_velocity = np.zeros(3)
                command_yaw = 0.0

            sensor_data = self.step(dt, command_velocity, command_yaw)

            if self.auv.battery_percentage <= 0:
                print("Mission terminated: Battery depleted")
                break

        return self.sensor_logs
