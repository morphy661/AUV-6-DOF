import numpy as np

from sensors.imu_sensor import IMUSensor
from sensors.dvl_sensor import DVLSensor
from sensors.current_sensor import CurrentSensor
from sensors.battery_sensor import BatterySensor


class DummyAUVState:
    def __init__(self):
        self.velocity = np.array([0.3, 0.0, -0.2])
        self.yaw = 0.1
        self.pitch = 0.02
        self.roll = -0.01


if __name__ == "__main__":
    auv_state = DummyAUVState()

    imu = IMUSensor()
    dvl = DVLSensor()
    current_sensor = CurrentSensor()
    battery_sensor = BatterySensor()

    cmd_vz = -0.8

    imu_data = imu.read(auv_state)
    dvl_data = dvl.read(auv_state)
    current_data = current_sensor.read(cmd_vz=cmd_vz, fault_mode=None)
    battery_data = battery_sensor.read(
        motor_current=current_data["measured_current"],
        dt=0.1
    )

    print("IMU Data:")
    print(imu_data)

    print("\nDVL Data:")
    print(dvl_data)

    print("\nCurrent Sensor Data:")
    print(current_data)

    print("\nBattery Data:")
    print(battery_data)

    print("\nEntangled Current:")
    print(current_sensor.read(cmd_vz=cmd_vz, fault_mode="entangled"))

    print("\nBroken Current:")
    print(current_sensor.read(cmd_vz=cmd_vz, fault_mode="broken"))