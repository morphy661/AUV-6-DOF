from actuators.thruster_motor_model import ThrusterMotorModel, ThrusterFaultMode


def main():
    model = ThrusterMotorModel()

    cmd = 0.8

    for mode in [
        ThrusterFaultMode.NORMAL,
        ThrusterFaultMode.ENTANGLED,
        ThrusterFaultMode.NO_OUTPUT,
        ThrusterFaultMode.THRUST_LOSS,
        ThrusterFaultMode.SHORT_CIRCUIT,
    ]:
        state = model.simulate(cmd, mode)

        print("\nMode:", mode.value)
        print("cmd:", state.cmd)
        print("expected_current:", state.expected_current)
        print("measured_current:", state.measured_current)
        print("current_residual:", state.current_residual)
        print("expected_omega:", state.expected_omega)
        print("measured_omega:", state.measured_omega)
        print("expected_thrust:", state.expected_thrust)
        print("actual_thrust:", state.actual_thrust)
        print("thrust_residual:", state.thrust_residual)


if __name__ == "__main__":
    main()