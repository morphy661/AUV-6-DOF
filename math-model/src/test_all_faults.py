from main import execute_mission
from faults.system_faults import FaultType


if __name__ == "__main__":
    test_faults = [
        FaultType.NO_FAULT,
        FaultType.BIAS,
        FaultType.DRIFT,
        FaultType.STUCK,
        FaultType.SPIKE,
        FaultType.NOISE_INCREASE,
        FaultType.THRUSTER_ENTANGLED,
        FaultType.THRUSTER_BROKEN,
    ]

    for fault in test_faults:
        print("=" * 80)
        print(f"Testing fault: {fault.name}")
        print("=" * 80)

        execute_mission(
            fault_type=fault,
            is_demo=False,
            duration_override=180
        )

        print(f"Finished: {fault.name}")