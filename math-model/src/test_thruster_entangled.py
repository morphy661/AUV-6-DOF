from main import execute_mission
from faults.system_faults import FaultType

if __name__ == "__main__":
    execute_mission(
        fault_type=FaultType.THRUSTER_ENTANGLED,
        is_demo=False,
        duration_override=180
    )