"""Mission-command adaptation after confirmed loss of vertical actuation."""

from dataclasses import dataclass

import numpy as np

from simple_control.six_dof_controller import PoseTarget


@dataclass(frozen=True)
class GuidanceDecision:
    target: PoseTarget
    mode: str
    entered_mode: bool


class FTCGuidanceManager:
    """Latch an estimated hold point when one vertical thruster is isolated."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mode = "normal"
        self.hold_position_ned = None
        self.hold_yaw_rad = None
        self.hold_context_id = None

    def update(self, requested_target, controller_state, ftc_decision):
        isolated = set(
            () if ftc_decision is None
            else ftc_decision.isolated_thruster_indices
        )
        vertical_isolated = bool(isolated.intersection({5, 6}))
        if not vertical_isolated:
            entered = self.mode != "normal"
            self.reset()
            return GuidanceDecision(requested_target, "normal", entered)

        entered = self.mode != "vertical_thruster_hold"
        if entered:
            self.mode = "vertical_thruster_hold"
            self.hold_position_ned = controller_state.position_ned.copy()
            self.hold_yaw_rad = float(controller_state.euler_rpy[2])
            self.hold_context_id = int(requested_target.guidance_context_id) + 100000
        target = PoseTarget(
            position_ned=self.hold_position_ned,
            euler_rpy=np.array([0.0, 0.0, self.hold_yaw_rad]),
            guidance_context_id=self.hold_context_id,
            linear_velocity_ned=np.zeros(3),
        )
        return GuidanceDecision(target, self.mode, entered)
