"""Mission-command adaptation after confirmed or corroborated actuation loss."""

from dataclasses import dataclass

import numpy as np

from simple_control.six_dof_controller import PoseTarget


@dataclass(frozen=True)
class GuidanceDecision:
    target: PoseTarget
    mode: str
    entered_mode: bool


class FTCGuidanceManager:
    """Latch an estimated hold point for vertical or horizontal degradation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mode = "normal"
        self.hold_position_ned = None
        self.hold_yaw_rad = None
        self.hold_context_id = None

    def update(self, requested_target, controller_state, ftc_decision):
        requested_mode = (
            "normal" if ftc_decision is None else ftc_decision.guidance_mode
        )
        if requested_mode == "normal":
            entered = self.mode != "normal"
            self.reset()
            return GuidanceDecision(requested_target, "normal", entered)
        if requested_mode not in {
            "vertical_thruster_hold", "horizontal_degraded_hold",
        }:
            raise ValueError(f"unknown FTC guidance mode: {requested_mode}")

        entered = self.mode != requested_mode
        if entered:
            self.mode = requested_mode
            self.hold_position_ned = controller_state.position_ned.copy()
            self.hold_yaw_rad = float(controller_state.euler_rpy[2])
            context_offset = (
                100000 if requested_mode == "vertical_thruster_hold"
                else 200000
            )
            self.hold_context_id = (
                int(requested_target.guidance_context_id) + context_offset
            )
        target = PoseTarget(
            position_ned=self.hold_position_ned,
            euler_rpy=np.array([0.0, 0.0, self.hold_yaw_rad]),
            guidance_context_id=self.hold_context_id,
            linear_velocity_ned=np.zeros(3),
        )
        return GuidanceDecision(target, self.mode, entered)
