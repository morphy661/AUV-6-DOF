"""Six-DOF simulation chain with optional oracle fault-tolerant allocation."""

from typing import Callable, Optional

import numpy as np

from actuators.six_dof_thruster_faults import (
    SingleThrusterFault,
    ThrusterActuatorBank,
)
from actuators.thruster_array import ThrusterArray, default_six_thruster_array
from environment.six_dof_dynamics import SixDOFDynamics
from simple_control.six_dof_controller import (
    CascadedSixDOFController,
    PoseTarget,
)
from sensors.six_dof_sensor_suite import SixDOFSensorSuite


class SixDOFSimulator:
    """Connect pose control, thrust allocation, actuator behavior, and dynamics."""

    def __init__(
        self,
        dynamics: Optional[SixDOFDynamics] = None,
        thruster_array: Optional[ThrusterArray] = None,
        controller: Optional[CascadedSixDOFController] = None,
        fault: Optional[SingleThrusterFault] = None,
        actuator_bank: Optional[ThrusterActuatorBank] = None,
        ideal_fault_tolerant_allocation: bool = False,
        sensor_suite: Optional[SixDOFSensorSuite] = None,
    ):
        self.dynamics = dynamics or SixDOFDynamics()
        self.thruster_array = thruster_array or default_six_thruster_array()
        self.controller = controller or CascadedSixDOFController()
        if fault is not None and actuator_bank is not None:
            raise ValueError("provide fault or actuator_bank, not both")
        self.actuator_bank = actuator_bank or ThrusterActuatorBank(
            self.thruster_array,
            fault=fault,
        )
        if self.actuator_bank.thruster_array.names != self.thruster_array.names:
            raise ValueError("actuator bank and thruster array must use the same layout")
        self.ideal_fault_tolerant_allocation = bool(
            ideal_fault_tolerant_allocation
        )
        self.sensor_suite = sensor_suite
        self.logs = []

    @property
    def state(self):
        return self.dynamics.state

    def reset(self, state=None):
        self.dynamics.reset(state)
        self.controller.reset()
        if self.sensor_suite is not None:
            self.sensor_suite.reset()
        self.logs = []
        return self.state.copy()

    def step(self, target: PoseTarget, dt, disturbance_body=None):
        previous_velocity = self.state.body_velocity.copy()
        control = self.controller.compute(self.state, target, dt)
        allocation_effectiveness = (
            self.actuator_bank.force_efficiencies_at(self.state.time)
            if self.ideal_fault_tolerant_allocation
            else np.ones(len(self.thruster_array.thrusters))
        )
        allocation = self.thruster_array.allocate(
            control.desired_wrench_body,
            thruster_effectiveness=allocation_effectiveness,
        )
        actuation = self.actuator_bank.apply(
            allocation.thruster_forces,
            time_s=self.state.time,
        )
        actual_wrench = self.thruster_array.wrench_from_forces(
            actuation.actual_forces
        )
        state = self.dynamics.step(
            actual_wrench,
            dt=dt,
            disturbance_body=disturbance_body,
        )
        sensor_packet = None
        if self.sensor_suite is not None:
            sensor_packet = self.sensor_suite.read(
                state,
                dt=dt,
                linear_acceleration_body=(
                    state.body_velocity[:3] - previous_velocity[:3]
                ) / float(dt),
            )

        log = {
            "time": float(state.time),
            "position_ned": state.position_ned.copy(),
            "euler_rpy": state.euler_rpy.copy(),
            "body_velocity": state.body_velocity.copy(),
            "target_position_ned": target.position_ned.copy(),
            "target_euler_rpy": target.euler_rpy.copy(),
            "position_error_ned": control.position_error_ned.copy(),
            "attitude_error_body": control.attitude_error_body.copy(),
            "desired_velocity_ned": control.desired_velocity_ned.copy(),
            "desired_angular_velocity_body": (
                control.desired_angular_velocity_body.copy()
            ),
            "desired_wrench_body": allocation.desired_wrench.copy(),
            "allocated_wrench_body": allocation.achieved_wrench.copy(),
            "achieved_wrench_body": actual_wrench.copy(),
            "allocation_residual_body": allocation.residual_wrench.copy(),
            "actuation_residual_body": (
                allocation.achieved_wrench - actual_wrench
            ),
            "thruster_forces": allocation.thruster_forces.copy(),
            "commanded_thruster_forces": actuation.commanded_forces.copy(),
            "actual_thruster_forces": actuation.actual_forces.copy(),
            "thruster_expected_currents": actuation.expected_currents.copy(),
            "thruster_measured_currents": actuation.measured_currents.copy(),
            "thruster_force_efficiencies": actuation.force_efficiencies.copy(),
            "thruster_fault_modes": actuation.fault_modes,
            "thruster_fault_active": actuation.fault_active,
            "faulted_thruster_index": actuation.faulted_thruster_index,
            "thruster_saturated": allocation.saturated.copy(),
            "allocation_thruster_effectiveness": (
                allocation.thruster_effectiveness.copy()
            ),
            "ideal_ftc_enabled": self.ideal_fault_tolerant_allocation,
            "ftc_active": bool(
                self.ideal_fault_tolerant_allocation
                and np.any(allocation_effectiveness < 1.0)
            ),
        }
        if sensor_packet is not None:
            log.update(sensor_packet)
        self.logs.append(log)
        return log

    def run(
        self,
        duration,
        dt,
        target_provider: Callable,
        disturbance_provider: Optional[Callable] = None,
    ):
        duration = float(duration)
        dt = float(dt)
        if duration <= 0 or dt <= 0:
            raise ValueError("duration and dt must be positive")
        steps = int(np.ceil(duration / dt))

        for _ in range(steps):
            target = target_provider(self.state.time, self.state.copy())
            if not isinstance(target, PoseTarget):
                raise TypeError("target_provider must return PoseTarget")
            disturbance = (
                None
                if disturbance_provider is None
                else disturbance_provider(self.state.time, self.state.copy())
            )
            self.step(target, dt=dt, disturbance_body=disturbance)

        return self.logs


# Backward-compatible name used by the nominal validation script and tests.
SixDOFNominalSimulator = SixDOFSimulator
