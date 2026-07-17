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
from diagnosis.sensor_health_monitor import SensorHealthMonitor
from diagnosis.sensor_fault_observer import SensorFaultObserver
from estimation.six_dof_state_estimator import SixDOFStateEstimator
from ftc.safety_supervisor import (
    FTCEvidence,
    build_rule_based_ftc_evidence,
)


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
        sensor_health_monitor: Optional[SensorHealthMonitor] = None,
        sensor_fault_observer: Optional[SensorFaultObserver] = None,
        state_estimator: Optional[SixDOFStateEstimator] = None,
        use_sensor_feedback: bool = True,
        ftc_supervisor=None,
        ftc_evidence_provider: Optional[Callable] = None,
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
        if self.ideal_fault_tolerant_allocation and ftc_supervisor is not None:
            raise ValueError(
                "ideal oracle FTC and deployable FTC supervisor are mutually exclusive"
            )
        self.ftc_supervisor = ftc_supervisor
        self.ftc_evidence_provider = ftc_evidence_provider
        self.ftc_decision = (
            None if self.ftc_supervisor is None else self.ftc_supervisor.reset()
        )
        self.sensor_suite = sensor_suite
        self.sensor_health_monitor = (
            sensor_health_monitor
            if sensor_health_monitor is not None
            else (SensorHealthMonitor() if sensor_suite is not None else None)
        )
        self.sensor_fault_observer = (
            sensor_fault_observer
            if sensor_fault_observer is not None
            else (SensorFaultObserver() if sensor_suite is not None else None)
        )
        if state_estimator is not None and sensor_suite is None:
            raise ValueError(
                "state_estimator requires a synchronized sensor_suite"
            )
        self.state_estimator = (
            state_estimator
            if state_estimator is not None
            else (
                SixDOFStateEstimator()
                if sensor_suite is not None
                else None
            )
        )
        self.sensor_feedback_enabled = bool(
            use_sensor_feedback and self.state_estimator is not None
        )
        self.state_estimate = (
            None
            if self.state_estimator is None
            else self.state_estimator.reset(self.state)
        )
        self.logs = []

    @property
    def state(self):
        return self.dynamics.state

    def reset(self, state=None):
        self.dynamics.reset(state)
        self.controller.reset()
        if hasattr(self.actuator_bank, "reset"):
            self.actuator_bank.reset()
        if self.sensor_suite is not None:
            self.sensor_suite.reset()
        if self.sensor_health_monitor is not None:
            self.sensor_health_monitor.reset()
        if self.sensor_fault_observer is not None:
            self.sensor_fault_observer.reset()
        if self.state_estimator is not None:
            self.state_estimate = self.state_estimator.reset(self.state)
        if self.ftc_supervisor is not None:
            self.ftc_decision = self.ftc_supervisor.reset()
        self.logs = []
        return self.state.copy()

    def step(self, target: PoseTarget, dt, disturbance_body=None):
        previous_velocity = self.state.body_velocity.copy()
        if self.sensor_feedback_enabled:
            controller_state = self.state_estimate.state.copy()
            controller_state_source = (
                "initial_pose_prior"
                if not self.logs
                else "sensor_estimate_previous_step"
            )
        else:
            controller_state = self.state.copy()
            controller_state_source = "simulator_truth"
        control = self.controller.compute(controller_state, target, dt)
        applied_ftc_decision = self.ftc_decision
        if self.ideal_fault_tolerant_allocation:
            allocation_effectiveness = (
                self.actuator_bank.force_efficiencies_at(self.state.time)
            )
        elif applied_ftc_decision is not None:
            allocation_effectiveness = (
                applied_ftc_decision.estimated_thruster_effectiveness
            )
        else:
            allocation_effectiveness = np.ones(
                len(self.thruster_array.thrusters)
            )
        wrench_scale = (
            1.0
            if applied_ftc_decision is None
            else applied_ftc_decision.wrench_scale
        )
        allocation = self.thruster_array.allocate(
            wrench_scale * control.desired_wrench_body,
            thruster_effectiveness=allocation_effectiveness,
        )
        actuation = self.actuator_bank.apply(
            allocation.thruster_forces,
            time_s=self.state.time,
            dt=dt,
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
        health = None
        observations = None
        if self.sensor_suite is not None:
            sensor_packet = self.sensor_suite.read(
                state,
                dt=dt,
                linear_acceleration_body=(
                    state.body_velocity[:3] - previous_velocity[:3]
                ) / float(dt),
            )
            motion_context = {
                "desired_velocity_ned": control.desired_velocity_ned,
                "desired_angular_velocity_body": (
                    control.desired_angular_velocity_body
                ),
            }
            if self.sensor_fault_observer is not None:
                observations = self.sensor_fault_observer.update(
                    state.time,
                    sensor_packet,
                    motion_context=motion_context,
                )
                sensor_packet["sensor_fault_observations"] = {
                    name: observation.to_dict()
                    for name, observation in observations.items()
                }
                sensor_packet["sensor_fault_observation_summary"] = (
                    self.sensor_fault_observer.summarize(observations)
                )
            if self.sensor_health_monitor is not None:
                health = self.sensor_health_monitor.update(
                    state.time,
                    sensor_packet,
                    motion_context=motion_context,
                    rebaseline_sensors=(
                        ()
                        if self.sensor_fault_observer is None
                        else self.sensor_fault_observer.rebaseline_sensors
                    ),
                )
                sensor_packet["sensor_health"] = {
                    name: result.to_dict()
                    for name, result in health.items()
                }
                sensor_packet["sensor_health_summary"] = (
                    self.sensor_health_monitor.summarize(health)
                )
            if self.state_estimator is not None:
                self.state_estimate = self.state_estimator.update(
                    time_s=state.time,
                    dt=dt,
                    sensor_packet=sensor_packet,
                    sensor_health=health,
                )

        log = {
            "time": float(state.time),
            "thruster_names": tuple(self.thruster_array.names),
            "position_ned": state.position_ned.copy(),
            "euler_rpy": state.euler_rpy.copy(),
            "body_velocity": state.body_velocity.copy(),
            "true_position_error_ned": (
                target.position_ned - state.position_ned
            ),
            "sensor_feedback_enabled": self.sensor_feedback_enabled,
            "controller_state_source": controller_state_source,
            "controller_position_ned": (
                controller_state.position_ned.copy()
            ),
            "controller_euler_rpy": controller_state.euler_rpy.copy(),
            "controller_body_velocity": (
                controller_state.body_velocity.copy()
            ),
            "target_position_ned": target.position_ned.copy(),
            "target_euler_rpy": target.euler_rpy.copy(),
            "guidance_context_id": target.guidance_context_id,
            "position_error_ned": control.position_error_ned.copy(),
            "attitude_error_body": control.attitude_error_body.copy(),
            "desired_velocity_ned": control.desired_velocity_ned.copy(),
            "desired_angular_velocity_body": (
                control.desired_angular_velocity_body.copy()
            ),
            "desired_wrench_body": allocation.desired_wrench.copy(),
            "controller_desired_wrench_body": (
                control.desired_wrench_body.copy()
            ),
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
            "thruster_expected_rpms": actuation.expected_rpms.copy(),
            "thruster_measured_rpms": actuation.measured_rpms.copy(),
            "thruster_measured_voltages": actuation.measured_voltages.copy(),
            "thruster_measured_temperatures": (
                actuation.measured_temperatures.copy()
            ),
            "thruster_telemetry_valid": actuation.telemetry_valid.copy(),
            "thruster_telemetry_age_s": actuation.telemetry_age_s.copy(),
            "thruster_force_limits": np.maximum(
                np.abs(self.thruster_array.min_forces),
                np.abs(self.thruster_array.max_forces),
            ),
            "thruster_allocation_matrix": (
                self.thruster_array.allocation_matrix.copy()
            ),
            "thruster_nominal_voltage": self.actuator_bank.nominal_voltage,
            "thruster_ambient_temperature": (
                self.actuator_bank.ambient_temperature
            ),
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
                (
                    self.ideal_fault_tolerant_allocation
                    and np.any(allocation_effectiveness < 1.0)
                )
                or (
                    applied_ftc_decision is not None
                    and applied_ftc_decision.intervention_required
                )
            ),
        }
        if sensor_packet is not None:
            log.update(sensor_packet)
        if self.state_estimate is not None:
            estimated_state = self.state_estimate.state
            log.update({
                "estimated_position_ned": (
                    estimated_state.position_ned.copy()
                ),
                "estimated_euler_rpy": estimated_state.euler_rpy.copy(),
                "estimated_body_velocity": (
                    estimated_state.body_velocity.copy()
                ),
                "state_estimate_quality": self.state_estimate.quality,
                "state_estimate_sources": dict(
                    self.state_estimate.sources
                ),
                "state_estimate_excluded_sensors": (
                    self.state_estimate.excluded_sensors
                ),
                "state_estimate_rejected_sensors": (
                    self.state_estimate.rejected_sensors
                ),
                "state_estimate_fallback_durations_s": dict(
                    self.state_estimate.fallback_durations_s
                ),
                "horizontal_position_reference": (
                    self.state_estimate.horizontal_position_reference
                ),
                "state_estimate_ftc_recommendation": (
                    self.state_estimate.ftc_recommendation
                ),
            })
        if self.ftc_supervisor is not None:
            provider = self.ftc_evidence_provider
            evidence = (
                build_rule_based_ftc_evidence(
                    log, config=self.ftc_supervisor.config
                )
                if provider is None
                else provider(log)
            )
            if not isinstance(evidence, FTCEvidence):
                raise TypeError("ftc_evidence_provider must return FTCEvidence")
            self.ftc_decision = self.ftc_supervisor.update(evidence)
            log.update({
                "ftc_applied_action": applied_ftc_decision.action.value,
                "ftc_action": self.ftc_decision.action.value,
                "ftc_reason": self.ftc_decision.reason,
                "ftc_intervention_requested": (
                    self.ftc_decision.intervention_required
                ),
                "ftc_mission_abort_requested": (
                    self.ftc_decision.mission_abort_requested
                ),
                "ftc_controlled_ascent_requested": (
                    self.ftc_decision.controlled_ascent_requested
                ),
                "ftc_targeted_thruster_index": (
                    self.ftc_decision.targeted_thruster_index
                ),
                "ftc_targeted_thruster_name": (
                    self.ftc_decision.targeted_thruster_name
                ),
                "ftc_estimated_effectiveness_next_step": (
                    self.ftc_decision
                    .estimated_thruster_effectiveness.copy()
                ),
                "ftc_evidence_source": evidence.source,
                "ftc_no_output_scores": evidence.no_output_scores.copy(),
                "ftc_untrusted_esc_channels": (
                    evidence.untrusted_esc_channels
                ),
                "ftc_tracking_error_ratio": evidence.tracking_error_ratio,
                "ftc_control_saturation_ratio": (
                    evidence.control_saturation_ratio
                ),
                "ftc_allocation_residual_ratio": (
                    evidence.allocation_residual_ratio
                ),
                "ftc_sensor_guard_action": evidence.sensor_guard_action,
                "ftc_untrusted_sensors": evidence.untrusted_sensors,
            })
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
