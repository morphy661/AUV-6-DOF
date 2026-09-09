"""Safety-gated rule-based FTC supervisor for the six-thruster AUV.

The learned detector is evidence, not an actuator command. This supervisor
allows targeted reallocation only when per-thruster current and RPM jointly
support a no-output fault under sufficient excitation. Thrust-loss evidence
is logged unless independent control-stress evidence requires a safer action.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

import numpy as np


THRUSTER_NAMES = ("H1", "H2", "H3", "H4", "V1", "V2")
NO_OUTPUT_MODE = 1
SENSOR_GUARD_ACTIONS = (
    "none",
    "observe",
    "reject_current_sample",
    "degraded_navigation",
    "safe_hold_or_abort",
)
SENSOR_NAMES = ("depth", "imu", "dvl")


class FTCAction(str, Enum):
    NORMAL_CONTROL = "normal_control"
    LOG_ONLY = "log_only"
    DEGRADED_OPERATION = "degraded_operation"
    TARGETED_REALLOCATION = "targeted_reallocation"
    SAFE_HOLD_OR_ABORT = "safe_hold_or_abort"
    CONTROLLED_ASCENT = "controlled_ascent"


@dataclass(frozen=True)
class FTCSupervisorConfig:
    """Safety thresholds; ratios are normalized by operational limits."""

    no_output_score_threshold: float = 0.80
    no_output_score_margin: float = 0.20
    no_output_confirmation_s: float = 0.50
    minimum_excitation_ratio: float = 0.20
    vertical_minimum_excitation_ratio: Optional[float] = 0.08
    require_fresh_esc_telemetry: bool = True
    maximum_esc_telemetry_age_s: float = 0.20
    minimum_expected_current_a: float = 0.30
    minimum_expected_rpm: float = 50.0
    degraded_tracking_error_ratio: float = 0.60
    degraded_control_saturation_ratio: float = 0.80
    degraded_allocation_residual_ratio: float = 0.15
    critical_tracking_error_ratio: float = 1.00
    critical_control_saturation_ratio: float = 0.95
    critical_allocation_residual_ratio: float = 0.30
    degraded_confirmation_s: float = 2.00
    critical_confirmation_s: float = 2.00
    stress_only_critical_confirmation_s: float = 5.00
    stress_only_recovery_fraction: float = 0.80
    recovery_confirmation_s: float = 5.00
    isolated_thruster_effectiveness: float = 0.0
    degraded_wrench_scale: float = 0.65
    safe_wrench_scale: float = 0.35
    horizontal_tracking_limit_m: float = 8.0
    depth_tracking_limit_m: float = 3.0
    tilt_limit_deg: float = 35.0
    vertical_fault_heave_weight: float = 0.10
    vertical_fault_pitch_weight: float = 5.0

    def __post_init__(self):
        if not isinstance(self.require_fresh_esc_telemetry, (bool, np.bool_)):
            raise ValueError("require_fresh_esc_telemetry must be boolean")
        unit_interval = (
            "no_output_score_threshold",
            "no_output_score_margin",
            "minimum_excitation_ratio",
            "degraded_control_saturation_ratio",
            "critical_control_saturation_ratio",
            "isolated_thruster_effectiveness",
            "degraded_wrench_scale",
            "safe_wrench_scale",
            "stress_only_recovery_fraction",
        )
        for name in unit_interval:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.vertical_minimum_excitation_ratio is not None:
            value = float(self.vertical_minimum_excitation_ratio)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    "vertical_minimum_excitation_ratio must be in [0, 1]"
                )
        non_negative = (
            "no_output_confirmation_s",
            "maximum_esc_telemetry_age_s",
            "minimum_expected_current_a",
            "minimum_expected_rpm",
            "degraded_tracking_error_ratio",
            "degraded_allocation_residual_ratio",
            "critical_tracking_error_ratio",
            "critical_allocation_residual_ratio",
            "degraded_confirmation_s",
            "critical_confirmation_s",
            "stress_only_critical_confirmation_s",
            "recovery_confirmation_s",
            "horizontal_tracking_limit_m",
            "depth_tracking_limit_m",
            "tilt_limit_deg",
            "vertical_fault_heave_weight",
            "vertical_fault_pitch_weight",
        )
        for name in non_negative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "horizontal_tracking_limit_m", "depth_tracking_limit_m",
            "tilt_limit_deg", "vertical_fault_pitch_weight",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if (
            self.critical_tracking_error_ratio
            < self.degraded_tracking_error_ratio
            or self.critical_control_saturation_ratio
            < self.degraded_control_saturation_ratio
            or self.critical_allocation_residual_ratio
            < self.degraded_allocation_residual_ratio
        ):
            raise ValueError("critical thresholds cannot be below degraded thresholds")

    @property
    def minimum_excitation_ratios(self):
        """Return per-thruster thresholds without relaxing horizontal gates."""

        thresholds = np.full(6, self.minimum_excitation_ratio, dtype=float)
        if self.vertical_minimum_excitation_ratio is not None:
            thresholds[4:] = self.vertical_minimum_excitation_ratio
        return thresholds


def _six_vector(values, name, default=0.0):
    if values is None:
        return np.full(6, default, dtype=float)
    vector = np.asarray(values, dtype=float)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain six finite values")
    return vector.copy()


def _six_bool_vector(values, name, default=True):
    if values is None:
        return np.full(6, bool(default), dtype=bool)
    vector = np.asarray(values)
    if vector.shape != (6,):
        raise ValueError(f"{name} must contain six boolean values")
    if vector.dtype == np.bool_:
        return vector.copy()
    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must contain six boolean values"
        ) from error
    if not np.all(np.isfinite(numeric)) or not np.all(
        (numeric == 0.0) | (numeric == 1.0)
    ):
        raise ValueError(f"{name} must contain six boolean values")
    return numeric.astype(bool)


@dataclass(frozen=True)
class FTCEvidence:
    """Observable detector and control evidence for one causal update."""

    time_s: float
    health_level: int = 0
    confirmed_mode: int = 0
    fault_probability: float = 0.0
    probable_group: str = "none"
    tracking_error_ratio: float = 0.0
    control_saturation_ratio: float = 0.0
    allocation_residual_ratio: float = 0.0
    no_output_scores: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=float)
    )
    excitation_ratios: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=float)
    )
    untrusted_esc_channels: tuple[str, ...] = ()
    vertical_control_unavailable: bool = False
    sensor_guard_action: str = "none"
    untrusted_sensors: tuple[str, ...] = ()
    source: str = "unknown"
    tracking_error_available: bool = True
    horizontal_tracking_error_m: Optional[float] = None
    depth_tracking_error_m: Optional[float] = None
    tilt_angle_deg: Optional[float] = None
    yaw_tracking_error_deg: Optional[float] = None
    position_tracking_safety_active: bool = False

    def __post_init__(self):
        time_s = float(self.time_s)
        if not isinstance(self.tracking_error_available, (bool, np.bool_)):
            raise ValueError("tracking_error_available must be boolean")
        object.__setattr__(
            self, "tracking_error_available", bool(self.tracking_error_available)
        )
        if not isinstance(
            self.position_tracking_safety_active, (bool, np.bool_)
        ):
            raise ValueError("position_tracking_safety_active must be boolean")
        object.__setattr__(
            self,
            "position_tracking_safety_active",
            bool(self.position_tracking_safety_active),
        )
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        object.__setattr__(self, "time_s", time_s)
        if int(self.health_level) not in range(4):
            raise ValueError("health_level must be between 0 and 3")
        if int(self.confirmed_mode) not in (0, 1, 2):
            raise ValueError("confirmed_mode must be 0, 1, or 2")
        object.__setattr__(self, "health_level", int(self.health_level))
        object.__setattr__(self, "confirmed_mode", int(self.confirmed_mode))
        for name in ("fault_probability", "control_saturation_ratio"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            object.__setattr__(self, name, value)
        for name in ("tracking_error_ratio", "allocation_residual_ratio"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "horizontal_tracking_error_m", "depth_tracking_error_m",
            "tilt_angle_deg", "yaw_tracking_error_deg",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        scores = _six_vector(self.no_output_scores, "no_output_scores")
        excitation = _six_vector(self.excitation_ratios, "excitation_ratios")
        if np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("no_output_scores must be within [0, 1]")
        if np.any(excitation < 0.0) or np.any(excitation > 1.0):
            raise ValueError("excitation_ratios must be within [0, 1]")
        object.__setattr__(self, "no_output_scores", scores)
        object.__setattr__(self, "excitation_ratios", excitation)
        untrusted_esc = tuple(sorted(set(
            str(channel) for channel in self.untrusted_esc_channels
        )))
        if any(channel not in THRUSTER_NAMES for channel in untrusted_esc):
            raise ValueError(
                "untrusted_esc_channels contains an unknown thruster"
            )
        object.__setattr__(self, "untrusted_esc_channels", untrusted_esc)
        sensor_action = str(self.sensor_guard_action)
        if sensor_action not in SENSOR_GUARD_ACTIONS:
            raise ValueError(
                f"sensor_guard_action must be one of {SENSOR_GUARD_ACTIONS}"
            )
        object.__setattr__(self, "sensor_guard_action", sensor_action)
        untrusted = tuple(sorted(set(
            str(sensor) for sensor in self.untrusted_sensors
        )))
        if any(sensor not in SENSOR_NAMES for sensor in untrusted):
            raise ValueError("untrusted_sensors contains an unknown sensor")
        object.__setattr__(self, "untrusted_sensors", untrusted)


@dataclass(frozen=True)
class FTCDecision:
    action: FTCAction
    reason: str
    estimated_thruster_effectiveness: np.ndarray
    isolated_thruster_indices: tuple[int, ...]
    targeted_thruster_index: Optional[int]
    targeted_thruster_name: Optional[str]
    wrench_scale: float
    intervention_required: bool
    mission_abort_requested: bool
    controlled_ascent_requested: bool
    stress_level: int
    guidance_mode: str
    allocation_wrench_weights: Optional[np.ndarray]


class FTCSafetySupervisor:
    """Causal safety gate between diagnosis evidence and control allocation."""

    def __init__(self, config=None):
        self.config = config or FTCSupervisorConfig()
        self.reset()

    def reset(self):
        self.last_time = None
        self.no_output_candidate = None
        self.no_output_candidate_start = None
        self.isolated_thrusters = set()
        self.stress_level = 0
        self.degraded_stress_start = None
        self.critical_stress_start = None
        self.corroborated_critical_start = None
        self.stress_only_peak_severity = 0.0
        self.critical_escalation_source = None
        self.stress_recovery_start = None
        self.last_decision = self._decision(
            FTCAction.NORMAL_CONTROL,
            "No fault or control-stress evidence.",
        )
        return self.last_decision

    def _effectiveness(self):
        effectiveness = np.ones(6, dtype=float)
        for index in self.isolated_thrusters:
            effectiveness[index] = self.config.isolated_thruster_effectiveness
        return effectiveness

    def _decision(self, action, reason, target=None):
        wrench_scale = 1.0
        abort = False
        ascent = False
        if action == FTCAction.DEGRADED_OPERATION:
            wrench_scale = self.config.degraded_wrench_scale
        elif action == FTCAction.SAFE_HOLD_OR_ABORT:
            wrench_scale = self.config.safe_wrench_scale
            abort = True
        elif action == FTCAction.CONTROLLED_ASCENT:
            wrench_scale = self.config.safe_wrench_scale
            abort = True
            ascent = True
        vertical_isolated = bool(self.isolated_thrusters.intersection({4, 5}))
        guidance_mode = (
            "vertical_thruster_hold" if vertical_isolated else "normal"
        )
        allocation_wrench_weights = None
        if vertical_isolated:
            allocation_wrench_weights = np.array([
                1.0, 1.0,
                self.config.vertical_fault_heave_weight,
                0.0,
                self.config.vertical_fault_pitch_weight,
                1.0,
            ])
        return FTCDecision(
            action=action,
            reason=reason,
            estimated_thruster_effectiveness=self._effectiveness(),
            isolated_thruster_indices=tuple(
                index + 1 for index in sorted(self.isolated_thrusters)
            ),
            targeted_thruster_index=(None if target is None else target + 1),
            targeted_thruster_name=(
                None if target is None else THRUSTER_NAMES[target]
            ),
            wrench_scale=wrench_scale,
            intervention_required=action not in (
                FTCAction.NORMAL_CONTROL,
                FTCAction.LOG_ONLY,
            ),
            mission_abort_requested=abort,
            controlled_ascent_requested=ascent,
            stress_level=self.stress_level,
            guidance_mode=guidance_mode,
            allocation_wrench_weights=allocation_wrench_weights,
        )

    def _direct_no_output_candidate(self, evidence):
        scores = evidence.no_output_scores
        ranking = np.argsort(-scores, kind="stable")
        best = int(ranking[0])
        second_score = float(scores[ranking[1]])
        if best in self.isolated_thrusters:
            return None
        if (
            evidence.excitation_ratios[best]
            < self.config.minimum_excitation_ratios[best]
        ):
            return None
        if scores[best] < self.config.no_output_score_threshold:
            return None
        if float(scores[best] - second_score) < self.config.no_output_score_margin:
            return None
        return best

    def _update_direct_no_output(self, evidence):
        candidate = self._direct_no_output_candidate(evidence)
        if candidate is None:
            self.no_output_candidate = None
            self.no_output_candidate_start = None
            return None
        if candidate != self.no_output_candidate:
            self.no_output_candidate = candidate
            self.no_output_candidate_start = evidence.time_s
        if (
            evidence.time_s - self.no_output_candidate_start
            >= self.config.no_output_confirmation_s
        ):
            self.isolated_thrusters.add(candidate)
            self.no_output_candidate = None
            self.no_output_candidate_start = None
            return candidate
        return None

    def _instantaneous_stress_level(self, evidence):
        tracking = evidence.tracking_error_ratio if evidence.tracking_error_available else 0.0
        if (
            tracking
            >= self.config.critical_tracking_error_ratio
            or evidence.allocation_residual_ratio
            >= self.config.critical_allocation_residual_ratio
            or (
                evidence.health_level >= 2
                and evidence.control_saturation_ratio
                >= self.config.critical_control_saturation_ratio
            )
        ):
            return 2
        if (
            tracking
            >= self.config.degraded_tracking_error_ratio
            or evidence.allocation_residual_ratio
            >= self.config.degraded_allocation_residual_ratio
            or (
                evidence.health_level >= 1
                and evidence.control_saturation_ratio
                >= self.config.degraded_control_saturation_ratio
            )
        ):
            return 1
        return 0

    def _has_thruster_fault_evidence(self, evidence):
        return bool(
            evidence.health_level >= 2
            or evidence.confirmed_mode in (1, 2)
            or self._direct_no_output_candidate(evidence) is not None
        )

    def _critical_stress_severity(self, evidence):
        config = self.config
        ratios = [
            (evidence.tracking_error_ratio if evidence.tracking_error_available else 0.0)
            / max(config.critical_tracking_error_ratio, 1e-12),
            evidence.allocation_residual_ratio
            / max(config.critical_allocation_residual_ratio, 1e-12),
        ]
        if evidence.health_level >= 2:
            ratios.append(
                evidence.control_saturation_ratio
                / max(config.critical_control_saturation_ratio, 1e-12)
            )
        return float(max(ratios))

    def _update_stress_state(self, evidence):
        instantaneous = self._instantaneous_stress_level(evidence)
        if instantaneous >= 1:
            if self.degraded_stress_start is None:
                self.degraded_stress_start = evidence.time_s
        else:
            self.degraded_stress_start = None

        has_fault_evidence = self._has_thruster_fault_evidence(evidence)
        if instantaneous >= 2:
            severity = self._critical_stress_severity(evidence)
            if self.critical_stress_start is None:
                self.critical_stress_start = evidence.time_s
                self.stress_only_peak_severity = severity
            else:
                self.stress_only_peak_severity = max(
                    self.stress_only_peak_severity, severity
                )
            if has_fault_evidence:
                if self.corroborated_critical_start is None:
                    self.corroborated_critical_start = evidence.time_s
            else:
                self.corroborated_critical_start = None
        else:
            severity = 0.0
            self.critical_stress_start = None
            self.corroborated_critical_start = None
            self.stress_only_peak_severity = 0.0

        degraded_ready = bool(
            self.degraded_stress_start is not None
            and evidence.time_s - self.degraded_stress_start
            >= self.config.degraded_confirmation_s
        )
        corroborated_critical_ready = bool(
            self.corroborated_critical_start is not None
            and evidence.time_s - self.corroborated_critical_start
            >= self.config.critical_confirmation_s
        )
        stress_only_critical_ready = bool(
            self.critical_stress_start is not None
            and evidence.time_s - self.critical_stress_start
            >= self.config.stress_only_critical_confirmation_s
        )
        if (
            stress_only_critical_ready
            and not corroborated_critical_ready
            and severity
            <= self.config.stress_only_recovery_fraction
            * self.stress_only_peak_severity
        ):
            # The disturbance remains above the critical threshold but is
            # clearly decaying. Keep degraded control and grant a new recovery
            # interval instead of issuing an abort request at the old peak.
            self.critical_stress_start = evidence.time_s
            self.stress_only_peak_severity = severity
            stress_only_critical_ready = False

        if corroborated_critical_ready:
            desired_level = 2
            escalation_source = "thruster_fault_evidence"
        elif stress_only_critical_ready:
            desired_level = 2
            escalation_source = "stress_only_timeout"
        elif degraded_ready:
            desired_level = 1
            escalation_source = None
        else:
            desired_level = 0
            escalation_source = None

        if desired_level > self.stress_level:
            self.stress_level = desired_level
            self.stress_recovery_start = None
            if desired_level == 2:
                self.critical_escalation_source = escalation_source
        elif desired_level < self.stress_level:
            if self.stress_recovery_start is None:
                self.stress_recovery_start = evidence.time_s
            if (
                evidence.time_s - self.stress_recovery_start
                >= self.config.recovery_confirmation_s
            ):
                self.stress_level = desired_level
                self.stress_recovery_start = None
                if self.stress_level < 2:
                    self.critical_escalation_source = None
        else:
            self.stress_recovery_start = None

    def update(self, evidence: FTCEvidence):
        if not isinstance(evidence, FTCEvidence):
            raise TypeError("evidence must be FTCEvidence")
        if self.last_time is not None and evidence.time_s < self.last_time:
            raise ValueError("evidence time_s must be non-decreasing")
        self.last_time = evidence.time_s
        new_isolation = self._update_direct_no_output(evidence)
        self._update_stress_state(evidence)

        vertical_unavailable = evidence.vertical_control_unavailable or {
            4, 5
        }.issubset(self.isolated_thrusters)
        if vertical_unavailable:
            decision = self._decision(
                FTCAction.CONTROLLED_ASCENT,
                "Vertical control authority is unavailable; request the independent safe-ascent system.",
            )
        elif new_isolation is not None:
            decision = self._decision(
                FTCAction.TARGETED_REALLOCATION,
                f"Direct current/RPM no-output evidence confirmed for {THRUSTER_NAMES[new_isolation]}.",
                target=new_isolation,
            )
        elif evidence.sensor_guard_action == "safe_hold_or_abort":
            decision = self._decision(
                FTCAction.SAFE_HOLD_OR_ABORT,
                "Confirmed IMU unavailability or stuck data makes attitude feedback untrustworthy; request safe hold or mission abort without guessing a thruster fault.",
            )
        elif self.stress_level >= 2:
            if self.critical_escalation_source == "stress_only_timeout":
                reason = (
                    "Critical control stress did not show sufficient recovery "
                    "within the stress-only safety interval; do not guess a "
                    "thruster location."
                )
            else:
                reason = (
                    "Persistent critical control stress is corroborated by "
                    "thruster fault evidence; do not guess an isolation "
                    "without direct ESC evidence."
                )
            decision = self._decision(
                FTCAction.SAFE_HOLD_OR_ABORT,
                reason,
            )
        elif self.isolated_thrusters:
            target = min(self.isolated_thrusters)
            decision = self._decision(
                FTCAction.TARGETED_REALLOCATION,
                "Previously confirmed no-output isolation remains latched for this mission.",
                target=target,
            )
        elif evidence.sensor_guard_action == "degraded_navigation":
            decision = self._decision(
                FTCAction.DEGRADED_OPERATION,
                "Confirmed depth or DVL sensor failure; reduce navigation demand and use the remaining state-estimation sources.",
            )
        elif self.stress_level == 1:
            decision = self._decision(
                FTCAction.DEGRADED_OPERATION,
                "Persistent control stress; reduce maneuver demand and observe recovery before requesting abort.",
            )
        elif evidence.sensor_guard_action in (
            "reject_current_sample",
            "observe",
        ):
            decision = self._decision(
                FTCAction.LOG_ONLY,
                "Sensor anomaly is isolated to a sample or still awaiting confirmation; reject or observe it without thruster isolation.",
            )
        elif evidence.untrusted_esc_channels:
            decision = self._decision(
                FTCAction.LOG_ONLY,
                "ESC telemetry is invalid or stale for "
                f"{', '.join(evidence.untrusted_esc_channels)}; record the "
                "communication anomaly without guessing a thruster fault.",
            )
        elif evidence.health_level >= 1 or evidence.confirmed_mode != 0:
            decision = self._decision(
                FTCAction.LOG_ONLY,
                "Fault evidence is not independently severe enough for FTC intervention; record it for inspection.",
            )
        else:
            decision = self._decision(
                FTCAction.NORMAL_CONTROL,
                "No persistent fault or control-stress evidence.",
            )
        self.last_decision = decision
        return decision


def _mapping_value(mapping, name, default=None):
    return mapping.get(name, default)


def _maintenance_value(result, name, default):
    if result is None:
        return default
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _optional_vector(mapping, name, size):
    value = mapping.get(name)
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        return None
    return vector.copy()


def _authority_normalized_allocation_residual(log, residual, limits):
    """Normalize residual by currently available actuator authority."""

    matrix_value = log.get("thruster_allocation_matrix")
    if matrix_value is None:
        return None
    try:
        matrix = np.asarray(matrix_value, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
        return None
    effectiveness = _optional_vector(
        log, "allocation_thruster_effectiveness", 6
    )
    if effectiveness is None:
        effectiveness = np.ones(6, dtype=float)
    if np.any(effectiveness < 0.0) or np.any(effectiveness > 1.0):
        return None
    effective_matrix = matrix * effectiveness[np.newaxis, :]
    authority = np.abs(effective_matrix) @ limits
    controlled_axes = np.array([0, 1, 2, 5])
    priority = _optional_vector(
        log, "ftc_allocation_wrench_weights_applied", 6
    )
    if priority is None:
        priority = np.ones(6, dtype=float)
    controlled_priority = priority[controlled_axes]
    priority_scale = max(float(np.max(controlled_priority)), 1e-12)
    controlled_priority = controlled_priority / priority_scale
    available = (
        (authority[controlled_axes] > 1e-9)
        & (controlled_priority > 0.0)
    )
    if not np.any(available):
        return None
    normalized = (
        np.abs(residual[controlled_axes][available])
        / authority[controlled_axes][available]
        * controlled_priority[available]
    )
    return float(np.max(normalized))


def _wrap_angle(angle):
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _observable_tracking_metrics(log, config):
    """Build safety errors from the same estimated state used for control."""

    position = _optional_vector(log, "controller_position_ned", 3)
    attitude = _optional_vector(log, "controller_euler_rpy", 3)
    target_position = _optional_vector(log, "target_position_ned", 3)
    target_attitude = _optional_vector(log, "target_euler_rpy", 3)
    if (
        position is None or attitude is None
        or target_position is None or target_attitude is None
    ):
        return None
    error = target_position - position
    horizontal = float(np.linalg.norm(error[:2]))
    depth = float(abs(error[2]))
    # Angle between the body z-axis and NED down is invariant to yaw and
    # remains meaningful near the Euler pitch singularity.
    tilt = float(np.degrees(np.arccos(np.clip(
        np.cos(attitude[0]) * np.cos(attitude[1]), -1.0, 1.0
    ))))
    yaw = float(abs(np.degrees(_wrap_angle(
        target_attitude[2] - attitude[2]
    ))))
    # A distant mission waypoint is not itself a tracking failure. Position
    # limits become safety evidence only after FTC has latched a local hold
    # point; tilt remains meaningful in every mission mode.
    position_safety_active = (
        str(log.get("ftc_guidance_mode_applied", "normal"))
        == "vertical_thruster_hold"
    )
    ratios = [tilt / config.tilt_limit_deg]
    if position_safety_active:
        ratios.extend((
            horizontal / config.horizontal_tracking_limit_m,
            depth / config.depth_tracking_limit_m,
        ))
    ratio = max(ratios)
    return {
        "tracking_error_ratio": float(ratio),
        "horizontal_tracking_error_m": horizontal,
        "depth_tracking_error_m": depth,
        "tilt_angle_deg": tilt,
        "yaw_tracking_error_deg": yaw,
        "position_tracking_safety_active": position_safety_active,
    }


def build_rule_based_ftc_evidence(
    log: Mapping[str, Any],
    maintenance_result=None,
    config=None,
):
    """Build evidence from ESC telemetry and observable control outputs.

    Simulator fault labels, actual thrust, and injected effectiveness are not
    read. Direct no-output evidence remains independent of the learned model.
    """

    config = config or FTCSupervisorConfig()
    time_s = float(_mapping_value(log, "time", 0.0))
    commanded = _six_vector(
        _mapping_value(log, "commanded_thruster_forces"),
        "commanded_thruster_forces",
    )
    limits = np.maximum(
        np.abs(_six_vector(
            _mapping_value(log, "thruster_force_limits"),
            "thruster_force_limits",
            default=1.0,
        )),
        1e-6,
    )
    expected_current = np.abs(_six_vector(
        _mapping_value(log, "thruster_expected_currents"),
        "thruster_expected_currents",
    ))
    measured_current = np.abs(_six_vector(
        _mapping_value(log, "thruster_measured_currents"),
        "thruster_measured_currents",
    ))
    expected_rpm = np.abs(_six_vector(
        _mapping_value(log, "thruster_expected_rpms"),
        "thruster_expected_rpms",
    ))
    measured_rpm = np.abs(_six_vector(
        _mapping_value(log, "thruster_measured_rpms"),
        "thruster_measured_rpms",
    ))
    telemetry_valid = _six_bool_vector(
        _mapping_value(log, "thruster_telemetry_valid"),
        "thruster_telemetry_valid",
        default=True,
    )
    telemetry_age_s = _six_vector(
        _mapping_value(log, "thruster_telemetry_age_s"),
        "thruster_telemetry_age_s",
        default=0.0,
    )
    if np.any(telemetry_age_s < 0.0):
        raise ValueError("thruster_telemetry_age_s must be non-negative")
    if config.require_fresh_esc_telemetry:
        telemetry_fresh = (
            telemetry_valid
            & (telemetry_age_s <= config.maximum_esc_telemetry_age_s)
        )
        untrusted_esc_channels = tuple(
            THRUSTER_NAMES[index]
            for index in np.flatnonzero(~telemetry_fresh)
        )
    else:
        telemetry_fresh = np.ones(6, dtype=bool)
        untrusted_esc_channels = ()

    excitation = np.clip(np.abs(commanded) / limits, 0.0, 1.0)
    current_dropout = np.clip(
        1.0 - measured_current / np.maximum(expected_current, 1e-6),
        0.0,
        1.0,
    )
    rpm_dropout = np.clip(
        1.0 - measured_rpm / np.maximum(expected_rpm, 1e-6),
        0.0,
        1.0,
    )
    active = (
        (excitation >= config.minimum_excitation_ratios)
        & (expected_current >= config.minimum_expected_current_a)
        & (expected_rpm >= config.minimum_expected_rpm)
        & telemetry_fresh
    )
    no_output_scores = np.where(
        active, np.minimum(current_dropout, rpm_dropout), 0.0
    )

    health_level = int(_maintenance_value(
        maintenance_result, "health_level", 0
    ))
    confirmed_mode = int(_maintenance_value(
        maintenance_result, "confirmed_mode", 0
    ))
    fault_probability = float(_maintenance_value(
        maintenance_result, "fault_probability", 0.0
    ))
    probable_group = str(_maintenance_value(
        maintenance_result, "suspected_group", "none"
    ))
    if float(np.max(no_output_scores)) >= config.no_output_score_threshold:
        health_level = max(health_level, 3)
        confirmed_mode = NO_OUTPUT_MODE
        fault_probability = max(
            fault_probability, float(np.max(no_output_scores))
        )

    tracking_metrics = _observable_tracking_metrics(log, config)
    explicit_tracking = _mapping_value(log, "tracking_error_ratio")
    if explicit_tracking is not None:
        tracking_error_ratio = float(explicit_tracking)
        tracking_error_available = True
    elif tracking_metrics is not None:
        tracking_error_ratio = tracking_metrics["tracking_error_ratio"]
        tracking_error_available = True
    else:
        tracking_error_ratio = 0.0
        tracking_error_available = False
    control_saturation_ratio = float(np.max(excitation))
    desired = _six_vector(
        _mapping_value(log, "desired_wrench_body"),
        "desired_wrench_body",
    )
    residual = _six_vector(
        _mapping_value(log, "allocation_residual_body"),
        "allocation_residual_body",
    )
    allocation_residual_ratio = _authority_normalized_allocation_residual(
        log, residual, limits
    )
    if allocation_residual_ratio is None:
        controlled_axes = np.array([0, 1, 2, 5])
        allocation_residual_ratio = float(
            np.linalg.norm(residual[controlled_axes])
            / max(np.linalg.norm(desired[controlled_axes]), 1.0)
        )
    vertical_unavailable = bool(np.all(
        no_output_scores[4:] >= config.no_output_score_threshold
    ))
    sensor_summary = _mapping_value(log, "sensor_health_summary", {})
    if not isinstance(sensor_summary, Mapping):
        sensor_summary = {}
    sensor_guard_action = str(
        sensor_summary.get("ftc_recommendation", "none")
    )
    if sensor_guard_action not in SENSOR_GUARD_ACTIONS:
        sensor_guard_action = "none"
    estimator_guard_action = str(_mapping_value(
        log, "state_estimate_ftc_recommendation", "none"
    ))
    if (
        sensor_guard_action == "none"
        and estimator_guard_action == "degraded_navigation"
    ):
        sensor_guard_action = estimator_guard_action
    untrusted_sensors = tuple(
        str(sensor)
        for sensor in sensor_summary.get("untrusted_sensors", ())
        if str(sensor) in SENSOR_NAMES
    )
    evidence_source = (
        "esc_rule+maintenance"
        if maintenance_result is not None
        else "esc_rule"
    )
    if sensor_guard_action != "none":
        evidence_source += "+sensor_guard"
    if estimator_guard_action == "degraded_navigation":
        evidence_source += "+state_estimator_guard"
    if untrusted_esc_channels:
        evidence_source += "+esc_telemetry_guard"
    if tracking_metrics is not None:
        evidence_source += "+tracking_safety"
    return FTCEvidence(
        time_s=time_s,
        health_level=health_level,
        confirmed_mode=confirmed_mode,
        fault_probability=np.clip(fault_probability, 0.0, 1.0),
        probable_group=probable_group,
        tracking_error_ratio=tracking_error_ratio,
        tracking_error_available=tracking_error_available,
        horizontal_tracking_error_m=(
            None if tracking_metrics is None
            else tracking_metrics["horizontal_tracking_error_m"]
        ),
        depth_tracking_error_m=(
            None if tracking_metrics is None
            else tracking_metrics["depth_tracking_error_m"]
        ),
        tilt_angle_deg=(
            None if tracking_metrics is None
            else tracking_metrics["tilt_angle_deg"]
        ),
        yaw_tracking_error_deg=(
            None if tracking_metrics is None
            else tracking_metrics["yaw_tracking_error_deg"]
        ),
        position_tracking_safety_active=(
            False if tracking_metrics is None
            else tracking_metrics["position_tracking_safety_active"]
        ),
        control_saturation_ratio=control_saturation_ratio,
        allocation_residual_ratio=allocation_residual_ratio,
        no_output_scores=no_output_scores,
        excitation_ratios=excitation,
        untrusted_esc_channels=untrusted_esc_channels,
        vertical_control_unavailable=vertical_unavailable,
        sensor_guard_action=sensor_guard_action,
        untrusted_sensors=untrusted_sensors,
        source=evidence_source,
    )
