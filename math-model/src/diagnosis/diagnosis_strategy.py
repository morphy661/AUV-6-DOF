from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


FAULT_NAMES = {
    0: "NO_FAULT",
    1: "BIAS",
    2: "DRIFT",
    3: "STUCK",
    4: "SPIKE",
    5: "NOISE_INCREASE",
    6: "THRUSTER_ENTANGLED",
    7: "THRUSTER_NO_OUTPUT",
    8: "THRUSTER_THRUST_LOSS",
}


@dataclass
class DiagnosisConfig:
    # Thruster thresholds
    cmd_vz_threshold: float = 0.6
    actual_vz_low_threshold: float = 0.10
    current_high_threshold: float = 3.0
    current_low_threshold: float = -3.0
    tracking_error_thruster_threshold: float = 3.0
    expected_thrust_active_threshold: float = 1.0
    actual_thrust_low_threshold: float = 0.3
    current_near_zero_threshold: float = 0.3
    thrust_loss_ratio_threshold: float = 0.6
    current_normal_residual_threshold: float = 1.0
    current_normal_ratio_low: float = 0.7
    current_normal_ratio_high: float = 1.3
    current_entangled_ratio_threshold: float = 1.4

    # Depth sensor thresholds
    spike_delta_threshold: float = 6.0
    stuck_depth_change_threshold: float = 0.02
    stuck_velocity_active_threshold: float = 0.20

    # BIAS / DRIFT thresholds
    bias_residual_threshold: float = 3.0
    bias_recent_residual_threshold: float = 3.0
    bias_max_slope_threshold: float = 0.12
    bias_step_threshold: float = 6.0

    # NOTE:
    # residual_slope is calculated over a short rolling history window.
    # Therefore, this threshold should not be too small, otherwise BIAS can be
    # confused with DRIFT during the transient response.
    drift_slope_threshold: float = 0.18
    drift_residual_range_threshold: float = 8.0
    drift_recent_residual_threshold: float = 5.0
    drift_min_recent_range: float = 4.0
    drift_min_samples: int = 10

    # NOISE / SPIKE thresholds
    noise_std_threshold: float = 1.5
    noise_large_diff_threshold: float = 2.0
    noise_large_diff_count_threshold: int = 8
    spike_max_jump_count: int = 2

    # History settings
    recent_window: int = 12
    min_history: int = 10


@dataclass
class DiagnosisResult:
    fault_id: int
    fault_name: str
    reason: str
    confidence: str
    source: str = "rule"

    def as_dict(self):
        return {
            "fault_id": self.fault_id,
            "fault_name": self.fault_name,
            "reason": self.reason,
            "confidence": self.confidence,
            "source": self.source,
        }


class DiagnosisStrategy:
    """
    Rule-based physical diagnosis.

    It uses residuals and sensor history to provide explainable diagnosis.
    The rule layer is designed to be combined with the Bi-LSTM Attention classifier.
    """

    def __init__(self, config: Optional[DiagnosisConfig] = None):
        self.config = config or DiagnosisConfig()

    def diagnose(
        self,
        sensor_data: Dict[str, Any],
        residuals: Dict[str, float],
        history: Optional[Sequence[Dict[str, Any]]] = None,
        ai_pred: int = 0,
    ) -> DiagnosisResult:

        # 1. Thruster faults are checked first because they have clear physical evidence:
        #    high command + low velocity + abnormal current.
        thruster_result = self._diagnose_thruster(residuals)
        if thruster_result.fault_id != 0:
            return thruster_result

        # 2. Then check depth sensor faults.
        sensor_result = self._diagnose_depth_sensor(
            residuals=residuals,
            history=history,
            ai_pred=ai_pred,
        )
        if sensor_result.fault_id != 0:
            return sensor_result

        # 3. If no strong rule evidence is found, use AI prediction as fallback.
        #
        # Important:
        # DRIFT is easy to confuse with long-route climb/descent trends, so
        # AI-only DRIFT fallback is blocked. BIAS and STUCK are allowed as
        # candidates here and should be debounced/confirmed in main.py.
        if ai_pred != 0:
            ai_pred = int(ai_pred)

            # Block AI-only DRIFT fallback, but do not block BIAS/STUCK.
            if ai_pred == 2:
                return DiagnosisResult(
                    fault_id=0,
                    fault_name=FAULT_NAMES[0],
                    reason=(
                        "AI predicted DRIFT, but no strong rule-based evidence was found. "
                        "AI-only DRIFT fallback is blocked to avoid long-route false alarms."
                    ),
                    confidence="Low",
                    source="rule_guard",
                )

            return DiagnosisResult(
                fault_id=ai_pred,
                fault_name=FAULT_NAMES.get(ai_pred, "UNKNOWN"),
                reason="No strong rule-based evidence; using AI classifier prediction as candidate.",
                confidence="Medium",
                source="ai_fallback",
            )

        return DiagnosisResult(
            fault_id=0,
            fault_name=FAULT_NAMES[0],
            reason="All residuals are within normal thresholds.",
            confidence="High",
            source="rule",
        )

    def _diagnose_thruster(self, residuals: Dict[str, float]) -> DiagnosisResult:
        cfg = self.config

        cmd_vz = float(residuals.get("cmd_vz", 0.0))
        actual_vz = float(residuals.get("actual_vz", 0.0))
        current_residual = float(residuals.get("current_residual", 0.0))
        tracking_error = abs(float(residuals.get("tracking_error", 0.0)))
        measured_current = float(residuals.get("measured_current", 0.0))
        expected_current = float(residuals.get("expected_current", 0.0))

        actual_thrust = float(residuals.get("actual_thrust", 0.0))
        expected_thrust = float(residuals.get("expected_thrust", 0.0))

        expected_thrust_abs = abs(expected_thrust)
        actual_thrust_abs = abs(actual_thrust)
        cmd_high = abs(cmd_vz) > cfg.cmd_vz_threshold
        velocity_low = abs(actual_vz) < cfg.actual_vz_low_threshold
        current_high = current_residual > cfg.current_high_threshold
        current_low = current_residual < cfg.current_low_threshold
        tracking_bad = tracking_error > cfg.tracking_error_thruster_threshold
        # --------------------------------------------------------
        # Motor-equation-based thruster diagnosis
        #
        # Priority is important:
        #   1) NO_OUTPUT: expected thrust exists, but current and thrust are near zero.
        #   2) ENTANGLED / overload: expected thrust exists, actual thrust is low,
        #      and current is clearly higher than expected.
        #   3) THRUST_LOSS: expected thrust exists, actual thrust is low,
        #      but current stays near normal.
        #
        # This prevents an ENTANGLED case from being incorrectly captured by
        # the THRUST_LOSS rule only because both cases have low actual thrust.
        # --------------------------------------------------------
        expected_thrust_active = expected_thrust_abs > cfg.expected_thrust_active_threshold
        actual_thrust_low = actual_thrust_abs < cfg.actual_thrust_low_threshold
        current_near_zero = abs(measured_current) < cfg.current_near_zero_threshold

        if abs(expected_current) > 1e-6:
            current_ratio = measured_current / (expected_current + 1e-6)
        else:
            current_ratio = 0.0

        current_high_for_entangled = (
            current_ratio > cfg.current_entangled_ratio_threshold
            or current_residual > cfg.current_high_threshold
        )

        current_near_normal = (
            cfg.current_normal_ratio_low <= current_ratio <= cfg.current_normal_ratio_high
            and abs(current_residual) < cfg.current_normal_residual_threshold
        )

        # 1) NO_OUTPUT: near-zero current + near-zero thrust.
        if expected_thrust_active and actual_thrust_low and current_near_zero:
            reason = (
                "Thruster no-output fault: expected thrust is active, "
                "but measured current and actual thrust are near zero."
            )

            return DiagnosisResult(
                fault_id=7,
                fault_name=FAULT_NAMES[7],
                reason=reason,
                confidence="High",
                source="rule",
            )

        # 2) ENTANGLED / overload: high current + low thrust.
        if expected_thrust_active and actual_thrust_low and current_high_for_entangled:
            reason = (
                "Thruster entanglement or overload-like fault: expected thrust is active, "
                "actual thrust is very low, and motor current is higher than expected."
            )
            if tracking_bad:
                reason += " Depth tracking error is also large."

            return DiagnosisResult(
                fault_id=6,
                fault_name=FAULT_NAMES[6],
                reason=reason,
                confidence="High",
                source="rule",
            )

        # 3) THRUST_LOSS: normal-like current + reduced thrust.
        thrust_ratio_low = (
            expected_thrust_abs > cfg.expected_thrust_active_threshold
            and actual_thrust_abs < cfg.thrust_loss_ratio_threshold * expected_thrust_abs
        )

        current_not_zero = abs(measured_current) >= cfg.current_near_zero_threshold

        if (
            thrust_ratio_low
            and current_near_normal
            and current_not_zero
            and not current_high_for_entangled
        ):
            reason = (
                "Thruster thrust-loss fault: expected thrust is active and motor current is near normal, "
                "but actual thrust is much lower than expected. "
                "This signal pattern may represent propeller blade damage, partial propeller loss, "
                "reduced propeller efficiency, biofouling, duct/guard-induced thrust reduction, "
                "or mild mechanical degradation."
            )

            return DiagnosisResult(
                fault_id=8,
                fault_name=FAULT_NAMES[8],
                reason=reason,
                confidence="High",
                source="rule",
            )
        if cmd_high and velocity_low and current_high:
            reason = (
                "Thruster entanglement: high vertical command, "
                "low actual vertical velocity, and motor current higher than expected."
            )
            if tracking_bad:
                reason += " Depth tracking error is also large."

            return DiagnosisResult(
                fault_id=6,
                fault_name=FAULT_NAMES[6],
                reason=reason,
                confidence="High",
                source="rule",
            )

        if cmd_high and velocity_low and current_low:
            reason = (
                "Thruster no-output fault: high vertical command, "
                "low actual vertical velocity, and motor current lower than expected."
            )
            if tracking_bad:
                reason += " Depth tracking error is also large."

            return DiagnosisResult(
                fault_id=7,
                fault_name=FAULT_NAMES[7],
                reason=reason,
                confidence="High",
                source="rule",
            )

        return DiagnosisResult(
            fault_id=0,
            fault_name=FAULT_NAMES[0],
            reason="No thruster inconsistency detected.",
            confidence="Low",
            source="rule",
        )

    def _diagnose_depth_sensor(
        self,
        residuals: Dict[str, float],
        history: Optional[Sequence[Dict[str, Any]]] = None,
        ai_pred: int = 0,
    ) -> DiagnosisResult:

        cfg = self.config

        if not history or len(history) < cfg.min_history:
            return DiagnosisResult(
                fault_id=0,
                fault_name=FAULT_NAMES[0],
                reason="Insufficient history for depth sensor diagnosis.",
                confidence="Low",
                source="rule",
            )

        depth_series = self._extract_series(history, "depth")
        vz_series = self._extract_series(history, "actual_vz")
        depth_residual_series = self._extract_series(history, "depth_residual")
        target_series = self._extract_series(history, "target_z")

        if len(depth_series) < cfg.min_history:
            return DiagnosisResult(
                fault_id=0,
                fault_name=FAULT_NAMES[0],
                reason="Insufficient depth history for depth sensor diagnosis.",
                confidence="Low",
                source="rule",
            )

        if len(depth_residual_series) == 0:
            depth_residual_series = np.array(
                [float(residuals.get("depth_residual", 0.0))],
                dtype=float,
            )

        depth_series = depth_series[np.isfinite(depth_series)]
        depth_residual_series = depth_residual_series[np.isfinite(depth_residual_series)]

        if len(depth_series) < cfg.min_history or len(depth_residual_series) < 2:
            return DiagnosisResult(
                fault_id=0,
                fault_name=FAULT_NAMES[0],
                reason="Insufficient valid residual history for depth sensor diagnosis.",
                confidence="Low",
                source="rule",
            )

        window_len = min(len(depth_series), len(depth_residual_series))
        depth_series = depth_series[-window_len:]
        depth_residual_series = depth_residual_series[-window_len:]

        # Basic statistics
        depth_change = abs(depth_series[-1] - depth_series[0])
        recent_depth_change = abs(depth_series[-1] - depth_series[-min(cfg.recent_window, len(depth_series))])
        depth_deltas = np.abs(np.diff(depth_series))

        residual_mean = float(np.mean(depth_residual_series))
        residual_std = float(np.std(depth_residual_series))
        residual_slope = self._linear_slope(depth_residual_series)
        residual_range = float(abs(depth_residual_series[-1] - depth_residual_series[0]))

        recent_n = min(cfg.recent_window, len(depth_residual_series))
        recent_residuals = depth_residual_series[-recent_n:]
        recent_abs_mean = float(np.mean(np.abs(recent_residuals)))
        latest_abs_residual = float(abs(depth_residual_series[-1]))
        recent_residual_range = float(abs(recent_residuals[-1] - recent_residuals[0])) if len(recent_residuals) >= 2 else 0.0
        recent_residual_slope = self._linear_slope(recent_residuals)

        residual_diff = np.abs(np.diff(depth_residual_series))
        max_residual_step = float(np.max(residual_diff)) if len(residual_diff) > 0 else 0.0
        large_residual_diff_count = int(np.sum(residual_diff > cfg.noise_large_diff_threshold))
        # --------------------------------------------------------------
        # Maneuver / waypoint-transition guard
        #
        # In long-route missions, normal waypoint transitions or large
        # commanded depth changes can create residual trends similar to DRIFT.
        # This guard uses mission/control information, not fault_start_time.
        # --------------------------------------------------------------
        if len(target_series) > 0:
            target_series = target_series[np.isfinite(target_series)]
        else:
            target_series = np.asarray([], dtype=float)

        if len(target_series) >= 2:
            target_window = target_series[-min(cfg.recent_window, len(target_series)):]
            recent_target_change = float(abs(target_window[-1] - target_window[0]))
            recent_target_slope = abs(self._linear_slope(target_window))
        else:
            recent_target_change = 0.0
            recent_target_slope = 0.0

        cmd_vz_for_guard = abs(float(residuals.get("cmd_vz", 0.0)))
        tracking_error_for_guard = abs(float(residuals.get("tracking_error", 0.0)))

        maneuver_guard_active = (
            recent_target_change > 2.0
            or recent_target_slope > 0.15
            or cmd_vz_for_guard > 0.8
            or tracking_error_for_guard > 4.0
        )
        if len(vz_series) > 0:
            vz_series = vz_series[np.isfinite(vz_series)]
            if len(vz_series) > 0:
                recent_vz = vz_series[-min(len(vz_series), cfg.recent_window):]
                velocity_active = np.mean(np.abs(recent_vz)) > cfg.stuck_velocity_active_threshold
            else:
                velocity_active = False
        else:
            velocity_active = False

        large_jump_count = int(np.sum(depth_deltas > cfg.spike_delta_threshold))
        max_delta = float(np.max(depth_deltas)) if len(depth_deltas) > 0 else 0.0

        # ------------------------------------------------------------------
        # 1. STUCK
        # Depth reading is almost constant while the vehicle is still moving.
        #
        # Important:
        # In a DRIFT case, the residual can grow quickly while the measured
        # depth may still look nearly constant in a short window. Therefore,
        # STUCK must be blocked when a clear residual trend is already forming.
        # ------------------------------------------------------------------
        drift_like_trend = (
            len(depth_residual_series) >= cfg.drift_min_samples
            and abs(recent_residual_slope) > cfg.drift_slope_threshold
            and recent_residual_range > cfg.drift_min_recent_range
        )

        cmd_vz = float(residuals.get("cmd_vz", 0.0))
        tracking_error = abs(float(residuals.get("tracking_error", 0.0)))

        command_active = abs(cmd_vz) > 0.25
        tracking_demand_exists = tracking_error > 2.0

        stuck_like = (
                recent_depth_change < cfg.stuck_depth_change_threshold
                and (command_active or tracking_demand_exists)
                and not drift_like_trend
        )

        if stuck_like:
            return DiagnosisResult(
                fault_id=3,
                fault_name=FAULT_NAMES[3],
                reason=(
                    "Stuck: depth reading is almost constant while the controller "
                    "still has vertical command or tracking demand."
                ),
                confidence="High",
                source="rule",
            )

        # ------------------------------------------------------------------
        # 2. BIAS / DRIFT
        #
        # Long-route missions can contain long descending / ascending segments
        # and waypoint transitions. A DRIFT fault can initially look like BIAS,
        # because the residual first becomes non-zero and then keeps changing.
        # Therefore, we first estimate whether the current residual pattern is
        # trend-like. If a clear trend exists, DRIFT should have priority over
        # BIAS confirmation.
        # ------------------------------------------------------------------
        early_drift_like = (
            len(depth_residual_series) >= cfg.drift_min_samples
            and latest_abs_residual > cfg.bias_residual_threshold
            and abs(recent_residual_slope) > cfg.bias_max_slope_threshold
            and recent_residual_range > 1.5
            and max_residual_step <= cfg.bias_step_threshold
        )

        gradual_drift_like = (
            len(depth_residual_series) >= cfg.drift_min_samples
            and latest_abs_residual > cfg.drift_recent_residual_threshold
            and abs(recent_residual_slope) > cfg.drift_slope_threshold
            and recent_residual_range > cfg.drift_min_recent_range
            and residual_range > cfg.drift_residual_range_threshold
            and max_residual_step <= cfg.bias_step_threshold
        )

        ai_drift_supported = (
            ai_pred == 2
            and latest_abs_residual > cfg.drift_recent_residual_threshold
            and recent_abs_mean > cfg.bias_recent_residual_threshold
            and recent_residual_range > cfg.drift_min_recent_range
            and residual_range > cfg.drift_residual_range_threshold
            and max_residual_step <= cfg.bias_step_threshold
        )

        # DRIFT can be confirmed by a clear residual trend.
        # AI support is helpful but not mandatory, because true DRIFT may cause
        # the controller state to deviate so much that the AI classifier becomes unreliable.
        # False alarms should be handled by the debounce/guard logic in main.py.
        drift_confirmed = (
            gradual_drift_like
            or ai_drift_supported
        )

        if drift_confirmed:
            return DiagnosisResult(
                fault_id=2,
                fault_name=FAULT_NAMES[2],
                reason=(
                    "Drift: AI predicts DRIFT and the depth residual shows "
                    "a continuous increasing or decreasing trend."
                ),
                confidence="High",
                source="rule",
            )

        # A weaker trend should not be immediately confirmed as DRIFT, but it
        # must block BIAS locking so that DRIFT has more time to develop.
        # Only block BIAS locking when AI also suggests DRIFT.
        # Otherwise normal route tracking or BIAS transient may be incorrectly
        # treated as drift-like.
        drift_like_before_bias = (
                ai_pred == 2
                and (
                        early_drift_like
                        or gradual_drift_like
                        or (
                                len(depth_residual_series) >= cfg.drift_min_samples
                                and latest_abs_residual > cfg.bias_residual_threshold
                                and (
                                        abs(recent_residual_slope) > 0.08
                                        or recent_residual_range > 1.5
                                )
                                and max_residual_step <= cfg.bias_step_threshold
                        )
                )
        )

        bias_step_like = (
            latest_abs_residual > cfg.bias_residual_threshold
            and max_residual_step > cfg.bias_step_threshold
            and recent_abs_mean > cfg.bias_recent_residual_threshold
        )

        bias_stable_like = (
            latest_abs_residual > cfg.bias_residual_threshold
            and recent_abs_mean > cfg.bias_recent_residual_threshold
            and abs(recent_residual_slope) < cfg.bias_max_slope_threshold
            and recent_residual_range < cfg.drift_min_recent_range
            and residual_std < cfg.noise_std_threshold * 2.5
        )

        # AI=1 is allowed to support BIAS only when the residual is clearly
        # non-zero and the residual trend is not drift-like. This prevents
        # AI-supported BIAS from locking too early in a DRIFT case.
        bias_ai_supported = (
            ai_pred == 1
            and latest_abs_residual > cfg.bias_residual_threshold
            and recent_abs_mean > cfg.bias_recent_residual_threshold
        )

        if (
                (bias_step_like or bias_stable_like or bias_ai_supported)
                and not drift_like_before_bias
        ):
            return DiagnosisResult(
                fault_id=1,
                fault_name=FAULT_NAMES[1],
                reason=(
                    "Bias: depth residual has a persistent non-zero offset "
                    "without a gradual drift trend."
                ),
                confidence="High",
                source="rule",
            )

        # ------------------------------------------------------------------
        # 4. SPIKE
        # Spike is an isolated one-step jump. Use max_delta instead of only
        # the last delta because the jump may already be inside the window.
        # ------------------------------------------------------------------
        # SPIKE should be isolated, not repeated high-frequency fluctuation.
        spike_like_count = int(np.sum(depth_deltas > cfg.spike_delta_threshold))
        spike_ratio = spike_like_count / max(len(depth_deltas), 1)

        if (
                max_delta > cfg.spike_delta_threshold
                and spike_like_count <= cfg.spike_max_jump_count
                and spike_ratio < 0.12
                and residual_std < cfg.noise_std_threshold * 3.0
        ):
            return DiagnosisResult(
                fault_id=4,
                fault_name=FAULT_NAMES[4],
                reason="Spike: isolated one-step depth jump exceeds threshold.",
                confidence="High",
                source="rule",
            )

        # ------------------------------------------------------------------
        # 5. NOISE_INCREASE
        # Noise is repeated high-frequency fluctuation, not isolated spikes.
        # ------------------------------------------------------------------
        residual_diff = np.abs(np.diff(depth_residual_series))
        high_freq_count = int(np.sum(np.abs(depth_residual_series - np.mean(depth_residual_series)) > 2.5))
        high_freq_ratio = high_freq_count / max(len(depth_residual_series), 1)

        if (
            residual_std > cfg.noise_std_threshold
            and high_freq_ratio > 0.35
            and spike_ratio < 0.12
            and large_residual_diff_count >= cfg.noise_large_diff_count_threshold
        ):
            return DiagnosisResult(
                fault_id=5,
                fault_name=FAULT_NAMES[5],
                reason="Noise increase: repeated high-frequency depth residual fluctuations are detected.",
                confidence="High",
                source="rule",
            )

        # AI can still help identify transient spike when the exact rule timing
        # misses the isolated jump.
        if ai_pred == 4 and max_delta > cfg.spike_delta_threshold:
            return DiagnosisResult(
                fault_id=4,
                fault_name=FAULT_NAMES[4],
                reason="Spike: isolated abnormal jump detected with AI support.",
                confidence="Medium",
                source="rule",
            )

        return DiagnosisResult(
            fault_id=0,
            fault_name=FAULT_NAMES[0],
            reason="No depth sensor rule exceeded threshold.",
            confidence="Low",
            source="rule",
        )

    @staticmethod
    def _linear_slope(values):
        values = np.asarray(values, dtype=float)

        if values.size < 2:
            return 0.0

        x = np.arange(values.size, dtype=float)
        slope, _ = np.polyfit(x, values, 1)

        return float(slope)

    @staticmethod
    def _extract_series(history, key: str):
        values = []

        for item in history:
            value = None

            if key in item:
                value = item.get(key)

            elif key == "actual_vz":
                value = item.get("thruster", {}).get("actual_vz")
            elif key == "target_z":
                value = item.get("target_z")
                if value is None:
                    value = item.get("target_depth")
                if value is None:
                    value = item.get("target")
            elif key in ["depth_residual", "tracking_error", "velocity_residual", "current_residual"]:
                value = item.get("residuals", {}).get(key)

            if value is not None:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    pass

        return np.asarray(values, dtype=float)
