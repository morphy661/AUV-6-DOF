"""Paired horizontal-thruster fault sweep for the six-DOF online FTC path."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "examples"):
    sys.path.insert(0, str(root))

from actuators.six_dof_thruster_faults import (
    SingleThrusterFault,
    SixDOFThrusterFaultMode,
)
from demo_six_dof_unified_diagnostics import (
    DEFAULT_CHECKPOINT,
    DEFAULT_TEMPORAL_CONFIG,
    MODEL_ROOT,
    _sha256,
    summarize_online_diagnosis,
)
from environment.six_dof_dynamics import SixDOFState
from environment.six_dof_simulator import SixDOFSimulator
from ftc.ai_evidence_gate import AIControlEvidenceGate, AIEvidenceGateConfig
from ftc.safety_supervisor import FTCSafetySupervisor
from presentation.six_dof_model_bridge import SixDOFModelBridge
from sensors.six_dof_sensor_suite import SixDOFSensorSuite
from simple_control.six_dof_controller import PoseTarget

sys.path.insert(0, str(MODEL_ROOT))
from model_six_dof_multitask import AUVSixDOFMultiTaskDetector


THRUSTERS = ("H1", "H2", "H3", "H4")
EFFICIENCIES = (0.20, 0.35, 0.50, 0.70)


def mission_target(depth_m, profile):
    def target(time_s, _state):
        speed = 1.35
        if profile == "straight":
            x = speed * time_s
            y = 0.0
            yaw = 0.0
        else:
            turn_rate = 0.10
            yaw = float(np.clip((time_s - 12.0) * turn_rate, 0.0, 0.8))
            if time_s <= 12.0:
                x = speed * time_s
                y = 0.0
            else:
                x = speed * (12.0 + np.sin(yaw) / turn_rate)
                y = speed * (1.0 - np.cos(yaw)) / turn_rate
        return PoseTarget(
            np.array([x, y, depth_m + 0.1 * time_s]),
            np.array([0.0, 0.0, yaw]),
            guidance_context_id=1,
        )

    return target


def disturbance_provider(enabled):
    def disturbance(time_s, _state):
        value = np.array([
            0.9 * np.sin(0.55 * time_s),
            0.6 * np.cos(0.37 * time_s),
            0.35 * np.sin(0.29 * time_s),
            0.02 * np.sin(0.5 * time_s),
            0.02 * np.cos(0.43 * time_s),
            0.04 * np.sin(0.31 * time_s),
        ])
        if enabled and 8.0 <= time_s < 14.0:
            value += np.array([35.0, 15.0, 3.0, 0.0, 0.0, 0.5])
        return value

    return disturbance


def run_case(case, mode, bridge, dt_s, duration_s):
    supervisor = FTCSafetySupervisor()
    fault = None
    if case.get("thruster"):
        fault = SingleThrusterFault(
            case["thruster"],
            SixDOFThrusterFaultMode.THRUST_LOSS,
            start_time=10.0,
            thrust_efficiency=case["efficiency"],
        )
    gate = AIControlEvidenceGate(
        AIEvidenceGateConfig(
            minimum_mode_probability=(
                bridge.decision.temporal.config.enter_fault_probability
            ),
            maximum_result_age_s=1.2 * bridge.inference_stride * dt_s,
        ),
        supervisor.config,
    )
    simulator = SixDOFSimulator(
        sensor_suite=SixDOFSensorSuite(seed=case["seed"]),
        fault=fault,
        ftc_supervisor=supervisor,
        diagnostic_observer=bridge,
        ai_ftc_fusion=gate if mode == "fusion" else None,
    )
    simulator.reset(SixDOFState(
        position_ned=np.array([0.0, 0.0, case["depth_m"]])
    ))
    return simulator.run(
        duration_s,
        dt_s,
        mission_target(case["depth_m"], case["profile"]),
        disturbance_provider(case.get("disturbance", False)),
    )


def _maximum(logs, key, default=0.0):
    values = [row.get(key) for row in logs if row.get(key) is not None]
    return float(max(values)) if values else float(default)


def metrics(logs, dt_s):
    position = np.asarray([row["position_ned"] for row in logs])
    target = np.asarray([row["target_position_ned"] for row in logs])
    euler_deg = np.degrees(np.asarray([row["euler_rpy"] for row in logs]))
    admitted = [
        row["time"] for row in logs
        if row.get("ai_ftc_fusion", {}).get("accepted", False)
    ]
    online = summarize_online_diagnosis(logs, dt_s)
    return {
        "actions": dict(Counter(row["ftc_action"] for row in logs)),
        "guidance_modes": dict(Counter(
            row["ftc_guidance_mode_applied"] for row in logs
        )),
        "isolated_thrusters": sorted({
            row["ftc_targeted_thruster_name"] for row in logs
            if row.get("ftc_targeted_thruster_name")
        }),
        "first_ai_admission_s": admitted[0] if admitted else None,
        "ai_admitted_steps": len(admitted),
        "first_horizontal_hold_s": next((
            row["time"] for row in logs
            if row["ftc_guidance_mode_applied"] == "horizontal_degraded_hold"
        ), None),
        "horizontal_hold_steps": int(sum(
            row["ftc_guidance_mode_applied"] == "horizontal_degraded_hold"
            for row in logs
        )),
        "max_abs_rpy_deg": np.max(np.abs(euler_deg), axis=0).tolist(),
        "max_yaw_tracking_error_deg": _maximum(
            logs, "ftc_yaw_tracking_error_deg"
        ),
        "max_yaw_rate_tracking_error_deg_s": _maximum(
            logs, "ftc_yaw_rate_tracking_error_deg_s"
        ),
        "max_cross_track_error_m": _maximum(
            logs, "ftc_cross_track_error_m"
        ),
        "max_horizontal_motion_error_ratio": _maximum(
            logs, "ftc_horizontal_motion_error_ratio"
        ),
        "max_control_saturation_ratio": _maximum(
            logs, "ftc_control_saturation_ratio"
        ),
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(
            (position - target) ** 2, axis=1
        )))),
        "max_position_error_m": float(np.max(np.linalg.norm(
            position - target, axis=1
        ))),
        "safe_abort_steps": int(sum(
            row["ftc_action"] == "safe_hold_or_abort" for row in logs
        )),
        "online": online,
    }


def case_record(case, shadow_logs, fusion_logs, dt_s):
    shadow = metrics(shadow_logs, dt_s)
    fusion = metrics(fusion_logs, dt_s)
    supervisor_config = FTCSafetySupervisor().config
    standard_hazard = bool(
        shadow["max_horizontal_motion_error_ratio"] >= 1.0
        and shadow["max_control_saturation_ratio"] >= 0.80
    )
    severe_horizontal_hazard = bool(
        shadow["max_horizontal_motion_error_ratio"] >= 1.5
        and shadow["max_control_saturation_ratio"] >= 0.55
    )
    hazard = standard_hazard or severe_horizontal_hazard
    intervention = fusion["first_horizontal_hold_s"] is not None
    yaw_reduction = (
        0.0
        if shadow["max_yaw_tracking_error_deg"] <= 1e-9
        else 1.0 - (
            fusion["max_yaw_tracking_error_deg"]
            / shadow["max_yaw_tracking_error_deg"]
        )
    )
    peak_yaw_within_limit = bool(
        fusion["max_yaw_tracking_error_deg"]
        <= supervisor_config.horizontal_yaw_error_limit_deg
    )
    position_not_worse = bool(
        fusion["max_position_error_m"]
        <= shadow["max_position_error_m"] + 0.05
    )
    materially_mitigated = bool(yaw_reduction >= 0.25)
    consequence_controlled = bool(
        intervention
        and fusion["safe_abort_steps"] == 0
        and position_not_worse
        and (peak_yaw_within_limit or materially_mitigated)
    )
    if hazard and intervention:
        if consequence_controlled and materially_mitigated:
            outcome = "materially_mitigated"
        elif consequence_controlled:
            outcome = "safe_bound_maintained"
        else:
            outcome = "intervened_no_gain"
    elif hazard:
        outcome = "missed_actionable"
    elif intervention:
        outcome = "unnecessary_intervention"
    else:
        outcome = "record_only_acceptable"
    return {
        "case": case,
        "shadow": shadow,
        "fusion": fusion,
        "actionable_horizontal_consequence": hazard,
        "standard_control_corroboration": standard_hazard,
        "severe_horizontal_corroboration": severe_horizontal_hazard,
        "horizontal_hold_entered": intervention,
        "peak_yaw_tracking_error_reduction_fraction": float(yaw_reduction),
        "configured_yaw_error_limit_deg": float(
            supervisor_config.horizontal_yaw_error_limit_deg
        ),
        "peak_yaw_within_configured_limit": peak_yaw_within_limit,
        "position_not_worse_than_shadow": position_not_worse,
        "materially_mitigated": materially_mitigated,
        "horizontal_consequence_controlled": consequence_controlled,
        "outcome": outcome,
        "no_ai_location_isolation": not fusion["isolated_thrusters"],
    }


def build_cases():
    cases = []
    seed = 20261000
    for thruster in THRUSTERS:
        for efficiency in EFFICIENCIES:
            cases.append({
                "name": f"{thruster}_eff{int(100 * efficiency):02d}_turn_200m",
                "thruster": thruster,
                "efficiency": efficiency,
                "depth_m": 200.0,
                "profile": "turn",
                "disturbance": False,
                "seed": seed,
            })
            seed += 1
    for thruster in THRUSTERS:
        for profile, disturbance in (("straight", False), ("turn", True)):
            label = "disturbance" if disturbance else profile
            cases.append({
                "name": f"{thruster}_eff35_{label}_200m",
                "thruster": thruster,
                "efficiency": 0.35,
                "depth_m": 200.0,
                "profile": profile,
                "disturbance": disturbance,
                "seed": seed,
            })
            seed += 1
    for thruster in THRUSTERS:
        for depth_m in (50.0, 280.0):
            cases.append({
                "name": f"{thruster}_eff35_turn_{int(depth_m)}m",
                "thruster": thruster,
                "efficiency": 0.35,
                "depth_m": depth_m,
                "profile": "turn",
                "disturbance": False,
                "seed": seed,
            })
            seed += 1
    normal_cases = []
    normal_seed = seed
    for depth_m in (50.0, 200.0, 280.0):
        for repeat in range(3):
            normal_cases.append({
                "name": (
                    f"normal_turn_{int(depth_m)}m_seed{normal_seed}"
                ),
                "depth_m": depth_m,
                "profile": "turn",
                "disturbance": False,
                "seed": normal_seed,
            })
            normal_seed += 1
    for repeat in range(2):
        normal_cases.append({
            "name": f"normal_straight_200m_seed{normal_seed}",
            "depth_m": 200.0,
            "profile": "straight",
            "disturbance": False,
            "seed": normal_seed,
        })
        normal_seed += 1
    for repeat in range(4):
        normal_cases.append({
            "name": f"normal_disturbance_200m_seed{normal_seed}",
            "depth_m": 200.0,
            "profile": "turn",
            "disturbance": True,
            "seed": normal_seed,
        })
        normal_seed += 1
    return normal_cases, cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=18.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dt_s = 0.05
    bridge = SixDOFModelBridge(
        DEFAULT_CHECKPOINT,
        DEFAULT_TEMPORAL_CONFIG,
        AUVSixDOFMultiTaskDetector,
        device=args.device,
    )
    normal_cases, fault_cases = build_cases()
    normal_records = []
    for case in normal_cases:
        print(f"Running {case['name']}", flush=True)
        logs = run_case(case, "fusion", bridge, dt_s, args.duration)
        record = {"case": case, "fusion": metrics(logs, dt_s)}
        normal_records.append(record)
        print(json.dumps({
            "case": case["name"],
            "hold": record["fusion"]["horizontal_hold_steps"],
            "ai": record["fusion"]["ai_admitted_steps"],
        }), flush=True)

    fault_records = []
    for case in fault_cases:
        print(f"Running {case['name']}", flush=True)
        shadow = run_case(case, "shadow", bridge, dt_s, args.duration)
        fusion = run_case(case, "fusion", bridge, dt_s, args.duration)
        record = case_record(case, shadow, fusion, dt_s)
        fault_records.append(record)
        print(json.dumps({
            "case": case["name"],
            "hazard": record["actionable_horizontal_consequence"],
            "hold": record["horizontal_hold_entered"],
            "outcome": record["outcome"],
            "yaw_reduction": round(
                record["peak_yaw_tracking_error_reduction_fraction"], 3
            ),
        }), flush=True)

    actionable = [
        row for row in fault_records
        if row["actionable_horizontal_consequence"]
    ]
    controlled = [
        row for row in actionable
        if row["horizontal_consequence_controlled"]
    ]
    materially_mitigated = [
        row for row in actionable if row["materially_mitigated"]
    ]
    unnecessary = [
        row for row in fault_records
        if row["outcome"] == "unnecessary_intervention"
    ]
    acceptance = {
        "normal_cases_without_horizontal_hold": all(
            row["fusion"]["horizontal_hold_steps"] == 0
            for row in normal_records
        ),
        "normal_cases_without_ai_control_admission": all(
            row["fusion"]["ai_admitted_steps"] == 0
            for row in normal_records
        ),
        "partial_loss_never_causes_ai_location_isolation": all(
            row["no_ai_location_isolation"] for row in fault_records
        ),
        "no_unnecessary_horizontal_intervention": not unnecessary,
        "actionable_case_control_rate_at_least_90_percent": (
            not actionable or len(controlled) / len(actionable) >= 0.90
        ),
        "controlled_cases_avoid_safe_abort": all(
            row["fusion"]["safe_abort_steps"] == 0 for row in controlled
        ),
    }
    supervisor_config = FTCSafetySupervisor().config
    report = {
        "validation_type": "paired horizontal fault parameter sweep",
        "checkpoint_sha256": _sha256(DEFAULT_CHECKPOINT),
        "temporal_policy_sha256": _sha256(DEFAULT_TEMPORAL_CONFIG),
        "supervisor_gate": asdict(AIEvidenceGateConfig()),
        "horizontal_control_thresholds": {
            "standard_motion_ratio": float(
                supervisor_config.horizontal_degraded_motion_ratio
            ),
            "standard_saturation_ratio": float(
                supervisor_config.degraded_control_saturation_ratio
            ),
            "severe_motion_ratio": float(
                supervisor_config.severe_horizontal_motion_ratio
            ),
            "severe_minimum_saturation_ratio": float(
                supervisor_config.severe_horizontal_minimum_saturation_ratio
            ),
            "configured_yaw_error_limit_deg": float(
                supervisor_config.horizontal_yaw_error_limit_deg
            ),
            "material_reduction_fraction": 0.25,
        },
        "dt_s": dt_s,
        "duration_s": args.duration,
        "normal_case_count": len(normal_records),
        "fault_case_count": len(fault_records),
        "actionable_case_count": len(actionable),
        "controlled_case_count": len(controlled),
        "materially_mitigated_case_count": len(materially_mitigated),
        "unnecessary_intervention_count": len(unnecessary),
        "acceptance": acceptance,
        "all_acceptance_checks_passed": all(acceptance.values()),
        "normal_cases": normal_records,
        "fault_cases": fault_records,
        "scope": (
            "Simulation engineering sweep. Probabilities are model scores, "
            "not calibrated real-world failure probabilities."
        ),
    }
    output = args.output_dir / "horizontal_fault_sweep_summary.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "actionable": len(actionable),
        "controlled": len(controlled),
        "materially_mitigated": len(materially_mitigated),
        "unnecessary": len(unnecessary),
        "acceptance": acceptance,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
