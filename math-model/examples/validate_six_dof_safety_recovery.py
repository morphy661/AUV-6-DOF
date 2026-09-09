"""Paired engineering validation for observable safety and vertical-fault recovery."""

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
)
from environment.six_dof_dynamics import SixDOFState
from environment.six_dof_simulator import SixDOFSimulator
from ftc.ai_evidence_gate import AIEvidenceGateConfig, AIControlEvidenceGate
from ftc.safety_supervisor import FTCSafetySupervisor
from presentation.six_dof_model_bridge import SixDOFModelBridge
from sensors.six_dof_sensor_suite import SixDOFSensorSuite
from simple_control.six_dof_controller import PoseTarget

sys.path.insert(0, str(MODEL_ROOT))
from model_six_dof_multitask import AUVSixDOFMultiTaskDetector


def mission_target(depth_m, vertical_speed_mps=0.10):
    def target(time_s, _state):
        speed = 1.35
        turn_rate = 0.10
        angle = float(np.clip((time_s - 12.0) * turn_rate, 0.0, 0.8))
        if time_s <= 12.0:
            north, east = speed * time_s, 0.0
        else:
            north = speed * (12.0 + np.sin(angle) / turn_rate)
            east = speed * (1.0 - np.cos(angle)) / turn_rate
            if time_s > 20.0:
                north += speed * (time_s - 20.0) * np.cos(0.8)
                east += speed * (time_s - 20.0) * np.sin(0.8)
        return PoseTarget(
            position_ned=np.array([
                north,
                east,
                depth_m + vertical_speed_mps * time_s,
            ]),
            euler_rpy=np.array([0.0, 0.0, angle]),
            guidance_context_id=1,
            linear_velocity_ned=np.array([
                speed * np.cos(angle),
                speed * np.sin(angle),
                vertical_speed_mps,
            ]),
        )

    return target


def disturbance(time_s, _state, transient=False):
    value = np.array([
        0.9 * np.sin(0.55 * time_s),
        0.6 * np.cos(0.37 * time_s),
        0.35 * np.sin(0.29 * time_s),
        0.02 * np.sin(0.50 * time_s),
        0.02 * np.cos(0.43 * time_s),
        0.04 * np.sin(0.31 * time_s),
    ])
    if transient and 8.0 <= time_s < 14.0:
        value += np.array([35.0, 15.0, 3.0, 0.0, 0.0, 0.5])
    return value


def run_case(case, recovery, mode, bridge, dt, duration):
    supervisor = FTCSafetySupervisor()
    fault = None
    if case.get("thruster"):
        fault = SingleThrusterFault(
            case["thruster"],
            SixDOFThrusterFaultMode.NO_OUTPUT,
            start_time=10.0,
            thrust_efficiency=0.0,
        )
    gate = None
    observer = None
    if mode == "fusion":
        observer = bridge
        gate = AIControlEvidenceGate(
            AIEvidenceGateConfig(
                minimum_mode_probability=(
                    bridge.decision.temporal.config.enter_fault_probability
                ),
                maximum_result_age_s=1.2 * bridge.inference_stride * dt,
            ),
            supervisor.config,
        )
    simulator = SixDOFSimulator(
        sensor_suite=SixDOFSensorSuite(seed=case["seed"]),
        fault=fault,
        ftc_supervisor=supervisor,
        diagnostic_observer=observer,
        ai_ftc_fusion=gate,
        vertical_fault_recovery=recovery,
    )
    simulator.reset(SixDOFState(
        position_ned=np.array([0.0, 0.0, case["depth_m"]])
    ))
    target = mission_target(
        case["depth_m"], case.get("vertical_speed_mps", 0.10)
    )
    return simulator.run(
        duration,
        dt,
        target,
        lambda time_s, state: disturbance(
            time_s, state, transient=case.get("transient", False)
        ),
    )


def metrics(logs):
    position = np.array([row["position_ned"] for row in logs])
    target = np.array([row["target_position_ned"] for row in logs])
    mission = np.array([row["mission_target_position_ned"] for row in logs])
    euler = np.array([row["euler_rpy"] for row in logs])
    tilt = np.degrees(np.arccos(np.clip(
        np.cos(euler[:, 0]) * np.cos(euler[:, 1]), -1.0, 1.0
    )))
    hold_mask = np.array([
        row["ftc_guidance_mode_applied"] == "vertical_thruster_hold"
        for row in logs
    ])
    updated = [
        row["maintenance_diagnosis"]
        for row in logs
        if row.get("maintenance_diagnosis", {}).get("updated")
    ]
    raw_fault_updates = [
        diagnosis for diagnosis in updated
        if int(np.argmax(diagnosis["raw_mode_probabilities"])) != 0
    ]
    no_output = np.array([row["ftc_no_output_scores"] for row in logs])
    evidence_mask = no_output >= 0.80
    dt = float(logs[1]["time"] - logs[0]["time"])

    def longest_true_run(values):
        longest = current = 0
        for value in values:
            current = current + 1 if value else 0
            longest = max(longest, current)
        return float(longest * dt)
    return {
        "actions": dict(Counter(row["ftc_action"] for row in logs)),
        "targeted_thrusters": sorted({
            row["ftc_targeted_thruster_name"]
            for row in logs
            if row.get("ftc_targeted_thruster_name")
        }),
        "first_targeted_reallocation_s": next((
            row["time"] for row in logs
            if row.get("ftc_targeted_thruster_name")
        ), None),
        "first_vertical_hold_s": next((
            row["time"] for row in logs
            if row["ftc_guidance_mode_applied"] == "vertical_thruster_hold"
        ), None),
        "vertical_hold_steps": int(np.sum(hold_mask)),
        "max_tilt_deg": float(np.max(tilt)),
        "final_tilt_deg": float(tilt[-1]),
        "max_abs_rpy_deg": np.max(np.abs(np.degrees(euler)), axis=0).tolist(),
        "max_effective_target_error_m": float(np.max(np.linalg.norm(
            position - target, axis=1
        ))),
        "max_mission_target_error_m": float(np.max(np.linalg.norm(
            position - mission, axis=1
        ))),
        "max_hold_horizontal_error_m": (
            None if not np.any(hold_mask)
            else float(np.max(np.linalg.norm(
                (position - target)[hold_mask, :2], axis=1
            )))
        ),
        "max_hold_depth_error_m": (
            None if not np.any(hold_mask)
            else float(np.max(np.abs((position - target)[hold_mask, 2])))
        ),
        "final_depth_m": float(position[-1, 2]),
        "tracking_evidence_available_steps": int(sum(
            bool(row["ftc_tracking_error_ratio_available"]) for row in logs
        )),
        "position_safety_active_steps": int(sum(
            bool(row["ftc_position_tracking_safety_active"]) for row in logs
        )),
        "max_tracking_error_ratio": float(max(
            row["ftc_tracking_error_ratio"] for row in logs
        )),
        "safe_hold_or_abort_steps": int(sum(
            row["ftc_action"] == "safe_hold_or_abort" for row in logs
        )),
        "ai_accepted_steps": int(sum(
            bool(row.get("ai_ftc_fusion", {}).get("accepted")) for row in logs
        )),
        "model_updates": len(updated),
        "raw_model_fault_updates": len(raw_fault_updates),
        "max_raw_model_fault_probability": (
            None if not updated
            else float(max(
                1.0 - diagnosis["raw_mode_probabilities"][0]
                for diagnosis in updated
            ))
        ),
        "max_no_output_score_by_thruster": dict(zip(
            ("H1", "H2", "H3", "H4", "V1", "V2"),
            np.max(no_output, axis=0).tolist(),
        )),
        "no_output_evidence_steps_by_thruster": dict(zip(
            ("H1", "H2", "H3", "H4", "V1", "V2"),
            np.sum(evidence_mask, axis=0).astype(int).tolist(),
        )),
        "longest_continuous_no_output_evidence_s_by_thruster": dict(zip(
            ("H1", "H2", "H3", "H4", "V1", "V2"),
            [longest_true_run(evidence_mask[:, index]) for index in range(6)],
        )),
    }


def trajectories_identical(first, second):
    return all(
        np.array_equal(a[key], b[key])
        for a, b in zip(first, second)
        for key in (
            "position_ned",
            "body_velocity",
            "commanded_thruster_forces",
            "achieved_wrench_body",
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dt = 0.05
    bridge = SixDOFModelBridge(
        DEFAULT_CHECKPOINT,
        DEFAULT_TEMPORAL_CONFIG,
        AUVSixDOFMultiTaskDetector,
        device=args.device,
    )
    cases = [
        {"name": "normal_50m", "depth_m": 50.0, "seed": 20260990},
        {"name": "normal_200m", "depth_m": 200.0, "seed": 20260991},
        {"name": "normal_280m", "depth_m": 280.0, "seed": 20260992},
        {
            "name": "normal_transient_200m",
            "depth_m": 200.0,
            "seed": 20260993,
            "transient": True,
        },
        {
            "name": "V1_no_output_200m",
            "depth_m": 200.0,
            "seed": 20260994,
            "thruster": "V1",
        },
        {
            "name": "V1_no_output_transient_200m",
            "depth_m": 200.0,
            "seed": 20260995,
            "thruster": "V1",
            "transient": True,
        },
        {
            "name": "V2_no_output_200m",
            "depth_m": 200.0,
            "seed": 20260996,
            "thruster": "V2",
            "vertical_speed_mps": -0.10,
        },
    ]
    records = []
    raw_logs = {}
    for case in cases:
        print(f"Running {case['name']}", flush=True)
        legacy = run_case(
            case, recovery=False, mode="rule", bridge=bridge,
            dt=dt, duration=args.duration,
        )
        recovery = run_case(
            case, recovery=True, mode="rule", bridge=bridge,
            dt=dt, duration=args.duration,
        )
        raw_logs[(case["name"], "recovery_rule")] = recovery
        record = {
            "case": case,
            "normal_paired_trajectory_identical": (
                trajectories_identical(legacy, recovery)
                if not case.get("thruster") else None
            ),
            "metrics": {
                "legacy_rule": metrics(legacy),
                "recovery_rule": metrics(recovery),
            },
        }
        if case["name"] == "V1_no_output_200m":
            fusion = run_case(
                case, recovery=True, mode="fusion", bridge=bridge,
                dt=dt, duration=args.duration,
            )
            record["metrics"]["recovery_fusion"] = metrics(fusion)
            record["fusion_matches_recovery_rule_trajectory"] = (
                trajectories_identical(recovery, fusion)
            )
            record["fusion_matches_recovery_rule_actions"] = all(
                first["ftc_action"] == second["ftc_action"]
                for first, second in zip(recovery, fusion)
            )
            raw_logs[(case["name"], "recovery_fusion")] = fusion
        records.append(record)
        print(json.dumps({
            "case": case["name"],
            "legacy_max_tilt_deg": record["metrics"]["legacy_rule"]["max_tilt_deg"],
            "recovery_max_tilt_deg": record["metrics"]["recovery_rule"]["max_tilt_deg"],
            "first_hold_s": record["metrics"]["recovery_rule"]["first_vertical_hold_s"],
        }), flush=True)

    normal_records = [
        item for item in records if not item["case"].get("thruster")
    ]
    v1_records = [
        item for item in records if item["case"].get("thruster") == "V1"
    ]
    v2_records = [
        item for item in records if item["case"].get("thruster") == "V2"
    ]
    fusion_record = next(
        item for item in records if "recovery_fusion" in item["metrics"]
    )
    acceptance = {
        "normal_trajectories_unchanged": all(
            item["normal_paired_trajectory_identical"]
            for item in normal_records
        ),
        "normal_cases_no_intervention": all(
            not item["metrics"]["recovery_rule"]["targeted_thrusters"]
            and item["metrics"]["recovery_rule"]["safe_hold_or_abort_steps"] == 0
            for item in normal_records
        ),
        "tracking_evidence_available_every_step": all(
            value["tracking_evidence_available_steps"]
            == int(np.ceil(args.duration / dt))
            for item in records
            for value in item["metrics"].values()
        ),
        "v1_hold_entered": all(
            item["metrics"]["recovery_rule"]["first_vertical_hold_s"] is not None
            for item in v1_records
        ),
        "v1_recovery_max_tilt_below_10_deg": all(
            item["metrics"]["recovery_rule"]["max_tilt_deg"] < 10.0
            for item in v1_records
        ),
        "v1_recovery_reduces_peak_tilt_by_80_percent": all(
            item["metrics"]["recovery_rule"]["max_tilt_deg"]
            < 0.2 * item["metrics"]["legacy_rule"]["max_tilt_deg"]
            for item in v1_records
        ),
        "v1_hold_errors_within_configured_limits": all(
            item["metrics"]["recovery_rule"]["max_hold_horizontal_error_m"] < 8.0
            and item["metrics"]["recovery_rule"]["max_hold_depth_error_m"] < 3.0
            for item in v1_records
        ),
        "v1_no_spurious_safe_abort": all(
            item["metrics"]["recovery_rule"]["safe_hold_or_abort_steps"] == 0
            for item in v1_records
        ),
        "v2_low_excitation_case_remains_attitude_safe": all(
            item["metrics"]["recovery_rule"]["max_tilt_deg"] < 10.0
            for item in v2_records
        ),
        "fusion_model_executed_online": (
            fusion_record["metrics"]["recovery_fusion"]["model_updates"] > 0
        ),
        "fusion_preserves_direct_esc_recovery": bool(
            fusion_record["fusion_matches_recovery_rule_trajectory"]
            and fusion_record["fusion_matches_recovery_rule_actions"]
        ),
    }
    report = {
        "validation_type": "paired corrective engineering validation",
        "checkpoint_sha256": _sha256(DEFAULT_CHECKPOINT),
        "temporal_policy_sha256": _sha256(DEFAULT_TEMPORAL_CONFIG),
        "ai_gate_config": asdict(AIEvidenceGateConfig()),
        "dt_s": dt,
        "duration_s": args.duration,
        "cases": records,
        "acceptance": acceptance,
        "all_acceptance_checks_passed": all(acceptance.values()),
        "scope": (
            "Simulator engineering validation only; not a blind model-accuracy "
            "claim and not evidence of real-sea certification."
        ),
    }
    output = args.output_dir / "safety_recovery_validation_summary.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(acceptance, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
