"""Paired rule/shadow/fusion integration checks with the frozen real model."""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "examples"):
    sys.path.insert(0, str(root))
from demo_six_dof_unified_diagnostics import (
    DEFAULT_CHECKPOINT, DEFAULT_TEMPORAL_CONFIG, MODEL_ROOT, _sha256,
    run_demo, save_online_trace, summarize_online_diagnosis,
)
from actuators.six_dof_thruster_faults import SingleThrusterFault, SixDOFThrusterFaultMode
from environment.six_dof_simulator import SixDOFSimulator
from environment.six_dof_dynamics import SixDOFState
from ftc.safety_supervisor import FTCSafetySupervisor
from ftc.ai_evidence_gate import AIControlEvidenceGate, AIEvidenceGateConfig
from sensors.six_dof_sensor_suite import SixDOFSensorSuite
from simple_control.six_dof_controller import PoseTarget
from presentation.six_dof_model_bridge import SixDOFModelBridge

sys.path.insert(0, str(MODEL_ROOT))
from model_six_dof_multitask import AUVSixDOFMultiTaskDetector


def mission_target(depth):
    def target(t, _state):
        speed, turn_rate = 1.35, .10
        angle = float(np.clip((t - 12.) * turn_rate, 0., .8))
        if t <= 12:
            x, y = speed * t, 0.
        else:
            x = speed * (12 + np.sin(angle) / turn_rate)
            y = speed * (1 - np.cos(angle)) / turn_rate
            if t > 20:
                x += speed * (t - 20) * np.cos(.8)
                y += speed * (t - 20) * np.sin(.8)
        return PoseTarget(np.array([x, y, depth + .1*t]), np.array([0., 0., angle]),
                          guidance_context_id=1)
    return target


def run_case(case, mode, bridge, dt, duration):
    if case.get("demo"):
        logs, _, _, _ = run_demo(duration, dt, case["seed"], injection_mode="random",
                                 model_bridge=None if mode == "rule" else bridge,
                                 diagnosis_mode="fusion" if mode == "fusion" else "shadow")
        return logs
    supervisor = FTCSafetySupervisor()
    fault = None
    if case.get("thruster"):
        fault = SingleThrusterFault(
            case["thruster"], SixDOFThrusterFaultMode(case["fault_mode"]),
            start_time=10., thrust_efficiency=case.get("efficiency", .35),
        )
    gate = AIControlEvidenceGate(
        AIEvidenceGateConfig(
            minimum_mode_probability=bridge.decision.temporal.config.enter_fault_probability,
            maximum_result_age_s=1.2 * bridge.inference_stride * dt), supervisor.config)
    simulator = SixDOFSimulator(
        sensor_suite=SixDOFSensorSuite(seed=case["seed"]), fault=fault,
        ftc_supervisor=supervisor, diagnostic_observer=None if mode == "rule" else bridge,
        ai_ftc_fusion=gate if mode == "fusion" else None,
    )
    simulator.reset(SixDOFState(position_ned=np.array([0., 0., case["depth"]])))

    def disturbance(t, _state):
        value = np.array([.9*np.sin(.55*t), .6*np.cos(.37*t), .35*np.sin(.29*t),
                          .02*np.sin(.5*t), .02*np.cos(.43*t), .04*np.sin(.31*t)])
        if case.get("disturbance") and 8 <= t < 14:
            value += np.array([35., 15., 3., 0., 0., .5])
        return value
    return simulator.run(duration, dt, mission_target(case["depth"]), disturbance)


def metrics(logs, dt):
    euler = np.rad2deg(np.array([log["euler_rpy"] for log in logs]))
    errors = np.array([log["position_ned"] - log["target_position_ned"] for log in logs])
    accepted = [log["time"] for log in logs if log.get("ai_ftc_fusion", {}).get("accepted")]
    raw = [log["maintenance_diagnosis"] for log in logs
           if log.get("maintenance_diagnosis", {}).get("updated")]
    return {
        "actions": dict(Counter(log["ftc_action"] for log in logs)),
        "guidance_modes": dict(Counter(
            log["ftc_guidance_mode_applied"] for log in logs
        )),
        "isolated_thrusters": sorted({log["ftc_targeted_thruster_name"] for log in logs
                                      if log["ftc_targeted_thruster_name"]}),
        "first_intervention_s": next((log["time"] for log in logs if log["ftc_intervention_requested"]), None),
        "first_ai_admission_s": accepted[0] if accepted else None,
        "first_horizontal_hold_s": next((
            log["time"] for log in logs
            if log["ftc_guidance_mode_applied"] == "horizontal_degraded_hold"
        ), None),
        "ai_admitted_steps": len(accepted),
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
        "max_position_error_m": float(np.max(np.linalg.norm(errors, axis=1))),
        "max_abs_roll_pitch_yaw_deg": np.max(np.abs(euler), axis=0).tolist(),
        "max_tracking_error_ratio": max((log["ftc_tracking_error_ratio"] for log in logs
                                          if log.get("ftc_tracking_error_ratio_available")), default=None),
        "max_horizontal_motion_error_ratio": max((
            log.get("ftc_horizontal_motion_error_ratio", 0.0)
            for log in logs
        ), default=0.0),
        "max_yaw_tracking_error_deg": max((
            log.get("ftc_yaw_tracking_error_deg")
            for log in logs
            if log.get("ftc_yaw_tracking_error_deg") is not None
        ), default=None),
        "max_yaw_rate_tracking_error_deg_s": max((
            log.get("ftc_yaw_rate_tracking_error_deg_s")
            for log in logs
            if log.get("ftc_yaw_rate_tracking_error_deg_s") is not None
        ), default=None),
        "max_cross_track_error_m": max((
            log.get("ftc_cross_track_error_m")
            for log in logs
            if log.get("ftc_cross_track_error_m") is not None
        ), default=None),
        "tracking_error_available_steps": sum(bool(log.get("ftc_tracking_error_ratio_available")) for log in logs),
        "raw_model_fault_predictions": int(sum(np.argmax(d["raw_mode_probabilities"]) != 0 for d in raw)),
        "online": summarize_online_diagnosis(logs, dt),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=40.)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dt = .05
    bridge = SixDOFModelBridge(DEFAULT_CHECKPOINT, DEFAULT_TEMPORAL_CONFIG,
                              AUVSixDOFMultiTaskDetector, device=args.device)
    cases = [dict(name=f"normal_{depth}m", depth=depth, seed=20260970+i)
             for i, depth in enumerate((50., 200., 280.))]
    cases += [dict(name="normal_disturbance_200m", depth=200., seed=20260973, disturbance=True)]
    for i, (thruster, fault_mode) in enumerate((("H2", "no_output"), ("V1", "no_output"),
                                               ("H1", "thrust_loss"), ("V2", "thrust_loss"))):
        cases.append(dict(name=f"{thruster}_{fault_mode}_200m", depth=200., seed=20260974+i,
                          thruster=thruster, fault_mode=fault_mode, efficiency=.35))
    cases += [dict(name="V2_loss_disturbance_200m", depth=200., seed=20260978,
                   thruster="V2", fault_mode="thrust_loss", efficiency=.35, disturbance=True),
              dict(name="mixed_demo_200m", depth=200., seed=20260911, demo=True)]
    results = []
    for case in cases:
        print("Running " + case["name"], flush=True)
        runs = {mode: run_case(case, mode, bridge, dt, args.duration)
                for mode in ("rule", "shadow", "fusion")}
        for a, b in zip(runs["rule"], runs["shadow"]):
            for key in ("position_ned", "body_velocity", "commanded_thruster_forces", "ftc_action"):
                np.testing.assert_equal(a[key], b[key])
        # The online outputs must be reproducible on their own causal trajectory.
        replay = deepcopy(runs["fusion"])
        bridge.enrich_logs(replay)
        for a, b in zip(replay, runs["fusion"]):
            for key in ("raw_mode_probabilities", "raw_location_probabilities", "temporal_state", "advisory_gate_active"):
                np.testing.assert_equal(a["maintenance_diagnosis"][key], b["maintenance_diagnosis"][key])
        for previous, current in zip(runs["fusion"], runs["fusion"][1:]):
            assert current["ftc_applied_action"] == previous["ftc_action"]
        base, fused = runs["shadow"], runs["fusion"]
        differing = [i for i, (a, b) in enumerate(zip(base, fused)) if a["ftc_action"] != b["ftc_action"]]
        added = sum(b["ftc_intervention_requested"] and not a["ftc_intervention_requested"]
                    for a, b in zip(base, fused))
        record = {"case": case, "metrics": {mode: metrics(logs, dt) for mode, logs in runs.items()},
                  "shadow_matches_rule": True, "fusion_replay_matches_online": True,
                  "next_step_application_passed": True,
                  "action_difference_steps": len(differing),
                  "first_action_difference_s": fused[differing[0]]["time"] if differing else None,
                  "additional_intervention_steps": int(added),
                  "max_paired_position_difference_m": float(max(np.linalg.norm(a["position_ned"]-b["position_ned"])
                                                                for a,b in zip(base,fused)))}
        if differing:
            assert any(log.get("ai_ftc_fusion", {}).get("accepted") for log in fused[:differing[0]+1])
        results.append(record)
        (args.output_dir / f"{case['name']}_metrics.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        save_online_trace(fused, args.output_dir / f"{case['name']}_fusion_trace.jsonl")
        print(json.dumps({"case": case["name"], "admitted": record["metrics"]["fusion"]["ai_admitted_steps"],
                          "changed_actions": len(differing), "additional_interventions": int(added)}), flush=True)
    normal = [r for r in results if r["case"]["name"].startswith("normal_")]
    h1_loss = next(
        r for r in results if r["case"]["name"] == "H1_thrust_loss_200m"
    )
    h2_no_output = next(
        r for r in results if r["case"]["name"] == "H2_no_output_200m"
    )
    v1_no_output = next(
        r for r in results if r["case"]["name"] == "V1_no_output_200m"
    )
    h1_shadow = h1_loss["metrics"]["shadow"]
    h1_fusion = h1_loss["metrics"]["fusion"]
    acceptance = {
        "normal_cases_without_additional_interventions": all(
            r["additional_intervention_steps"] == 0 for r in normal
        ),
        "direct_no_output_isolation_preserved": (
            h2_no_output["metrics"]["fusion"]["isolated_thrusters"] == ["H2"]
            and v1_no_output["metrics"]["fusion"]["isolated_thrusters"] == ["V1"]
        ),
        "h1_loss_ai_evidence_admitted": h1_fusion["ai_admitted_steps"] > 0,
        "h1_loss_horizontal_hold_entered": (
            h1_fusion["first_horizontal_hold_s"] is not None
        ),
        "h1_loss_does_not_guess_isolation": not h1_fusion["isolated_thrusters"],
        "h1_loss_max_yaw_below_35_deg": (
            h1_fusion["max_abs_roll_pitch_yaw_deg"][2] < 35.0
        ),
        "h1_loss_reduces_peak_yaw_by_50_percent": (
            h1_fusion["max_abs_roll_pitch_yaw_deg"][2]
            < 0.5 * h1_shadow["max_abs_roll_pitch_yaw_deg"][2]
        ),
    }
    report = {
        "checkpoint_sha256": _sha256(DEFAULT_CHECKPOINT),
        "temporal_policy_sha256": _sha256(DEFAULT_TEMPORAL_CONFIG),
        "fusion_config": asdict(AIEvidenceGateConfig()),
        "dt_s": dt, "duration_s": args.duration, "case_count": len(cases),
        "functional_checks_passed": all(acceptance.values()),
        "acceptance": acceptance,
        "normal_cases_without_additional_interventions": all(r["additional_intervention_steps"] == 0 for r in normal),
        "normal_additional_intervention_steps": sum(r["additional_intervention_steps"] for r in normal),
        "cases": results,
        "scope": "Integration experiment, not a new blind performance acceptance or a hard real-time guarantee.",
    }
    (args.output_dir / "fusion_validation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Paired fusion validation complete.", flush=True)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
