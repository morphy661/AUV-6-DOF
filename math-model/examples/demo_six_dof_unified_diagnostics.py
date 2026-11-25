"""Generate a unified six-DOF diagnosis and FTC demonstration video."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_ROOT = (
    REPOSITORY_ROOT
    / "depth-sensor-fault-detection"
    / "depth_fault_detection"
)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from actuators.six_dof_thruster_faults import (
    SingleThrusterFault,
    SixDOFThrusterFaultMode,
)
from actuators.esc_telemetry_faults import (
    ESCTelemetryFaultInjector,
    THRUSTER_NAMES,
)
from environment.six_dof_simulator import SixDOFSimulator
from environment.six_dof_dynamics import SixDOFState
from ftc.safety_supervisor import (
    FTCSafetySupervisor,
    build_rule_based_ftc_evidence,
)
from ftc.ai_evidence_gate import AIControlEvidenceGate, AIEvidenceGateConfig
from presentation.six_dof_demo_adapter import (
    adapt_logs,
    extract_demo_events,
    summarize_demo,
)
from presentation.six_dof_demo_renderer import SixDOFDemoRenderer
from presentation.six_dof_model_bridge import SixDOFModelBridge
from sensors.sensor_faults import (
    SensorFaultEvent,
    SensorFaultInjector,
    SensorFaultMode,
)
from sensors.six_dof_sensor_suite import SixDOFSensorSuite
from simple_control.six_dof_controller import PoseTarget


DEFAULT_CHECKPOINT = (
    MODEL_ROOT
    / "results"
    / "six_dof_v3_5_high_yaw_seed43_20260904"
    / "best_model.pth"
)
DEFAULT_TEMPORAL_CONFIG = (
    MODEL_ROOT
    / "results"
    / "six_dof_hybrid_telemetry_0_300m_focus_v3_4_seed43_calibrated_20260903"
    / "temporal_decision_config.json"
)
DEFAULT_ACCEPTANCE_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "six_dof_unified_acceptance_v4_20260717"
    / "unified_acceptance_v4_summary.json"
)


MISSION_TARGETS = (
    (0.0, (0.0, 0.0, 200.0), (0.0, 0.0, 0.0)),
    (4.0, (4.0, 1.0, 200.5), (0.0, 0.0, 0.6)),
    (9.0, (-2.0, 3.0, 201.0), (0.0, 0.0, -0.6)),
    (15.0, (0.0, 0.0, 202.5), (0.0, 0.0, 0.0)),
    (20.0, (2.0, -1.0, 200.5), (0.0, 0.0, 0.3)),
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_acceptance_badge(path):
    """Load an offline V4 badge without mixing it into live diagnosis."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("benchmark") != "six_dof_unified_acceptance_v4":
        raise ValueError("acceptance summary is not the unified V4 benchmark")
    summary = payload.get("summary", {})
    passed = int(summary.get("passed_count", -1))
    total = int(summary.get("check_count", -1))
    accepted = bool(summary.get("all_acceptance_checks_passed", False))
    if total <= 0 or passed < 0 or passed > total:
        raise ValueError("invalid unified V4 acceptance counts")
    expected_decision = "accepted" if accepted else "not_accepted"
    if summary.get("decision") != expected_decision:
        raise ValueError("unified V4 decision and pass flag disagree")
    return {
        "name": "six_dof_unified_acceptance_v4",
        "scope": "offline_simulation_evidence",
        "accepted": accepted,
        "passed_count": passed,
        "check_count": total,
        "protocol_sha256": str(payload.get("protocol_sha256", "")),
    }


def target_provider(time_s, _state):
    selected_index = 0
    for index, candidate in enumerate(MISSION_TARGETS):
        if time_s >= candidate[0]:
            selected_index = index
        else:
            break
    selected = MISSION_TARGETS[selected_index]
    return PoseTarget(
        np.asarray(selected[1], dtype=float),
        np.asarray(selected[2], dtype=float),
        guidance_context_id=selected_index + 1,
    )


def disturbance_provider(time_s, _state):
    return np.array([
        0.9 * np.sin(0.55 * time_s),
        0.6 * np.cos(0.37 * time_s),
        0.35 * np.sin(0.29 * time_s),
        0.02 * np.sin(0.50 * time_s),
        0.02 * np.cos(0.43 * time_s),
        0.04 * np.sin(0.31 * time_s),
    ])


def _event_manifest(event):
    return {
        "event_id": event.event_id,
        "sensor": event.sensor,
        "mode": event.mode.value,
        "start_time_s": event.start_time_s,
        "end_time_s": event.end_time_s,
        "channels": list(event.channels),
        "magnitude": event.magnitude,
    }


def fixed_fault_schedule(seed):
    events = [
        SensorFaultEvent(
            "depth", SensorFaultMode.SPIKE, 3.0, 3.5,
            magnitude=0.30, event_id="depth_weak_spike",
        ),
        SensorFaultEvent(
            "dvl", SensorFaultMode.BIAS, 5.0, 11.0,
            channels=(0,), magnitude=0.30, event_id="dvl_bias",
        ),
        SensorFaultEvent(
            "imu", SensorFaultMode.UNAVAILABLE, 12.0, 12.15,
            event_id="imu_dropout_1",
        ),
        SensorFaultEvent(
            "imu", SensorFaultMode.UNAVAILABLE, 13.0, 13.15,
            event_id="imu_dropout_2",
        ),
        SensorFaultEvent(
            "imu", SensorFaultMode.UNAVAILABLE, 14.0, 14.15,
            event_id="imu_dropout_3",
        ),
    ]
    thruster_fault = SingleThrusterFault(
        "V1", SixDOFThrusterFaultMode.NO_OUTPUT, start_time=17.0
    )
    return events, thruster_fault


def safety_recovery_fault_schedule(seed):
    """Single-event story for explaining online diagnosis and FTC recovery."""

    return [], SingleThrusterFault(
        "V1",
        SixDOFThrusterFaultMode.NO_OUTPUT,
        start_time=10.0,
        thrust_efficiency=0.0,
    )


def esc_telemetry_schedule(seed, injection_mode):
    """Create an observable ESC-link anomaly before the physical fault."""

    if injection_mode == "fixed":
        return [{
            "event_id": "v2_esc_packet_loss",
            "thruster_name": "V2",
            "mode": "continuous_packet_loss",
            "start_time_s": 15.35,
            "end_time_s": 16.35,
        }]
    if injection_mode == "safety_recovery":
        return []
    if injection_mode != "random":
        raise ValueError(
            "injection_mode must be fixed, random, or safety_recovery"
        )
    rng = np.random.default_rng(int(seed) ^ 0x5EC0F)
    start = float(rng.uniform(14.75, 15.10))
    duration = float(rng.uniform(0.90, 1.15))
    return [{
        "event_id": "random_esc_link_anomaly",
        "thruster_name": str(rng.choice(THRUSTER_NAMES)),
        "mode": str(rng.choice((
            "continuous_packet_loss", "communication_freeze"
        ))),
        "start_time_s": start,
        "end_time_s": start + duration,
    }]


def build_esc_telemetry_evidence_provider(events, config):
    """Inject link faults into observable logs before building FTC evidence."""

    injector = ESCTelemetryFaultInjector(events)

    def provider(log):
        injector.apply(log)
        return build_rule_based_ftc_evidence(log, config=config)

    return provider


def random_fault_schedule(seed):
    """Create a reproducible randomized sequence for a stress demonstration."""

    rng = np.random.default_rng(seed)
    weak_sensor, ambiguous_sensor, intermittent_sensor = tuple(
        rng.permutation(("depth", "imu", "dvl"))
    )
    weak_specs = {
        "depth": ((), 0.30),
        "imu": ((2,), 0.15),
        "dvl": ((0,), 0.40),
    }
    ambiguous_mode = (
        SensorFaultMode.BIAS
        if rng.random() < 0.5
        else SensorFaultMode.DRIFT
    )
    ambiguous_specs = {
        SensorFaultMode.BIAS: {
            "depth": ((), 0.35), "imu": ((2,), 0.15),
            "dvl": ((0,), 0.30),
        },
        SensorFaultMode.DRIFT: {
            "depth": ((), 0.04), "imu": ((2,), 0.02),
            "dvl": ((0,), 0.04),
        },
    }
    weak_channels, weak_magnitude = weak_specs[weak_sensor]
    ambiguous_channels, ambiguous_magnitude = (
        ambiguous_specs[ambiguous_mode][ambiguous_sensor]
    )
    weak_start = float(rng.uniform(2.6, 3.4))
    ambiguous_start = float(rng.uniform(5.0, 5.8))
    intermittent_start = float(rng.uniform(11.8, 12.2))
    events = [
        SensorFaultEvent(
            weak_sensor, SensorFaultMode.SPIKE,
            weak_start, weak_start + 0.5,
            channels=weak_channels, magnitude=weak_magnitude,
            event_id=f"random_{weak_sensor}_weak_spike",
        ),
        SensorFaultEvent(
            ambiguous_sensor, ambiguous_mode,
            ambiguous_start, ambiguous_start + 6.0,
            channels=ambiguous_channels, magnitude=ambiguous_magnitude,
            event_id=(
                f"random_{ambiguous_sensor}_{ambiguous_mode.value}"
            ),
        ),
    ]
    for index in range(3):
        start = intermittent_start + index
        events.append(SensorFaultEvent(
            intermittent_sensor, SensorFaultMode.UNAVAILABLE,
            start, start + 0.15,
            event_id=(
                f"random_{intermittent_sensor}_dropout_{index + 1}"
            ),
        ))
    thruster_name = str(rng.choice(("H1", "H2", "H3", "H4", "V1", "V2")))
    mode = (
        SixDOFThrusterFaultMode.NO_OUTPUT
        if rng.random() < 0.65
        else SixDOFThrusterFaultMode.THRUST_LOSS
    )
    thruster_fault = SingleThrusterFault(
        thruster_name,
        mode,
        start_time=float(rng.uniform(16.5, 17.5)),
        thrust_efficiency=(
            0.0 if mode is SixDOFThrusterFaultMode.NO_OUTPUT
            else float(rng.uniform(0.35, 0.70))
        ),
    )
    return events, thruster_fault


def build_fault_scenario(seed, injection_mode):
    if injection_mode == "fixed":
        events, thruster_fault = fixed_fault_schedule(seed)
    elif injection_mode == "random":
        events, thruster_fault = random_fault_schedule(seed)
    elif injection_mode == "safety_recovery":
        events, thruster_fault = safety_recovery_fault_schedule(seed)
    else:
        raise ValueError(
            "injection_mode must be fixed, random, or safety_recovery"
        )
    suite = SixDOFSensorSuite(
        fault_injector=SensorFaultInjector(events), seed=seed
    )
    manifest = {
        "injection_mode": injection_mode,
        "seed": int(seed),
        "sensor_events": [_event_manifest(event) for event in events],
        "esc_telemetry_events": esc_telemetry_schedule(seed, injection_mode),
        "thruster_fault": {
            "thruster_name": thruster_fault.thruster_name,
            "mode": thruster_fault.mode.value,
            "start_time_s": thruster_fault.start_time,
            "thrust_efficiency": thruster_fault.thrust_efficiency,
        },
    }
    return suite, thruster_fault, manifest


def run_demo(
    duration, dt, seed, *, injection_mode="fixed", model_bridge=None,
    diagnosis_mode="shadow",
):
    if diagnosis_mode not in ("shadow", "fusion"):
        raise ValueError("diagnosis_mode must be shadow or fusion")
    if diagnosis_mode == "fusion" and model_bridge is None:
        raise ValueError("fusion requires an online model")
    sensor_suite, thruster_fault, manifest = build_fault_scenario(
        seed, injection_mode
    )
    supervisor = FTCSafetySupervisor()
    simulator = SixDOFSimulator(
        fault=thruster_fault,
        sensor_suite=sensor_suite,
        ftc_supervisor=supervisor,
        ftc_evidence_provider=build_esc_telemetry_evidence_provider(
            manifest["esc_telemetry_events"], supervisor.config
        ),
        diagnostic_observer=model_bridge,
        ai_ftc_fusion=(
            AIControlEvidenceGate(
                AIEvidenceGateConfig(
                    minimum_mode_probability=model_bridge.decision.temporal.config.enter_fault_probability,
                    maximum_result_age_s=1.2 * model_bridge.inference_stride * dt,
                ), supervisor.config,
            ) if diagnosis_mode == "fusion" else None
        ),
    )
    simulator.reset(SixDOFState(
        position_ned=np.asarray(MISSION_TARGETS[0][1], dtype=float),
    ))
    logs = simulator.run(
        duration, dt, target_provider,
        disturbance_provider=disturbance_provider,
    )
    frames = adapt_logs(logs)
    events = extract_demo_events(frames)
    return logs, frames, events, manifest


def summarize_online_diagnosis(logs, dt):
    """Report measured synchronous timing, including first-inference startup."""
    diagnoses = [log["maintenance_diagnosis"] for log in logs
                 if "maintenance_diagnosis" in log]

    def timing(values, budget_ms):
        return {
            "count": len(values),
            "mean_ms": float(np.mean(values)) if values else None,
            "p95_ms": float(np.percentile(values, 95)) if values else None,
            "max_ms": float(np.max(values)) if values else None,
            "budget_ms": budget_ms,
            "over_budget_count": sum(value > budget_ms for value in values),
        }

    return {
        "execution_mode": diagnoses[0]["execution_mode"] if diagnoses else "disabled",
        "affects_ftc": any(d.get("affects_ftc", False) for d in diagnoses),
        "fusion_verdict_counts": {
            verdict: sum(log.get("ai_ftc_fusion", {}).get("verdict") == verdict for log in logs)
            for verdict in sorted({log["ai_ftc_fusion"]["verdict"] for log in logs if "ai_ftc_fusion" in log})
        },
        "probability_note": "Raw softmax scores; entropy is not calibrated error probability.",
        "scope": ["normal", "no_output", "thrust_loss"],
        "sample_count": len(diagnoses),
        "inference_count": sum(bool(d.get("updated")) for d in diagnoses),
        "available_count": sum(bool(d.get("available")) for d in diagnoses),
        "observer_update": timing([d["update_wall_ms"] for d in diagnoses], dt * 1000),
        "inference_updates": timing([d["update_wall_ms"] for d in diagnoses
                                     if d.get("updated")], dt * 1000),
        "full_simulation_step": timing([log["step_wall_ms"] for log in logs], dt * 1000),
    }


def save_online_trace(logs, path):
    """Retain actual per-step scores, freshness, gating and FTC on one timeline."""
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    with path.open("w", encoding="utf-8") as output:
        for log in logs:
            output.write(json.dumps(json_safe({
                "time_s": float(log["time"]),
                "model": log.get("maintenance_diagnosis"),
                "ftc_action": log.get("ftc_action"),
                "ftc_applied_action": log.get("ftc_applied_action"),
                "ftc_evidence_source": log.get("ftc_evidence_source"),
                "fusion": log.get("ai_ftc_fusion"),
                "step_wall_ms": log["step_wall_ms"],
            }), ensure_ascii=False, allow_nan=False) + "\n")


def save_json(
    frames, events, summary, manifest, offline_acceptance, path
):
    path.write_text(json.dumps({
        "demo": "six_dof_unified_diagnostics",
        "diagnostic_contract": (
            "Sensor and thruster tiers use causal onboard fields only; injected truth is excluded."
        ),
        "offline_acceptance_baseline": offline_acceptance,
        "summary": summary,
        "injection_manifest": manifest,
        "events": events,
        "frames": frames,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(frames, path):
    handoff_keys = ("state", "recovery_phase", "pause_abort_requested", "original_mission_aborted",
                    "external_recovery_required", "external_request_sent", "external_takeover_confirmed")
    fields = [
        "time_s", "north_m", "east_m", "depth_m", "roll_deg",
        "pitch_deg", "yaw_deg", "overall_tier", "depth_tier",
        "imu_tier", "dvl_tier", "ftc_action", "ftc_target_thruster",
        "model_health_state", "model_fault_probability",
        "model_probable_mode", "model_suspected_group",
        "model_advisory_gate_active", "model_advisory_suppressed",
        "model_advisory_gate_reasons",
        "ftc_untrusted_esc_channels",
        "model_top1", "model_top1_probability",
        "model_top2", "model_top2_probability",
        *[
            f"{name}_no_output_score"
            for name in ("H1", "H2", "H3", "H4", "V1", "V2")
        ],
        *[
            f"{name}_telemetry_status"
            for name in ("H1", "H2", "H3", "H4", "V1", "V2")
        ],
        *["mission_" + key for key in handoff_keys],
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for frame in frames:
            position = frame["pose"]["position_ned_m"]
            rpy_deg = np.rad2deg(frame["pose"]["euler_rpy_rad"])
            row = {
                "time_s": frame["time_s"],
                "north_m": position[0],
                "east_m": position[1],
                "depth_m": position[2],
                "roll_deg": rpy_deg[0],
                "pitch_deg": rpy_deg[1],
                "yaw_deg": rpy_deg[2],
                "overall_tier": frame["overall_tier"],
                "depth_tier": frame["sensors"]["depth"]["tier"],
                "imu_tier": frame["sensors"]["imu"]["tier"],
                "dvl_tier": frame["sensors"]["dvl"]["tier"],
                "ftc_action": frame["ftc"]["action"],
                "ftc_target_thruster": frame["ftc"]["target_thruster"],
                "ftc_untrusted_esc_channels": "|".join(
                    frame["ftc"]["untrusted_esc_channels"]
                ),
                "model_health_state": frame["maintenance"]["health_state"],
                "model_fault_probability": frame["maintenance"]["fault_probability"],
                "model_probable_mode": frame["maintenance"]["probable_mode"],
                "model_suspected_group": frame["maintenance"]["suspected_group"],
                "model_advisory_gate_active": frame["maintenance"][
                    "advisory_gate_active"
                ],
                "model_advisory_suppressed": frame["maintenance"][
                    "advisory_suppressed"
                ],
                "model_advisory_gate_reasons": "|".join(
                    frame["maintenance"]["advisory_gate_reasons"]
                ),
            }
            candidates = frame["maintenance"]["candidates"]
            for rank in range(2):
                candidate = candidates[rank] if rank < len(candidates) else None
                row[f"model_top{rank + 1}"] = (
                    None if candidate is None else candidate["name"]
                )
                row[f"model_top{rank + 1}_probability"] = (
                    None if candidate is None else candidate["probability"]
                )
            row.update({
                f"{card['name']}_no_output_score": card["no_output_score"]
                for card in frame["thrusters"]
            })
            row.update({
                f"{card['name']}_telemetry_status": card["telemetry_status"]
                for card in frame["thrusters"]
            })
            for key in handoff_keys:
                value = frame.get("mission_handoff", {}).get(key)
                row["mission_" + key] = "not_reported" if value is None else value
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=22.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--injection-mode",
        choices=("fixed", "random", "safety_recovery"),
        default="fixed",
    )
    parser.add_argument("--disable-model", action="store_true")
    parser.add_argument("--diagnosis-mode", choices=("shadow", "fusion"), default="shadow")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--temporal-config", type=Path, default=DEFAULT_TEMPORAL_CONFIG
    )
    parser.add_argument(
        "--model-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--acceptance-summary",
        type=Path,
        default=DEFAULT_ACCEPTANCE_SUMMARY,
    )
    badge_group = parser.add_mutually_exclusive_group()
    badge_group.add_argument("--show-acceptance-badge", action="store_true")
    badge_group.add_argument(
        "--hide-acceptance-badge",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-video-frames", type=int, default=240)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "results" / "six_dof_unified_diagnostics_demo",
    )
    args = parser.parse_args()
    if args.disable_model and args.diagnosis_mode == "fusion":
        parser.error("--diagnosis-mode fusion requires the model")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    acceptance_badge = (
        load_acceptance_badge(args.acceptance_summary)
        if args.show_acceptance_badge and not args.hide_acceptance_badge
        else None
    )
    model_bridge = None
    if not args.disable_model:
        if str(MODEL_ROOT) not in sys.path:
            sys.path.insert(0, str(MODEL_ROOT))
        from model_six_dof_multitask import AUVSixDOFMultiTaskDetector

        model_bridge = SixDOFModelBridge(
            args.checkpoint,
            args.temporal_config,
            AUVSixDOFMultiTaskDetector,
            device=(None if args.model_device == "auto" else args.model_device),
        )
    logs, frames, events, manifest = run_demo(
        args.duration,
        args.dt,
        args.seed,
        injection_mode=args.injection_mode,
        model_bridge=model_bridge,
        diagnosis_mode=args.diagnosis_mode,
    )
    summary = summarize_demo(frames, events)
    summary.update({
        "injection_mode": args.injection_mode,
        "injection_seed": args.seed,
        "injected_thruster": manifest["thruster_fault"]["thruster_name"],
        "injected_thruster_mode": manifest["thruster_fault"]["mode"],
        "model_enabled": model_bridge is not None,
        "online_diagnosis": summarize_online_diagnosis(logs, args.dt),
        "model_runtime_preparation_ms": (
            None if model_bridge is None else model_bridge.runtime_preparation_ms
        ),
        "model_device": (
            None if model_bridge is None else str(model_bridge.device)
        ),
        "model_checkpoint": (
            None if model_bridge is None else str(args.checkpoint)
        ),
        "model_checkpoint_sha256": (
            None if model_bridge is None else _sha256(args.checkpoint)
        ),
        "model_temporal_config": (
            None if model_bridge is None else str(args.temporal_config)
        ),
        "model_temporal_config_sha256": (
            None if model_bridge is None else _sha256(args.temporal_config)
        ),
        "offline_acceptance_baseline": acceptance_badge,
    })

    json_path = args.output_dir / "six_dof_unified_diagnostics.json"
    csv_path = args.output_dir / "six_dof_unified_diagnostics.csv"
    image_path = args.output_dir / "six_dof_unified_diagnostics.png"
    detail_image_path = (
        args.output_dir / "six_dof_unified_diagnostics_detail.png"
    )
    esc_image_path = args.output_dir / "six_dof_unified_diagnostics_esc_link.png"
    video_path = args.output_dir / "six_dof_unified_diagnostics.mp4"
    save_json(
        frames,
        events,
        summary,
        manifest,
        acceptance_badge,
        json_path,
    )
    save_csv(frames, csv_path)
    save_online_trace(logs, args.output_dir / "online_model_trace.jsonl")

    renderer = SixDOFDemoRenderer(
        frames,
        events,
        injection_manifest=manifest,
        view_mode="summary",
        acceptance_badge=acceptance_badge,
    )
    renderer.save_snapshot(image_path)
    if not args.skip_video:
        renderer.save_video(
            video_path, fps=args.fps, max_frames=args.max_video_frames
        )
    renderer.close()

    detail_renderer = SixDOFDemoRenderer(
        frames,
        events,
        injection_manifest=manifest,
        view_mode="technical",
        acceptance_badge=acceptance_badge,
    )
    detail_renderer.save_snapshot(detail_image_path)
    esc_indices = [
        index for index, frame in enumerate(frames)
        if frame["ftc"]["untrusted_esc_channels"]
    ]
    if esc_indices:
        detail_renderer.save_snapshot(esc_image_path, index=esc_indices[0])
    detail_renderer.close()

    print(json.dumps(summary, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Image: {image_path}")
    print(f"Technical detail image: {detail_image_path}")
    if esc_indices:
        print(f"ESC image: {esc_image_path}")
    if not args.skip_video:
        print(f"Video: {video_path}")


if __name__ == "__main__":
    main()
