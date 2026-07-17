"""Evaluate recovered short disturbances without injected thruster faults."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
MATH_MODEL_ROOT = REPO_ROOT / "math-model"
MATH_MODEL_SRC = MATH_MODEL_ROOT / "src"
if str(MATH_MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MATH_MODEL_SRC))

from actuators.six_dof_thruster_faults import ThrusterActuatorBank
from actuators.thruster_array import default_six_thruster_array
from diagnosis.maintenance_health_decision import (
    apply_maintenance_decision_layer,
)
from diagnosis.maintenance_ticket_policy import (
    MaintenanceTicketConfig,
    apply_maintenance_ticket_policy,
)
from diagnosis.temporal_fault_decision import TemporalDecisionConfig
from environment.six_dof_dynamics import SixDOFDynamics, SixDOFState
from environment.six_dof_simulator import SixDOFSimulator
from evaluation.transient_recovery import (
    default_transient_scenarios,
    summarize_transient_recovery,
)
from ftc.safety_supervisor import (
    FTCSafetySupervisor,
    build_rule_based_ftc_evidence,
)
from sensors.depth_sensor import DepthSensor
from sensors.dvl_sensor import DVLSensor
from sensors.imu_sensor import IMUSensor
from sensors.six_dof_sensor_suite import SixDOFSensorSuite
from simple_control.six_dof_controller import PoseTarget
from utils.six_dof_dataset_builder import build_six_dof_sequence_dataset

from model_six_dof_multitask import AUVSixDOFMultiTaskDetector
from train_six_dof_multitask import evaluate, make_loader, set_seed


TARGET_POSITION = np.array([0.0, 0.0, 2.0])
TARGET_ATTITUDE = np.zeros(3)


def _sensor_suite(seed, dvl_dropout_windows=()):
    return SixDOFSensorSuite(
        depth_sensor=DepthSensor(
            noise_std=0.05,
            drift_std=0.001,
            seed=seed + 1,
        ),
        imu_sensor=IMUSensor(
            attitude_noise_std=0.003,
            gyro_noise_std=0.0015,
            accel_noise_std=0.015,
            seed=seed + 2,
        ),
        dvl_sensor=DVLSensor(
            velocity_noise_std=0.03,
            dropout_prob=0.01,
            dropout_windows=dvl_dropout_windows,
            seed=seed + 3,
        ),
    )


def _tracking_aware_ftc_provider(log):
    """Supply normalized observable pose error to the rule supervisor."""

    enriched = dict(log)
    position_ratio = np.linalg.norm(log["position_error_ned"]) / 0.50
    attitude_ratio = np.linalg.norm(log["attitude_error_body"]) / 0.25
    enriched["tracking_error_ratio"] = float(
        max(position_ratio, attitude_ratio)
    )
    return build_rule_based_ftc_evidence(enriched)


def _disturbance_provider(scenario, seed):
    rng = np.random.default_rng(seed)
    amplitudes = rng.uniform(
        np.zeros(6),
        np.array([0.8, 0.8, 0.5, 0.02, 0.02, 0.05]),
    )
    frequencies = rng.uniform(0.04, 0.10, size=6)
    phases = rng.uniform(-np.pi, np.pi, size=6)

    def provider(time_s, _state):
        ambient = amplitudes * np.sin(frequencies * time_s + phases)
        pulse = (
            np.zeros(6)
            if scenario is None
            else scenario.wrench_at(time_s)
        )
        return ambient + pulse

    return provider, {
        "ambient_amplitudes": amplitudes.tolist(),
        "ambient_frequencies_radps": frequencies.tolist(),
        "ambient_phases_rad": phases.tolist(),
    }


def run_transient_mission(
    scenario,
    duration,
    dt,
    seed,
    *,
    dvl_dropout_windows=(),
):
    thruster_array = default_six_thruster_array()
    dynamics = SixDOFDynamics(initial_state=SixDOFState(
        position_ned=TARGET_POSITION,
    ))
    actuator_bank = ThrusterActuatorBank(
        thruster_array,
        current_noise_std=0.05,
        rpm_noise_std=25.0,
        voltage_noise_std=0.05,
        temperature_noise_std=0.15,
        seed=seed + 4,
    )
    supervisor = FTCSafetySupervisor()
    simulator = SixDOFSimulator(
        dynamics=dynamics,
        thruster_array=thruster_array,
        actuator_bank=actuator_bank,
        sensor_suite=_sensor_suite(seed, dvl_dropout_windows),
        ftc_supervisor=supervisor,
        ftc_evidence_provider=_tracking_aware_ftc_provider,
    )
    target = PoseTarget(TARGET_POSITION, TARGET_ATTITUDE)
    disturbance, ambient_metadata = _disturbance_provider(scenario, seed + 5)
    logs = simulator.run(
        duration=duration,
        dt=dt,
        target_provider=lambda _time, _state: target,
        disturbance_provider=disturbance,
    )
    return logs, ambient_metadata


def _build_dataset(missions, seq_len, stride):
    chunks = []
    metadata = {}
    for mission_id, item in missions.items():
        chunks.append(build_six_dof_sequence_dataset(
            {mission_id: item["logs"]},
            seq_len=seq_len,
            stride=stride,
        ))
        metadata[mission_id] = {
            "scenario": item["scenario"].name,
            "seed": item["seed"],
            "split": "transient_recovery_benchmark",
            "parameters": {
                "thrusters": {
                    "horizontal_force_limit_n": 40.0,
                    "vertical_force_limit_n": 35.0,
                },
                "transient": item["scenario"].as_dict(),
                "ambient": item["ambient"],
            },
        }
    keys = (
        "X",
        "y_mode",
        "y_location",
        "y_joint",
        "mission_ids",
        "window_end_times",
        "guidance_context_ids",
        "guidance_context_stable",
    )
    payload = {
        key: torch.from_numpy(np.concatenate([
            chunk[key] for chunk in chunks
        ], axis=0))
        for key in keys
    }
    payload.update({
        key: chunks[0][key]
        for key in (
            "feature_names",
            "raw_feature_dim",
            "model_input_dim",
            "sequence_length",
        )
    })
    payload["mission_metadata"] = metadata
    return payload


def _load_json_config(path, config_type):
    return config_type(**json.loads(path.read_text(encoding="utf-8")))


def _model_predictions(dataset, checkpoint_path, batch_size, device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = AUVSixDOFMultiTaskDetector(
        input_dim=int(checkpoint["input_dim"]),
        structured_fusion=bool(checkpoint.get("structured_fusion", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    indices = np.arange(len(dataset["X"]), dtype=np.int64)
    loader = make_loader(
        dataset,
        indices,
        checkpoint["mean"],
        checkpoint["std"],
        batch_size,
        shuffle=False,
        seed=0,
    )
    return indices, evaluate(
        model,
        loader,
        device,
        torch.nn.CrossEntropyLoss(),
        torch.nn.CrossEntropyLoss(),
    )


def _mission_level_results(missions, decisions, dataset):
    mission_ids = dataset["mission_ids"].cpu().numpy()
    health = np.asarray(decisions["health_level_pred"], dtype=np.int64)
    tickets = decisions["maintenance_tickets"]
    tickets_by_mission = Counter(
        int(ticket["mission_id"]) for ticket in tickets
    )
    events_by_mission = Counter(
        int(event["mission_id"])
        for event in decisions["maintenance_events"]
    )
    rows = []
    for mission_id, item in missions.items():
        logs = item["logs"]
        recovery = summarize_transient_recovery(
            logs, item["scenario"]
        )
        actions = Counter(log["ftc_action"] for log in logs)
        isolated = sorted({
            log["ftc_targeted_thruster_name"]
            for log in logs
            if log["ftc_targeted_thruster_name"] is not None
        })
        positions = mission_ids == mission_id
        row = {
            "mission_id": int(mission_id),
            "scenario": item["scenario"].name,
            "seed": int(item["seed"]),
            **recovery,
            "maximum_health_level": int(np.max(health[positions])),
            "raw_health_event_count": int(events_by_mission[mission_id]),
            "formal_maintenance_ticket_count": int(
                tickets_by_mission[mission_id]
            ),
            "ftc_action_counts": dict(sorted(actions.items())),
            "ftc_intervention_steps": int(sum(
                bool(log["ftc_intervention_requested"]) for log in logs
            )),
            "ftc_abort_requested": bool(any(
                log["ftc_mission_abort_requested"] for log in logs
            )),
            "ftc_controlled_ascent_requested": bool(any(
                log["ftc_controlled_ascent_requested"] for log in logs
            )),
            "ftc_targeted_thrusters": isolated,
        }
        row["passed"] = bool(
            row["response_observed"]
            and row["recovered"]
            and row["formal_maintenance_ticket_count"] == 0
            and row["ftc_intervention_steps"] == 0
            and not row["ftc_abort_requested"]
            and not row["ftc_controlled_ascent_requested"]
            and not row["ftc_targeted_thrusters"]
        )
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=8421)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            THIS_DIR / "results" / "six_dof_hybrid_telemetry"
            / "best_model.pth"
        ),
    )
    parser.add_argument(
        "--temporal-config",
        type=Path,
        default=(
            THIS_DIR / "results" / "six_dof_hybrid_telemetry_temporal"
            / "temporal_decision_config.json"
        ),
    )
    parser.add_argument(
        "--ticket-config",
        type=Path,
        default=(
            THIS_DIR / "results" / "six_dof_hybrid_telemetry_temporal"
            / "maintenance_ticket_config.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            THIS_DIR / "results" / "six_dof_transient_recovery"
        ),
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.duration <= 25.0 or args.dt <= 0.0:
        parser.error("--duration must exceed 25 s and --dt must be positive")
    if args.seq_len <= 0 or args.stride <= 0:
        parser.error("--seq-len and --stride must be positive")

    set_seed(args.seed)
    missions = {}
    mission_id = 0
    for scenario_index, scenario in enumerate(default_transient_scenarios()):
        for repeat in range(args.repeats):
            seed = args.seed + scenario_index * 1000 + repeat
            logs, ambient = run_transient_mission(
                scenario, args.duration, args.dt, seed
            )
            missions[mission_id] = {
                "scenario": scenario,
                "seed": seed,
                "logs": logs,
                "ambient": ambient,
            }
            print(
                f"Mission {mission_id + 1:02d}/"
                f"{len(default_transient_scenarios()) * args.repeats}: "
                f"{scenario.name}, repeat={repeat + 1}"
            )
            mission_id += 1

    dataset = _build_dataset(missions, args.seq_len, args.stride)
    if np.any(dataset["y_mode"].cpu().numpy() != 0):
        raise RuntimeError("transient benchmark must not inject fault labels")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices, predictions = _model_predictions(
        dataset, args.checkpoint, args.batch_size, device
    )
    temporal_config = _load_json_config(
        args.temporal_config, TemporalDecisionConfig
    )
    ticket_config = _load_json_config(
        args.ticket_config, MaintenanceTicketConfig
    )
    decisions = apply_maintenance_decision_layer(
        dataset, indices, predictions, temporal_config
    )
    decisions = apply_maintenance_ticket_policy(
        dataset, indices, decisions, ticket_config
    )
    mission_results = _mission_level_results(
        missions, decisions, dataset
    )

    summary = {
        "benchmark": "six_dof_transient_recovery_v1",
        "device": str(device),
        "acceptance_policy": (
            "a short observable pose excursion must recover; raw health logs "
            "are allowed, but no formal maintenance ticket, FTC intervention, "
            "thruster isolation, abort, or ascent request is allowed"
        ),
        "scenario_count": len(default_transient_scenarios()),
        "mission_count": len(mission_results),
        "window_count": int(len(dataset["X"])),
        "raw_health_event_count": len(decisions["maintenance_events"]),
        "formal_maintenance_ticket_count": len(
            decisions["maintenance_tickets"]
        ),
        "recovered_missions": int(sum(
            row["recovered"] for row in mission_results
        )),
        "missions_with_ftc_intervention": int(sum(
            row["ftc_intervention_steps"] > 0 for row in mission_results
        )),
        "passed_missions": int(sum(
            row["passed"] for row in mission_results
        )),
        "all_passed": bool(all(
            row["passed"] for row in mission_results
        )),
        "temporal_config": vars(temporal_config),
        "maintenance_ticket_config": vars(ticket_config),
        "mission_results": mission_results,
        "raw_health_events": decisions["maintenance_events"],
        "formal_maintenance_tickets": decisions["maintenance_tickets"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "transient_recovery_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if args.strict and not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
