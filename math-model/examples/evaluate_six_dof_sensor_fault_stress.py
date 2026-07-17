"""Run the hash-locked six-DOF sensor-fault stress benchmark once."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from environment.six_dof_simulator import SixDOFSimulator
from evaluation.protocol import (
    canonical_sha256,
    prepare_locked_protocol,
)
from evaluation.sensor_fault_stress_benchmark import (
    default_sensor_fault_stress_scenarios,
    evaluate_sensor_fault_stress_mission,
    summarize_sensor_fault_stress_benchmark,
)
from ftc.safety_supervisor import FTCSafetySupervisor
from sensors.depth_sensor import DepthSensor
from sensors.dvl_sensor import DVLSensor
from sensors.imu_sensor import IMUSensor
from sensors.sensor_faults import SensorFaultInjector
from sensors.six_dof_sensor_suite import SixDOFSensorSuite
from simple_control.six_dof_controller import PoseTarget


DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT / "docs" / "six_dof_sensor_fault_stress_protocol_v1.json"
)


def target_provider(time_s, _state):
    if time_s < 4.0:
        return PoseTarget(np.zeros(3), np.zeros(3), guidance_context_id=0)
    if time_s < 8.0:
        return PoseTarget(
            np.array([4.0, 1.0, 1.5]),
            np.array([0.0, 0.0, 0.6]),
            guidance_context_id=1,
        )
    if time_s < 12.0:
        return PoseTarget(
            np.array([-2.0, 3.0, 2.5]),
            np.array([0.0, 0.0, -0.6]),
            guidance_context_id=2,
        )
    return PoseTarget(
        np.array([0.0, 0.0, 1.0]),
        np.zeros(3),
        guidance_context_id=3,
    )


def sensor_suite(scenario, seed):
    rng = np.random.default_rng(seed)
    parameters = {
        "depth_noise_std_m": float(rng.uniform(0.01, 0.12)),
        "depth_drift_std_m_per_sqrt_s": float(
            rng.uniform(0.0003, 0.0040)
        ),
        "imu_attitude_noise_std_rad": float(rng.uniform(0.0005, 0.008)),
        "imu_gyro_noise_std_radps": float(rng.uniform(0.0005, 0.006)),
        "imu_accel_noise_std_mps2": float(rng.uniform(0.005, 0.05)),
        "dvl_velocity_noise_std_mps": float(rng.uniform(0.005, 0.08)),
    }
    suite = SixDOFSensorSuite(
        depth_sensor=DepthSensor(
            noise_std=parameters["depth_noise_std_m"],
            drift_std=parameters["depth_drift_std_m_per_sqrt_s"],
            seed=seed + 1,
        ),
        imu_sensor=IMUSensor(
            attitude_noise_std=parameters["imu_attitude_noise_std_rad"],
            gyro_noise_std=parameters["imu_gyro_noise_std_radps"],
            accel_noise_std=parameters["imu_accel_noise_std_mps2"],
            seed=seed + 2,
        ),
        dvl_sensor=DVLSensor(
            velocity_noise_std=parameters["dvl_velocity_noise_std_mps"],
            dropout_prob=0.0,
            seed=seed + 3,
        ),
        fault_injector=SensorFaultInjector(scenario.events),
    )
    return suite, parameters


def disturbance_provider(seed, scale):
    rng = np.random.default_rng(seed)
    amplitudes = float(scale) * rng.uniform(
        np.zeros(6),
        np.array([1.5, 1.5, 1.0, 0.04, 0.04, 0.12]),
    )
    frequencies = rng.uniform(0.05, 0.18, size=6)
    phases = rng.uniform(-np.pi, np.pi, size=6)

    def provider(time_s, _state):
        return amplitudes * np.sin(frequencies * time_s + phases)

    return provider, {
        "amplitudes": amplitudes.tolist(),
        "frequencies_radps": frequencies.tolist(),
        "phases_rad": phases.tolist(),
    }


def run_mission(scenario, *, duration, dt, seed):
    suite, sensor_parameters = sensor_suite(scenario, seed)
    disturbance, disturbance_parameters = disturbance_provider(
        seed + 10, scenario.disturbance_scale
    )
    simulator = SixDOFSimulator(
        sensor_suite=suite,
        ftc_supervisor=FTCSafetySupervisor(),
    )
    logs = simulator.run(
        duration,
        dt,
        target_provider,
        disturbance_provider=disturbance,
    )
    row = evaluate_sensor_fault_stress_mission(logs, scenario)
    row.update({
        "seed": seed,
        "sensor_parameters": sensor_parameters,
        "disturbance_parameters": disturbance_parameters,
    })
    return row


def validate_protocol(protocol, protocol_path, scenarios):
    configuration, output_dir, protocol_hash = prepare_locked_protocol(
        protocol,
        protocol_path,
        REPOSITORY_ROOT,
        "six_dof_sensor_fault_stress_v1",
        code_message="code changed after freeze: {relative}",
        output_message=(
            "frozen V1 output already exists; it cannot be overwritten"
        ),
    )
    expected_matrix_hash = protocol.get("scenario_matrix_sha256")
    actual_matrix_hash = canonical_sha256([
        scenario.as_dict() for scenario in scenarios
    ])
    if actual_matrix_hash != expected_matrix_hash:
        raise RuntimeError("scenario matrix differs from the frozen protocol")
    if int(configuration["missions_per_scenario"]) <= 0:
        raise ValueError("missions_per_scenario must be positive")
    if float(configuration["duration_s"]) <= 12.0:
        raise ValueError("duration_s must exceed the final fault interval")
    if float(configuration["dt_s"]) <= 0.0:
        raise ValueError("dt_s must be positive")
    return configuration, output_dir, protocol_hash


def save_csv(rows, path):
    columns = (
        "scenario",
        "category",
        "sensor",
        "truth_mode",
        "seed",
        "disturbance_scale",
        "confirmed_event_count",
        "target_confirmed_event_count",
        "exact_event_detected",
        "conflicting_confirmed_event_count",
        "possible_evidence_observed",
        "any_fault_evidence_observed",
        "expected_ftc_action",
        "correct_ftc_action_observed",
        "protective_action_observed",
        "post_recovery_protective_action_observed",
        "wrong_thruster_target_count",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    scenarios = default_sensor_fault_stress_scenarios()
    configuration, output_dir, protocol_hash = validate_protocol(
        protocol, protocol_path, scenarios
    )
    missions_per_scenario = int(configuration["missions_per_scenario"])
    duration = float(configuration["duration_s"])
    dt = float(configuration["dt_s"])
    base_seed = int(configuration["base_seed"])

    rows = []
    for scenario_index, scenario in enumerate(scenarios):
        for repetition in range(missions_per_scenario):
            seed = base_seed + scenario_index * 1000 + repetition
            row = run_mission(
                scenario,
                duration=duration,
                dt=dt,
                seed=seed,
            )
            rows.append(row)
            print(
                f"{scenario.name:<44} seed={seed} "
                f"exact={row['exact_event_detected']} "
                f"evidence={row['any_fault_evidence_observed']} "
                f"conflict={row['conflicting_confirmed_event_count']}"
            )

    summary = summarize_sensor_fault_stress_benchmark(rows)
    payload = {
        "benchmark": "six_dof_sensor_fault_stress_v1",
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "configuration": configuration,
        "scenario_matrix": [scenario.as_dict() for scenario in scenarios],
        "summary": summary,
        "missions": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "sensor_fault_stress_summary.json"
    csv_path = output_dir / "sensor_fault_stress_missions.csv"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_csv(rows, csv_path)
    print("\nFrozen V1 summary")
    for key in (
        "strong_direct_event_recall",
        "strong_direct_event_precision",
        "strong_direct_ftc_action_match_rate",
        "normal_false_confirmed_missions",
        "normal_false_protective_missions",
        "ambiguous_fault_evidence_observation_rate",
        "ambiguous_possible_only_evidence_rate",
        "ambiguous_exact_classification_rate",
        "ambiguous_conflicting_certainty_rate",
        "ambiguous_post_recovery_protective_missions",
        "wrong_thruster_target_count",
        "all_acceptance_checks_passed",
    ):
        print(f"  {key}: {summary[key]}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
