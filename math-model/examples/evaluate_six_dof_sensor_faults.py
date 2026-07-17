"""Run the depth/IMU/DVL development benchmark with causal FTC feedback."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from environment.six_dof_simulator import SixDOFSimulator
from evaluation.sensor_fault_benchmark import (
    default_sensor_fault_scenarios,
    evaluate_sensor_fault_mission,
    summarize_sensor_fault_benchmark,
)
from ftc.safety_supervisor import FTCSafetySupervisor
from sensors.depth_sensor import DepthSensor
from sensors.dvl_sensor import DVLSensor
from sensors.imu_sensor import IMUSensor
from sensors.sensor_faults import SensorFaultInjector
from sensors.six_dof_sensor_suite import SixDOFSensorSuite
from simple_control.six_dof_controller import PoseTarget


def target_provider(time_s, _state):
    """Use multi-axis maneuvers so every stuck test is observable."""

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
    event = scenario.fault_event()
    parameters = {
        "depth_noise_std_m": float(rng.uniform(0.01, 0.08)),
        "depth_drift_std_m_per_sqrt_s": float(
            rng.uniform(0.0003, 0.0020)
        ),
        "imu_attitude_noise_std_rad": float(rng.uniform(0.0005, 0.005)),
        "imu_gyro_noise_std_radps": float(rng.uniform(0.0005, 0.003)),
        "imu_accel_noise_std_mps2": float(rng.uniform(0.005, 0.03)),
        "dvl_velocity_noise_std_mps": float(rng.uniform(0.005, 0.05)),
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
            velocity_noise_std=parameters[
                "dvl_velocity_noise_std_mps"
            ],
            dropout_prob=0.0,
            seed=seed + 3,
        ),
        fault_injector=SensorFaultInjector(
            () if event is None else (event,)
        ),
    )
    return suite, parameters


def disturbance_provider(seed):
    rng = np.random.default_rng(seed)
    amplitudes = rng.uniform(
        np.zeros(6),
        np.array([1.0, 1.0, 0.7, 0.02, 0.02, 0.08]),
    )
    frequencies = rng.uniform(0.05, 0.15, size=6)
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
    disturbance, disturbance_parameters = disturbance_provider(seed + 10)
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
    result = evaluate_sensor_fault_mission(logs, scenario)
    result.update({
        "seed": seed,
        "sensor_parameters": sensor_parameters,
        "disturbance_parameters": disturbance_parameters,
    })
    return result


def save_csv(rows, path):
    columns = (
        "scenario",
        "sensor",
        "mode",
        "seed",
        "exact_event_detected",
        "detection_delay_s",
        "confirmed_event_count",
        "spurious_event_count",
        "expected_ftc_action",
        "correct_ftc_action_observed",
        "sensor_health_recovered",
        "estimate_integrity_restored",
        "absolute_trajectory_recovered",
        "wrong_thruster_target_count",
        "protective_action_observed",
        "final_true_position_error_m",
        "maximum_estimation_position_error_m",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in columns})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--missions-per-scenario", type=int, default=5)
    parser.add_argument("--duration", type=float, default=18.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT / "results" / "six_dof_sensor_fault_benchmark"
        ),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.missions_per_scenario <= 0:
        raise ValueError("missions-per-scenario must be positive")
    if args.duration <= 10.0 or args.dt <= 0.0:
        raise ValueError("duration must exceed 10 s and dt must be positive")

    scenarios = default_sensor_fault_scenarios()
    rows = []
    for scenario_index, scenario in enumerate(scenarios):
        for repetition in range(args.missions_per_scenario):
            seed = args.seed + scenario_index * 1000 + repetition
            row = run_mission(
                scenario,
                duration=args.duration,
                dt=args.dt,
                seed=seed,
            )
            rows.append(row)
            print(
                f"{scenario.name:<22} seed={seed} "
                f"detected={row['exact_event_detected']} "
                f"delay={row['detection_delay_s']} "
                f"spurious={row['spurious_event_count']} "
                f"action_ok={row['correct_ftc_action_observed']}"
            )

    summary = summarize_sensor_fault_benchmark(rows)
    payload = {
        "benchmark": "six_dof_sensor_fault_development",
        "configuration": {
            "missions_per_scenario": args.missions_per_scenario,
            "duration_s": args.duration,
            "dt_s": args.dt,
            "base_seed": args.seed,
            "scenario_matrix": [
                scenario.as_dict() for scenario in scenarios
            ],
        },
        "summary": summary,
        "missions": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "sensor_fault_summary.json"
    mission_path = args.output_dir / "sensor_fault_missions.csv"
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_csv(rows, mission_path)

    print("\nSummary")
    for key in (
        "event_recall",
        "event_precision",
        "mean_detection_delay_s",
        "ftc_action_match_rate",
        "sensor_health_recovery_rate",
        "estimate_integrity_restoration_rate",
        "absolute_trajectory_recovery_rate",
        "normal_false_event_missions",
        "normal_false_protective_missions",
        "wrong_thruster_target_count",
        "maximum_estimation_position_error_m",
        "all_acceptance_checks_passed",
    ):
        print(f"  {key}: {summary[key]}")
    print(f"JSON: {summary_path}")
    print(f"CSV: {mission_path}")
    if args.strict and not summary["all_acceptance_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
