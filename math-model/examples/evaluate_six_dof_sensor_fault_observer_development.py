"""Development-only evaluation of the ambiguous sensor observation layer."""

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from environment.six_dof_simulator import SixDOFSimulator
from evaluate_six_dof_sensor_fault_stress import (
    disturbance_provider,
    sensor_suite,
    target_provider,
)
from evaluation.sensor_fault_observer_benchmark import (
    evaluate_sensor_fault_observer_mission,
    summarize_sensor_fault_observer_benchmark,
)
from evaluation.sensor_fault_stress_benchmark import (
    default_sensor_fault_stress_scenarios,
)
from ftc.safety_supervisor import FTCSafetySupervisor


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
    row = evaluate_sensor_fault_observer_mission(logs, scenario)
    row.update({
        "seed": seed,
        "sensor_parameters": sensor_parameters,
        "disturbance_parameters": disturbance_parameters,
    })
    return row


def save_csv(rows, path):
    columns = (
        "scenario",
        "category",
        "sensor",
        "truth_mode",
        "seed",
        "display_tier",
        "exact_event_detected",
        "observer_possible_evidence_observed",
        "observer_operator_possible_evidence_observed",
        "observer_possible_hypotheses",
        "observer_all_possible_hypotheses",
        "observer_operator_possible_sample_count",
        "conflicting_confirmed_event_count",
        "observer_confirmed_count",
        "observer_protective_count",
        "wrong_thruster_target_count",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--missions-per-scenario", type=int, default=2)
    parser.add_argument("--duration", type=float, default=18.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026071800)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "six_dof_sensor_fault_observer_development"
        ),
    )
    args = parser.parse_args()
    if args.missions_per_scenario <= 0:
        raise ValueError("missions-per-scenario must be positive")
    scenarios = default_sensor_fault_stress_scenarios()
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
                f"{scenario.name:<44} seed={seed} "
                f"tier={row['display_tier']:<9} "
                f"possible={row['observer_possible_evidence_observed']} "
                f"conflict={row['conflicting_confirmed_event_count']}"
            )

    summary = summarize_sensor_fault_observer_benchmark(rows)
    payload = {
        "benchmark": "sensor_fault_observer_development_not_blind",
        "configuration": {
            "missions_per_scenario": args.missions_per_scenario,
            "duration_s": args.duration,
            "dt_s": args.dt,
            "base_seed": args.seed,
            "scenario_matrix": [item.as_dict() for item in scenarios],
        },
        "summary": summary,
        "missions": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "sensor_fault_observer_summary.json"
    csv_path = args.output_dir / "sensor_fault_observer_missions.csv"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_csv(rows, csv_path)
    print("\nDevelopment summary")
    for key, value in summary.items():
        if key not in ("per_scenario", "acceptance_checks"):
            print(f"  {key}: {value}")
    print(f"  acceptance_checks: {summary['acceptance_checks']}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
