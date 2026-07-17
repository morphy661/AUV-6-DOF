"""Run the hash-locked one-shot V2 sensor observation benchmark."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluate_six_dof_sensor_fault_observer_development import (
    run_mission,
    save_csv,
)
from evaluate_six_dof_sensor_fault_stress import (
    canonical_sha256,
    sha256_file,
)
from evaluation.sensor_fault_observer_benchmark import (
    summarize_sensor_fault_observer_benchmark,
)
from evaluation.sensor_fault_stress_benchmark import (
    default_sensor_fault_stress_scenarios,
)


DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT
    / "docs"
    / "six_dof_sensor_fault_observer_protocol_v2.json"
)


def validate_protocol(protocol, protocol_path, scenarios):
    if protocol.get("protocol_id") != "six_dof_sensor_fault_observer_v2":
        raise ValueError("unexpected protocol_id")
    if not protocol.get("locked_before_execution", False):
        raise ValueError("protocol is not marked as locked")
    actual_matrix_hash = canonical_sha256([
        scenario.as_dict() for scenario in scenarios
    ])
    if actual_matrix_hash != protocol.get("scenario_matrix_sha256"):
        raise RuntimeError("scenario matrix differs from the V2 protocol")
    for relative, expected_hash in protocol.get("code_sha256", {}).items():
        path = REPOSITORY_ROOT / relative
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"code changed after V2 freeze: {relative}")
    configuration = protocol["configuration"]
    if int(configuration["missions_per_scenario"]) <= 0:
        raise ValueError("missions_per_scenario must be positive")
    if float(configuration["duration_s"]) <= 12.0:
        raise ValueError("duration_s must exceed the final fault interval")
    if float(configuration["dt_s"]) <= 0.0:
        raise ValueError("dt_s must be positive")
    expected_missions = (
        len(scenarios) * int(configuration["missions_per_scenario"])
    )
    if int(configuration["mission_count"]) != expected_missions:
        raise ValueError("mission_count does not match the scenario matrix")
    output_dir = REPOSITORY_ROOT / protocol["output_directory"]
    if output_dir.exists():
        raise FileExistsError(
            "frozen V2 output already exists; it cannot be overwritten"
        )
    return configuration, output_dir, sha256_file(protocol_path)


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
                f"tier={row['display_tier']:<9} "
                f"raw={row['observer_possible_evidence_observed']} "
                f"operator={row['observer_operator_possible_evidence_observed']} "
                f"conflict={row['conflicting_confirmed_event_count']}"
            )

    summary = summarize_sensor_fault_observer_benchmark(rows)
    payload = {
        "benchmark": "six_dof_sensor_fault_observer_v2",
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "configuration": configuration,
        "scenario_matrix": [scenario.as_dict() for scenario in scenarios],
        "summary": summary,
        "missions": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "sensor_fault_observer_v2_summary.json"
    csv_path = output_dir / "sensor_fault_observer_v2_missions.csv"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_csv(rows, csv_path)

    print("\nFrozen V2 summary")
    for key, value in summary.items():
        if key not in ("per_scenario", "acceptance_checks"):
            print(f"  {key}: {value}")
    print(f"  acceptance_checks: {summary['acceptance_checks']}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
