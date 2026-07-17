"""Validate the deployed vertical FTC gate with new paired scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT / "examples", SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_six_dof_thruster_stratified as v1
from evaluation.protocol import prepare_locked_protocol
from ftc.safety_supervisor import FTCSupervisorConfig


DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT / "docs" / "six_dof_thruster_stratified_protocol_v2.json"
)


def validate_protocol(protocol, protocol_path):
    configuration, output_dir, protocol_hash = prepare_locked_protocol(
        protocol,
        protocol_path,
        REPOSITORY_ROOT,
        "six_dof_thruster_stratified_v2",
        output_message="locked benchmark output already exists",
    )
    if tuple(configuration["thrusters"]) != v1.THRUSTER_NAMES:
        raise ValueError("all six thrusters must be declared in layout order")
    if int(configuration["replicate_count"]) < 3:
        raise ValueError("at least three paired replicates are required")
    return configuration, output_dir, protocol_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    configuration, output_dir, protocol_hash = validate_protocol(
        protocol, protocol_path
    )
    strategy_configs = {
        item["name"]: v1._strategy_config(item)
        for item in configuration["strategies"]
    }
    deployed_name = configuration["candidate_strategy"]
    if strategy_configs[deployed_name] != FTCSupervisorConfig():
        raise RuntimeError(
            "candidate strategy no longer matches the deployed default config"
        )
    baseline_config = strategy_configs[configuration["baseline_strategy"]]
    rows = []
    starts = configuration["fault_start_times_s"]
    for context_index, context in enumerate(configuration["contexts"]):
        for replicate in range(int(configuration["replicate_count"])):
            seed = int(configuration["base_seed"]) + 100 * context_index + replicate
            fault_start = float(starts[replicate])
            for thruster in v1.THRUSTER_NAMES:
                scenario_id = f"{context}:r{replicate}:{thruster}"
                logs = v1._run_logs(
                    thruster_name=thruster,
                    fault_start_s=fault_start,
                    context=context,
                    seed=seed,
                    configuration=configuration,
                    baseline_config=baseline_config,
                )
                for strategy_name, strategy_config in strategy_configs.items():
                    row = v1._replay_strategy(
                        logs,
                        strategy_name=strategy_name,
                        config=strategy_config,
                        truth_thruster=thruster,
                        fault_start_s=fault_start,
                        scenario_id=scenario_id,
                    )
                    row.update({
                        "context": context,
                        "replicate": replicate,
                        "seed": seed,
                    })
                    rows.append(row)
            healthy_id = f"{context}:r{replicate}:healthy"
            healthy_logs = v1._run_logs(
                thruster_name=None,
                fault_start_s=fault_start,
                context=context,
                seed=seed,
                configuration=configuration,
                baseline_config=baseline_config,
            )
            for strategy_name, strategy_config in strategy_configs.items():
                row = v1._replay_strategy(
                    healthy_logs,
                    strategy_name=strategy_name,
                    config=strategy_config,
                    truth_thruster=None,
                    fault_start_s=fault_start,
                    scenario_id=healthy_id,
                )
                row.update({
                    "context": context,
                    "replicate": replicate,
                    "seed": seed,
                })
                rows.append(row)
            print(f"completed context={context} replicate={replicate + 1}")

    summary = v1.summarize(
        rows, configuration, protocol["acceptance_thresholds"]
    )
    payload = {
        "benchmark": "six_dof_thruster_stratified_v2",
        "evaluation_type": "paired_stratified_deployed_config_validation",
        "real_sea_trial_claim": False,
        "independent_blind_test_claim": False,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "configuration": configuration,
        "deployed_config_verified": True,
        "summary": summary,
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "thruster_stratified_v2_summary.json"
    csv_path = output_dir / "thruster_stratified_v2_rows.csv"
    plot_path = output_dir / "thruster_stratified_v2_delay.png"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    v1.save_csv(rows, csv_path)
    v1.save_plot(rows, summary, configuration, plot_path)
    print(json.dumps(summary, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
