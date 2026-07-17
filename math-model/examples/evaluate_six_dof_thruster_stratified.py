"""Run a paired, balanced six-thruster no-output FTC latency benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from actuators.six_dof_thruster_faults import (
    SingleThrusterFault,
    SixDOFThrusterFaultMode,
    ThrusterActuatorBank,
)
from actuators.thruster_array import default_six_thruster_array
from demo_six_dof_unified_diagnostics import (
    disturbance_provider,
    random_fault_schedule,
    target_provider,
)
from environment.six_dof_simulator import SixDOFSimulator
from evaluation.protocol import prepare_locked_protocol
from ftc.safety_supervisor import (
    FTCSafetySupervisor,
    FTCSupervisorConfig,
    build_rule_based_ftc_evidence,
)
from sensors.sensor_faults import SensorFaultInjector
from sensors.six_dof_sensor_suite import SixDOFSensorSuite


DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT / "docs" / "six_dof_thruster_stratified_protocol_v1.json"
)
THRUSTER_NAMES = ("H1", "H2", "H3", "H4", "V1", "V2")


def validate_protocol(protocol, protocol_path):
    configuration, output_dir, protocol_hash = prepare_locked_protocol(
        protocol,
        protocol_path,
        REPOSITORY_ROOT,
        "six_dof_thruster_stratified_v1",
        output_message="locked benchmark output already exists",
    )
    if tuple(configuration["thrusters"]) != THRUSTER_NAMES:
        raise ValueError("all six thrusters must be declared in layout order")
    if int(configuration["replicate_count"]) < 3:
        raise ValueError("at least three paired replicates are required")
    return configuration, output_dir, protocol_hash


def _strategy_config(strategy):
    vertical = strategy.get("vertical_minimum_excitation_ratio")
    return FTCSupervisorConfig(
        minimum_excitation_ratio=float(strategy["minimum_excitation_ratio"]),
        vertical_minimum_excitation_ratio=(
            None if vertical is None else float(vertical)
        ),
    )


def _sensor_suite(context, seed):
    if context == "clean":
        return None
    if context != "sensor_stress":
        raise ValueError(f"unknown context: {context}")
    events, _ = random_fault_schedule(seed)
    return SixDOFSensorSuite(
        fault_injector=SensorFaultInjector(events), seed=seed
    )


def _run_logs(
    *, thruster_name, fault_start_s, context, seed, configuration, baseline_config
):
    array = default_six_thruster_array()
    fault = None if thruster_name is None else SingleThrusterFault(
        thruster_name,
        SixDOFThrusterFaultMode.NO_OUTPUT,
        start_time=float(fault_start_s),
    )
    noise = configuration["esc_noise"]
    bank = ThrusterActuatorBank(
        array,
        fault=fault,
        current_noise_std=float(noise["current_noise_std_a"]),
        rpm_noise_std=float(noise["rpm_noise_std"]),
        voltage_noise_std=float(noise["voltage_noise_std_v"]),
        temperature_noise_std=float(noise["temperature_noise_std_c"]),
        seed=seed,
    )
    simulator = SixDOFSimulator(
        thruster_array=array,
        actuator_bank=bank,
        sensor_suite=_sensor_suite(context, seed),
        ftc_supervisor=FTCSafetySupervisor(baseline_config),
    )
    return simulator.run(
        float(configuration["duration_s"]),
        float(configuration["dt_s"]),
        target_provider,
        disturbance_provider=disturbance_provider,
    )


def _first_time(logs, predicate):
    for log in logs:
        if predicate(log):
            return float(log["time"])
    return None


def _replay_strategy(
    logs, *, strategy_name, config, truth_thruster, fault_start_s, scenario_id
):
    supervisor = FTCSafetySupervisor(config)
    truth_index = (
        None if truth_thruster is None else THRUSTER_NAMES.index(truth_thruster)
    )
    evidence_time = None
    target_time = None
    target_names = []
    max_pre_evidence_excitation = 0.0
    for log in logs:
        evidence = build_rule_based_ftc_evidence(log, config=config)
        time_s = float(log["time"])
        if truth_index is not None and time_s >= float(fault_start_s):
            if evidence_time is None:
                max_pre_evidence_excitation = max(
                    max_pre_evidence_excitation,
                    float(evidence.excitation_ratios[truth_index]),
                )
                if (
                    evidence.no_output_scores[truth_index]
                    >= config.no_output_score_threshold
                ):
                    evidence_time = time_s
            elif evidence_time is not None:
                max_pre_evidence_excitation = max(
                    max_pre_evidence_excitation,
                    float(evidence.excitation_ratios[truth_index]),
                )
        decision = supervisor.update(evidence)
        if decision.targeted_thruster_name is not None:
            target_names.append(decision.targeted_thruster_name)
            if (
                truth_thruster is not None
                and decision.targeted_thruster_name == truth_thruster
                and target_time is None
            ):
                target_time = time_s

    wrong_target = any(
        truth_thruster is None or name != truth_thruster
        for name in target_names
    )
    return {
        "scenario_id": scenario_id,
        "strategy": strategy_name,
        "truth_thruster": truth_thruster or "healthy",
        "fault_start_s": (
            None if truth_thruster is None else float(fault_start_s)
        ),
        "direct_evidence_observed": evidence_time is not None,
        "direct_evidence_delay_s": (
            None if evidence_time is None else evidence_time - fault_start_s
        ),
        "correct_target_observed": target_time is not None,
        "target_delay_s": (
            None if target_time is None else target_time - fault_start_s
        ),
        "confirmation_delay_s": (
            None
            if target_time is None or evidence_time is None
            else target_time - evidence_time
        ),
        "wrong_target_observed": bool(wrong_target),
        "any_target_observed": bool(target_names),
        "first_target_name": None if not target_names else target_names[0],
        "max_pre_evidence_excitation_ratio": max_pre_evidence_excitation,
    }


def _rate(rows, key):
    return mean(bool(row[key]) for row in rows) if rows else None


def _median(rows, key):
    values = [float(row[key]) for row in rows if row[key] is not None]
    return None if not values else median(values)


def _percentile(rows, key, percentile):
    values = [float(row[key]) for row in rows if row[key] is not None]
    return None if not values else float(np.percentile(values, percentile))


def summarize(rows, configuration, thresholds):
    fault_rows = [row for row in rows if row["truth_thruster"] != "healthy"]
    healthy_rows = [row for row in rows if row["truth_thruster"] == "healthy"]
    summaries = {}
    for strategy in configuration["strategies"]:
        name = strategy["name"]
        selected = [row for row in fault_rows if row["strategy"] == name]
        healthy = [row for row in healthy_rows if row["strategy"] == name]
        per_thruster = {}
        for thruster in THRUSTER_NAMES:
            subset = [row for row in selected if row["truth_thruster"] == thruster]
            per_thruster[thruster] = {
                "mission_count": len(subset),
                "target_recall": _rate(subset, "correct_target_observed"),
                "median_evidence_delay_s": _median(
                    subset, "direct_evidence_delay_s"
                ),
                "median_target_delay_s": _median(subset, "target_delay_s"),
                "p90_target_delay_s": _percentile(
                    subset, "target_delay_s", 90
                ),
                "median_confirmation_delay_s": _median(
                    subset, "confirmation_delay_s"
                ),
            }
        horizontal = [
            row for row in selected if row["truth_thruster"].startswith("H")
        ]
        vertical = [
            row for row in selected if row["truth_thruster"].startswith("V")
        ]
        sensor_stress = [
            row for row in selected if row["context"] == "sensor_stress"
        ]
        summaries[name] = {
            "fault_mission_count": len(selected),
            "healthy_mission_count": len(healthy),
            "target_recall": _rate(selected, "correct_target_observed"),
            "wrong_target_mission_rate": _rate(
                selected, "wrong_target_observed"
            ),
            "healthy_false_target_rate": _rate(
                healthy, "any_target_observed"
            ),
            "sensor_stress_target_recall": _rate(
                sensor_stress, "correct_target_observed"
            ),
            "horizontal_median_target_delay_s": _median(
                horizontal, "target_delay_s"
            ),
            "vertical_median_target_delay_s": _median(
                vertical, "target_delay_s"
            ),
            "vertical_p90_target_delay_s": _percentile(
                vertical, "target_delay_s", 90
            ),
            "per_thruster": per_thruster,
        }

    baseline_name = configuration["baseline_strategy"]
    candidate_name = configuration["candidate_strategy"]
    baseline_rows = {
        row["scenario_id"]: row
        for row in fault_rows if row["strategy"] == baseline_name
    }
    candidate_rows = {
        row["scenario_id"]: row
        for row in fault_rows if row["strategy"] == candidate_name
    }
    vertical_improvements = []
    horizontal_differences = []
    for scenario_id, baseline in baseline_rows.items():
        candidate = candidate_rows[scenario_id]
        if baseline["target_delay_s"] is None or candidate["target_delay_s"] is None:
            continue
        difference = (
            float(baseline["target_delay_s"])
            - float(candidate["target_delay_s"])
        )
        if baseline["truth_thruster"].startswith("V"):
            vertical_improvements.append(difference)
        else:
            horizontal_differences.append(abs(difference))
    paired = {
        "median_vertical_target_delay_improvement_s": (
            None if not vertical_improvements else median(vertical_improvements)
        ),
        "minimum_vertical_target_delay_improvement_s": (
            None if not vertical_improvements else min(vertical_improvements)
        ),
        "maximum_horizontal_target_delay_change_s": (
            None if not horizontal_differences else max(horizontal_differences)
        ),
    }
    baseline = summaries[baseline_name]
    candidate = summaries[candidate_name]
    checks = {
        "baseline_target_recall_at_least": (
            baseline["target_recall"] >= thresholds["target_recall_min"]
        ),
        "candidate_target_recall_at_least": (
            candidate["target_recall"] >= thresholds["target_recall_min"]
        ),
        "candidate_sensor_stress_recall_at_least": (
            candidate["sensor_stress_target_recall"]
            >= thresholds["sensor_stress_target_recall_min"]
        ),
        "all_strategies_zero_wrong_targets": all(
            value["wrong_target_mission_rate"]
            <= thresholds["wrong_target_mission_rate_max"]
            for value in summaries.values()
        ),
        "all_strategies_zero_healthy_false_targets": all(
            value["healthy_false_target_rate"]
            <= thresholds["healthy_false_target_rate_max"]
            for value in summaries.values()
        ),
        "candidate_vertical_median_delay_at_most": (
            candidate["vertical_median_target_delay_s"]
            <= thresholds["candidate_vertical_median_target_delay_s_max"]
        ),
        "paired_vertical_improvement_at_least": (
            paired["median_vertical_target_delay_improvement_s"]
            >= thresholds["paired_vertical_median_improvement_s_min"]
        ),
        "horizontal_delay_unchanged": (
            paired["maximum_horizontal_target_delay_change_s"]
            <= thresholds["horizontal_target_delay_change_s_max"]
        ),
    }
    return {
        "fault_mission_count": len(fault_rows) // len(configuration["strategies"]),
        "healthy_mission_count": len(healthy_rows) // len(configuration["strategies"]),
        "strategy_metrics": summaries,
        "paired_comparison": paired,
        "acceptance_thresholds": thresholds,
        "acceptance_checks": checks,
        "all_acceptance_checks_passed": all(checks.values()),
    }


def save_plot(rows, summary, configuration, path):
    strategies = [item["name"] for item in configuration["strategies"]]
    contexts = configuration["contexts"]
    colors = ("#607d95", "#f1b94b", "#31c7b5")
    figure, axes = plt.subplots(1, len(contexts), figsize=(14.0, 5.8), sharey=True)
    if len(contexts) == 1:
        axes = [axes]
    figure.patch.set_facecolor("#07131f")
    width = 0.24
    x = np.arange(len(THRUSTER_NAMES))
    for axis, context in zip(axes, contexts):
        axis.set_facecolor("#0d2031")
        for strategy_index, (strategy, color) in enumerate(zip(strategies, colors)):
            offsets = x + (strategy_index - 1) * width
            medians = []
            for thruster in THRUSTER_NAMES:
                subset = [
                    row for row in rows
                    if row["strategy"] == strategy
                    and row["context"] == context
                    and row["truth_thruster"] == thruster
                ]
                medians.append(_median(subset, "target_delay_s"))
            bars = axis.bar(
                offsets, medians, width, label=strategy, color=color, alpha=0.9
            )
            for bar, value in zip(bars, medians):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.06,
                    f"{value:.2f}",
                    ha="center", va="bottom", color="#eef6fb", fontsize=7.5,
                )
        axis.set_xticks(x, THRUSTER_NAMES, color="#eef6fb")
        axis.set_title(context.replace("_", " "), color="#eef6fb", loc="left")
        axis.grid(axis="y", color="#294256", alpha=0.7)
        axis.tick_params(colors="#91a8ba")
        for spine in axis.spines.values():
            spine.set_color("#294256")
    axes[0].set_ylabel("Median correct FTC target delay (s)", color="#91a8ba")
    legend = axes[0].legend(frameon=False, loc="upper left")
    for value in legend.get_texts():
        value.set_color("#eef6fb")
    figure.suptitle(
        "Paired stratified no-output benchmark by thruster",
        color="#eef6fb", x=0.055, ha="left", fontsize=14,
    )
    figure.text(
        0.055, 0.01,
        "Equal replicates per thruster; ESC noise enabled; learned model excluded from FTC.",
        color="#91a8ba", fontsize=9,
    )
    figure.tight_layout(rect=(0.04, 0.05, 0.99, 0.92))
    figure.savefig(path, dpi=170, facecolor=figure.get_facecolor())
    plt.close(figure)


def save_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
        item["name"]: _strategy_config(item)
        for item in configuration["strategies"]
    }
    baseline_config = strategy_configs[configuration["baseline_strategy"]]
    rows = []
    starts = configuration["fault_start_times_s"]
    for context_index, context in enumerate(configuration["contexts"]):
        for replicate in range(int(configuration["replicate_count"])):
            seed = int(configuration["base_seed"]) + 100 * context_index + replicate
            fault_start = float(starts[replicate])
            for thruster in THRUSTER_NAMES:
                scenario_id = f"{context}:r{replicate}:{thruster}"
                logs = _run_logs(
                    thruster_name=thruster,
                    fault_start_s=fault_start,
                    context=context,
                    seed=seed,
                    configuration=configuration,
                    baseline_config=baseline_config,
                )
                for strategy_name, strategy_config in strategy_configs.items():
                    row = _replay_strategy(
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
            healthy_logs = _run_logs(
                thruster_name=None,
                fault_start_s=fault_start,
                context=context,
                seed=seed,
                configuration=configuration,
                baseline_config=baseline_config,
            )
            for strategy_name, strategy_config in strategy_configs.items():
                row = _replay_strategy(
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

    summary = summarize(
        rows, configuration, protocol["acceptance_thresholds"]
    )
    payload = {
        "benchmark": "six_dof_thruster_stratified_v1",
        "evaluation_type": "paired_stratified_development_benchmark",
        "real_sea_trial_claim": False,
        "independent_blind_test_claim": False,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "configuration": configuration,
        "summary": summary,
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "thruster_stratified_summary.json"
    csv_path = output_dir / "thruster_stratified_rows.csv"
    plot_path = output_dir / "thruster_stratified_delay.png"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_csv(rows, csv_path)
    save_plot(rows, summary, configuration, plot_path)
    print(json.dumps(summary, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
