"""Locked paired stress test for ESC telemetry validity and freshness gates."""

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
EXAMPLES_ROOT = PROJECT_ROOT / "examples"
for path in (SRC_ROOT, EXAMPLES_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_six_dof_thruster_stratified as stratified
from actuators.esc_telemetry_faults import ESCTelemetryFaultInjector
from evaluation.protocol import prepare_locked_protocol
from ftc.safety_supervisor import (
    FTCAction,
    FTCSafetySupervisor,
    FTCSupervisorConfig,
    build_rule_based_ftc_evidence,
)


THRUSTER_NAMES = ("H1", "H2", "H3", "H4", "V1", "V2")
DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT / "docs" / "six_dof_esc_telemetry_stress_protocol_v1.json"
)
COMMUNICATION_EVENTS = (
    "continuous_packet_loss",
    "communication_freeze",
)
BENIGN_TELEMETRY_EVENTS = (
    "bus_voltage_dip",
    "quantization",
)
ALL_STRESS_EVENTS = COMMUNICATION_EVENTS + BENIGN_TELEMETRY_EVENTS


def validate_protocol(protocol, protocol_path):
    configuration, output_dir, protocol_hash = prepare_locked_protocol(
        protocol,
        protocol_path,
        REPOSITORY_ROOT,
        (
            "six_dof_esc_telemetry_stress_v1",
            "six_dof_esc_telemetry_stress_v2",
        ),
        output_message="locked benchmark output already exists",
    )
    if tuple(configuration["thrusters"]) != THRUSTER_NAMES:
        raise ValueError("all six thrusters must be declared in layout order")
    if tuple(configuration["stress_events"]) != ALL_STRESS_EVENTS:
        raise ValueError("unexpected ESC telemetry stress event declaration")
    if int(configuration["replicate_count"]) < 3:
        raise ValueError("at least three paired replicates are required")
    return configuration, output_dir, protocol_hash


def strategy_config(strategy):
    return FTCSupervisorConfig(
        minimum_excitation_ratio=float(
            strategy.get("minimum_excitation_ratio", 0.20)
        ),
        vertical_minimum_excitation_ratio=float(
            strategy.get("vertical_minimum_excitation_ratio", 0.08)
        ),
        require_fresh_esc_telemetry=bool(
            strategy["require_fresh_esc_telemetry"]
        ),
        maximum_esc_telemetry_age_s=float(
            strategy.get("maximum_esc_telemetry_age_s", 0.20)
        ),
    )


def inject_telemetry_stress(
    logs,
    *,
    event_name,
    thruster_name,
    start_s,
    duration_s,
    configuration,
):
    """Return copied logs with one causal, observable ESC telemetry stress."""

    event = {
        "event_id": f"stress_{event_name}_{thruster_name}",
        "thruster_name": thruster_name,
        "mode": event_name,
        "start_time_s": float(start_s),
        "end_time_s": float(start_s) + float(duration_s),
        "signal_scale": float(configuration["bus_voltage_dip_signal_scale"]),
        "voltage_scale": float(
            configuration["bus_voltage_dip_voltage_scale"]
        ),
        "current_step_a": float(configuration["current_quantization_a"]),
        "rpm_step": float(configuration["rpm_quantization"]),
    }
    return ESCTelemetryFaultInjector([event]).apply_logs(logs)


def replay(
    logs,
    *,
    strategy_name,
    config,
    scenario_id,
    scenario_type,
    event_name,
    affected_thruster,
    event_start_s,
    event_duration_s,
):
    supervisor = FTCSafetySupervisor(config)
    first_target_time = None
    first_target_name = None
    correct_target_time = None
    communication_logged = False
    untrusted_observed = False
    max_affected_score = 0.0
    affected_index = THRUSTER_NAMES.index(affected_thruster)
    event_stop_s = float(event_start_s) + float(event_duration_s)
    for log in logs:
        evidence = build_rule_based_ftc_evidence(log, config=config)
        decision = supervisor.update(evidence)
        time_s = float(log["time"])
        in_event = float(event_start_s) <= time_s <= event_stop_s
        if in_event:
            max_affected_score = max(
                max_affected_score,
                float(evidence.no_output_scores[affected_index]),
            )
            channel_untrusted = (
                affected_thruster in evidence.untrusted_esc_channels
            )
            untrusted_observed = untrusted_observed or channel_untrusted
            communication_logged = communication_logged or bool(
                channel_untrusted
                and decision.action == FTCAction.LOG_ONLY
                and "ESC telemetry" in decision.reason
            )
        if decision.targeted_thruster_name is not None:
            if first_target_time is None:
                first_target_time = time_s
                first_target_name = decision.targeted_thruster_name
            if (
                scenario_type == "real_no_output"
                and decision.targeted_thruster_name == affected_thruster
                and correct_target_time is None
            ):
                correct_target_time = time_s
    false_target = bool(
        scenario_type != "real_no_output" and first_target_name is not None
    )
    wrong_target = bool(
        scenario_type == "real_no_output"
        and first_target_name is not None
        and first_target_name != affected_thruster
    )
    return {
        "scenario_id": scenario_id,
        "strategy": strategy_name,
        "scenario_type": scenario_type,
        "event_name": event_name,
        "affected_thruster": affected_thruster,
        "event_start_s": float(event_start_s),
        "event_duration_s": float(event_duration_s),
        "any_target_observed": first_target_name is not None,
        "first_target_name": first_target_name,
        "target_delay_s": (
            None
            if first_target_time is None
            else float(first_target_time - event_start_s)
        ),
        "false_target_observed": false_target,
        "correct_target_observed": correct_target_time is not None,
        "correct_target_delay_s": (
            None
            if correct_target_time is None
            else float(correct_target_time - event_start_s)
        ),
        "wrong_target_observed": wrong_target,
        "untrusted_channel_observed": untrusted_observed,
        "communication_log_observed": communication_logged,
        "max_affected_no_output_score": max_affected_score,
    }


def rate(rows, key):
    return None if not rows else mean(bool(row[key]) for row in rows)


def median_value(rows, key):
    values = [float(row[key]) for row in rows if row[key] is not None]
    return None if not values else median(values)


def summarize(rows, configuration, thresholds):
    strategy_metrics = {}
    for strategy in configuration["strategies"]:
        name = strategy["name"]
        selected = [row for row in rows if row["strategy"] == name]
        by_event = {}
        for event_name in ALL_STRESS_EVENTS + ("real_no_output",):
            subset = [
                row for row in selected if row["event_name"] == event_name
            ]
            by_event[event_name] = {
                "mission_count": len(subset),
                "false_target_rate": rate(subset, "false_target_observed"),
                "correct_target_recall": rate(
                    subset, "correct_target_observed"
                ),
                "wrong_target_rate": rate(subset, "wrong_target_observed"),
                "untrusted_channel_observed_rate": rate(
                    subset, "untrusted_channel_observed"
                ),
                "communication_log_rate": rate(
                    subset, "communication_log_observed"
                ),
                "median_target_delay_s": median_value(
                    subset, "correct_target_delay_s"
                ),
                "median_max_no_output_score": median_value(
                    subset, "max_affected_no_output_score"
                ),
            }
        telemetry_stress = [
            row for row in selected if row["scenario_type"] == "telemetry_stress"
        ]
        real_fault = [
            row for row in selected if row["scenario_type"] == "real_no_output"
        ]
        strategy_metrics[name] = {
            "telemetry_stress_mission_count": len(telemetry_stress),
            "real_fault_mission_count": len(real_fault),
            "overall_false_target_rate": rate(
                telemetry_stress, "false_target_observed"
            ),
            "real_fault_target_recall": rate(
                real_fault, "correct_target_observed"
            ),
            "real_fault_wrong_target_rate": rate(
                real_fault, "wrong_target_observed"
            ),
            "real_fault_median_target_delay_s": median_value(
                real_fault, "correct_target_delay_s"
            ),
            "by_event": by_event,
        }

    legacy = strategy_metrics[configuration["legacy_strategy"]]
    deployed = strategy_metrics[configuration["deployed_strategy"]]
    deployed_events = deployed["by_event"]
    legacy_events = legacy["by_event"]
    checks = {
        "legacy_continuous_loss_vulnerability_reproduced": (
            legacy_events["continuous_packet_loss"]["false_target_rate"]
            >= thresholds["legacy_continuous_loss_false_target_rate_min"]
        ),
        "deployed_zero_false_targets_all_telemetry_stress": (
            deployed["overall_false_target_rate"]
            <= thresholds["deployed_overall_false_target_rate_max"]
        ),
        "deployed_logs_all_communication_anomalies": all(
            deployed_events[event]["untrusted_channel_observed_rate"]
            >= thresholds["deployed_communication_record_rate_min"]
            for event in COMMUNICATION_EVENTS
        ),
        "deployed_preserves_real_no_output_recall": (
            deployed["real_fault_target_recall"]
            >= thresholds["deployed_real_fault_target_recall_min"]
        ),
        "deployed_has_zero_wrong_real_fault_targets": (
            deployed["real_fault_wrong_target_rate"]
            <= thresholds["deployed_real_fault_wrong_target_rate_max"]
        ),
        "deployed_real_fault_delay_within_limit": (
            deployed["real_fault_median_target_delay_s"]
            <= thresholds["deployed_real_fault_median_delay_s_max"]
        ),
        "benign_voltage_and_quantization_never_isolate": all(
            deployed_events[event]["false_target_rate"] == 0.0
            for event in BENIGN_TELEMETRY_EVENTS
        ),
    }
    return {
        "paired_scenario_count": len(rows) // len(configuration["strategies"]),
        "strategy_metrics": strategy_metrics,
        "acceptance_thresholds": thresholds,
        "acceptance_checks": checks,
        "all_acceptance_checks_passed": all(checks.values()),
    }


def save_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(summary, configuration, path):
    strategies = [item["name"] for item in configuration["strategies"]]
    events = ALL_STRESS_EVENTS + ("real_no_output",)
    labels = (
        "Continuous\npacket loss",
        "Communication\nfreeze",
        "Bus voltage\ndip",
        "Quantization",
        "Real\nno-output",
    )
    figure, axis = plt.subplots(figsize=(11.5, 5.6))
    figure.patch.set_facecolor("#07131f")
    axis.set_facecolor("#0d2031")
    x = np.arange(len(events))
    width = 0.34
    colors = ("#f1b94b", "#31c7b5")
    for offset, (strategy, color) in enumerate(zip(strategies, colors)):
        metrics = summary["strategy_metrics"][strategy]["by_event"]
        values = []
        for event in events:
            key = (
                "correct_target_recall"
                if event == "real_no_output"
                else "false_target_rate"
            )
            values.append(float(metrics[event][key]))
        bars = axis.bar(
            x + (offset - 0.5) * width,
            np.asarray(values) * 100.0,
            width,
            label=strategy,
            color=color,
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value * 100.0 + 2.0,
                f"{value * 100.0:.0f}%",
                ha="center",
                va="bottom",
                color="#eef6fb",
                fontsize=8,
            )
    axis.set_xticks(x, labels, color="#eef6fb")
    axis.set_ylim(0.0, 112.0)
    axis.set_ylabel("Rate (%)", color="#91a8ba")
    axis.set_title(
        "False isolation under ESC telemetry stress; recall for real fault",
        color="#eef6fb",
        loc="left",
    )
    axis.grid(axis="y", color="#294256", alpha=0.7)
    axis.tick_params(colors="#91a8ba")
    for spine in axis.spines.values():
        spine.set_color("#294256")
    legend = axis.legend(frameon=False, loc="upper right")
    for text in legend.get_texts():
        text.set_color("#eef6fb")
    figure.tight_layout()
    figure.savefig(path, dpi=170, facecolor=figure.get_facecolor())
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    configuration, output_dir, protocol_hash = validate_protocol(
        protocol, protocol_path
    )
    strategies = {
        item["name"]: strategy_config(item)
        for item in configuration["strategies"]
    }
    deployed_name = configuration["deployed_strategy"]
    if strategies[deployed_name] != FTCSupervisorConfig():
        raise RuntimeError(
            "deployed strategy no longer matches FTCSupervisorConfig defaults"
        )
    simulation_config = strategies[deployed_name]
    rows = []
    starts = configuration["event_start_times_s"]
    for context_index, context in enumerate(configuration["contexts"]):
        for replicate in range(int(configuration["replicate_count"])):
            seed = int(configuration["base_seed"]) + 100 * context_index + replicate
            event_start_s = float(starts[replicate])
            healthy_logs = stratified._run_logs(
                thruster_name=None,
                fault_start_s=event_start_s,
                context=context,
                seed=seed,
                configuration=configuration,
                baseline_config=simulation_config,
            )
            for thruster_name in THRUSTER_NAMES:
                for event_name in ALL_STRESS_EVENTS:
                    duration_s = float(
                        configuration["event_durations_s"][event_name]
                    )
                    stressed_logs = inject_telemetry_stress(
                        healthy_logs,
                        event_name=event_name,
                        thruster_name=thruster_name,
                        start_s=event_start_s,
                        duration_s=duration_s,
                        configuration=configuration,
                    )
                    scenario_id = (
                        f"{context}:r{replicate}:{thruster_name}:{event_name}"
                    )
                    for strategy_name, config in strategies.items():
                        row = replay(
                            stressed_logs,
                            strategy_name=strategy_name,
                            config=config,
                            scenario_id=scenario_id,
                            scenario_type="telemetry_stress",
                            event_name=event_name,
                            affected_thruster=thruster_name,
                            event_start_s=event_start_s,
                            event_duration_s=duration_s,
                        )
                        row.update({
                            "context": context,
                            "replicate": replicate,
                            "seed": seed,
                        })
                        rows.append(row)

                fault_logs = stratified._run_logs(
                    thruster_name=thruster_name,
                    fault_start_s=event_start_s,
                    context=context,
                    seed=seed,
                    configuration=configuration,
                    baseline_config=simulation_config,
                )
                duration_s = float(configuration["duration_s"]) - event_start_s
                scenario_id = (
                    f"{context}:r{replicate}:{thruster_name}:real_no_output"
                )
                for strategy_name, config in strategies.items():
                    row = replay(
                        fault_logs,
                        strategy_name=strategy_name,
                        config=config,
                        scenario_id=scenario_id,
                        scenario_type="real_no_output",
                        event_name="real_no_output",
                        affected_thruster=thruster_name,
                        event_start_s=event_start_s,
                        event_duration_s=duration_s,
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
        "benchmark": protocol["protocol_id"],
        "evaluation_type": "paired_locked_esc_telemetry_stress",
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
    json_path = output_dir / "esc_telemetry_stress_summary.json"
    csv_path = output_dir / "esc_telemetry_stress_rows.csv"
    plot_path = output_dir / "esc_telemetry_stress.png"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_csv(rows, csv_path)
    save_plot(summary, configuration, plot_path)
    print(json.dumps(summary, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
