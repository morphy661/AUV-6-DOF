"""Evaluate V2 context-gated advice with new hash-locked random seeds."""

from __future__ import annotations

import argparse
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
MODEL_ROOT = (
    REPOSITORY_ROOT
    / "depth-sensor-fault-detection"
    / "depth_fault_detection"
)
for path in (PROJECT_ROOT / "examples", SRC_ROOT, MODEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_six_dof_unified_random_batch as v1
from demo_six_dof_unified_diagnostics import run_demo
from model_six_dof_multitask import AUVSixDOFMultiTaskDetector
from presentation.six_dof_model_bridge import SixDOFModelBridge


DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT
    / "docs"
    / "six_dof_unified_random_batch_protocol_v2.json"
)


def validate_protocol(protocol, protocol_path):
    if protocol.get("protocol_id") != "six_dof_unified_random_batch_v2":
        raise ValueError("unexpected protocol_id")
    if not protocol.get("locked_before_execution", False):
        raise ValueError("protocol is not locked")
    if protocol.get("evaluation_type") != "development_random_batch":
        raise ValueError("this script is only for the declared development batch")
    for relative, expected in protocol.get("code_sha256", {}).items():
        actual = v1.sha256_file(REPOSITORY_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"code hash mismatch: {relative}")
    for relative, expected in protocol.get("artifact_sha256", {}).items():
        actual = v1.sha256_file(REPOSITORY_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"artifact hash mismatch: {relative}")
    configuration = protocol["configuration"]
    count = int(configuration["mission_count"])
    base_seed = int(configuration["base_seed"])
    if count < 20:
        raise ValueError("development batch must contain at least 20 missions")
    seeds = list(range(base_seed, base_seed + count))
    output_dir = REPOSITORY_ROOT / protocol["output_directory"]
    if output_dir.exists():
        raise FileExistsError("locked batch output already exists")
    return configuration, seeds, output_dir, v1.sha256_file(protocol_path)


def _rate(rows, key):
    return mean(bool(row[key]) for row in rows) if rows else None


def evaluate_mission(seed, frames, manifest):
    row = v1.evaluate_mission(seed, frames, manifest)
    fault_start = float(manifest["thruster_fault"]["start_time_s"])
    pre_fault = [frame for frame in frames if frame["time_s"] < fault_start]
    post_fault = [frame for frame in frames if frame["time_s"] >= fault_start]
    advice_tiers = ("log_only", "possible")
    first_advice = v1._first_time(
        post_fault,
        lambda frame: frame["maintenance"]["tier"] in advice_tiers,
    )
    row.update({
        "context_gate_active_before_thruster": any(
            frame["maintenance"].get("advisory_gate_active", False)
            for frame in pre_fault
        ),
        "raw_advisory_suppressed_before_thruster": any(
            frame["maintenance"].get("advisory_suppressed", False)
            for frame in pre_fault
        ),
        "raw_advisory_suppressed_anytime": any(
            frame["maintenance"].get("advisory_suppressed", False)
            for frame in frames
        ),
        "first_post_fault_advisory_delay_s": (
            None if first_advice is None else first_advice - fault_start
        ),
    })
    return row


def summarize(rows, thresholds):
    summary = v1.summarize(rows, thresholds)
    no_output = [row for row in rows if row["thruster_mode"] == "no_output"]
    thrust_loss = [row for row in rows if row["thruster_mode"] == "thrust_loss"]
    delays = [
        row["first_post_fault_advisory_delay_s"] for row in rows
        if row["first_post_fault_advisory_delay_s"] is not None
    ]
    model = summary["model_advice"]
    model.update({
        "no_output_post_fault_advisory_rate": _rate(
            no_output, "model_advisory_observed"
        ),
        "thrust_loss_post_fault_advisory_rate": _rate(
            thrust_loss, "model_advisory_observed"
        ),
        "no_output_peak_top2_location_accuracy": _rate(
            no_output, "model_top2_correct"
        ),
        "thrust_loss_peak_top2_location_accuracy": _rate(
            thrust_loss, "model_top2_correct"
        ),
        "median_first_post_fault_advisory_delay_s": (
            None if not delays else median(delays)
        ),
    })
    summary["context_gate"] = {
        "active_before_thruster_mission_rate": _rate(
            rows, "context_gate_active_before_thruster"
        ),
        "raw_advisory_suppressed_before_thruster_mission_rate": _rate(
            rows, "raw_advisory_suppressed_before_thruster"
        ),
        "raw_advisory_suppressed_anytime_mission_rate": _rate(
            rows, "raw_advisory_suppressed_anytime"
        ),
    }
    checks = summary["acceptance_checks"]
    checks.update({
        "pre_thruster_advisory_rate_at_most": (
            model["pre_thruster_advisory_mission_rate"]
            <= thresholds["pre_thruster_advisory_mission_rate_max"]
        ),
        "post_fault_advisory_rate_at_least": (
            model["post_fault_advisory_rate"]
            >= thresholds["post_fault_advisory_rate_min"]
        ),
        "no_output_post_fault_advisory_at_least": (
            model["no_output_post_fault_advisory_rate"]
            >= thresholds["no_output_post_fault_advisory_rate_min"]
        ),
        "thrust_loss_post_fault_advisory_at_least": (
            model["thrust_loss_post_fault_advisory_rate"]
            >= thresholds["thrust_loss_post_fault_advisory_rate_min"]
        ),
    })
    summary["all_acceptance_checks_passed"] = all(checks.values())
    return summary


def save_comparison_plot(baseline, current, path):
    labels = (
        "No premature\nadvice",
        "Post-fault\nadvice",
        "No-output\nFTC target",
        "Safe FTC\nbehaviour",
        "Model Top-2\nlocation",
    )
    v1_values = np.array([
        1.0 - baseline["model_advice"]["pre_thruster_advisory_mission_rate"],
        baseline["model_advice"]["post_fault_advisory_rate"],
        baseline["thruster_ftc"]["no_output_target_recall"],
        baseline["thruster_ftc"]["safe_ftc_behavior_rate"],
        baseline["model_advice"]["peak_top2_location_accuracy"],
    ])
    v2_values = np.array([
        1.0 - current["model_advice"]["pre_thruster_advisory_mission_rate"],
        current["model_advice"]["post_fault_advisory_rate"],
        current["thruster_ftc"]["no_output_target_recall"],
        current["thruster_ftc"]["safe_ftc_behavior_rate"],
        current["model_advice"]["peak_top2_location_accuracy"],
    ])
    x = np.arange(len(labels))
    width = 0.34
    figure, axis = plt.subplots(figsize=(11.5, 5.8))
    figure.patch.set_facecolor("#07131f")
    axis.set_facecolor("#0d2031")
    bars_v1 = axis.bar(x - width / 2, v1_values, width, label="V1", color="#607d95")
    bars_v2 = axis.bar(x + width / 2, v2_values, width, label="V2 gated", color="#31c7b5")
    axis.set_ylim(0.0, 1.08)
    axis.set_xticks(x, labels, color="#eef6fb")
    axis.set_ylabel("Mission rate", color="#91a8ba")
    axis.set_title(
        "Unified random-batch comparison: context-gated learned advice",
        color="#eef6fb", loc="left", pad=14,
    )
    axis.grid(axis="y", color="#294256", alpha=0.7)
    axis.tick_params(colors="#91a8ba")
    for spine in axis.spines.values():
        spine.set_color("#294256")
    legend = axis.legend(frameon=False)
    for value in legend.get_texts():
        value.set_color("#eef6fb")
    for bars in (bars_v1, bars_v2):
        for bar in bars:
            height = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(height + 0.025, 1.045),
                f"{100 * height:.1f}%",
                ha="center", va="bottom", color="#eef6fb", fontsize=9,
            )
    figure.text(
        0.075, 0.01,
        "Separate 30-mission development batches; learned advice remains non-authoritative.",
        color="#91a8ba", fontsize=9,
    )
    figure.tight_layout(rect=(0.04, 0.05, 0.99, 0.98))
    figure.savefig(path, dpi=170, facecolor=figure.get_facecolor())
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--model-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    configuration, seeds, output_dir, protocol_hash = validate_protocol(
        protocol, protocol_path
    )
    bridge = SixDOFModelBridge(
        REPOSITORY_ROOT / protocol["model_checkpoint_path"],
        REPOSITORY_ROOT / protocol["temporal_config_path"],
        AUVSixDOFMultiTaskDetector,
        advisory_stabilization_time_s=float(
            configuration["advisory_stabilization_time_s"]
        ),
        device=(None if args.model_device == "auto" else args.model_device),
    )
    rows = []
    for mission_index, seed in enumerate(seeds, start=1):
        _, frames, _, manifest = run_demo(
            float(configuration["duration_s"]),
            float(configuration["dt_s"]),
            seed,
            injection_mode="random",
            model_bridge=bridge,
        )
        row = evaluate_mission(seed, frames, manifest)
        rows.append(row)
        print(
            f"{mission_index:02d}/{len(seeds)} seed={seed} "
            f"thruster={row['thruster_name']}:{row['thruster_mode']} "
            f"pre_advice={row['pre_thruster_model_advisory']} "
            f"post_advice={row['model_advisory_observed']} "
            f"ftc_safe={row['safe_ftc_behavior']}"
        )
    summary = summarize(rows, protocol["acceptance_thresholds"])
    baseline_payload = json.loads(
        (REPOSITORY_ROOT / protocol["baseline_summary_path"]).read_text(
            encoding="utf-8"
        )
    )
    payload = {
        "benchmark": "six_dof_unified_random_batch_v2",
        "evaluation_type": "development_random_batch",
        "real_sea_trial_claim": False,
        "independent_blind_test_claim": False,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "configuration": configuration,
        "model_device": str(bridge.device),
        "summary": summary,
        "baseline_v1_summary_path": protocol["baseline_summary_path"],
        "missions": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "unified_random_batch_v2_summary.json"
    csv_path = output_dir / "unified_random_batch_v2_missions.csv"
    plot_path = output_dir / "unified_random_batch_v1_v2_comparison.png"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    v1.save_csv(rows, csv_path)
    save_comparison_plot(baseline_payload["summary"], summary, plot_path)
    print(json.dumps(summary, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
