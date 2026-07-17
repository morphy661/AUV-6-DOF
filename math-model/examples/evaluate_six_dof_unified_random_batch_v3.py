"""Evaluate the fresh-inference context gate with a third locked seed batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
import evaluate_six_dof_unified_random_batch_v2 as v2
from demo_six_dof_unified_diagnostics import run_demo
from evaluation.protocol import prepare_locked_protocol
from model_six_dof_multitask import AUVSixDOFMultiTaskDetector
from presentation.six_dof_model_bridge import SixDOFModelBridge


DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT
    / "docs"
    / "six_dof_unified_random_batch_protocol_v3.json"
)


def validate_protocol(protocol, protocol_path):
    configuration, output_dir, protocol_hash = prepare_locked_protocol(
        protocol,
        protocol_path,
        REPOSITORY_ROOT,
        "six_dof_unified_random_batch_v3",
        evaluation_type="development_random_batch",
        output_message="locked batch output already exists",
    )
    count = int(configuration["mission_count"])
    base_seed = int(configuration["base_seed"])
    if count < 20:
        raise ValueError("development batch must contain at least 20 missions")
    seeds = list(range(base_seed, base_seed + count))
    return configuration, seeds, output_dir, protocol_hash


def _comparison_values(summary):
    return np.array([
        1.0 - summary["model_advice"]["pre_thruster_advisory_mission_rate"],
        summary["model_advice"]["post_fault_advisory_rate"],
        summary["thruster_ftc"]["no_output_target_recall"],
        summary["thruster_ftc"]["safe_ftc_behavior_rate"],
        summary["model_advice"]["peak_top2_location_accuracy"],
    ])


def save_comparison_plot(v1_summary, v2_summary, v3_summary, path):
    labels = (
        "No premature\nadvice",
        "Post-fault\nadvice",
        "No-output\nFTC target",
        "Safe FTC\nbehaviour",
        "Model Top-2\nlocation",
    )
    series = (
        ("V1", _comparison_values(v1_summary), "#607d95"),
        ("V2 timed gate", _comparison_values(v2_summary), "#f1b94b"),
        ("V3 fresh inference", _comparison_values(v3_summary), "#31c7b5"),
    )
    x = np.arange(len(labels))
    width = 0.25
    figure, axis = plt.subplots(figsize=(12.2, 5.9))
    figure.patch.set_facecolor("#07131f")
    axis.set_facecolor("#0d2031")
    offsets = (-width, 0.0, width)
    for offset, (name, values, color) in zip(offsets, series):
        bars = axis.bar(x + offset, values, width, label=name, color=color)
        for bar in bars:
            height = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(height + 0.022, 1.047),
                f"{100 * height:.1f}%",
                ha="center", va="bottom", color="#eef6fb", fontsize=8.5,
            )
    axis.set_ylim(0.0, 1.08)
    axis.set_xticks(x, labels, color="#eef6fb")
    axis.set_ylabel("Mission rate", color="#91a8ba")
    axis.set_title(
        "Context-gated learned advice: three locked development batches",
        color="#eef6fb", loc="left", pad=14,
    )
    axis.grid(axis="y", color="#294256", alpha=0.7)
    axis.tick_params(colors="#91a8ba")
    for spine in axis.spines.values():
        spine.set_color("#294256")
    legend = axis.legend(frameon=False, ncol=3, loc="lower left")
    for value in legend.get_texts():
        value.set_color("#eef6fb")
    figure.text(
        0.075, 0.01,
        "Each version uses a separate 30-mission seed batch; learned output remains advisory only.",
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
        row = v2.evaluate_mission(seed, frames, manifest)
        rows.append(row)
        print(
            f"{mission_index:02d}/{len(seeds)} seed={seed} "
            f"thruster={row['thruster_name']}:{row['thruster_mode']} "
            f"pre_advice={row['pre_thruster_model_advisory']} "
            f"post_advice={row['model_advisory_observed']} "
            f"ftc_safe={row['safe_ftc_behavior']}"
        )
    summary = v2.summarize(rows, protocol["acceptance_thresholds"])
    v1_payload = json.loads(
        (REPOSITORY_ROOT / protocol["baseline_v1_summary_path"]).read_text(
            encoding="utf-8"
        )
    )
    v2_payload = json.loads(
        (REPOSITORY_ROOT / protocol["baseline_v2_summary_path"]).read_text(
            encoding="utf-8"
        )
    )
    payload = {
        "benchmark": "six_dof_unified_random_batch_v3",
        "evaluation_type": "development_random_batch",
        "real_sea_trial_claim": False,
        "independent_blind_test_claim": False,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "configuration": configuration,
        "model_device": str(bridge.device),
        "summary": summary,
        "baseline_v1_summary_path": protocol["baseline_v1_summary_path"],
        "baseline_v2_summary_path": protocol["baseline_v2_summary_path"],
        "missions": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "unified_random_batch_v3_summary.json"
    csv_path = output_dir / "unified_random_batch_v3_missions.csv"
    plot_path = output_dir / "unified_random_batch_v1_v2_v3_comparison.png"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    v1.save_csv(rows, csv_path)
    save_comparison_plot(
        v1_payload["summary"], v2_payload["summary"], summary, plot_path
    )
    print(json.dumps(summary, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
