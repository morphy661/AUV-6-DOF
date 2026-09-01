"""Summarize controlled six-DOF feature ablations across training seeds.

The script treats validation metrics as decision evidence and test metrics as
descriptive evidence only.  It compares a candidate feature mask against the
same-seed baseline, applies fixed adoption gates, and writes machine-readable
CSV and JSON artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


DEFAULT_CANDIDATE = "motion_loss_evidence"
SPLITS = ("validation", "test")


def _load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric(split_result, *path):
    value = split_result
    for key in path:
        value = value[key]
    return value


def _result_row(seed, split, baseline, candidate):
    fields = {
        "raw_false_alarm_rate": ("false_alarm", "raw"),
        "temporal_false_alarm_rate": ("false_alarm", "temporal"),
        "joint_macro_f1": ("joint_macro_f1",),
        "no_output_ticket_recall": (
            "formal_ticket_metrics",
            "no_output_ticket_recall",
        ),
        "thrust_loss_ticket_recall": (
            "formal_ticket_metrics",
            "thrust_loss_ticket_recall",
        ),
        "false_maintenance_tickets": (
            "formal_ticket_metrics",
            "false_maintenance_tickets",
        ),
    }
    row = {"seed": seed, "split": split}
    for name, path in fields.items():
        baseline_value = _metric(baseline, *path)
        candidate_value = _metric(candidate, *path)
        row[f"baseline_{name}"] = baseline_value
        row[f"candidate_{name}"] = candidate_value
        row[f"delta_{name}"] = candidate_value - baseline_value
    return row


def _aggregate(rows):
    aggregate = {}
    delta_fields = sorted(
        key for key in rows[0] if key.startswith("delta_")
    )
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        metrics = {}
        for field in delta_fields:
            values = [float(row[field]) for row in split_rows]
            metrics[field] = {
                "mean": statistics.mean(values),
                "sample_standard_deviation": (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                ),
                "minimum": min(values),
                "maximum": max(values),
            }
        aggregate[split] = metrics
    return aggregate


def _validation_gates(rows, joint_f1_tolerance, thrust_ticket_tolerance):
    validation = [row for row in rows if row["split"] == "validation"]
    gates = {
        "raw_false_alarm_rate_lower_for_every_seed": all(
            row["delta_raw_false_alarm_rate"] < 0.0 for row in validation
        ),
        "joint_macro_f1_within_tolerance_for_every_seed": all(
            row["delta_joint_macro_f1"] >= -joint_f1_tolerance
            for row in validation
        ),
        "no_output_ticket_recall_not_lower_for_every_seed": all(
            row["delta_no_output_ticket_recall"] >= 0.0
            for row in validation
        ),
        "thrust_loss_ticket_recall_within_tolerance_for_every_seed": all(
            row["delta_thrust_loss_ticket_recall"]
            >= -thrust_ticket_tolerance
            for row in validation
        ),
        "false_maintenance_tickets_not_higher_for_every_seed": all(
            row["delta_false_maintenance_tickets"] <= 0
            for row in validation
        ),
    }
    return {
        "gates": gates,
        "adopt_candidate_mask": all(gates.values()),
        "policy": (
            "Validation is used for adoption. Test remains descriptive and "
            "does not change the decision."
        ),
        "joint_macro_f1_absolute_tolerance": joint_f1_tolerance,
        "thrust_loss_ticket_recall_absolute_tolerance": (
            thrust_ticket_tolerance
        ),
    }


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--joint-f1-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--thrust-ticket-tolerance",
        type=float,
        default=(1.0 / 18.0),
        help="Absolute validation event-recall tolerance; default is one of 18 events.",
    )
    args = parser.parse_args()

    experiments = [_load(path) for path in args.inputs]
    seeds = [int(item["training"]["seed"]) for item in experiments]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate training seeds: {seeds}")

    dataset_versions = {item["dataset_version"] for item in experiments}
    dataset_hashes = {item["dataset_sha256"] for item in experiments}
    split_missions = {
        json.dumps(item["split_missions"], sort_keys=True)
        for item in experiments
    }
    if len(dataset_versions) != 1 or len(dataset_hashes) != 1:
        raise ValueError("inputs do not use the same dataset")
    if len(split_missions) != 1:
        raise ValueError("inputs do not use the same mission split")

    rows = []
    for experiment in sorted(
        experiments, key=lambda item: int(item["training"]["seed"])
    ):
        seed = int(experiment["training"]["seed"])
        variants = experiment["variants"]
        if "baseline" not in variants or args.candidate not in variants:
            raise ValueError(
                f"seed {seed} lacks baseline or {args.candidate}"
            )
        for split in SPLITS:
            rows.append(
                _result_row(
                    seed,
                    split,
                    variants["baseline"][split],
                    variants[args.candidate][split],
                )
            )

    summary = {
        "experiment_type": "controlled_multiseed_feature_ablation_summary",
        "candidate_variant": args.candidate,
        "dataset_version": next(iter(dataset_versions)),
        "dataset_sha256": next(iter(dataset_hashes)),
        "seeds": sorted(seeds),
        "input_files": [str(path) for path in args.inputs],
        "aggregate_deltas": _aggregate(rows),
        "adoption_decision": _validation_gates(
            rows,
            args.joint_f1_tolerance,
            args.thrust_ticket_tolerance,
        ),
        "rows": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "multiseed_feature_ablation_rows.csv", rows)
    with (args.output_dir / "multiseed_feature_ablation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary["adoption_decision"], indent=2))


if __name__ == "__main__":
    main()
