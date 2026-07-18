"""Aggregate the final six-DOF simulation evidence under one V4 protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.protocol import prepare_locked_protocol
from evaluation.unified_acceptance import evaluate_unified_acceptance


DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT / "docs" / "six_dof_unified_acceptance_protocol_v4.json"
)


def validate_protocol(protocol, protocol_path):
    configuration, output_dir, protocol_hash = prepare_locked_protocol(
        protocol,
        protocol_path,
        REPOSITORY_ROOT,
        "six_dof_unified_acceptance_v4",
        evaluation_type="simulation_evidence_aggregation",
        output_message="unified V4 output already exists",
    )
    source_paths = {
        name: item["path"]
        for name, item in protocol["evidence_sources"].items()
    }
    if tuple(source_paths) != tuple(configuration["source_names"]):
        raise ValueError("source_names must match evidence_sources in order")
    if set(source_paths.values()) != set(protocol["artifact_sha256"]):
        raise ValueError("every evidence source must have one locked artifact hash")
    return configuration, output_dir, protocol_hash, source_paths


def load_sources(source_paths):
    return {
        name: json.loads(
            (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
        )
        for name, relative in source_paths.items()
    }


def write_checks_csv(checks, path):
    columns = (
        "id", "category", "description", "source", "value_path",
        "observed", "operator", "threshold", "passed",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in checks)


def render_report(payload):
    summary = payload["summary"]
    lines = [
        "# Six-DOF unified acceptance V4",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "This is a simulation evidence aggregation, not a real-sea result or a new independent blind test.",
        "",
        "## Acceptance categories",
        "",
        "| Category | Passed | Total | Result |",
        "|---|---:|---:|---|",
    ]
    for name, category in summary["categories"].items():
        result = "PASS" if category["all_passed"] else "FAIL"
        lines.append(
            f"| {name} | {category['passed_count']} | "
            f"{category['check_count']} | {result} |"
        )
    lines.extend(["", "## Non-gating model information", ""])
    for metric in summary["informational_metrics"]:
        lines.append(
            f"- {metric['description']}: `{metric['observed']}`"
        )
    lines.extend([
        "",
        "## Diagnostic boundary",
        "",
        "- Confirmed sensor and complete thruster failures may trigger FTC.",
        "- Weak thrust loss remains a recorded probability-based maintenance clue.",
        "- Exact weak-fault thruster location is not an automatic safety claim.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    configuration, output_dir, protocol_hash, source_paths = validate_protocol(
        protocol, protocol_path
    )
    sources = load_sources(source_paths)
    summary = evaluate_unified_acceptance(protocol, sources)
    payload = {
        "benchmark": "six_dof_unified_acceptance_v4",
        "evaluation_type": protocol["evaluation_type"],
        "real_sea_trial_claim": False,
        "independent_blind_test_claim": False,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "configuration": configuration,
        "evidence_sources": {
            name: {
                "path": relative,
                "sha256": protocol["artifact_sha256"][relative],
            }
            for name, relative in source_paths.items()
        },
        "summary": summary,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "unified_acceptance_v4_summary.json"
    csv_path = output_dir / "unified_acceptance_v4_checks.csv"
    report_path = output_dir / "unified_acceptance_v4_report.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_checks_csv(summary["checks"], csv_path)
    report_path.write_text(render_report(payload), encoding="utf-8")

    print(
        f"Unified V4: {summary['passed_count']}/{summary['check_count']} "
        f"checks passed -> {summary['decision']}"
    )
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")
    if not summary["all_acceptance_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
