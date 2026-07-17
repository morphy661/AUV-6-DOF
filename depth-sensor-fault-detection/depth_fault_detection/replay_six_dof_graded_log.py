"""Retrospectively replay a frozen result through the presentation-only log.

This is deliberately not a new blind evaluation.  It verifies that the frozen
model, temporal decision, and ticket metrics reproduce the archived result,
then measures only the post-hoc event merging and display-level policy.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
MATH_MODEL_SRC = REPO_ROOT / "math-model" / "src"
if str(MATH_MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MATH_MODEL_SRC))

from diagnosis.maintenance_health_decision import (
    apply_maintenance_decision_layer,
    maintenance_event_metrics,
)
from diagnosis.maintenance_log_policy import (
    MaintenanceLogConfig,
    apply_maintenance_log_policy,
    maintenance_log_metrics,
)
from diagnosis.maintenance_ticket_policy import (
    MaintenanceTicketConfig,
    apply_maintenance_ticket_policy,
    maintenance_ticket_metrics,
)
from diagnosis.temporal_fault_decision import TemporalDecisionConfig
from model_six_dof_multitask import AUVSixDOFMultiTaskDetector
from train_six_dof_multitask import evaluate, make_loader, set_seed


DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "six_dof_final_blind_protocol_v2.json"
)
DEFAULT_OUTPUT = (
    THIS_DIR / "results" / "six_dof_log_presentation_replay_v2_20260717"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_config(path, config_type):
    return config_type(**json.loads(Path(path).read_text(encoding="utf-8")))


def _verify_frozen_input(protocol, key):
    relative_path = protocol[key]
    artifact_hashes = {
        artifact["path"]: artifact["sha256"]
        for artifact in protocol["frozen_artifacts"]
    }
    expected = artifact_hashes.get(relative_path)
    if expected is None:
        raise ValueError(f"{key} is not listed as a frozen artifact")
    path = REPO_ROOT / relative_path
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"frozen input hash mismatch: {relative_path}")
    return path, actual


def _assert_reproduced(name, current, archived, keys):
    for key in keys:
        current_value = current[key]
        archived_value = archived[key]
        if isinstance(current_value, (float, np.floating)):
            matched = np.isclose(
                current_value,
                archived_value,
                rtol=0.0,
                atol=1e-12,
            )
        else:
            matched = current_value == archived_value
        if not matched:
            raise RuntimeError(
                f"{name}.{key} did not reproduce archived V2: "
                f"current={current_value}, archived={archived_value}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    summary_path = args.output_dir / "graded_log_replay_summary.json"
    if summary_path.exists():
        raise FileExistsError("graded-log replay result will not be overwritten")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_hash = _sha256(args.protocol)
    archived_dir = REPO_ROOT / protocol["output_directory"]
    archived_summary = json.loads(
        (archived_dir / "final_blind_summary.json").read_text(encoding="utf-8")
    )
    if protocol_hash != archived_summary["protocol_sha256"]:
        raise RuntimeError("source protocol no longer matches archived V2 result")

    checkpoint_path, checkpoint_hash = _verify_frozen_input(
        protocol, "model_checkpoint_path"
    )
    temporal_path, temporal_hash = _verify_frozen_input(
        protocol, "temporal_config_path"
    )
    ticket_path, ticket_hash = _verify_frozen_input(
        protocol, "ticket_config_path"
    )

    dataset_path = REPO_ROOT / protocol["dataset"]["output_path"]
    dataset_hash = _sha256(dataset_path)
    if dataset_hash != archived_summary["dataset_sha256"]:
        raise RuntimeError("source dataset no longer matches archived V2 result")

    set_seed(int(protocol["evaluation_seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    indices = np.arange(len(dataset["X"]), dtype=np.int64)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = AUVSixDOFMultiTaskDetector(
        input_dim=int(checkpoint["input_dim"]),
        structured_fusion=bool(checkpoint.get("structured_fusion", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = make_loader(
        dataset,
        indices,
        checkpoint["mean"],
        checkpoint["std"],
        args.batch_size,
        shuffle=False,
        seed=int(protocol["evaluation_seed"]),
    )
    predictions = evaluate(
        model,
        loader,
        device,
        torch.nn.CrossEntropyLoss(),
        torch.nn.CrossEntropyLoss(),
    )

    temporal_config = _load_config(temporal_path, TemporalDecisionConfig)
    ticket_config = _load_config(ticket_path, MaintenanceTicketConfig)
    maintenance = apply_maintenance_decision_layer(
        dataset, indices, predictions, temporal_config
    )
    raw_metrics = maintenance_event_metrics(dataset, indices, maintenance)
    decisions = apply_maintenance_ticket_policy(
        dataset, indices, maintenance, ticket_config
    )
    ticket_metrics = maintenance_ticket_metrics(dataset, indices, decisions)
    _assert_reproduced(
        "maintenance_log",
        raw_metrics,
        archived_summary["maintenance_log"],
        (
            "event_recall",
            "event_precision",
            "false_advisory_events",
            "normal_window_advisory_rate",
        ),
    )
    _assert_reproduced(
        "maintenance_tickets",
        ticket_metrics,
        archived_summary["maintenance_tickets"],
        (
            "no_output_ticket_recall",
            "thrust_loss_ticket_recall",
            "false_maintenance_tickets",
            "formal_maintenance_ticket_count",
        ),
    )

    log_config = MaintenanceLogConfig()
    decisions = apply_maintenance_log_policy(
        dataset, indices, decisions, log_config
    )
    graded_metrics = maintenance_log_metrics(dataset, indices, decisions)
    summary = {
        "benchmark": "six_dof_graded_log_retrospective_replay",
        "source_protocol_id": protocol["protocol_id"],
        "source_protocol_sha256": protocol_hash,
        "source_dataset_sha256": dataset_hash,
        "frozen_input_hashes": {
            "model_checkpoint": checkpoint_hash,
            "temporal_config": temporal_hash,
            "ticket_config": ticket_hash,
        },
        "device": str(device),
        "retrospective_replay": True,
        "new_blind_acceptance_claim": False,
        "training_calibration_or_threshold_selection_performed": False,
        "archived_v2_metrics_reproduced_before_presentation_policy": True,
        "maintenance_log_config": asdict(log_config),
        "archived_raw_maintenance_log": raw_metrics,
        "archived_maintenance_tickets": ticket_metrics,
        "graded_maintenance_log": graded_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "graded_log_events.json").write_text(json.dumps({
        "graded_health_events": decisions["maintenance_graded_events"],
        "operator_attention_events": decisions["maintenance_operator_events"],
        "collapsed_observations": decisions["maintenance_observation_events"],
        "background_traces": decisions["maintenance_trace_events"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
