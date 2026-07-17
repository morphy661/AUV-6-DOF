"""Evaluate a frozen six-DOF blind set without calibration or retraining."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
MATH_MODEL_SRC = REPO_ROOT / "math-model" / "src"
if str(MATH_MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MATH_MODEL_SRC))

from diagnosis.maintenance_health_decision import (
    apply_maintenance_decision_layer,
    maintenance_event_metrics,
)
from diagnosis.maintenance_ticket_policy import (
    MaintenanceTicketConfig,
    apply_maintenance_ticket_policy,
    maintenance_ticket_metrics,
)
from diagnosis.maintenance_log_policy import (
    MaintenanceLogConfig,
    apply_maintenance_log_policy,
    maintenance_log_metrics,
)
from diagnosis.temporal_fault_decision import (
    TemporalDecisionConfig,
    apply_temporal_decision_layer,
)
from utils.six_dof_feature_extractor import (
    FAULT_MODE_NAMES,
    JOINT_FAULT_NAMES,
    SIX_DOF_MODEL_INPUT_DIM,
    SIX_DOF_RAW_FEATURE_DIM,
    SIX_DOF_RAW_FEATURE_NAMES,
)

from model_six_dof_multitask import AUVSixDOFMultiTaskDetector
from train_six_dof_multitask import (
    evaluate,
    make_loader,
    operational_test_metrics,
    set_seed,
)


DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "six_dof_final_blind_protocol.json"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_config(path, config_type):
    return config_type(**json.loads(Path(path).read_text(encoding="utf-8")))


def _verify_protocol(path):
    protocol_path = Path(path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("state") != "frozen_before_evaluation":
        raise ValueError("blind protocol is not frozen before evaluation")
    for artifact in protocol["frozen_artifacts"]:
        artifact_path = REPO_ROOT / artifact["path"]
        if _sha256(artifact_path) != artifact["sha256"]:
            raise RuntimeError(
                f"frozen artifact hash mismatch: {artifact['path']}"
            )
    return protocol, _sha256(protocol_path)


def _validate_dataset(dataset, protocol):
    required = {
        "X", "y_mode", "y_location", "y_joint", "mission_ids",
        "window_end_times", "feature_names", "raw_feature_dim",
        "model_input_dim", "sequence_length", "mission_metadata",
        "blind_indices", "blind_protocol_id", "guidance_context_ids",
        "guidance_context_stable",
    }
    missing = sorted(required - set(dataset))
    if missing:
        raise ValueError(f"blind dataset missing keys: {missing}")
    if dataset["blind_protocol_id"] != protocol["protocol_id"]:
        raise ValueError("blind dataset protocol id mismatch")
    if tuple(dataset["X"].shape[1:]) != (
        int(protocol["dataset"]["sequence_length"]),
        SIX_DOF_RAW_FEATURE_DIM,
    ):
        raise ValueError("blind feature tensor shape mismatch")
    if int(dataset["raw_feature_dim"]) != SIX_DOF_RAW_FEATURE_DIM:
        raise ValueError("blind raw feature dimension mismatch")
    if int(dataset["model_input_dim"]) != SIX_DOF_MODEL_INPUT_DIM:
        raise ValueError("blind model input dimension mismatch")
    if tuple(dataset["feature_names"]) != SIX_DOF_RAW_FEATURE_NAMES:
        raise ValueError("blind feature schema mismatch")
    indices = torch.as_tensor(dataset["blind_indices"], dtype=torch.long)
    if not torch.equal(indices, torch.arange(len(dataset["X"]))):
        raise ValueError("blind indices must cover every window exactly once")
    expected_missions = (
        len(JOINT_FAULT_NAMES)
        * int(protocol["dataset"]["repeats_per_scenario"])
    )
    if len(dataset["mission_metadata"]) != expected_missions:
        raise ValueError("blind mission count does not match protocol")
    scenario_counts = Counter(
        item["scenario"] for item in dataset["mission_metadata"].values()
    )
    if set(scenario_counts.values()) != {
        int(protocol["dataset"]["repeats_per_scenario"])
    } or len(scenario_counts) != len(JOINT_FAULT_NAMES):
        raise ValueError("blind scenario coverage does not match protocol")
    if set(dataset["y_joint"].tolist()) != set(range(len(JOINT_FAULT_NAMES))):
        raise ValueError("blind set does not cover all joint labels")
    contexts = torch.as_tensor(
        dataset["guidance_context_ids"], dtype=torch.long
    )
    context_stable = torch.as_tensor(
        dataset["guidance_context_stable"], dtype=torch.bool
    )
    if contexts.shape != (len(dataset["X"]),):
        raise ValueError("blind guidance context shape mismatch")
    if context_stable.shape != contexts.shape:
        raise ValueError("blind context stability shape mismatch")
    if torch.any(contexts < 0) or not torch.any(contexts > 0):
        raise ValueError("blind guidance contexts do not contain transitions")
    if not torch.any(context_stable) or not torch.any(~context_stable):
        raise ValueError(
            "blind guidance context stability does not cover both states"
        )
    return indices.numpy(), dict(sorted(scenario_counts.items()))


def _classification_metrics(predictions):
    return {
        "mode_macro_f1": float(f1_score(
            predictions["mode_true"], predictions["mode_pred"],
            labels=range(len(FAULT_MODE_NAMES)), average="macro",
            zero_division=0,
        )),
        "location_macro_f1": float(f1_score(
            predictions["location_true"], predictions["location_pred"],
            labels=range(7), average="macro", zero_division=0,
        )),
        "joint_macro_f1": float(f1_score(
            predictions["joint_true"], predictions["joint_pred"],
            labels=range(len(JOINT_FAULT_NAMES)), average="macro",
            zero_division=0,
        )),
        "joint_accuracy": float(accuracy_score(
            predictions["joint_true"], predictions["joint_pred"]
        )),
    }


def _acceptance(protocol, maintenance_metrics, ticket_metrics):
    criteria = protocol["acceptance_criteria"]
    checks = {
        "raw_health_event_recall": {
            "value": maintenance_metrics["event_recall"],
            "threshold": criteria["minimum_raw_health_event_recall"],
            "passed": maintenance_metrics["event_recall"]
            >= criteria["minimum_raw_health_event_recall"],
        },
        "no_output_ticket_recall": {
            "value": ticket_metrics["no_output_ticket_recall"],
            "threshold": criteria["minimum_no_output_ticket_recall"],
            "passed": ticket_metrics["no_output_ticket_recall"]
            >= criteria["minimum_no_output_ticket_recall"],
        },
        "false_maintenance_tickets": {
            "value": ticket_metrics["false_maintenance_tickets"],
            "threshold": criteria["maximum_false_maintenance_tickets"],
            "passed": ticket_metrics["false_maintenance_tickets"]
            <= criteria["maximum_false_maintenance_tickets"],
        },
    }
    return {
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks.values()),
        "thrust_loss_ticket_recall_is_informational": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    protocol, protocol_hash = _verify_protocol(args.protocol)
    dataset_path = REPO_ROOT / protocol["dataset"]["output_path"]
    sidecar_path = dataset_path.with_suffix(dataset_path.suffix + ".sha256.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    dataset_hash = _sha256(dataset_path)
    if dataset_hash != sidecar["dataset_sha256"]:
        raise RuntimeError("blind dataset hash does not match its sidecar")
    output_dir = REPO_ROOT / protocol["output_directory"]
    summary_path = output_dir / "final_blind_summary.json"
    if summary_path.exists():
        raise FileExistsError(
            "final blind result already exists; one-shot evaluation will "
            "not overwrite it"
        )

    set_seed(int(protocol["evaluation_seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    indices, scenario_counts = _validate_dataset(dataset, protocol)
    checkpoint_path = REPO_ROOT / protocol["model_checkpoint_path"]
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = AUVSixDOFMultiTaskDetector(
        input_dim=int(checkpoint["input_dim"]),
        structured_fusion=bool(checkpoint.get("structured_fusion", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = make_loader(
        dataset, indices, checkpoint["mean"], checkpoint["std"],
        args.batch_size, shuffle=False,
        seed=int(protocol["evaluation_seed"]),
    )
    mode_loss = torch.nn.CrossEntropyLoss()
    location_loss = torch.nn.CrossEntropyLoss()
    predictions = evaluate(
        model, loader, device, mode_loss, location_loss
    )
    temporal_path = REPO_ROOT / protocol["temporal_config_path"]
    ticket_path = REPO_ROOT / protocol["ticket_config_path"]
    temporal_config = _load_config(temporal_path, TemporalDecisionConfig)
    ticket_config = _load_config(ticket_path, MaintenanceTicketConfig)
    temporal = apply_temporal_decision_layer(
        dataset, indices, predictions, temporal_config
    )
    maintenance = apply_maintenance_decision_layer(
        dataset, indices, predictions, temporal_config
    )
    maintenance_metrics = maintenance_event_metrics(
        dataset, indices, maintenance
    )
    decisions = apply_maintenance_ticket_policy(
        dataset, indices, maintenance, ticket_config
    )
    log_config = MaintenanceLogConfig()
    decisions = apply_maintenance_log_policy(
        dataset, indices, decisions, log_config
    )
    ticket_metrics = maintenance_ticket_metrics(
        dataset, indices, decisions
    )
    graded_log_metrics = maintenance_log_metrics(
        dataset, indices, decisions
    )
    summary = {
        "benchmark": protocol.get(
            "benchmark", "six_dof_final_blind_v2"
        ),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "dataset_sha256": dataset_hash,
        "device": str(device),
        "calibration_or_training_performed": False,
        "mission_count": len(dataset["mission_metadata"]),
        "window_count": len(dataset["X"]),
        "scenario_counts": scenario_counts,
        "guidance_context_side_channel": {
            "available": bool(
                decisions["maintenance_guidance_context_available"]
            ),
            "context_count": int(torch.unique(
                dataset["guidance_context_ids"]
            ).numel()),
            "stable_window_rate": float(torch.as_tensor(
                dataset["guidance_context_stable"], dtype=torch.float32
            ).mean()),
        },
        "raw": {
            "classification": _classification_metrics(predictions),
            "operations": operational_test_metrics(
                dataset, indices, predictions
            ),
        },
        "temporal": {
            "classification": _classification_metrics(temporal),
            "operations": operational_test_metrics(
                dataset, indices, temporal
            ),
        },
        "maintenance_log": maintenance_metrics,
        "graded_maintenance_log": graded_log_metrics,
        "maintenance_tickets": ticket_metrics,
        "acceptance": _acceptance(
            protocol, maintenance_metrics, ticket_metrics
        ),
        "frozen_temporal_config": asdict(temporal_config),
        "frozen_ticket_config": asdict(ticket_config),
        "maintenance_log_config": asdict(log_config),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "final_blind_event_log.json").write_text(json.dumps({
        "raw_health_events": decisions["maintenance_events"],
        "graded_health_events": decisions["maintenance_graded_events"],
        "operator_attention_events": decisions["maintenance_operator_events"],
        "collapsed_observations": decisions["maintenance_observation_events"],
        "background_traces": decisions["maintenance_trace_events"],
        "pending_maintenance_observations": decisions[
            "maintenance_pending_observations"
        ],
        "maintenance_advisories": decisions["maintenance_advisories"],
        "formal_maintenance_tickets": decisions["maintenance_tickets"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
