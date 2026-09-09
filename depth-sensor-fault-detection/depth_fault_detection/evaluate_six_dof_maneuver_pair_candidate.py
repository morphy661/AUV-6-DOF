"""Compare locked diagnosis models on paired 150--300 m maneuvers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
MATH_MODEL_SRC = REPO_ROOT / "math-model" / "src"
if str(MATH_MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MATH_MODEL_SRC))

from diagnosis.maintenance_health_decision import (  # noqa: E402
    apply_maintenance_decision_layer,
)
from diagnosis.maintenance_log_policy import (  # noqa: E402
    MaintenanceLogConfig,
    apply_maintenance_log_policy,
    maintenance_log_metrics,
)
from diagnosis.maintenance_ticket_policy import (  # noqa: E402
    MaintenanceTicketConfig,
    apply_maintenance_ticket_policy,
    extract_maintenance_ticket_evidence,
    maintenance_ticket_metrics,
)
from diagnosis.temporal_fault_decision import (  # noqa: E402
    TemporalDecisionConfig,
    apply_temporal_decision_layer,
)
from model_six_dof_multitask import (  # noqa: E402
    AUVSixDOFMultiTaskDetector,
)
from train_six_dof_multitask import (  # noqa: E402
    evaluate,
    load_and_validate_dataset,
    make_loader,
    operational_test_metrics,
    set_seed,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metadata(dataset, mission_id):
    metadata = dataset["mission_metadata"]
    return metadata.get(int(mission_id), metadata.get(str(int(mission_id))))


def _subset_predictions(predictions, positions):
    subset = {}
    sample_count = len(predictions["mode_true"])
    for key, value in predictions.items():
        if isinstance(value, np.ndarray) and len(value) == sample_count:
            subset[key] = value[positions]
        else:
            subset[key] = value
    return subset


def _classification_metrics(predictions):
    mode_true = np.asarray(predictions["mode_true"], dtype=int)
    mode_pred = np.asarray(predictions["mode_pred"], dtype=int)
    joint_true = np.asarray(predictions["joint_true"], dtype=int)
    joint_pred = np.asarray(predictions["joint_pred"], dtype=int)
    present_mode = sorted(np.unique(mode_true).tolist())
    present_joint = sorted(np.unique(joint_true).tolist())
    thrust_mask = mode_true == 2
    return {
        "window_count": int(len(mode_true)),
        "mode_macro_f1_present_classes": float(f1_score(
            mode_true,
            mode_pred,
            labels=present_mode,
            average="macro",
            zero_division=0,
        )),
        "joint_macro_f1_present_classes": float(f1_score(
            joint_true,
            joint_pred,
            labels=present_joint,
            average="macro",
            zero_division=0,
        )),
        "joint_accuracy": float(np.mean(joint_true == joint_pred)),
        "thrust_loss_mode_recall": (
            float(np.mean(mode_pred[thrust_mask] == 2))
            if np.any(thrust_mask) else None
        ),
        "thrust_loss_exact_window_accuracy": (
            float(np.mean(joint_pred[thrust_mask] == joint_true[thrust_mask]))
            if np.any(thrust_mask) else None
        ),
    }


def _mission_outcomes(dataset, indices, raw, temporal, maintenance):
    indices = np.asarray(indices, dtype=np.int64)
    mission_ids = dataset["mission_ids"][indices].cpu().numpy().astype(int)
    times = dataset["window_end_times"][indices].cpu().numpy().astype(float)
    true_modes = np.asarray(raw["mode_true"], dtype=int)
    raw_modes = np.asarray(raw["mode_pred"], dtype=int)
    temporal_modes = np.asarray(temporal["mode_pred"], dtype=int)
    log_levels = np.asarray(
        maintenance["maintenance_log_level_pred"], dtype=int
    )
    ticket_active = np.asarray(
        maintenance["maintenance_ticket_active"], dtype=bool
    )
    tickets = maintenance.get("maintenance_tickets", [])
    pending = maintenance.get("maintenance_pending_observations", [])
    outcomes = []
    for mission_id in sorted(np.unique(mission_ids)):
        positions = np.flatnonzero(mission_ids == mission_id)
        metadata = _metadata(dataset, mission_id)
        fault_start = metadata["parameters"].get("fault_start_time_s")
        mission_tickets = [
            ticket for ticket in tickets
            if int(ticket["mission_id"]) == int(mission_id)
        ]
        mission_pending = [
            observation for observation in pending
            if int(observation["mission_id"]) == int(mission_id)
        ]
        status_counts = {}
        for observation in mission_pending:
            status = observation["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        outcomes.append({
            "mission_id": int(mission_id),
            "scenario": metadata["scenario"],
            "pair_id": metadata.get("pair_id"),
            "pair_condition": metadata.get("pair_condition"),
            "paired_thruster": metadata.get("paired_thruster"),
            "fault_start_time_s": fault_start,
            "window_count": int(len(positions)),
            "fault_label_window_count": int(np.sum(
                true_modes[positions] != 0
            )),
            "raw_fault_window_count": int(np.sum(
                raw_modes[positions] != 0
            )),
            "temporal_fault_window_count": int(np.sum(
                temporal_modes[positions] != 0
            )),
            "operator_attention_window_count": int(np.sum(
                log_levels[positions] >= 3
            )),
            "formal_ticket_window_count": int(np.sum(
                ticket_active[positions]
            )),
            "formal_ticket_count": int(len(mission_tickets)),
            "pending_observation_status_counts": status_counts,
            "first_raw_fault_time_s": (
                float(times[positions[np.flatnonzero(
                    raw_modes[positions] != 0
                )[0]]])
                if np.any(raw_modes[positions] != 0) else None
            ),
            "first_temporal_fault_time_s": (
                float(times[positions[np.flatnonzero(
                    temporal_modes[positions] != 0
                )[0]]])
                if np.any(temporal_modes[positions] != 0) else None
            ),
        })
    return outcomes


def _decision_metrics(
    dataset,
    indices,
    raw_predictions,
    temporal_config,
    ticket_config,
):
    temporal = apply_temporal_decision_layer(
        dataset, indices, raw_predictions, temporal_config
    )
    maintenance = apply_maintenance_decision_layer(
        dataset, indices, raw_predictions, temporal_config
    )
    evidence = extract_maintenance_ticket_evidence(
        dataset, indices, ticket_config
    )
    maintenance = apply_maintenance_ticket_policy(
        dataset,
        indices,
        maintenance,
        ticket_config,
        ticket_evidence=evidence,
    )
    maintenance = apply_maintenance_log_policy(
        dataset, indices, maintenance, MaintenanceLogConfig()
    )
    return {
        "raw": {
            "classification": _classification_metrics(raw_predictions),
            "operations": operational_test_metrics(
                dataset, indices, raw_predictions
            ),
        },
        "temporal": {
            "classification": _classification_metrics(temporal),
            "operations": operational_test_metrics(
                dataset, indices, temporal
            ),
        },
        "graded_log": maintenance_log_metrics(
            dataset, indices, maintenance
        ),
        "formal_tickets": maintenance_ticket_metrics(
            dataset, indices, maintenance
        ),
        "missions": _mission_outcomes(
            dataset, indices, raw_predictions, temporal, maintenance
        ),
    }


def _group_positions(dataset, test_indices, augmentation_tag):
    mission_ids = dataset["mission_ids"][test_indices].cpu().numpy()
    groups = {
        "full_test": np.ones(len(test_indices), dtype=bool),
        "legacy_test": np.zeros(len(test_indices), dtype=bool),
        "maneuver_pair_test": np.zeros(len(test_indices), dtype=bool),
        "maneuver_normal_test": np.zeros(len(test_indices), dtype=bool),
        "maneuver_fault_test": np.zeros(len(test_indices), dtype=bool),
    }
    thruster_masks = {}
    for position, mission_id in enumerate(mission_ids):
        metadata = _metadata(dataset, mission_id)
        augmented = metadata.get("augmentation") == augmentation_tag
        if not augmented:
            groups["legacy_test"][position] = True
            continue
        groups["maneuver_pair_test"][position] = True
        condition = metadata["pair_condition"]
        if condition == "normal":
            groups["maneuver_normal_test"][position] = True
        else:
            groups["maneuver_fault_test"][position] = True
            thruster = metadata["paired_thruster"]
            thruster_masks.setdefault(
                f"maneuver_fault_{thruster}",
                np.zeros(len(test_indices), dtype=bool),
            )[position] = True
    groups.update(thruster_masks)
    return {
        name: np.flatnonzero(mask) for name, mask in groups.items()
    }


def _evaluate_model(
    label,
    checkpoint_path,
    dataset,
    test_indices,
    groups,
    temporal_config,
    ticket_config,
    device,
    batch_size,
    seed,
):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = AUVSixDOFMultiTaskDetector(
        input_dim=int(checkpoint["input_dim"]),
        structured_fusion=bool(checkpoint.get("structured_fusion", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    raw_full = evaluate(
        model,
        make_loader(
            dataset,
            test_indices,
            checkpoint["mean"],
            checkpoint["std"],
            batch_size,
            shuffle=False,
            seed=seed,
        ),
        device,
        torch.nn.CrossEntropyLoss(),
        torch.nn.CrossEntropyLoss(),
    )
    results = {}
    for group_name, positions in groups.items():
        indices = test_indices[positions]
        predictions = _subset_predictions(raw_full, positions)
        results[group_name] = _decision_metrics(
            dataset,
            indices,
            predictions,
            temporal_config,
            ticket_config,
        )
    return {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_dataset_version": checkpoint.get("dataset_version"),
        "groups": results,
    }


def _maneuver_details(dataset, augmentation_tag):
    metadata = [
        value for value in dataset["mission_metadata"].values()
        if value.get("augmentation") == augmentation_tag
        and value.get("pair_condition") == "normal"
    ]
    turns = np.asarray([
        abs(delta)
        for item in metadata
        for delta in item["parameters"]["mission"]["heading_deltas_rad"]
    ])
    return {
        "pair_count": len(metadata),
        "planned_yaw_change_abs_rad": {
            "minimum": float(np.min(turns)),
            "median": float(np.median(turns)),
            "mean": float(np.mean(turns)),
            "maximum": float(np.max(turns)),
        },
        "planned_yaw_change_abs_deg": {
            "minimum": float(np.degrees(np.min(turns))),
            "median": float(np.degrees(np.median(turns))),
            "mean": float(np.degrees(np.mean(turns))),
            "maximum": float(np.degrees(np.max(turns))),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--rehearsal-checkpoint", type=Path)
    parser.add_argument("--rehearsal-label", default="rehearsal candidate")
    parser.add_argument(
        "--augmentation-tag", default="v3_2_high_yaw_turn_pair"
    )
    parser.add_argument(
        "--experiment-id",
        default="six_dof_v3_2_maneuver_pairs_20260902",
    )
    parser.add_argument("--temporal-config", type=Path, required=True)
    parser.add_argument("--ticket-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset, split_indices, _ = load_and_validate_dataset(args.dataset)
    test_indices = split_indices["test"]
    groups = _group_positions(
        dataset, test_indices, args.augmentation_tag
    )
    temporal_config = TemporalDecisionConfig(**_json(args.temporal_config))
    ticket_config = MaintenanceTicketConfig(**_json(args.ticket_config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [
        _evaluate_model(
            args.baseline_label,
            args.baseline_checkpoint,
            dataset,
            test_indices,
            groups,
            temporal_config,
            ticket_config,
            device,
            args.batch_size,
            args.seed,
        )
    ]
    if args.candidate_checkpoint is not None:
        models.append(_evaluate_model(
            args.candidate_label,
            args.candidate_checkpoint,
            dataset,
            test_indices,
            groups,
            temporal_config,
            ticket_config,
            device,
            args.batch_size,
            args.seed,
        ))
    if args.rehearsal_checkpoint is not None:
        models.append(_evaluate_model(
            args.rehearsal_label,
            args.rehearsal_checkpoint,
            dataset,
            test_indices,
            groups,
            temporal_config,
            ticket_config,
            device,
            args.batch_size,
            args.seed,
        ))
    result = {
        "experiment_id": args.experiment_id,
        "evaluation_policy": (
            "All models use the supplied locked temporal, maintenance-ticket, "
            "and logging rules. The paired test missions are evaluated after "
            "the candidate and rules have been selected."
        ),
        "device": str(device),
        "dataset": str(args.dataset),
        "dataset_sha256": _sha256(args.dataset),
        "dataset_version": dataset.get("dataset_version"),
        "temporal_config": asdict(temporal_config),
        "ticket_config": asdict(ticket_config),
        "group_window_counts": {
            name: int(len(positions)) for name, positions in groups.items()
        },
        "augmentation_tag": args.augmentation_tag,
        "maneuver_details": _maneuver_details(
            dataset, args.augmentation_tag
        ),
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    def concise_group(group):
        return {key: value for key, value in group.items() if key != "missions"}

    print(json.dumps({
        model["label"]: {
            "maneuver_normal": concise_group(
                model["groups"]["maneuver_normal_test"]
            ),
            "maneuver_fault": concise_group(
                model["groups"]["maneuver_fault_test"]
            ),
        }
        for model in models
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
