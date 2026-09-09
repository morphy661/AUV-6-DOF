"""Fine-tune V3.1 with old/new rehearsal and validation-only selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
MATH_MODEL_SRC = REPO_ROOT / "math-model" / "src"
if str(MATH_MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MATH_MODEL_SRC))

from diagnosis.maintenance_health_decision import (  # noqa: E402
    apply_maintenance_decision_layer,
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
    class_weights,
    evaluate,
    load_and_validate_dataset,
    make_loader,
    multitask_loss,
    operational_test_metrics,
    set_seed,
)
from utils.six_dof_feature_extractor import (  # noqa: E402
    FAULT_MODE_NAMES,
    SIX_DOF_MODEL_INPUT_DIM,
    SIX_DOF_RAW_FEATURE_DIM,
    SIX_DOF_RAW_FEATURE_NAMES,
    THRUSTER_NAMES,
)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _validation_metrics(
    model,
    loader,
    dataset,
    indices,
    device,
    mode_loss_fn,
    location_loss_fn,
    temporal_config,
    ticket_config,
):
    raw = evaluate(
        model, loader, device, mode_loss_fn, location_loss_fn
    )
    temporal = apply_temporal_decision_layer(
        dataset, indices, raw, temporal_config
    )
    maintenance = apply_maintenance_decision_layer(
        dataset, indices, raw, temporal_config
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
    tickets = maintenance_ticket_metrics(dataset, indices, maintenance)
    temporal_operations = operational_test_metrics(
        dataset, indices, temporal
    )
    return raw, temporal_operations, tickets


def _row(epoch, train_loss, learning_rate, raw, operations, tickets):
    return {
        "epoch": int(epoch),
        "train_loss": train_loss,
        "learning_rate": float(learning_rate),
        "validation_loss": float(raw["loss"]),
        "validation_mode_macro_f1": float(raw["mode_macro_f1"]),
        "validation_location_macro_f1": float(raw["location_macro_f1"]),
        "validation_joint_macro_f1": float(raw["joint_macro_f1"]),
        "validation_temporal_normal_false_alarm_rate": float(
            operations["normal_window_false_alarm_rate"]
        ),
        "validation_no_output_ticket_recall": tickets[
            "no_output_ticket_recall"
        ],
        "validation_thrust_loss_ticket_recall": tickets[
            "thrust_loss_ticket_recall"
        ],
        "validation_false_maintenance_tickets": int(
            tickets["false_maintenance_tickets"]
        ),
        "validation_ticket_event_precision": tickets[
            "ticket_event_precision"
        ],
    }


def _eligible(row, constraints):
    return (
        row["validation_no_output_ticket_recall"]
        >= constraints["minimum_no_output_ticket_recall"]
        and row["validation_thrust_loss_ticket_recall"]
        >= constraints["minimum_thrust_loss_ticket_recall"]
        and row["validation_false_maintenance_tickets"]
        <= constraints["maximum_false_maintenance_tickets"]
        and row["validation_temporal_normal_false_alarm_rate"]
        <= constraints["maximum_temporal_normal_false_alarm_rate"]
    )


def _fallback_key(row):
    return (
        row["validation_no_output_ticket_recall"],
        row["validation_thrust_loss_ticket_recall"],
        -row["validation_false_maintenance_tickets"],
        -row["validation_temporal_normal_false_alarm_rate"],
        row["validation_joint_macro_f1"],
    )


def _checkpoint(
    model,
    initial,
    dataset,
    dataset_path,
    initial_path,
    epoch,
    row,
    constraints,
    training_policy,
):
    return {
        "model_state_dict": model.state_dict(),
        "input_dim": SIX_DOF_MODEL_INPUT_DIM,
        "raw_feature_dim": SIX_DOF_RAW_FEATURE_DIM,
        "feature_names": SIX_DOF_RAW_FEATURE_NAMES,
        "mode_names": FAULT_MODE_NAMES,
        "location_names": THRUSTER_NAMES,
        "mean": initial["mean"].cpu(),
        "std": initial["std"].cpu(),
        "best_epoch": int(epoch),
        "validation_joint_macro_f1": row[
            "validation_joint_macro_f1"
        ],
        "validation_selection_metrics": row,
        "validation_constraints": constraints,
        "structured_fusion": bool(
            initial.get("structured_fusion", True)
        ),
        "dataset_version": dataset.get("dataset_version"),
        "diagnostic_feature_profile": dataset.get(
            "diagnostic_feature_profile",
            "fixed_nominal_generalized_mass_v3",
        ),
        "initial_checkpoint": str(initial_path),
        "initial_checkpoint_sha256": _sha256(initial_path),
        "training_dataset": str(dataset_path),
        "training_dataset_sha256": _sha256(dataset_path),
        "training_policy": training_policy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--temporal-config", type=Path, required=True)
    parser.add_argument("--ticket-config", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, split_indices, split_missions = load_and_validate_dataset(
        args.dataset
    )
    initial = torch.load(
        args.initial_checkpoint, map_location="cpu", weights_only=False
    )
    mean, std = initial["mean"].float(), initial["std"].float()
    temporal_config = TemporalDecisionConfig(
        **_json(args.temporal_config)
    )
    ticket_config = MaintenanceTicketConfig(**_json(args.ticket_config))
    args.results_dir.mkdir(parents=True, exist_ok=True)

    loaders = {
        name: make_loader(
            dataset,
            split_indices[name],
            mean,
            std,
            args.batch_size,
            shuffle=name == "train",
            seed=args.seed,
        )
        for name in ("train", "validation")
    }
    mode_weights, mode_counts = class_weights(
        dataset["y_mode"][split_indices["train"]], len(FAULT_MODE_NAMES)
    )
    train_modes = dataset["y_mode"][split_indices["train"]]
    train_locations = dataset["y_location"][split_indices["train"]]
    location_weights, location_counts = class_weights(
        train_locations[train_modes != 0] - 1, len(THRUSTER_NAMES)
    )
    mode_loss_fn = nn.CrossEntropyLoss(weight=mode_weights.to(device))
    location_loss_fn = nn.CrossEntropyLoss(
        weight=location_weights.to(device)
    )
    model = AUVSixDOFMultiTaskDetector(
        input_dim=int(initial["input_dim"]),
        structured_fusion=bool(initial.get("structured_fusion", True)),
    ).to(device)
    model.load_state_dict(initial["model_state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    constraints = {
        "minimum_no_output_ticket_recall": 1.0,
        "minimum_thrust_loss_ticket_recall": 0.90,
        "maximum_false_maintenance_tickets": 1,
        "maximum_temporal_normal_false_alarm_rate": 0.08,
    }
    training_policy = {
        "initialization": "frozen V3.1 checkpoint",
        "normalization": "preserve V3.1 mean and std",
        "rehearsal": "combined old and maneuver-pair training split",
        "candidate_selection": (
            "validation constraints first, then maximum joint macro F1"
        ),
        "test_usage": "none during training and model selection",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "seed": args.seed,
    }
    history = []
    best_eligible = None
    best_fallback = None
    best_path = args.results_dir / "best_model.pth"
    fallback_path = args.results_dir / "best_fallback_model.pth"

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    for name in ("train", "validation", "test"):
        print(
            f"{name}: {len(split_indices[name])} windows, "
            f"{len(split_missions[name])} missions"
        )
    print(f"Training mode counts: {mode_counts.tolist()}")
    print(f"Training location counts: {location_counts.tolist()}")

    for epoch in range(0, args.epochs + 1):
        train_loss = None
        if epoch > 0:
            model.train()
            loss_sum = 0.0
            sample_count = 0
            for X, y_mode, y_location in loaders["train"]:
                X = X.to(device)
                y_mode = y_mode.to(device)
                y_location = y_location.to(device)
                optimizer.zero_grad(set_to_none=True)
                mode_logits, location_logits = model(X)
                loss = multitask_loss(
                    mode_logits,
                    location_logits,
                    y_mode,
                    y_location,
                    mode_loss_fn,
                    location_loss_fn,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0
                )
                optimizer.step()
                loss_sum += loss.item() * len(X)
                sample_count += len(X)
            train_loss = loss_sum / max(sample_count, 1)

        raw, operations, tickets = _validation_metrics(
            model,
            loaders["validation"],
            dataset,
            split_indices["validation"],
            device,
            mode_loss_fn,
            location_loss_fn,
            temporal_config,
            ticket_config,
        )
        row = _row(
            epoch,
            train_loss,
            optimizer.param_groups[0]["lr"],
            raw,
            operations,
            tickets,
        )
        row["eligible"] = _eligible(row, constraints)
        history.append(row)
        print(
            f"Epoch {epoch:02d} | joint F1 "
            f"{row['validation_joint_macro_f1']:.4f} | temporal FAR "
            f"{100 * row['validation_temporal_normal_false_alarm_rate']:.2f}% "
            f"| tickets NO {row['validation_no_output_ticket_recall']:.3f} "
            f"TL {row['validation_thrust_loss_ticket_recall']:.3f} "
            f"false {row['validation_false_maintenance_tickets']} "
            f"| eligible {row['eligible']}"
        )

        checkpoint = _checkpoint(
            model,
            initial,
            dataset,
            args.dataset,
            args.initial_checkpoint,
            epoch,
            row,
            constraints,
            training_policy,
        )
        if (
            best_fallback is None
            or _fallback_key(row) > _fallback_key(best_fallback)
        ):
            best_fallback = row
            torch.save(checkpoint, fallback_path)
        if row["eligible"] and (
            best_eligible is None
            or row["validation_joint_macro_f1"]
            > best_eligible["validation_joint_macro_f1"] + 1e-8
        ):
            best_eligible = row
            torch.save(checkpoint, best_path)

    with (args.results_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (args.results_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    selected_path = best_path if best_eligible is not None else fallback_path
    selected_row = best_eligible or best_fallback
    summary = {
        "selection_status": (
            "eligible_candidate_selected"
            if best_eligible is not None
            else "no_eligible_candidate_fallback_only"
        ),
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": _sha256(selected_path),
        "selected_validation_metrics": selected_row,
        "best_eligible_validation_metrics": best_eligible,
        "best_fallback_validation_metrics": best_fallback,
        "validation_constraints": constraints,
        "training_policy": training_policy,
        "temporal_config": asdict(temporal_config),
        "ticket_config": asdict(ticket_config),
        "dataset": str(args.dataset),
        "dataset_version": dataset.get("dataset_version"),
        "initial_checkpoint": str(args.initial_checkpoint),
    }
    (args.results_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
