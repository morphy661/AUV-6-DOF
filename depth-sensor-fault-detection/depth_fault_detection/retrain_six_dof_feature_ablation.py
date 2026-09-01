"""Retrain controlled feature-ablation variants of the 0-300 m V3 model.

All variants use the same dataset split, architecture, initialization seed,
loss, optimizer, and validation joint-macro-F1 checkpoint rule.  The only
experimental factor is which normalized raw and temporal-difference channels
are masked.  Test is evaluated only after each validation-selected checkpoint
is frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
MATH_MODEL_SRC = REPO_ROOT / "math-model" / "src"
if str(MATH_MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MATH_MODEL_SRC))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from ablate_six_dof_false_alarm_features import (  # noqa: E402
    ABLATION_GROUPS,
    MaskedNormalizedWindowDataset,
    _compact_split_result,
    _load_json,
    _sha256,
)
from audit_six_dof_false_alarms import _build_split_audit  # noqa: E402
from diagnosis.maintenance_ticket_policy import (  # noqa: E402
    MaintenanceTicketConfig,
)
from diagnosis.temporal_fault_decision import (  # noqa: E402
    TemporalDecisionConfig,
)
from model_six_dof_multitask import AUVSixDOFMultiTaskDetector  # noqa: E402
from train_six_dof_multitask import (  # noqa: E402
    class_weights,
    evaluate,
    fit_training_statistics,
    load_and_validate_dataset,
    multitask_loss,
    set_seed,
)
from utils.six_dof_feature_extractor import (  # noqa: E402
    FAULT_MODE_NAMES,
    SIX_DOF_MODEL_INPUT_DIM,
    SIX_DOF_RAW_FEATURE_DIM,
    SIX_DOF_RAW_FEATURE_NAMES,
    THRUSTER_NAMES,
)


DEFAULT_VARIANTS = (
    "baseline",
    "motion_loss_evidence",
    "simplified_response_residual",
    "motion_plus_response",
)


def _loader(
    dataset,
    indices,
    mean,
    std,
    masked_indices,
    batch_size,
    shuffle,
    seed,
):
    windows = MaskedNormalizedWindowDataset(
        dataset["X"],
        dataset["y_mode"].long(),
        dataset["y_location"].long(),
        indices,
        mean,
        std,
        masked_raw_indices=masked_indices,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        windows,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def _train_variant(
    variant,
    masked_names,
    dataset,
    split_indices,
    mean,
    std,
    mode_loss,
    location_loss,
    temporal_config,
    ticket_config,
    device,
    args,
):
    set_seed(args.seed)
    feature_names = tuple(dataset["feature_names"])
    missing = sorted(set(masked_names) - set(feature_names))
    if missing:
        raise ValueError(f"{variant} has unknown features: {missing}")
    masked_indices = tuple(
        feature_names.index(name) for name in masked_names
    )
    loaders = {
        split: _loader(
            dataset,
            split_indices[split],
            mean,
            std,
            masked_indices,
            args.batch_size,
            shuffle=split == "train",
            seed=args.seed,
        )
        for split in ("train", "validation", "test")
    }
    model = AUVSixDOFMultiTaskDetector(
        input_dim=SIX_DOF_MODEL_INPUT_DIM
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    variant_dir = args.output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = variant_dir / "best_model.pth"
    history = []
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"\n=== {variant} ({len(masked_names)} raw features masked) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for values, mode, location in loaders["train"]:
            values = values.to(device)
            mode = mode.to(device)
            location = location.to(device)
            optimizer.zero_grad(set_to_none=True)
            mode_logits, location_logits = model(values)
            loss = multitask_loss(
                mode_logits,
                location_logits,
                mode,
                location,
                mode_loss,
                location_loss,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * len(values)
            total_samples += len(values)

        validation = evaluate(
            model,
            loaders["validation"],
            device,
            mode_loss,
            location_loss,
        )
        scheduler.step(validation["joint_macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_samples, 1),
            "validation_loss": validation["loss"],
            "validation_mode_macro_f1": validation["mode_macro_f1"],
            "validation_location_macro_f1": validation[
                "location_macro_f1"
            ],
            "validation_joint_macro_f1": validation["joint_macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"{variant} epoch {epoch:02d} | "
            f"train {row['train_loss']:.4f} | "
            f"val joint {row['validation_joint_macro_f1']:.4f} | "
            f"val mode {row['validation_mode_macro_f1']:.4f}",
            flush=True,
        )

        if validation["joint_macro_f1"] > best_score + 1e-6:
            best_score = float(validation["joint_macro_f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": SIX_DOF_MODEL_INPUT_DIM,
                "raw_feature_dim": SIX_DOF_RAW_FEATURE_DIM,
                "feature_names": SIX_DOF_RAW_FEATURE_NAMES,
                "mode_names": FAULT_MODE_NAMES,
                "location_names": THRUSTER_NAMES,
                "mean": mean,
                "std": std,
                "best_epoch": best_epoch,
                "validation_joint_macro_f1": best_score,
                "structured_fusion": model.structured_fusion,
                "ablation_variant": variant,
                "masked_raw_features": list(masked_names),
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"{variant}: early stopping after epoch {epoch}",
                    flush=True,
                )
                break

    with (variant_dir / "training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    result = {
        "masked_raw_features": list(masked_names),
        "best_epoch": best_epoch,
        "validation_selection_joint_macro_f1": best_score,
        "history": history,
        "checkpoint_sha256": _sha256(checkpoint_path),
    }
    for split in ("validation", "test"):
        predictions = evaluate(
            model,
            loaders[split],
            device,
            mode_loss,
            location_loss,
        )
        audit = _build_split_audit(
            dataset,
            split,
            split_indices[split].cpu().numpy(),
            predictions,
            temporal_config,
            ticket_config,
        )
        result[split] = _compact_split_result(predictions, audit)
    return result


def _summary_rows(result):
    rows = []
    for variant, values in result["variants"].items():
        for split in ("validation", "test"):
            metrics = values[split]
            false_alarm = metrics["false_alarm"]
            windows = metrics["window_metrics"]
            tickets = metrics["formal_ticket_metrics"]
            rows.append({
                "variant": variant,
                "split": split,
                "best_epoch": values["best_epoch"],
                "masked_raw_feature_count": len(
                    values["masked_raw_features"]
                ),
                "raw_false_alarm_rate": false_alarm["raw"],
                "temporal_false_alarm_rate": false_alarm["temporal"],
                "wholly_normal_mission_raw_far": false_alarm[
                    "wholly_normal_mission_raw"
                ],
                "pre_fault_segment_raw_far": false_alarm[
                    "pre_fault_segment_raw"
                ],
                "raw_episode_count": false_alarm["raw_episode_count"],
                "raw_episodes_at_least_8_s": false_alarm[
                    "raw_episodes_at_least_8_s"
                ],
                "mode_macro_f1": metrics["mode_macro_f1"],
                "joint_macro_f1": metrics["joint_macro_f1"],
                "normal_recall": windows["normal_recall"],
                "no_output_recall": windows["no_output_recall"],
                "thrust_loss_recall": windows["thrust_loss_recall"],
                "fault_detection_recall": windows[
                    "fault_detection_recall"
                ],
                "formal_ticket_count": tickets[
                    "formal_maintenance_ticket_count"
                ],
                "false_maintenance_tickets": tickets[
                    "false_maintenance_tickets"
                ],
                "no_output_ticket_recall": tickets[
                    "no_output_ticket_recall"
                ],
                "thrust_loss_ticket_recall": tickets[
                    "thrust_loss_ticket_recall"
                ],
                "mean_ticket_detection_delay_s": tickets[
                    "mean_ticket_detection_delay_s"
                ],
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            THIS_DIR / "data" /
            "simulation_dataset_six_dof_hybrid_telemetry_0_300m_focus_v3.pth"
        ),
    )
    parser.add_argument(
        "--temporal-config",
        type=Path,
        default=(
            THIS_DIR / "results" /
            "six_dof_hybrid_telemetry_0_300m_focus_v3_temporal" /
            "temporal_decision_config.json"
        ),
    )
    parser.add_argument(
        "--ticket-config",
        type=Path,
        default=(
            THIS_DIR / "results" /
            "six_dof_hybrid_telemetry_0_300m_focus_v3_temporal" /
            "maintenance_ticket_config.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            THIS_DIR / "results" /
            "six_dof_feature_ablation_retrained_v1_20260821"
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(ABLATION_GROUPS),
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, split_indices, split_missions = load_and_validate_dataset(
        args.dataset
    )
    mean, std = fit_training_statistics(
        dataset["X"], split_indices["train"]
    )
    train_modes = dataset["y_mode"][split_indices["train"]]
    train_locations = dataset["y_location"][split_indices["train"]]
    mode_weights, _ = class_weights(train_modes, len(FAULT_MODE_NAMES))
    location_weights, _ = class_weights(
        train_locations[train_modes != 0] - 1, len(THRUSTER_NAMES)
    )
    mode_loss = nn.CrossEntropyLoss(weight=mode_weights.to(device))
    location_loss = nn.CrossEntropyLoss(
        weight=location_weights.to(device)
    )
    temporal_config = TemporalDecisionConfig(**_load_json(
        args.temporal_config
    ))
    ticket_config = MaintenanceTicketConfig(**_load_json(
        args.ticket_config
    ))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variants = {}
    for variant in args.variants:
        variants[variant] = _train_variant(
            variant,
            ABLATION_GROUPS[variant],
            dataset,
            split_indices,
            mean,
            std,
            mode_loss,
            location_loss,
            temporal_config,
            ticket_config,
            device,
            args,
        )

    result = {
        "experiment_id": "six_dof_feature_ablation_retrained_v1_20260821",
        "experiment_type": "controlled_retraining_feature_ablation",
        "selection_policy": (
            "validation joint macro F1 only; test evaluated after checkpoint "
            "freeze and not used for model selection"
        ),
        "controlled_factors": (
            "same dataset splits, model architecture, seed, optimizer, loss, "
            "normalization, temporal policy, and ticket policy"
        ),
        "device": str(device),
        "dataset_version": dataset["dataset_version"],
        "dataset_sha256": _sha256(args.dataset),
        "split_missions": {
            name: len(values) for name, values in split_missions.items()
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "early_stopping_patience": args.early_stopping_patience,
            "seed": args.seed,
        },
        "variants": variants,
    }
    (args.output_dir / "retrained_feature_ablation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = _summary_rows(result)
    with (args.output_dir / "retrained_feature_ablation_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\n=== final summary ===", flush=True)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
