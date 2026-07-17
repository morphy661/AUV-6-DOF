"""Mission-level cross-validation without touching the held-out OOD test."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model_six_dof_multitask import AUVSixDOFMultiTaskDetector
from train_six_dof_multitask import (
    class_weights,
    evaluate,
    fit_training_statistics,
    load_and_validate_dataset,
    make_loader,
    multitask_loss,
    set_seed,
)
from utils.six_dof_feature_extractor import (
    FAULT_MODE_NAMES,
    THRUSTER_NAMES,
)


THIS_DIR = Path(__file__).resolve().parent
BASELINE_RAW_FEATURE_DIM = 61


def select_feature_set(dataset, feature_set):
    """Return a shallow dataset view for a same-mission feature ablation."""

    if feature_set == "hybrid":
        return dataset
    if feature_set != "baseline":
        raise ValueError(f"unsupported feature set {feature_set!r}")
    selected = dict(dataset)
    selected["X"] = dataset["X"][..., :BASELINE_RAW_FEATURE_DIM]
    return selected


def build_scenario_stratified_mission_folds(
    dataset,
    fold_count=5,
    seed=42,
):
    """Assign whole in-domain missions to folds within each scenario."""

    metadata = dataset["mission_metadata"]
    in_domain = {
        int(mission_id): values
        for mission_id, values in metadata.items()
        if values["split"] in {"train", "validation"}
    }
    test_missions = {
        int(mission_id)
        for mission_id, values in metadata.items()
        if values["split"] == "test"
    }
    scenarios = sorted({values["scenario"] for values in in_domain.values()})
    rng = np.random.default_rng(seed)
    validation_missions = [set() for _ in range(fold_count)]

    for scenario in scenarios:
        mission_group = np.array([
            mission_id
            for mission_id, values in in_domain.items()
            if values["scenario"] == scenario
        ], dtype=np.int64)
        if len(mission_group) < fold_count:
            raise ValueError(
                f"scenario {scenario!r} has {len(mission_group)} in-domain "
                f"missions, fewer than {fold_count} folds"
            )
        rng.shuffle(mission_group)
        for fold_index, chunk in enumerate(
            np.array_split(mission_group, fold_count)
        ):
            validation_missions[fold_index].update(chunk.tolist())

    all_in_domain = set(in_domain)
    mission_ids = dataset["mission_ids"].cpu().numpy()
    folds = []
    for fold_index, validation_set in enumerate(validation_missions):
        training_set = all_in_domain - validation_set
        if training_set & validation_set:
            raise RuntimeError("cross-validation mission leakage")
        if (training_set | validation_set) & test_missions:
            raise RuntimeError("held-out test mission entered cross-validation")
        folds.append({
            "fold": fold_index + 1,
            "train_missions": sorted(training_set),
            "validation_missions": sorted(validation_set),
            "train_indices": np.flatnonzero(
                np.isin(mission_ids, sorted(training_set))
            ),
            "validation_indices": np.flatnonzero(
                np.isin(mission_ids, sorted(validation_set))
            ),
        })
    return folds


def train_fold(dataset, fold, args, device):
    fold_seed = args.seed + fold["fold"]
    set_seed(fold_seed)
    train_indices = fold["train_indices"]
    validation_indices = fold["validation_indices"]
    mean, std = fit_training_statistics(dataset["X"], train_indices)
    train_loader = make_loader(
        dataset,
        train_indices,
        mean,
        std,
        args.batch_size,
        shuffle=True,
        seed=fold_seed,
    )
    validation_loader = make_loader(
        dataset,
        validation_indices,
        mean,
        std,
        args.batch_size,
        shuffle=False,
        seed=fold_seed,
    )

    train_modes = dataset["y_mode"][train_indices]
    train_locations = dataset["y_location"][train_indices]
    mode_weights, _ = class_weights(train_modes, len(FAULT_MODE_NAMES))
    location_weights, _ = class_weights(
        train_locations[train_modes != 0] - 1,
        len(THRUSTER_NAMES),
    )
    mode_loss_fn = torch.nn.CrossEntropyLoss(weight=mode_weights.to(device))
    location_loss_fn = torch.nn.CrossEntropyLoss(
        weight=location_weights.to(device)
    )
    model_input_dim = 2 * int(dataset["X"].shape[-1])
    structured_fusion = (
        args.architecture == "structured"
        or (
            args.architecture == "auto"
            and model_input_dim
            == AUVSixDOFMultiTaskDetector.HYBRID_INPUT_DIM
        )
    )
    model = AUVSixDOFMultiTaskDetector(
        input_dim=model_input_dim,
        structured_fusion=structured_fusion,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for X, y_mode, y_location in train_loader:
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        validation = evaluate(
            model,
            validation_loader,
            device,
            mode_loss_fn,
            location_loss_fn,
        )
        scheduler.step(validation["joint_macro_f1"])
        score = validation["joint_macro_f1"]
        print(
            f"Fold {fold['fold']} epoch {epoch:02d}: "
            f"mode={validation['mode_macro_f1']:.4f}, "
            f"location={validation['location_macro_f1']:.4f}, "
            f"joint={score:.4f}"
        )
        if best is None or score > best["joint_macro_f1"] + 1e-6:
            best = {
                "best_epoch": epoch,
                "mode_macro_f1": validation["mode_macro_f1"],
                "location_macro_f1": validation["location_macro_f1"],
                "joint_macro_f1": score,
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                break

    best.update({
        "fold": fold["fold"],
        "train_missions": len(fold["train_missions"]),
        "validation_missions": len(fold["validation_missions"]),
        "train_windows": len(train_indices),
        "validation_windows": len(validation_indices),
    })
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            THIS_DIR
            / "data"
            / "simulation_dataset_six_dof_hybrid_telemetry.pth"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            THIS_DIR
            / "results"
            / "six_dof_hybrid_telemetry_cv"
            / "cross_validation_summary.json"
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-set",
        choices=("hybrid", "baseline"),
        default="hybrid",
        help="Use all 109 raw features or the original first 61 features.",
    )
    parser.add_argument(
        "--architecture",
        choices=("auto", "flat", "structured"),
        default="auto",
    )
    args = parser.parse_args()

    dataset, _, _ = load_and_validate_dataset(args.dataset)
    dataset = select_feature_set(dataset, args.feature_set)
    if args.architecture == "structured" and args.feature_set != "hybrid":
        parser.error("structured architecture requires --feature-set hybrid")
    folds = build_scenario_stratified_mission_folds(
        dataset, fold_count=args.folds, seed=args.seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; folds: {len(folds)}")
    fold_results = [
        train_fold(dataset, fold, args, device) for fold in folds
    ]
    summary = {
        "feature_set": args.feature_set,
        "architecture": (
            "structured"
            if args.architecture == "structured"
            or (args.architecture == "auto" and args.feature_set == "hybrid")
            else "flat"
        ),
        "raw_feature_dim": int(dataset["X"].shape[-1]),
        "model_input_dim": 2 * int(dataset["X"].shape[-1]),
        "folds": fold_results,
    }
    for metric in (
        "mode_macro_f1",
        "location_macro_f1",
        "joint_macro_f1",
    ):
        values = np.array([result[metric] for result in fold_results])
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
