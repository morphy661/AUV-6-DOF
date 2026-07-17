"""Train and evaluate the leakage-safe six-thruster multi-task detector."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
MATH_MODEL_SRC = REPO_ROOT / "math-model" / "src"
if str(MATH_MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MATH_MODEL_SRC))

from model_six_dof_multitask import (
    AUVSixDOFMultiTaskDetector,
    combine_multitask_predictions,
)
from utils.six_dof_feature_extractor import (
    FAULT_LOCATION_NAMES,
    FAULT_MODE_NAMES,
    JOINT_FAULT_NAMES,
    PRIVILEGED_SIMULATOR_FIELDS,
    SIX_DOF_MODEL_INPUT_DIM,
    SIX_DOF_RAW_FEATURE_DIM,
    SIX_DOF_RAW_FEATURE_NAMES,
    THRUSTER_NAMES,
)


class NormalizedWindowDataset(Dataset):
    """Add temporal differences and training-only normalization on demand."""

    def __init__(self, X, y_mode, y_location, indices, mean, std):
        self.X = X
        self.y_mode = y_mode
        self.y_location = y_location
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.mean = mean.float()
        self.std = std.float()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        raw = self.X[index].float()
        difference = torch.zeros_like(raw)
        difference[1:] = raw[1:] - raw[:-1]
        augmented = torch.cat((raw, difference), dim=-1)
        normalized = (augmented - self.mean) / (self.std + 1e-8)
        return normalized, self.y_mode[index], self.y_location[index]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_and_validate_dataset(path):
    dataset = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "X",
        "y_mode",
        "y_location",
        "y_joint",
        "mission_ids",
        "split_indices",
        "feature_names",
        "raw_feature_dim",
        "model_input_dim",
    }
    missing = sorted(required - set(dataset))
    if missing:
        raise ValueError(f"dataset is missing required keys: {missing}")

    X = dataset["X"]
    if X.ndim != 3 or tuple(X.shape[1:]) != (
        int(dataset["sequence_length"]), SIX_DOF_RAW_FEATURE_DIM
    ):
        raise ValueError(f"unexpected raw feature tensor shape {tuple(X.shape)}")
    if int(dataset["raw_feature_dim"]) != SIX_DOF_RAW_FEATURE_DIM:
        raise ValueError("raw feature dimension metadata mismatch")
    if int(dataset["model_input_dim"]) != SIX_DOF_MODEL_INPUT_DIM:
        raise ValueError("model input dimension metadata mismatch")
    if tuple(dataset["feature_names"]) != SIX_DOF_RAW_FEATURE_NAMES:
        raise ValueError("feature schema does not match the six-DOF extractor")

    feature_text = " ".join(dataset["feature_names"])
    forbidden_names = [
        name for name in PRIVILEGED_SIMULATOR_FIELDS if name in feature_text
    ]
    if forbidden_names:
        raise ValueError(f"privileged feature names detected: {forbidden_names}")

    sample_count = len(X)
    for key in ("y_mode", "y_location", "y_joint", "mission_ids"):
        if dataset[key].shape != (sample_count,):
            raise ValueError(f"{key} shape does not match X")

    modes = dataset["y_mode"].long()
    locations = dataset["y_location"].long()
    expected_joint = combine_multitask_predictions(modes, locations)
    if not torch.equal(expected_joint, dataset["y_joint"].long()):
        raise ValueError("joint labels are inconsistent with mode/location labels")

    split_indices = {
        name: torch.as_tensor(indices, dtype=torch.long)
        for name, indices in dataset["split_indices"].items()
    }
    if set(split_indices) != {"train", "validation", "test"}:
        raise ValueError("dataset split names must be train/validation/test")
    index_sets = {name: set(value.tolist()) for name, value in split_indices.items()}
    if (
        index_sets["train"] & index_sets["validation"]
        or index_sets["train"] & index_sets["test"]
        or index_sets["validation"] & index_sets["test"]
    ):
        raise ValueError("window indices overlap across dataset splits")
    if set.union(*index_sets.values()) != set(range(sample_count)):
        raise ValueError("dataset splits do not cover every window exactly once")

    mission_ids = dataset["mission_ids"].long()
    split_missions = {
        name: set(mission_ids[indices].tolist())
        for name, indices in split_indices.items()
    }
    if (
        split_missions["train"] & split_missions["validation"]
        or split_missions["train"] & split_missions["test"]
        or split_missions["validation"] & split_missions["test"]
    ):
        raise ValueError("mission leakage detected across dataset splits")

    for name, indices in split_indices.items():
        present_joint = set(dataset["y_joint"][indices].tolist())
        if present_joint != set(range(len(JOINT_FAULT_NAMES))):
            raise ValueError(
                f"{name} split has incomplete joint labels: {sorted(present_joint)}"
            )
    return dataset, split_indices, split_missions


def fit_training_statistics(X, train_indices, batch_size=512):
    model_input_dim = 2 * int(X.shape[-1])
    feature_sum = torch.zeros(model_input_dim, dtype=torch.float64)
    feature_square_sum = torch.zeros_like(feature_sum)
    observation_count = 0
    indices = torch.as_tensor(train_indices, dtype=torch.long)
    for start in range(0, len(indices), batch_size):
        raw = X[indices[start:start + batch_size]].double()
        difference = torch.zeros_like(raw)
        difference[:, 1:] = raw[:, 1:] - raw[:, :-1]
        augmented = torch.cat((raw, difference), dim=-1)
        feature_sum += augmented.sum(dim=(0, 1))
        feature_square_sum += (augmented * augmented).sum(dim=(0, 1))
        observation_count += augmented.shape[0] * augmented.shape[1]

    mean = feature_sum / observation_count
    variance = feature_square_sum / observation_count - mean * mean
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    return mean.float(), std.float()


def class_weights(labels, class_count):
    counts = torch.bincount(labels.long(), minlength=class_count).float()
    if torch.any(counts == 0):
        raise ValueError(f"training labels have empty classes: {counts.tolist()}")
    weights = 1.0 / torch.sqrt(counts)
    return weights / weights.mean(), counts.long()


def multitask_loss(
    mode_logits,
    location_logits,
    y_mode,
    y_location,
    mode_loss_fn,
    location_loss_fn,
):
    """Train location only where a thruster fault is actually present."""

    loss = mode_loss_fn(mode_logits, y_mode)
    fault_mask = y_mode != 0
    if torch.any(fault_mask):
        loss = loss + location_loss_fn(
            location_logits[fault_mask],
            y_location[fault_mask] - 1,
        )
    return loss


def make_loader(dataset, indices, mean, std, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    windows = NormalizedWindowDataset(
        dataset["X"],
        dataset["y_mode"].long(),
        dataset["y_location"].long(),
        indices,
        mean,
        std,
    )
    return DataLoader(
        windows,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def evaluate(model, loader, device, mode_loss_fn, location_loss_fn):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    mode_true, mode_pred = [], []
    location_true, location_pred = [], []
    mode_probabilities = []
    location_probabilities = []
    with torch.no_grad():
        for X, y_mode, y_location in loader:
            X = X.to(device)
            y_mode = y_mode.to(device)
            y_location = y_location.to(device)
            mode_logits, location_logits = model(X)
            loss = multitask_loss(
                mode_logits,
                location_logits,
                y_mode,
                y_location,
                mode_loss_fn,
                location_loss_fn,
            )
            total_loss += loss.item() * len(X)
            total_samples += len(X)
            mode_true.extend(y_mode.cpu().tolist())
            location_true.extend(y_location.cpu().tolist())
            mode_pred.extend(mode_logits.argmax(dim=1).cpu().tolist())
            mode_probabilities.append(
                torch.softmax(mode_logits, dim=1).cpu().numpy()
            )
            location_probabilities.append(
                torch.softmax(location_logits, dim=1).cpu().numpy()
            )
            predicted_mode = mode_logits.argmax(dim=1)
            predicted_location = location_logits.argmax(dim=1) + 1
            predicted_location = torch.where(
                predicted_mode == 0,
                torch.zeros_like(predicted_location),
                predicted_location,
            )
            location_pred.extend(predicted_location.cpu().tolist())

    mode_true = np.asarray(mode_true, dtype=np.int64)
    mode_pred = np.asarray(mode_pred, dtype=np.int64)
    location_true = np.asarray(location_true, dtype=np.int64)
    location_pred = np.asarray(location_pred, dtype=np.int64)
    joint_true = combine_multitask_predictions(
        mode_true, location_true
    ).numpy()
    joint_pred = combine_multitask_predictions(
        mode_pred, location_pred
    ).numpy()
    mode_probabilities = np.concatenate(mode_probabilities, axis=0)
    location_probabilities = np.concatenate(
        location_probabilities, axis=0
    )
    return {
        "loss": total_loss / max(total_samples, 1),
        "mode_macro_f1": f1_score(
            mode_true, mode_pred, labels=range(3), average="macro", zero_division=0
        ),
        "location_macro_f1": f1_score(
            location_true,
            location_pred,
            labels=range(7),
            average="macro",
            zero_division=0,
        ),
        "joint_macro_f1": f1_score(
            joint_true,
            joint_pred,
            labels=range(13),
            average="macro",
            zero_division=0,
        ),
        "mode_true": mode_true,
        "mode_pred": mode_pred,
        "location_true": location_true,
        "location_pred": location_pred,
        "joint_true": joint_true,
        "joint_pred": joint_pred,
        "mode_probabilities": mode_probabilities,
        "location_probabilities": location_probabilities,
    }


def save_report(y_true, y_pred, names, path):
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(names))),
        target_names=list(names),
        digits=4,
        zero_division=0,
    )
    path.write_text(report, encoding="utf-8")
    return report


def operational_test_metrics(dataset, test_indices, test_predictions):
    """Report mission-level false alarms, detection success, and delay."""

    indices = torch.as_tensor(test_indices, dtype=torch.long)
    mission_ids = dataset["mission_ids"][indices].cpu().numpy()
    end_times = dataset["window_end_times"][indices].cpu().numpy()
    joint_true = test_predictions["joint_true"]
    joint_pred = test_predictions["joint_pred"]
    mode_pred = test_predictions.get("mode_pred")
    if mode_pred is None:
        mode_pred = np.where(
            joint_pred == 0, 0, np.where(joint_pred <= 6, 1, 2)
        )
    metadata = dataset["mission_metadata"]

    false_alarm_count = 0
    normal_window_count = 0
    mode_detection_delays = []
    exact_diagnosis_delays = []
    mode_detected_missions = 0
    exactly_diagnosed_missions = 0
    total_fault_missions = 0
    severity_records = {
        "0.30-0.45": [],
        "0.45-0.60": [],
        "0.60-0.75": [],
    }

    for mission_id in np.unique(mission_ids):
        mask = mission_ids == mission_id
        mission_true = joint_true[mask]
        mission_pred = joint_pred[mask]
        mission_mode_pred = mode_pred[mask]
        mission_times = end_times[mask]
        normal_mask = mission_true == 0
        false_alarm_count += int(np.sum(mission_pred[normal_mask] != 0))
        normal_window_count += int(np.sum(normal_mask))

        fault_labels = np.unique(mission_true[mission_true != 0])
        if len(fault_labels) == 0:
            continue
        target = int(fault_labels[0])
        target_mode = 1 if target <= 6 else 2
        total_fault_missions += 1
        fault_start = float(
            metadata[int(mission_id)]["parameters"]["fault_start_time_s"]
        )
        correct_mode_times = mission_times[
            (mission_times >= fault_start)
            & (mission_mode_pred == target_mode)
        ]
        if len(correct_mode_times):
            mode_detected_missions += 1
            mode_detection_delays.append(
                max(0.0, float(np.min(correct_mode_times) - fault_start))
            )
        correct_joint_times = mission_times[
            (mission_times >= fault_start) & (mission_pred == target)
        ]
        if len(correct_joint_times):
            exactly_diagnosed_missions += 1
            exact_diagnosis_delays.append(
                max(0.0, float(np.min(correct_joint_times) - fault_start))
            )

        efficiency = metadata[int(mission_id)]["parameters"].get(
            "thrust_efficiency"
        )
        if target >= 7 and efficiency is not None:
            if efficiency < 0.45:
                bin_name = "0.30-0.45"
            elif efficiency < 0.60:
                bin_name = "0.45-0.60"
            else:
                bin_name = "0.60-0.75"
            fault_mask = mission_true == target
            severity_records[bin_name].extend(
                (mission_pred[fault_mask] == target).astype(float).tolist()
            )

    return {
        "normal_window_false_alarm_rate": (
            false_alarm_count / max(normal_window_count, 1)
        ),
        "mode_detection_rate": (
            mode_detected_missions / max(total_fault_missions, 1)
        ),
        "exact_diagnosis_rate": (
            exactly_diagnosed_missions / max(total_fault_missions, 1)
        ),
        "mode_detected_missions": mode_detected_missions,
        "exactly_diagnosed_missions": exactly_diagnosed_missions,
        "total_fault_missions": total_fault_missions,
        "mean_mode_detection_delay_s": (
            float(np.mean(mode_detection_delays))
            if mode_detection_delays else None
        ),
        "median_mode_detection_delay_s": (
            float(np.median(mode_detection_delays))
            if mode_detection_delays else None
        ),
        "mean_exact_diagnosis_delay_s": (
            float(np.mean(exact_diagnosis_delays))
            if exact_diagnosis_delays else None
        ),
        "median_exact_diagnosis_delay_s": (
            float(np.median(exact_diagnosis_delays))
            if exact_diagnosis_delays else None
        ),
        "thrust_loss_exact_accuracy_by_efficiency": {
            name: (float(np.mean(values)) if values else None)
            for name, values in severity_records.items()
        },
    }


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
        "--results-dir",
        type=Path,
        default=THIS_DIR / "results" / "six_dof_hybrid_telemetry",
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
    dataset, split_indices, split_missions = load_and_validate_dataset(args.dataset)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Raw dataset shape: {tuple(dataset['X'].shape)}")
    for name in ("train", "validation", "test"):
        print(
            f"{name}: {len(split_indices[name])} windows, "
            f"{len(split_missions[name])} missions"
        )

    mean, std = fit_training_statistics(dataset["X"], split_indices["train"])
    np.save(args.results_dir / "mean.npy", mean.numpy())
    np.save(args.results_dir / "std.npy", std.numpy())

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
        for name in ("train", "validation", "test")
    }
    mode_weights, mode_counts = class_weights(
        dataset["y_mode"][split_indices["train"]], len(FAULT_MODE_NAMES)
    )
    train_mode_labels = dataset["y_mode"][split_indices["train"]]
    train_location_labels = dataset["y_location"][split_indices["train"]]
    fault_train_locations = train_location_labels[train_mode_labels != 0] - 1
    location_weights, location_counts = class_weights(
        fault_train_locations,
        len(THRUSTER_NAMES),
    )
    print(f"Training mode counts: {mode_counts.tolist()}")
    print(f"Training location counts: {location_counts.tolist()}")

    model = AUVSixDOFMultiTaskDetector(
        input_dim=SIX_DOF_MODEL_INPUT_DIM
    ).to(device)
    mode_loss_fn = nn.CrossEntropyLoss(weight=mode_weights.to(device))
    location_loss_fn = nn.CrossEntropyLoss(weight=location_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    checkpoint_path = args.results_dir / "best_model.pth"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * len(X)
            total_samples += len(X)

        validation = evaluate(
            model,
            loaders["validation"],
            device,
            mode_loss_fn,
            location_loss_fn,
        )
        train_loss = total_loss / total_samples
        scheduler.step(validation["joint_macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation["loss"],
            "validation_mode_macro_f1": validation["mode_macro_f1"],
            "validation_location_macro_f1": validation["location_macro_f1"],
            "validation_joint_macro_f1": validation["joint_macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train {train_loss:.4f} | "
            f"val {validation['loss']:.4f} | "
            f"mode F1 {validation['mode_macro_f1']:.4f} | "
            f"location F1 {validation['location_macro_f1']:.4f} | "
            f"joint F1 {validation['joint_macro_f1']:.4f}"
        )

        if validation["joint_macro_f1"] > best_score + 1e-6:
            best_score = validation["joint_macro_f1"]
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
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping after epoch {epoch}")
                break

    with (args.results_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate(
        model,
        loaders["test"],
        device,
        mode_loss_fn,
        location_loss_fn,
    )
    reports = {
        "mode": save_report(
            test["mode_true"],
            test["mode_pred"],
            FAULT_MODE_NAMES,
            args.results_dir / "test_mode_report.txt",
        ),
        "location": save_report(
            test["location_true"],
            test["location_pred"],
            FAULT_LOCATION_NAMES,
            args.results_dir / "test_location_report.txt",
        ),
        "joint": save_report(
            test["joint_true"],
            test["joint_pred"],
            JOINT_FAULT_NAMES,
            args.results_dir / "test_joint_report.txt",
        ),
    }
    np.save(
        args.results_dir / "test_mode_confusion.npy",
        confusion_matrix(test["mode_true"], test["mode_pred"], labels=range(3)),
    )
    np.save(
        args.results_dir / "test_location_confusion.npy",
        confusion_matrix(
            test["location_true"], test["location_pred"], labels=range(7)
        ),
    )
    np.save(
        args.results_dir / "test_joint_confusion.npy",
        confusion_matrix(test["joint_true"], test["joint_pred"], labels=range(13)),
    )
    summary = {
        "device": str(device),
        "best_epoch": best_epoch,
        "validation_joint_macro_f1": best_score,
        "test_loss": test["loss"],
        "test_mode_macro_f1": test["mode_macro_f1"],
        "test_location_macro_f1": test["location_macro_f1"],
        "test_joint_macro_f1": test["joint_macro_f1"],
        "test_invalid_fault_location_rate": float(np.mean(
            (test["mode_pred"] != 0) & (test["location_pred"] == 0)
        )),
        "operational_metrics": operational_test_metrics(
            dataset,
            split_indices["test"],
            test,
        ),
        "reports": reports,
    }
    (args.results_dir / "test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))


if __name__ == "__main__":
    main()
