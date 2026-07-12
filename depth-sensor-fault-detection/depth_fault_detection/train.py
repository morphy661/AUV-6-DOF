import csv
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
MATH_MODEL_ROOT = REPO_ROOT / "math-model"
MATH_MODEL_SRC = MATH_MODEL_ROOT / "src"

for import_path in (MATH_MODEL_ROOT, MATH_MODEL_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from model import AUVFaultDetector
from my_config import *
print("Using device:", DEVICE)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("PyTorch CUDA version:", torch.version.cuda)
from my_utils import (
    plot_sample_sequences,
    plot_training_history,
    plot_confusion_matrix,
    plot_fault_examples,
    plot_residual_examples,
    label_names
)

# =======================================================
# Stage 3: 9-class Multi-sensor AI Fusion Diagnosis Training
# =======================================================
DATA_DIR = THIS_DIR / "data"
RESULTS_DIR = THIS_DIR / "results"
STAGE3_DATASET_PATH = DATA_DIR / "simulation_dataset_stage3_9class.pth"

EXPECTED_RAW_FEATURE_DIM = 20
EXPECTED_MODEL_INPUT_DIM = 40
SEQ_LEN = 50

BEST_MODEL_PATH = RESULTS_DIR / "best_model_stage3_9class.pth"
TRAINING_PLOT_PATH = RESULTS_DIR / "training_plot_stage3_9class.png"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history_stage3_9class.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "test_confusion_matrix_stage3_9class.png"
NORMALIZED_CONFUSION_MATRIX_PATH = RESULTS_DIR / "test_confusion_matrix_normalized_stage3_9class.png"
SEQUENCE_PLOT_PATH = RESULTS_DIR / "stage3_9class_sequences.png"
FAULT_EXAMPLE_DIR = RESULTS_DIR / "fault_examples_stage3_9class"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "test_classification_report_stage3_9class.txt"
CLASSIFICATION_REPORT_CSV_PATH = RESULTS_DIR / "test_metrics_stage3_9class.csv"
SPLIT_COUNTS_PATH = RESULTS_DIR / "dataset_split_counts_stage3_9class.csv"
SPLIT_MANIFEST_PATH = RESULTS_DIR / "dataset_split_manifest_stage3_9class.json"
MEAN_PATH = RESULTS_DIR / "mean_stage3_9class.npy"
STD_PATH = RESULTS_DIR / "std_stage3_9class.npy"


# ===============================
# 创建结果目录
# ===============================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_global_seed(RANDOM_SEED)

# ===============================
# 1. Load the Stage 3 9-class multi-sensor dataset.
# ===============================
dataset = torch.load(STAGE3_DATASET_PATH, map_location="cpu", weights_only=False)
print("Dataset keys:", dataset.keys())
data = dataset["X"]  # Expected raw shape: [N, 50, 20]
labels = dataset["y"]

print("=" * 70)
print("Stage 3 9-class multi-sensor dataset loaded.")
print(f"Original X shape: {data.shape}")
print(f"Label shape: {labels.shape}")

if "feature_names" in dataset:
    print(f"Feature names ({len(dataset['feature_names'])}): {dataset['feature_names']}")

raw_feature_dim = int(dataset.get("raw_feature_dim", EXPECTED_RAW_FEATURE_DIM))
model_input_dim = int(dataset.get("model_input_dim", EXPECTED_MODEL_INPUT_DIM))

print(f"Raw feature dimension from dataset: {raw_feature_dim}")
print(f"Model input dimension from dataset: {model_input_dim}")

if data.ndim != 3:
    raise ValueError(
        "Expected a raw Stage 3 dataset with shape [N, 50, 20]. "
        f"Got {data.shape}. If the last dimension is 2, this is the legacy "
        "globally normalized dataset; regenerate it with math-model/src/main.py."
    )

if data.shape[1] != SEQ_LEN:
    raise ValueError(
        f"Expected sequence length {SEQ_LEN}, but got {data.shape[1]}"
    )

if data.shape[2] != EXPECTED_RAW_FEATURE_DIM:
    raise ValueError(
        f"Expected raw feature dimension {EXPECTED_RAW_FEATURE_DIM}, "
        f"but got {data.shape[2]}"
    )

if model_input_dim != EXPECTED_MODEL_INPUT_DIM:
    raise ValueError(
        f"Dataset metadata model_input_dim is {model_input_dim}, "
        f"expected {EXPECTED_MODEL_INPUT_DIM}"
    )

dataset_labels = labels.cpu().numpy()
present_labels = set(np.unique(dataset_labels).tolist())
expected_labels = set(range(NUM_CLASSES))
missing_labels = sorted(expected_labels - present_labels)
unexpected_labels = sorted(present_labels - expected_labels)

if missing_labels or unexpected_labels:
    raise ValueError(
        "Stage 3 dataset label validation failed. "
        f"Missing labels: {missing_labels}; unexpected labels: {unexpected_labels}"
    )

print(f"Verified dataset labels: {sorted(present_labels)}")
print("Raw Stage 3 9-class multi-sensor dataset check passed.")
print("=" * 70)

# ===============================
# 2. Mission-level train/validation/test split
# ===============================
if "mission_ids" not in dataset:
    raise ValueError(
        "mission_ids are required for leakage-free evaluation. Regenerate the dataset."
    )

if VALIDATION_MISSION_FRACTION <= 0 or TEST_MISSION_FRACTION <= 0:
    raise ValueError("Validation and test mission fractions must both be positive.")

if VALIDATION_MISSION_FRACTION + TEST_MISSION_FRACTION >= 1:
    raise ValueError("Validation and test mission fractions must sum to less than 1.")

groups = dataset["mission_ids"].cpu().numpy()
all_indices = np.arange(len(labels))

test_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_MISSION_FRACTION,
    random_state=RANDOM_SEED,
)
train_val_idx, test_idx = next(
    test_splitter.split(all_indices, groups=groups)
)

relative_val_fraction = (
    VALIDATION_MISSION_FRACTION / (1.0 - TEST_MISSION_FRACTION)
)
validation_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=relative_val_fraction,
    random_state=RANDOM_SEED + 1,
)
train_local_idx, val_local_idx = next(
    validation_splitter.split(
        train_val_idx,
        groups=groups[train_val_idx],
    )
)
train_idx = train_val_idx[train_local_idx]
val_idx = train_val_idx[val_local_idx]

split_indices = {
    "train": train_idx,
    "validation": val_idx,
    "test": test_idx,
}
split_missions = {
    name: set(groups[indices].tolist())
    for name, indices in split_indices.items()
}

if (
    split_missions["train"] & split_missions["validation"]
    or split_missions["train"] & split_missions["test"]
    or split_missions["validation"] & split_missions["test"]
):
    raise ValueError("Mission leakage detected between train, validation, and test splits.")

for name in ("train", "validation", "test"):
    split_labels = labels[split_indices[name]].cpu().numpy()
    missing = sorted(expected_labels - set(np.unique(split_labels).tolist()))
    if missing:
        raise ValueError(f"{name} split is missing labels {missing}.")
    print(
        f"{name.title()} missions: {len(split_missions[name])}; "
        f"windows: {len(split_indices[name])}"
    )


def add_differences(raw_tensor):
    zero_pad = torch.zeros_like(raw_tensor[:, :1, :])
    differences = torch.cat(
        (zero_pad, raw_tensor[:, 1:, :] - raw_tensor[:, :-1, :]),
        dim=1,
    )
    return torch.stack((raw_tensor, differences), dim=-1)


X_train_raw = data[train_idx].float()
X_val_raw = data[val_idx].float()
X_test_raw = data[test_idx].float()
y_train = labels[train_idx]
y_val = labels[val_idx]
y_test = labels[test_idx]

X_train_augmented = add_differences(X_train_raw)
train_mean = X_train_augmented.mean(dim=(0, 1))
train_std = X_train_augmented.std(dim=(0, 1), unbiased=False)

if train_mean.shape != (EXPECTED_RAW_FEATURE_DIM, 2):
    raise ValueError(f"Unexpected normalization mean shape: {train_mean.shape}")


def normalize_and_flatten(raw_tensor):
    augmented = add_differences(raw_tensor)
    normalized = (augmented - train_mean) / (train_std + 1e-8)
    return normalized.reshape(
        normalized.shape[0],
        normalized.shape[1],
        EXPECTED_MODEL_INPUT_DIM,
    )


X_train = ((X_train_augmented - train_mean) / (train_std + 1e-8)).reshape(
    X_train_augmented.shape[0],
    X_train_augmented.shape[1],
    EXPECTED_MODEL_INPUT_DIM,
)
X_val = normalize_and_flatten(X_val_raw)
X_test = normalize_and_flatten(X_test_raw)

np.save(MEAN_PATH, train_mean.cpu().numpy().reshape(-1).astype(np.float32))
np.save(STD_PATH, train_std.cpu().numpy().reshape(-1).astype(np.float32))

with open(SPLIT_COUNTS_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "class_id",
        "class_name",
        "train_missions",
        "train_windows",
        "validation_missions",
        "validation_windows",
        "test_missions",
        "test_windows",
    ])
    for class_id in range(NUM_CLASSES):
        row = [class_id, label_names[class_id]]
        for split_name in ("train", "validation", "test"):
            indices = split_indices[split_name]
            split_y = dataset_labels[indices]
            class_mask = split_y == class_id
            class_missions = np.unique(groups[indices][class_mask])
            row.extend([len(class_missions), int(class_mask.sum())])
        writer.writerow(row)

split_manifest = {
    "random_seed": RANDOM_SEED,
    "validation_mission_fraction": VALIDATION_MISSION_FRACTION,
    "test_mission_fraction": TEST_MISSION_FRACTION,
    "normalization_fit_partition": "train",
    "normalization_epsilon": 1e-8,
    "dataset_path": str(STAGE3_DATASET_PATH),
    "feature_names": dataset.get("feature_names", []),
    "sequence_length": SEQ_LEN,
    "raw_feature_dim": EXPECTED_RAW_FEATURE_DIM,
    "model_input_dim": EXPECTED_MODEL_INPUT_DIM,
    "splits": {
        name: {
            "mission_count": len(split_missions[name]),
            "window_count": len(indices),
            "mission_ids": sorted(int(value) for value in split_missions[name]),
        }
        for name, indices in split_indices.items()
    },
}
with open(SPLIT_MANIFEST_PATH, "w", encoding="utf-8") as file:
    json.dump(split_manifest, file, indent=2)

del data, X_train_raw, X_val_raw, X_test_raw, X_train_augmented

train_generator = torch.Generator().manual_seed(RANDOM_SEED)
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=train_generator,
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# ===============================
# 3. 故障示例与可视化
# ===============================
print(
    "Residual-example plotting is skipped during model training. "
    "Physical residual figures should be generated from representative raw mission logs."
)

# 绘制常规故障示例
plot_fault_examples(
    X_train,
    y_train,
    label_names_dict=label_names,
    n_samples=3,
    save_dir=FAULT_EXAMPLE_DIR
)

# 训练前序列可视化
plot_sample_sequences(
    X_train,
    y_train,
    n_samples=1,
    save_path=SEQUENCE_PLOT_PATH
)

# ===============================
# 4. 模型
# ===============================
model = AUVFaultDetector(
    input_dim=EXPECTED_MODEL_INPUT_DIM,
    seq_len=SEQ_LEN,
    num_classes=NUM_CLASSES
).to(DEVICE)

# =======================================================
# 自动计算并应用类别权重 Class Weights
# =======================================================
label_counts = np.bincount(y_train.cpu().numpy(), minlength=NUM_CLASSES)
if len(label_counts) != NUM_CLASSES or np.any(label_counts == 0):
    raise ValueError(
        "Every Stage 3 class must appear in the training split before class "
        f"weights are calculated. Counts: {label_counts.tolist()}"
    )

weights = 1.0 / np.sqrt(label_counts)
weights = weights / weights.sum() * len(label_counts)
class_weights = torch.FloatTensor(weights).to(DEVICE)

print("\nStage 3 training class counts:", label_counts)
print("Stage 3 class weights:", class_weights.cpu().numpy())

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5
)

# ===============================
# 5. 训练
# ===============================
train_losses = []
val_losses = []
val_accuracies = []
val_macro_f1_scores = []

best_acc = 0.0
best_epoch = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # ===============================
    # 验证
    # ===============================
    model.eval()
    correct, total = 0, 0
    validation_loss = 0.0
    validation_predictions = []
    validation_labels = []

    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x = val_x.to(DEVICE)
            val_y = val_y.to(DEVICE)

            outputs = model(val_x)
            loss = criterion(outputs, val_y)
            _, predicted = torch.max(outputs, 1)

            validation_loss += loss.item()
            total += val_y.size(0)
            correct += (predicted == val_y).sum().item()
            validation_predictions.extend(predicted.cpu().numpy())
            validation_labels.extend(val_y.cpu().numpy())

    acc = correct / total
    avg_val_loss = validation_loss / len(val_loader)
    macro_f1 = f1_score(
        validation_labels,
        validation_predictions,
        average="macro",
        zero_division=0,
    )

    scheduler.step(acc)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {avg_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {acc:.4f} | "
        f"Val Macro-F1: {macro_f1:.4f} | "
        f"LR: {current_lr:.6f}"
    )

    train_losses.append(avg_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(acc)
    val_macro_f1_scores.append(macro_f1)

    # ===============================
    # Save the best Stage 3 9-class model.
    # ===============================
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Saved the current best Stage 3 9-class model! Best Acc: {best_acc:.4f}")

# ===============================
# 6. 训练曲线
# ===============================
plot_training_history(
    train_losses,
    val_accuracies,
    save_path=TRAINING_PLOT_PATH,
    val_losses=val_losses,
    val_macro_f1_scores=val_macro_f1_scores,
    best_epoch=best_epoch,
)

with open(TRAINING_HISTORY_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_accuracy",
        "validation_macro_f1",
        "best_epoch",
    ])
    for index in range(len(train_losses)):
        writer.writerow([
            index + 1,
            train_losses[index],
            val_losses[index],
            val_accuracies[index],
            val_macro_f1_scores[index],
            best_epoch,
        ])

# ===============================
# 7. 混淆矩阵
# ===============================
model.load_state_dict(
    torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
)
print(f"Reloaded best checkpoint from epoch {best_epoch} for test evaluation.")

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for test_x, test_y in test_loader:
        test_x = test_x.to(DEVICE)
        test_y = test_y.to(DEVICE)

        outputs = model(test_x)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(test_y.cpu().numpy())

plot_confusion_matrix(
    all_labels,
    all_preds,
    label_names_dict=label_names,
    save_path=CONFUSION_MATRIX_PATH
)
plot_confusion_matrix(
    all_labels,
    all_preds,
    label_names_dict=label_names,
    save_path=NORMALIZED_CONFUSION_MATRIX_PATH,
    normalize=True,
)
target_names = [label_names[i] for i in range(NUM_CLASSES)]

report_dict = classification_report(
    all_labels,
    all_preds,
    labels=list(range(NUM_CLASSES)),
    target_names=target_names,
    zero_division=0,
    digits=4,
    output_dict=True,
)
report = classification_report(
    all_labels,
    all_preds,
    labels=list(range(NUM_CLASSES)),
    target_names=target_names,
    zero_division=0,
    digits=4,
)

print("\nClassification Report:")
print(report)

with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

with open(CLASSIFICATION_REPORT_CSV_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["class", "precision", "recall", "f1_score", "support"])
    for name in target_names:
        metrics = report_dict[name]
        writer.writerow([
            name,
            metrics["precision"],
            metrics["recall"],
            metrics["f1-score"],
            int(metrics["support"]),
        ])
    for name in ("macro avg", "weighted avg"):
        metrics = report_dict[name]
        writer.writerow([
            name,
            metrics["precision"],
            metrics["recall"],
            metrics["f1-score"],
            int(metrics["support"]),
        ])
    writer.writerow(["accuracy", report_dict["accuracy"], "", "", len(all_labels)])
print("=" * 70)
print("Stage 3 9-class training complete.")
print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")
print(f"Best model saved to: {BEST_MODEL_PATH}")
print(f"Training plot saved to: {TRAINING_PLOT_PATH}")
print(f"Test confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
print(f"Test classification report saved to: {CLASSIFICATION_REPORT_PATH}")
print("=" * 70)
