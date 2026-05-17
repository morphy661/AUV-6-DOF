import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

from model import AUVFaultDetector
from my_config import *
print("Using device:", DEVICE)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("PyTorch CUDA version:", torch.version.cuda)
from src.my_utils import (
    plot_sample_sequences,
    plot_training_history,
    plot_confusion_matrix,
    plot_fault_examples,
    plot_residual_examples,
    label_names
)

# =======================================================
# Stage 2: Multi-sensor AI Fusion Diagnosis Training
# =======================================================
STAGE2_DATASET_PATH = (
    r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model"
    r"\depth_fault_detection\data\simulation_dataset_stage2_multisensor.pth"
)

EXPECTED_RAW_FEATURE_DIM = 20
EXPECTED_MODEL_INPUT_DIM = 40
SEQ_LEN = 50

BEST_MODEL_PATH = "results/best_model_stage2_multisensor.pth"
TRAINING_PLOT_PATH = "results/training_plot_stage2_multisensor.png"
CONFUSION_MATRIX_PATH = "results/confusion_matrix_stage2_multisensor.png"
SEQUENCE_PLOT_PATH = "results/stage2_multisensor_sequences.png"
FAULT_EXAMPLE_DIR = "results/fault_examples_stage2_multisensor"


# ===============================
# 创建结果目录
# ===============================
os.makedirs("results", exist_ok=True)

# ===============================
# 1. 加载 Stage 2 多传感器数据集
# ===============================
dataset = torch.load(STAGE2_DATASET_PATH, map_location="cpu")
print("Dataset keys:", dataset.keys())
data = dataset["X"]  # Stage 2 expected shape: [N, 50, 20, 2]
labels = dataset["y"]

print("=" * 70)
print("Stage 2 multi-sensor dataset loaded.")
print(f"Original X shape: {data.shape}")
print(f"Label shape: {labels.shape}")

if "feature_names" in dataset:
    print(f"Feature names ({len(dataset['feature_names'])}): {dataset['feature_names']}")

raw_feature_dim = int(dataset.get("raw_feature_dim", EXPECTED_RAW_FEATURE_DIM))
model_input_dim = int(dataset.get("model_input_dim", EXPECTED_MODEL_INPUT_DIM))

print(f"Raw feature dimension from dataset: {raw_feature_dim}")
print(f"Model input dimension from dataset: {model_input_dim}")

if data.ndim != 4:
    raise ValueError(
        f"Expected Stage 2 dataset X shape [N, 50, 20, 2], but got {data.shape}"
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

if data.shape[3] != 2:
    raise ValueError(
        f"Expected last dimension 2 for [raw, diff], but got {data.shape[3]}"
    )

# 立即进行维度重塑，适配 LSTM 输入: [N, 50, 20, 2] -> [N, 50, 40]
data = data.view(data.shape[0], data.shape[1], -1)
print(f"Data reshaping complete, new shape: {data.shape}")

if data.shape[-1] != EXPECTED_MODEL_INPUT_DIM:
    raise ValueError(
        f"Stage 2 input dimension error: got {data.shape[-1]}, "
        f"expected {EXPECTED_MODEL_INPUT_DIM}"
    )

if model_input_dim != EXPECTED_MODEL_INPUT_DIM:
    raise ValueError(
        f"Dataset metadata model_input_dim is {model_input_dim}, "
        f"expected {EXPECTED_MODEL_INPUT_DIM}"
    )

print("Stage 2 multi-sensor dataset check passed.")
print("=" * 70)

# ===============================
# 2. 数据划分
# ===============================
if "mission_ids" in dataset:
    print("Using GroupShuffleSplit by mission_ids to avoid same-mission leakage.")

    groups = dataset["mission_ids"].numpy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, val_idx = next(
        splitter.split(data, labels, groups=groups)
    )

    X_train = data[train_idx]
    X_val = data[val_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]

    train_missions = set(groups[train_idx])
    val_missions = set(groups[val_idx])
    overlap = train_missions & val_missions

    print(f"Train missions: {len(train_missions)}")
    print(f"Validation missions: {len(val_missions)}")
    print(f"Mission overlap: {overlap}")

    if len(overlap) != 0:
        raise ValueError("Mission leakage detected: train and validation missions overlap!")

else:
    print("WARNING: mission_ids not found. Using random window-level train_test_split.")
    print("This may cause same-mission leakage. Use this only for quick testing.")

    X_train, X_val, y_train, y_val = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# ===============================
# 3. 故障示例与可视化
# ===============================
# 安全处理残差绘图
if "X_true" in dataset:
    X_true = dataset["X_true"]

    if len(X_true.shape) == 4:
        X_true = X_true.view(X_true.shape[0], X_true.shape[1], -1)

    plot_residual_examples(
        data=data,
        true_data=X_true,
        labels=labels,
        label_names_dict=label_names,
        n_samples=3
    )
else:
    print("X_true was not found in the dataset, so residual plotting has been skipped.")

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
    num_classes=8
).to(DEVICE)

# =======================================================
# 自动计算并应用类别权重 Class Weights
# =======================================================
label_counts = np.bincount(labels.cpu().numpy(), minlength=8)
label_counts[label_counts == 0] = 1

weights = 1.0 / np.sqrt(label_counts)
weights = weights / weights.sum() * len(label_counts)
# Manually strengthen difficult STUCK class
weights[3] *= 1.0

# Re-normalize to keep average weight around 1
weights = weights / weights.sum() * len(weights)
class_weights = torch.FloatTensor(weights).to(DEVICE)

print(
    "\nThe weights for each category in the category imbalance penalty mechanism are as follows:",
    class_weights.cpu().numpy()
)

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
val_accuracies = []

best_acc = 0.0

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

    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x = val_x.to(DEVICE)
            val_y = val_y.to(DEVICE)

            outputs = model(val_x)
            _, predicted = torch.max(outputs, 1)

            total += val_y.size(0)
            correct += (predicted == val_y).sum().item()

    acc = correct / total

    scheduler.step(acc)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"Val Acc: {acc:.4f} | "
        f"LR: {current_lr:.6f}"
    )

    train_losses.append(avg_loss)
    val_accuracies.append(acc)

    # ===============================
    # 保存最佳 Stage 2 模型
    # ===============================
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Save the current best Stage 2 multi-sensor model! Best Acc: {best_acc:.4f}")

# ===============================
# 6. 训练曲线
# ===============================
plot_training_history(
    train_losses,
    val_accuracies,
    save_path=TRAINING_PLOT_PATH
)

# ===============================
# 7. 混淆矩阵
# ===============================
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for val_x, val_y in val_loader:
        val_x = val_x.to(DEVICE)
        val_y = val_y.to(DEVICE)

        outputs = model(val_x)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(val_y.cpu().numpy())

plot_confusion_matrix(
    all_labels,
    all_preds,
    label_names_dict=label_names,
    save_path=CONFUSION_MATRIX_PATH
)
target_names = [label_names[i] for i in range(NUM_CLASSES)]

report = classification_report(
    all_labels,
    all_preds,
    target_names=target_names,
    digits=4
)

print("\nClassification Report:")
print(report)

with open("results/classification_report_stage2_multisensor.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("=" * 70)
print("Stage 2 training complete.")
print(f"Best validation accuracy: {best_acc:.4f}")
print(f"Best model saved to: {BEST_MODEL_PATH}")
print(f"Training plot saved to: {TRAINING_PLOT_PATH}")
print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
print("=" * 70)
