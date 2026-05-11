import os
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model import AUVFaultDetector
from my_config import *

from src.my_utils import (
    plot_sample_sequences,
    plot_training_history,
    plot_confusion_matrix,
    plot_fault_examples,
    plot_residual_examples,
    label_names
)

# ===============================
# 创建结果目录
# ===============================
os.makedirs("results", exist_ok=True)
# 1. 加载数据
# ===============================
dataset = torch.load(r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model\depth_fault_detection\data\simulation_dataset.pth")

data = dataset["X"]  # 原始形状 [N, 50, 5, 2]
labels = dataset["y"]

# 【关键点 1】：立即进行维度重塑，适配 LSTM 输入 (N, 50, 10)
data = data.view(data.shape[0], data.shape[1], -1)
print(f"Data reshaping complete, new shape: {data.shape}")

# ===============================
# 2. 数据划分
# ===============================
X_train, X_val, y_train, y_val = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(TensorDataset(X_val, y_val),
                        batch_size=BATCH_SIZE)

# ===============================
# 7. 故障示例与可视化
# ===============================

# 【关键点 2】：安全处理残差绘图
if "X_true" in dataset:
    X_true = dataset["X_true"]
    # 如果 X_true 也是 4 维的，同样需要重塑
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
    print(" X_true was not found in the dataset, so residual plotting has been skipped.")

# 绘制常规故障示例
plot_fault_examples(
    X_train, y_train,
    label_names_dict=label_names,
    n_samples=3,
    save_dir="results/fault_examples"
)

# 训练前序列可视化
plot_sample_sequences(
    X_train, y_train,
    n_samples=1,
    save_path="results/depth_sensor_sequences.png"
)
# ===============================
# 3. 模型
# ===============================
#  适配全新的 1D-CNN + LSTM 模型参数
model = AUVFaultDetector(
    input_dim=14,
    seq_len=50,
    num_classes=8
).to(DEVICE)
import numpy as np

# =======================================================
#  核心修复：自动计算并应用类别权重 (Class Weights)
# =======================================================
# 1. 统计 labels 里每个类别的样本数量
label_counts = np.bincount(labels.numpy())

# 2. 防止除以 0 的情况（如果有某个类别恰好为 0 个）
label_counts[label_counts == 0] = 1

# 3. 计算权重：数量越少的类别，权重越大平方根平滑
weights = 1.0 / np.sqrt(label_counts)

# 4. 归一化权重（保持梯度稳定）
weights = weights / weights.sum() * len(label_counts)

# 5. 转换为 Tensor 并送到 GPU/CPU
class_weights = torch.FloatTensor(weights).to(DEVICE)

print("\n The weights for each category in the category imbalance penalty mechanism are as follows:", class_weights.cpu().numpy())

# 6. 把计算好的权重传给损失函数
criterion = nn.CrossEntropyLoss(weight=class_weights)
# =======================================================

# 优化器与调度器 (保持你原来的代码不变)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# ===============================
# 4. 训练
# ===============================
train_losses = []
val_accuracies = []

best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

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
            val_x, val_y = val_x.to(DEVICE), val_y.to(DEVICE)

            outputs = model(val_x)
            _, predicted = torch.max(outputs, 1)

            total += val_y.size(0)
            correct += (predicted == val_y).sum().item()

    # 真实的验证集准确率叫 acc
    acc = correct / total

    # =======================================================
    #  调度器在这里发力：根据这轮跑出的 acc 决定要不要降学习率
    # =======================================================
    scheduler.step(acc)
    current_lr = optimizer.param_groups[0]['lr']

    # 打印本轮成绩单，顺便把当前学习率也打出来看看
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | LR: {current_lr:.6f}")

    train_losses.append(avg_loss)
    val_accuracies.append(acc)

    # ===============================
    #  保存最佳模型
    # ===============================
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "results/best_model.pth")
        print("Save the current best model!")

# ===============================
# 5. 训练曲线
# ===============================
plot_training_history(
    train_losses,
    val_accuracies,
    save_path="results/training_plot.png"
)

# ===============================
# 6. 混淆矩阵
# ===============================
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for val_x, val_y in val_loader:
        val_x, val_y = val_x.to(DEVICE), val_y.to(DEVICE)

        outputs = model(val_x)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(val_y.cpu().numpy())

plot_confusion_matrix(
    all_labels,
    all_preds,
    label_names_dict=label_names,
    save_path="results/confusion_matrix.png"
)

print("Training complete, all results saved. results/")