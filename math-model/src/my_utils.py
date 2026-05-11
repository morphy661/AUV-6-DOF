import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ===============================
# Label mapping
# ===============================
label_names = {
    0: "Normal",
    1: "Bias",
    2: "Drift",
    3: "Stuck",
    4: "Spike",
    5: "Increased Noise",
    6: "THRUSTER_ENTANGLED",  # 故障 6：海草缠绕
    7: "THRUSTER_BROKEN"  # 故障 7：桨叶断裂
}


# ===============================
# Helper: tensor处理（通用）
# ===============================
def _prepare_data(data, labels):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()

    if data.dim() == 3 and data.shape[-1] == 1:
        data = data.squeeze(-1)

    return data, labels


# ===============================
# 1. 样本序列可视化
# ===============================
def plot_sample_sequences(data, labels, n_samples=1, save_path=None):
    data, labels = _prepare_data(data, labels)
    seq_len = data.shape[1]

    plt.figure(figsize=(16, 10))

    for label in sorted(label_names.keys()):
        indices = (labels == label).nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            continue

        for i in range(min(n_samples, len(indices))):
            idx = indices[i]

            # 核心修复：如果是 10 维/14 维特征，只取第 0 列（深度）画图，防止画面爆炸
            if data[idx].ndim == 2:
                seq_to_plot = data[idx][:, 0].numpy()
            else:
                seq_to_plot = data[idx].numpy()

            if i == 0:
                plt.plot(range(seq_len), seq_to_plot, label=label_names[label], alpha=0.8)
            else:
                plt.plot(range(seq_len), seq_to_plot, alpha=0.8)

    plt.title("Depth Sensor Sequences by Fault Type")
    plt.xlabel("Time Step")
    plt.ylabel("Depth Value (Normalized)")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Image saved to: {save_path}")

    plt.show()


# ===============================
# 2. 训练过程可视化
# ===============================
def plot_training_history(train_losses, val_accuracies, save_path=None):
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss(red)')
    ax1.plot(epochs, train_losses, marker='o', label='Train Loss(red)', color='red')
    ax1.tick_params(axis='y')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Val Accuracy')
    ax2.plot(epochs, val_accuracies, marker='s', label='Val Acc')
    ax2.tick_params(axis='y')

    plt.title('Training Loss and Validation Accuracy')
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Image saved to: {save_path}")

    plt.show()


# ===============================
# 3. 混淆矩阵
# ===============================
def plot_confusion_matrix(y_true, y_pred, label_names_dict=label_names, save_path=None):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    labels_order = sorted(label_names_dict.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    tick_labels = [label_names_dict[i] for i in labels_order]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=tick_labels, yticklabels=tick_labels)

    plt.xlabel("Predicted Fault")
    plt.ylabel("Actual Fault")
    plt.title("Fault Classification Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()


# ===============================
# 4. 故障样本特征分析 (还原物理单位)
# ===============================
def plot_fault_examples(data, labels, label_names_dict=label_names, n_samples=3, save_dir="results/fault_examples"):
    os.makedirs(save_dir, exist_ok=True)

    # 核心修复：换成你本地正确的绝对路径！
    std_path = r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model\depth_fault_detection\results\std.npy"
    std_arr = np.load(std_path)
    s_depth = std_arr[0]  # 第 0 维是深度

    data, labels = _prepare_data(data, labels)
    seq_len = data.shape[1]

    for label, name in label_names_dict.items():
        indices = (labels == label).nonzero(as_tuple=True)[0]
        if len(indices) == 0: continue

        plt.figure(figsize=(10, 5))
        perm = torch.randperm(len(indices))
        selected_idx = indices[perm[:min(n_samples, len(indices))]]

        for i, idx in enumerate(selected_idx):
            if data[idx].ndim == 2:
                raw_seq = data[idx][:, 0].numpy()  # 只取深度特征
            else:
                raw_seq = data[idx].numpy()

            physical_seq = raw_seq * s_depth
            relative_seq = physical_seq - physical_seq[0]

            plt.plot(range(seq_len), relative_seq, alpha=0.8, linewidth=1.5, label=f"Sample {i + 1}")

        plt.title(f"Fault Feature Analysis - {name} (Relative Change)")
        plt.xlabel("Time Step (within 25s window)")
        plt.ylabel("Depth Variation (m)")
        plt.ylim(-5, 10)
        plt.axhline(0, color='black', linestyle='--', alpha=0.3)
        plt.legend(loc='upper right')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{name}_examples.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f" Image saved to:: {save_path}")


# ===============================
# 5. 实时检测残差分析 (逆归一化)
# ===============================
def plot_residual_examples(data, true_data, labels, label_names_dict=label_names, n_samples=2,
                           save_dir="results/residuals"):
    os.makedirs(save_dir, exist_ok=True)

    # 换成你本地正确的绝对路径！
    base_dir = r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model\depth_fault_detection\results"
    mean_arr = np.load(os.path.join(base_dir, "mean.npy"))
    std_arr = np.load(os.path.join(base_dir, "std.npy"))
    m_depth, s_depth = mean_arr[0], std_arr[0]

    if torch.is_tensor(data): data = data.cpu().numpy()
    if torch.is_tensor(true_data): true_data = true_data.cpu().numpy()

    # 处理输入数据 data (N, 50, 10/14) -> (N, 50) 取深度列
    if data.ndim == 3:
        data_depth_only = data[:, :, 0]
    else:
        data_depth_only = data

    data_physical = (data_depth_only * s_depth) + m_depth

    # 处理真实数据 true_data
    if true_data.ndim == 3:
        true_physical = true_data[:, :, 0]
    else:
        true_physical = true_data

    residuals = data_physical - true_physical
    seq_len = data_physical.shape[1]

    for label, name in label_names_dict.items():
        if torch.is_tensor(labels):
            indices = (labels == label).nonzero(as_tuple=True)[0].numpy()
        else:
            indices = np.where(labels == label)[0]

        if len(indices) == 0: continue

        plt.figure(figsize=(10, 5))
        selected_idx = np.random.choice(indices, min(n_samples, len(indices)), replace=False)

        for i, idx in enumerate(selected_idx):
            res_seq = residuals[idx]
            if name == "Drift":
                display_seq = res_seq - res_seq[0]
                label_y = "Relative Depth Error (m)"
            else:
                display_seq = res_seq
                label_y = "Depth Error (m)"

            plt.plot(range(seq_len), display_seq, alpha=0.8, linewidth=1.5, label=f"Sample {i + 1}")

        plt.title(f"Fault Residual Analysis - {name} (Sensor minus True Depth)")
        plt.xlabel("Time Step (within 25s window)")
        plt.ylabel(label_y)
        plt.ylim(-5, 12)
        plt.axhline(0, color='red', linestyle='--', alpha=0.6, label="Zero Error (Ideal)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{name}_residual.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f" 残差Image saved to:: {save_path}")


# =========================================================
# 6. 论文/汇报专用：容错控制 (FTC) 动态响应曲线
# =========================================================
def plot_ftc_response(logs, fault_time, ai_intervention_time, save_path,
                      true_fault_name="Unknown", ai_diagnosis="None", ai_action="None", spike_times=None):
    """
    豪华版 FTC 响应曲线绘制（带 AI 诊断信息面板）
    """
    times = [log["time"] for log in logs]

    # 🌟 终极修复：去掉了负号！直接读取 true_depth 保证 100% 同频
    true_depths = [log["true_depth"] for log in logs]
    sensor_depths = [log["depth"] for log in logs]

    plt.figure(figsize=(12, 6))

    # 画线
    plt.plot(times, sensor_depths, label="Sensor Reading (Faulty)", color="#f59b9b", linewidth=2)
    plt.plot(times, true_depths, label="True Physical Depth", color="blue", linewidth=2.5)

    # 画故障注入线
    if fault_time is not None and fault_time > 0:
        plt.axvline(x=fault_time, color="orange", linestyle="-.", linewidth=2,
                    label=f"Fault Injected ({fault_time:.1f}s)")

    # 画 AI 介入线
    if ai_intervention_time is not None and ai_intervention_time > 0:
        plt.axvline(x=ai_intervention_time, color="green", linestyle="-.", linewidth=2,
                    label=f"AI Activated ({ai_intervention_time:.1f}s)")

    # ==========================================
    # 在图表右上角绘制 AI 诊断信息面板
    # ==========================================
    info_text = (
        f"[ Mission Report ]\n"
        f"------------------------\n"
        f" True Fault : {true_fault_name}\n"
        f" AI Predict : {ai_diagnosis}\n"
        f" AI Action  : {ai_action}"
    )

    # 使用 bbox 添加一个半透明的文本框
    plt.gca().text(
        0.95, 0.05, info_text,
        transform=plt.gca().transAxes,
        fontsize=11, fontweight='bold',
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.85)
    )

    if spike_times and len(spike_times) > 0:
        for st in spike_times:
            # 画一条贯穿上下的亮绿色细线，对齐红色的跳变
            plt.axvline(x=st, color='limegreen', linestyle=':', alpha=0.6, linewidth=1.5)

        # 偷偷在图例里加一个标签，这样别人才知道这绿线是什么意思
        plt.plot([], [], color='limegreen', linestyle=':', linewidth=1.5, label='SPIKE Filtered by AI')

    plt.title(f"Fault-Tolerant Control (FTC) Response - {true_fault_name}", fontsize=16, fontweight='bold')
    plt.xlabel("Mission Time (s)", fontsize=12)
    plt.ylabel("Depth (m)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()