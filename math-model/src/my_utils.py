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
    6: "THRUSTER_ENTANGLED",  # Training/raw class 6: entanglement cause
    7: "THRUSTER_NO_OUTPUT",
    8: "THRUSTER_THRUST_LOSS",
}


# =========================================================
# FTC / monitoring-level display mapping
# =========================================================
# Training and confusion-matrix analysis still use the raw 9 classes above.
# For FTC monitoring, plots, and reports, raw class 6 (entanglement cause)
# and raw class 8 (thrust loss symptom) are merged into the same control-level
# class: THRUSTER_THRUST_LOSS.
FTC_LABEL_NAMES = {
    0: "Normal",
    1: "Bias",
    2: "Drift",
    3: "Stuck",
    4: "Spike",
    5: "Increased Noise",
    6: "THRUSTER_THRUST_LOSS",
    7: "THRUSTER_NO_OUTPUT",
}


def merge_fault_for_monitoring_and_ftc(fault_id):
    """Map raw 9-class fault IDs to FTC / monitoring-level IDs."""
    try:
        fid = int(fault_id)
    except (TypeError, ValueError):
        return fault_id

    if fid in [6, 8]:
        return 6
    return fid


def merge_fault_array_for_display(values):
    """Vectorized version of merge_fault_for_monitoring_and_ftc for plots."""
    arr = np.asarray(values).copy()
    arr = np.where(np.isin(arr, [6, 8]), 6, arr)
    return arr


def canonical_fault_name_for_display(fault_name):
    """Return the FTC / monitoring-level name used in figures and reports."""
    text = "" if fault_name is None else str(fault_name)
    name = text.upper().replace(" ", "_")

    if "ENTANGLED" in name or "THRUST_LOSS" in name:
        return "THRUSTER_THRUST_LOSS"

    if "NO_OUTPUT" in name:
        return "THRUSTER_NO_OUTPUT"

    return text


def get_candidate_raw_fault_ids_for_display_name(fault_name):
    """Return raw 9-class IDs that correspond to a displayed FTC fault name."""
    name = str(fault_name).upper().replace(" ", "_")

    if "NO_FAULT" in name or "NORMAL" in name:
        return [0]
    if "BIAS" in name:
        return [1]
    if "DRIFT" in name:
        return [2]
    if "STUCK" in name:
        return [3]
    if "SPIKE" in name:
        return [4]
    if "NOISE" in name:
        return [5]
    if "ENTANGLED" in name or "THRUST_LOSS" in name:
        return [6, 8]
    if "NO_OUTPUT" in name:
        return [7]

    return []


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
    std_path = r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model\depth_fault_detection\results\std_stage3_9class.npy"
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
    mean_arr = np.load(os.path.join(base_dir, "mean_stage3_9class.npy"))
    std_arr = np.load(os.path.join(base_dir, "std_stage3_9class.npy"))
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
# Helper: presentation-friendly FTC / diagnosis text
# =========================================================
def format_ftc_action_for_display(action_text, final_fault_name=None):
    """
    Convert old FTC action names into more realistic AUV recovery actions
    for presentation and plotting.
    """

    if action_text is None:
        action_text = ""

    action_text = str(action_text)
    display_fault_name = canonical_fault_name_for_display(final_fault_name).upper()

    # FTC / monitoring-level merge: raw entanglement and thrust-loss cases
    # are reported with one unified thrust-loss recovery action.
    if "THRUSTER_THRUST_LOSS" in display_fault_name:
        return "Degraded Thrust Mode + Controlled Emergency Ascent + Acoustic Beacon"

    if "THRUSTER_NO_OUTPUT" in display_fault_name:
        return "Power Cut + Emergency Buoyancy Ascent + Acoustic Beacon"

    if "Power Cut" in action_text and "USV Winch Recovery" in action_text:
        return "Power Cut + Emergency Buoyancy Ascent + Acoustic Beacon"

    if "USV Winch Recovery" in action_text:
        return "Emergency Buoyancy Ascent + Acoustic Beacon"

    if "Abort & Slow Surface" in action_text:
        return "Abort Mission + Controlled Emergency Ascent"

    if "Hover Locked" in action_text:
        return "Safe Hover / Depth-Hold Using Estimated Depth"

    if "Filtered" in action_text and "Adaptive" not in action_text:
        return "Spike Rejection Filter"

    if "Adaptive Filtering" in action_text:
        return "Adaptive Smoothing Filter"

    if "Normal Cruising" in action_text:
        return "Normal Cruising"

    return action_text


def build_diagnosis_basis(final_fault_name):
    """
    Build a concise diagnosis criterion explanation for the mission report box.
    """

    name = str(final_fault_name).upper()

    if "NO_FAULT" in name or "NORMAL" in name:
        return (
            "Criterion: all residuals remain within normal thresholds.\n"
            "Signals: depth, DVL velocity, and motor current are consistent."
        )

    if "BIAS" in name:
        return (
            "Criterion: persistent non-zero depth residual with small slope.\n"
            "Signal: depth sensor residual."
        )

    if "DRIFT" in name:
        return (
            "Criterion: depth residual shows continuous increasing/decreasing trend.\n"
            "Signals: residual slope and residual range."
        )

    if "STUCK" in name:
        return (
            "Criterion: depth reading is nearly constant while vehicle motion continues.\n"
            "Recovery: isolate sensor, abort mission, and command emergency ascent."
        )

    if "SPIKE" in name:
        return (
            "Criterion: isolated one-step depth jump exceeds threshold.\n"
            "FTC: reject spike and keep normal control."
        )

    if "NOISE" in name:
        return (
            "Criterion: repeated high-frequency depth residual fluctuation.\n"
            "FTC: activate adaptive smoothing filter."
        )

    if "ENTANGLED" in name or "THRUST_LOSS" in name:
        return (
            "Criterion: commanded thrust is active, but the effective thrust / "
            "velocity response is lower than expected.\n"
            "Possible causes: entanglement, propeller damage, efficiency loss, "
            "biofouling, or duct/guard-induced thrust reduction."
        )

    if "NO_OUTPUT" in name:
        return (
            "Criterion: high cmd_vz + near-zero thrust / current response.\n"
            "Sources: DVL velocity + motor current sensor."
        )
    return "Criterion: AI-rule fused diagnosis."


def _safe_last_valid(values, default="N/A"):
    """
    Return the last non-empty / non-NaN value from a sequence.
    """
    for value in reversed(values):
        if value is None:
            continue

        if isinstance(value, float) and np.isnan(value):
            continue

        if str(value) == "":
            continue

        return value

    return default




# =========================================================
# Helper: compact diagnosis / evidence / recovery codes
# =========================================================
def get_diagnosis_evidence_recovery_codes(final_fault_name, action_text=None):
    """
    Return compact codes for figure display:
        D-code: final diagnosis class
        E-code: diagnosis evidence / sensor basis
        R-code: recovery or FTC action
    The full meaning is listed in the code explanation table.
    """

    name = str(final_fault_name).upper()

    if "NO_FAULT" in name or "NORMAL" in name:
        return "D0", "E7", "R0"

    if "BIAS" in name:
        return "D1", "E1", "R5"

    if "DRIFT" in name:
        return "D2", "E2", "R4"

    if "STUCK" in name:
        return "D3", "E3", "R3"

    if "SPIKE" in name:
        return "D4", "E4", "R1"

    if "NOISE" in name:
        return "D5", "E5", "R2"

    if "ENTANGLED" in name or "THRUST_LOSS" in name:
        return "D6", "E6", "R4"

    if "NO_OUTPUT" in name:
        return "D7", "E7", "R3"

    return "D?", "E?", "R?"
# FTC / monitoring-level codes:
# D6 = THRUSTER_THRUST_LOSS  (merged raw 6 THRUSTER_ENTANGLED + raw 8 THRUSTER_THRUST_LOSS)
# D7 = THRUSTER_NO_OUTPUT
#
# E6 = Commanded thrust is active, but effective thrust / velocity response is reduced
# E7 = Near-zero thrust/current output despite command demand
#
# R3 = Power cut + emergency buoyancy ascent + acoustic beacon
# R4 = Degraded thrust mode + controlled emergency ascent + acoustic beacon

# =========================================================
# 6. 论文/汇报专用：容错控制 (FTC) 动态响应曲线
# =========================================================
def plot_ftc_response(logs, fault_time, ai_intervention_time, save_path,
                      true_fault_name="Unknown", ai_diagnosis="None", ai_action="None", spike_times=None):
    """
    豪华版 FTC 响应曲线绘制（带 AI 诊断信息面板）
    """
    times = [log["time"] for log in logs]
    display_true_fault_name = canonical_fault_name_for_display(true_fault_name)
    display_ai_diagnosis = canonical_fault_name_for_display(ai_diagnosis)
    display_ai_action = format_ftc_action_for_display(ai_action, display_ai_diagnosis)

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
        f" True Fault : {display_true_fault_name}\n"
        f" AI Predict : {display_ai_diagnosis}\n"
        f" FTC Action : {display_ai_action}"
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

    plt.title(f"Fault-Tolerant Control (FTC) Response - {display_true_fault_name}", fontsize=16, fontweight='bold')
    plt.xlabel("Mission Time (s)", fontsize=12)
    plt.ylabel("Depth (m)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
def plot_ftc_diagnosis_response(
        logs,
        fault_time,
        ai_intervention_time,
        save_path,
        true_fault_name="Unknown",
        ai_diagnosis="None",
        ai_action="None",
        spike_times=None
):
    """
    Enhanced FTC diagnosis response figure.

    This figure is designed for thesis / presentation use.
    It shows:
        1. Depth response
        2. Residuals
        3. Thruster behavior
        4. AI / rule / final diagnosis and FTC state
    """

    if logs is None or len(logs) == 0:
        print("No logs to plot.")
        return

    display_true_fault_name = canonical_fault_name_for_display(true_fault_name)
    display_ai_diagnosis = canonical_fault_name_for_display(ai_diagnosis)
    display_ai_action = format_ftc_action_for_display(ai_action, display_ai_diagnosis)

    times = np.array([log.get("time", 0.0) for log in logs])

    # =========================================================
    # 1. Depth data
    # =========================================================
    true_depths = np.array([log.get("true_depth", np.nan) for log in logs])
    sensor_depths = np.array([log.get("depth", np.nan) for log in logs])
    target_depths = np.array([log.get("target_z", np.nan) for log in logs])

    # =========================================================
    # 2. Residual data
    # =========================================================
    tracking_errors = np.array([
        log.get("residuals", {}).get("tracking_error", np.nan)
        for log in logs
    ])

    velocity_residuals = np.array([
        log.get("residuals", {}).get("velocity_residual", np.nan)
        for log in logs
    ])

    current_residuals = np.array([
        log.get("residuals", {}).get("current_residual", np.nan)
        for log in logs
    ])

    # =========================================================
    # 3. Thruster data
    # =========================================================
    cmd_vz = np.array([
        log.get("thruster", {}).get("cmd_vz", np.nan)
        for log in logs
    ])

    actual_vz = np.array([
        log.get("thruster", {}).get("actual_vz", np.nan)
        for log in logs
    ])

    motor_current = np.array([
        log.get("thruster", {}).get("current", np.nan)
        for log in logs
    ])

    expected_current = np.array([
        log.get("residuals", {}).get("expected_current", np.nan)
        for log in logs
    ])

    # =========================================================
    # 4. Diagnosis data
    # =========================================================
    ai_pred_raw = np.array([
        log.get("ai_pred", 0)
        for log in logs
    ])

    rule_pred_raw = np.array([
        log.get("rule_pred", 0)
        for log in logs
    ])

    final_pred_raw = np.array([
        log.get("final_pred", 0)
        for log in logs
    ])

    # FTC / monitoring-level display merge:
    # raw 6 (entanglement) and raw 8 (thrust loss) are shown together as
    # THRUSTER_THRUST_LOSS. Training/evaluation utilities above still use
    # raw 9-class labels.
    ai_pred = merge_fault_array_for_display(ai_pred_raw)
    rule_pred = merge_fault_array_for_display(rule_pred_raw)
    final_pred = merge_fault_array_for_display(final_pred_raw)

    ftc_locked = np.array([
        1 if log.get("ftc_is_locked", False) else 0
        for log in logs
    ])

    # =========================================================
    # Get final diagnosis reason
    # =========================================================
    final_reason = "No strong diagnosis reason recorded."

    # Convert final diagnosis name to one or more raw 9-class fault ids.
    # THRUSTER_ENTANGLED and THRUSTER_THRUST_LOSS are merged only for FTC /
    # monitoring display, so the reason lookup accepts both raw IDs [6, 8].
    target_fault_ids = get_candidate_raw_fault_ids_for_display_name(display_ai_diagnosis)
    target_fault_id = target_fault_ids[0] if len(target_fault_ids) > 0 else None

    invalid_reasons = [
        "",
        "No diagnosis has been triggered yet.",
        "All residuals are within normal thresholds.",
    ]

    # 1) First, find the reason at the moment when final_pred equals
    #    the final diagnosis. This is the most reliable for locked faults.
    if target_fault_id is not None:
        matched_reasons = [
            log.get("diagnosis_reason", "")
            for log in logs
            if (
                log.get("final_pred", 0) in target_fault_ids
                or merge_fault_for_monitoring_and_ftc(log.get("final_pred", 0))
                == merge_fault_for_monitoring_and_ftc(target_fault_id)
            )
            and log.get("diagnosis_reason", "") not in invalid_reasons
        ]

        if len(matched_reasons) > 0:
            final_reason = matched_reasons[-1]

    # 2) For transient faults such as SPIKE, final_pred may quickly return
    #    to 0. Therefore, also search rule_pred for the same target fault.
    if final_reason == "No strong diagnosis reason recorded." and target_fault_id is not None:
        matched_reasons = [
            log.get("diagnosis_reason", "")
            for log in logs
            if (
                log.get("rule_pred", 0) in target_fault_ids
                or merge_fault_for_monitoring_and_ftc(log.get("rule_pred", 0))
                == merge_fault_for_monitoring_and_ftc(target_fault_id)
            )
            and log.get("diagnosis_reason", "") not in invalid_reasons
        ]

        if len(matched_reasons) > 0:
            final_reason = matched_reasons[-1]

    # 3) Fallback: use any valid diagnosis reason.
    if final_reason == "No strong diagnosis reason recorded.":
        diagnosis_reasons = [
            log.get("diagnosis_reason", "")
            for log in logs
            if log.get("diagnosis_reason", "") not in invalid_reasons
        ]

        if len(diagnosis_reasons) > 0:
            final_reason = diagnosis_reasons[-1]

    # Prevent very long text box
    if len(final_reason) > 120:
        final_reason = final_reason[:120] + "..."

    # =========================================================
    # Plot
    # =========================================================
    fig, axes = plt.subplots(
        4, 1,
        figsize=(15, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.1, 1.1, 0.9]}
    )

    fig.suptitle(
        f"Enhanced FTC Diagnosis and Recovery Response - {display_true_fault_name}",
        fontsize=16,
        fontweight="bold"
    )

    # ---------------------------------------------------------
    # Subplot 1: Depth response
    # ---------------------------------------------------------
    axes[0].plot(times, target_depths, label="Target Depth", linestyle="--", linewidth=2)
    axes[0].plot(times, true_depths, label="True Physical Depth", linewidth=2.2)
    axes[0].plot(times, sensor_depths, label="Measured Depth", linewidth=1.8, alpha=0.85)

    axes[0].set_ylabel("Depth (m)")
    axes[0].set_title("1) Depth Response")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left")

    # ---------------------------------------------------------
    # Subplot 2: Residuals
    # ---------------------------------------------------------
    axes[1].plot(times, tracking_errors, label="Tracking Error $e_z$")
    axes[1].plot(times, velocity_residuals, label="Velocity Residual $r_v$")
    axes[1].plot(times, current_residuals, label="Current Residual $r_I$")

    # Example threshold lines
    axes[1].axhline(3.0, linestyle="--", linewidth=1.2, alpha=0.7, label="Tracking Threshold")
    axes[1].axhline(-3.0, linestyle="--", linewidth=1.2, alpha=0.7)

    axes[1].set_ylabel("Residual")
    axes[1].set_title("2) Residuals for Diagnosis")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper left")

    # ---------------------------------------------------------
    # Subplot 3: Thruster behavior
    # ---------------------------------------------------------
    axes[2].plot(times, cmd_vz, label="Commanded $v_z$")
    axes[2].plot(times, actual_vz, label="Actual $v_z$")
    axes[2].plot(times, motor_current, label="Measured Motor Current")
    axes[2].plot(times, expected_current, label="Expected Current", linestyle="--")

    axes[2].set_ylabel("Thruster")
    axes[2].set_title("3) Thruster Command, Velocity, and Current")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper left")

    # ---------------------------------------------------------
    # Subplot 4: Diagnosis state
    # ---------------------------------------------------------
    axes[3].step(
        times,
        ai_pred,
        where="post",
        label="AI Prediction",
        alpha=0.28,
        linewidth=1.0
    )
    axes[3].step(
        times,
        rule_pred,
        where="post",
        label="Rule Prediction",
        alpha=0.35,
        linewidth=1.0
    )
    axes[3].step(
        times,
        final_pred,
        where="post",
        label="Final Diagnosis",
        linewidth=2.6
    )
    # Recovery lock is a binary state:
    #   0 = recovery mode OFF
    #   1 = recovery mode ON
    # It is plotted at y=8 only to make it visible together with fault IDs 0-7.
    FTC_LOCK_VIS_LEVEL = max(FTC_LABEL_NAMES.keys()) + 1

    axes[3].step(
        times,
        ftc_locked * FTC_LOCK_VIS_LEVEL,
        where="post",
        label="FTC Recovery Mode ON",
        linestyle="--",
        linewidth=1.8,
        color="red"
    )

    axes[3].set_ylabel("Fault ID")
    axes[3].set_xlabel("Mission Time (s)")
    axes[3].set_title("4) Diagnosis Result and FTC State")

    # Left y-axis: diagnosis classes
    axes[3].set_ylim(-0.5, FTC_LOCK_VIS_LEVEL + 0.5)
    axes[3].set_yticks(list(FTC_LABEL_NAMES.keys()))
    axes[3].set_yticklabels([FTC_LABEL_NAMES[i] for i in FTC_LABEL_NAMES.keys()], fontsize=8)

    # Add a small visual explanation inside the plot
    axes[3].text(
        0.985,
        0.86,
        "Red dashed line:\nFTC recovery mode ON\n(binary state, shown at top)",
        transform=axes[3].transAxes,
        fontsize=8.5,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="red",
            alpha=0.78
        )
    )

    # Right y-axis: FTC lock state explanation
    lock_axis = axes[3].twinx()
    lock_axis.set_ylim(axes[3].get_ylim())
    lock_axis.set_yticks([0, FTC_LOCK_VIS_LEVEL])
    lock_axis.set_yticklabels(["OFF", "ON"], fontsize=8)
    lock_axis.set_ylabel("FTC Recovery Mode", fontsize=9)
    lock_axis.tick_params(axis="y", colors="red")
    lock_axis.spines["right"].set_color("red")
    lock_axis.yaxis.label.set_color("red")

    axes[3].grid(True, linestyle="--", alpha=0.5)
    axes[3].legend(loc="upper left")

    # =========================================================
    # Vertical lines: fault and AI/FTC intervention
    # =========================================================
    for ax in axes:
        if fault_time is not None and fault_time > 0:
            ax.axvline(
                x=fault_time,
                linestyle="-.",
                linewidth=1.8,
                alpha=0.9,
                label="Fault Injected"
            )

        if ai_intervention_time is not None and ai_intervention_time > 0:
            ax.axvline(
                x=ai_intervention_time,
                linestyle="-.",
                linewidth=1.8,
                alpha=0.9,
                label="AI / FTC Activated"
            )

    # Spike markers
    if spike_times and len(spike_times) > 0:
        # Remove dense repeated spike markers.
        # A single spike can stay inside the history window for many steps,
        # so plotting every logged time makes the figure unreadable.
        filtered_spike_times = []
        min_gap = 3.0  # seconds

        for st in sorted(spike_times):
            if len(filtered_spike_times) == 0:
                filtered_spike_times.append(st)
            elif st - filtered_spike_times[-1] >= min_gap:
                filtered_spike_times.append(st)

        for st in filtered_spike_times:
            for ax in axes:
                ax.axvline(
                    x=st,
                    linestyle=":",
                    alpha=0.45,
                    linewidth=1.2
                )

    # =========================================================
    # Text box: compact mission report using diagnosis codes
    # =========================================================
    diagnosis_code, evidence_code, recovery_code = get_diagnosis_evidence_recovery_codes(
        display_ai_diagnosis,
        display_ai_action
    )

    # If final diagnosis is normal, avoid showing an old transient reason.
    if str(display_ai_diagnosis).upper() in ["NO_FAULT", "NORMAL"]:
        final_reason = "All residuals are within normal thresholds."

    info_text = (
        f"[Mission Report]\n"
        f"True Fault: {display_true_fault_name}\n"
        f"Final Diagnosis: {display_ai_diagnosis}\n"
        f"Diagnosis Code: {diagnosis_code}\n"
        f"Evidence Code: {evidence_code}\n"
        f"Recovery Code: {recovery_code}"
    )

    axes[0].text(
        0.985, 0.06,
        info_text,
        transform=axes[0].transAxes,
        fontsize=9.2,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="gray",
            alpha=0.88
        )
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Enhanced FTC diagnosis response saved to: {save_path}")
