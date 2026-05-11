import numpy as np

def build_sequences(sensor_logs, seq_len=50):
    X = []
    y = []

    # 1. 提取特征和标签
    feature_list = []
    labels = []

    for log in sensor_logs:
        # ==========================================
        #  核心升级：14维 PINN 物理先验特征
        # ==========================================
        # 先提取核心深度数据
        f_depth = log["depth"]
        # 安全获取目标深度 (假设 Simulator 日志里已经记录了 target_z)
        f_target = log.get("target_z", 0.0)
        #  物理残差：目标深度与实际深度的差值
        f_error = f_target - f_depth

        features = [
            f_depth,                                   # 1. 实际深度
            f_target,                                  # 2. 目标深度 (意图)
            f_error,                                   # 3. 追踪误差 (物理残差，抓 Bias/Drift 神器)
            log["thruster"]["current"],                # 4. 电机电流
            log["thruster"]["actual_vz"],              # 5. 实际垂向速度
            log["relative_pos"]["delta_x"],            # 6. 相对 X 偏移
            log["relative_pos"]["delta_y"]             # 7. 相对 Y 偏移
        ]
        feature_list.append(features)
        labels.append(log.get("fault_label", 0))

    features_np = np.array(feature_list)
    labels_np = np.array(labels)

    # =======================================================
    # 2. 核心采样逻辑
    # =======================================================

    # A. 捕捉故障跳变瞬间 (Transition Window)
    label_diff = np.diff(labels_np)
    change_indices = np.where(label_diff != 0)[0]

    for idx in change_indices:
        start = idx - (seq_len // 2)
        end = start + seq_len

        if start >= 0 and end <= len(features_np):
            X.append(features_np[start:end])
            y.append(labels_np[end - 1])

    # B. 稳态采样 (Steady State Window)
    stride = 15
    for i in range(0, len(features_np) - seq_len, stride):
        X.append(features_np[i: i + seq_len])
        y.append(labels_np[i + seq_len - 1])

    return np.array(X), np.array(y)


def preprocess_dataset(X, epsilon=1e-8):
    """
    对切片后的 X 序列进行特征增强和归一化
     [升级版维度说明]
    X 初始 shape: (N, 50, 7) -> 7个原始基础物理维度
    返回 shape: (N, 50, 7, 2) -> 增加了沿时间轴的一阶差分维度
    (注：后续在 train.py 中会被 view 展平为 (N, 50, 14))
    """
    # 沿时间轴 (axis=1) 做一阶差分
    X_diff = np.diff(X, axis=1)
    # 在时间轴头部补 0，保持序列长度为 50
    X_diff = np.insert(X_diff, 0, 0, axis=1)

    # 堆叠原始数据和差分数据，此时 shape 变为 (N, 50, 7, 2)
    X_combined = np.stack((X, X_diff), axis=-1)

    # 计算均值和标准差，在 N 和 seq_len 维度上压缩
    # 现在的 means 和 stds 形状将是 (7, 2)
    means = np.mean(X_combined, axis=(0, 1))
    stds = np.std(X_combined, axis=(0, 1))

    # 利用 Numpy 的广播机制直接归一化
    X_combined = (X_combined - means) / (stds + epsilon)

    return X_combined, {'mean': means, 'std': stds}