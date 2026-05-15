import numpy as np

from utils.feature_extractor import (
    extract_ai_features,
    RAW_FEATURE_DIM,
    MODEL_INPUT_DIM,
)


def build_sequences(
        sensor_logs,
        seq_len=50,
        steady_stride=15,
        enable_spike_centered_sampling=True,
        spike_label=4,
        spike_center_stride=1,
        max_spike_windows_per_mission=30,
):
    """
    Build sequence samples from simulator sensor logs.

    Stage 2 feature setting:
        raw feature dimension = 20
        after adding first-order temporal differences = 40

    Sampling strategy:
        A. Transition windows:
            Capture windows around fault label changes.

        B. Spike-centered windows:
            Extra sampling for transient SPIKE faults.
            This is important because SPIKE may only appear in one or a few frames.

        C. Steady-state windows:
            Regular sliding windows with a fixed stride.
    """

    X = []
    y = []

    # ======================================================
    # 1. Extract Stage-2 multi-sensor features and labels
    # ======================================================
    feature_list = []
    labels = []

    for log in sensor_logs:
        features = extract_ai_features(log)

        feature_list.append(features)
        labels.append(log.get("fault_label", 0))

    features_np = np.array(feature_list, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.int64)

    if len(features_np) < seq_len:
        return np.empty((0, seq_len, RAW_FEATURE_DIM), dtype=np.float32), np.empty((0,), dtype=np.int64)

    # Safety check
    if features_np.shape[1] != RAW_FEATURE_DIM:
        raise ValueError(
            f"Feature dimension mismatch: got {features_np.shape[1]}, "
            f"expected {RAW_FEATURE_DIM}"
        )

    # Used to avoid adding exactly duplicated windows.
    # Key format: (start, end, label)
    sampled_windows = set()

    def add_window(start, label):
        """Safely add one sequence window."""
        end = start + seq_len

        if start < 0 or end > len(features_np):
            return

        label = int(label)
        key = (int(start), int(end), label)

        if key in sampled_windows:
            return

        sampled_windows.add(key)
        X.append(features_np[start:end])
        y.append(label)

    # ======================================================
    # 2A. Transition windows
    # ======================================================
    label_diff = np.diff(labels_np)
    change_indices = np.where(label_diff != 0)[0]

    for idx in change_indices:
        # Put the transition close to the center of the window
        start = idx - (seq_len // 2)
        end = start + seq_len

        if start >= 0 and end <= len(features_np):
            add_window(start, labels_np[end - 1])

    # ======================================================
    # 2B. Spike-centered windows
    # ======================================================
    # SPIKE is a transient fault. If we only use steady-state windows,
    # the spike point may be diluted by many normal frames.
    #
    # This section forces windows to be centered around spike frames,
    # so the Bi-LSTM Attention model can learn the transient pattern.
    if enable_spike_centered_sampling:
        spike_indices = np.where(labels_np == spike_label)[0]

        spike_indices = spike_indices[::spike_center_stride]

        if len(spike_indices) > max_spike_windows_per_mission:
            spike_indices = np.random.choice(
                spike_indices,
                size=max_spike_windows_per_mission,
                replace=False
            )
            spike_indices = np.sort(spike_indices)

        for idx in spike_indices:
            start = idx - (seq_len // 2)
            add_window(start, spike_label)

    # ======================================================
    # 2C. Steady-state windows
    # ======================================================
    for start in range(0, len(features_np) - seq_len + 1, steady_stride):
        end = start + seq_len
        add_window(start, labels_np[end - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def preprocess_dataset(X, epsilon=1e-8):
    """
    Stage-2 multi-sensor preprocessing.

    X initial shape:
        (N, 50, 20)

    After adding first-order temporal differences:
        (N, 50, 20, 2)

    After flattening in train.py:
        (N, 50, 40)
    """

    if X.ndim != 3:
        raise ValueError(
            f"Expected X with shape (N, seq_len, feature_dim), got {X.shape}"
        )

    if X.shape[-1] != RAW_FEATURE_DIM:
        raise ValueError(
            f"Expected raw feature dimension {RAW_FEATURE_DIM}, got {X.shape[-1]}"
        )

    # ======================================================
    # 1. First-order temporal difference
    # ======================================================
    X_diff = np.diff(X, axis=1)

    # Add zero difference at the first time step
    zero_pad = np.zeros((X.shape[0], 1, X.shape[2]), dtype=X.dtype)
    X_diff = np.concatenate([zero_pad, X_diff], axis=1)

    # Shape: (N, 50, 20, 2)
    X_combined = np.stack((X, X_diff), axis=-1)

    # ======================================================
    # 2. Normalization
    # ======================================================
    means = np.mean(X_combined, axis=(0, 1))
    stds = np.std(X_combined, axis=(0, 1))

    X_combined = (X_combined - means) / (stds + epsilon)

    # Safety check
    flattened_dim = X_combined.shape[2] * X_combined.shape[3]
    if flattened_dim != MODEL_INPUT_DIM:
        raise ValueError(
            f"Flattened model input dimension mismatch: got {flattened_dim}, "
            f"expected {MODEL_INPUT_DIM}"
        )

    return X_combined.astype(np.float32), {
        "mean": means.astype(np.float32),
        "std": stds.astype(np.float32),
    }