import numpy as np
import torch

def generate_sequence(seq_len=50, fault_type="normal"):
    t = np.linspace(0, 10, seq_len)
    base = np.sin(0.5 * t) + 10  # 模拟深度变化

    if fault_type == "normal":
        signal = base + np.random.normal(0, 0.1, size=seq_len)

    elif fault_type == "offset_drift":
        drift = np.linspace(0, 2, seq_len)
        signal = base + drift + np.random.normal(0, 0.1, size=seq_len)

    elif fault_type == "spike":
        signal = base + np.random.normal(0, 0.1, size=seq_len)
        spike_index = np.random.randint(5, seq_len - 5)
        signal[spike_index:spike_index+3] += np.random.uniform(3, 5)

    elif fault_type == "constant":
        constant_value = base[0]
        signal = np.ones(seq_len) * constant_value

    elif fault_type == "noise":
        signal = base + np.random.normal(0, 1.0, size=seq_len)

    else:
        raise ValueError("Unknown fault type")

    return signal

def generate_dataset(n_samples=1000, seq_len=50):
    fault_types = ["normal", "offset_drift", "spike", "constant", "noise"]
    data = []
    labels = []

    for _ in range(n_samples):
        label = np.random.randint(0, len(fault_types))
        signal = generate_sequence(seq_len=seq_len, fault_type=fault_types[label])
        data.append(signal)
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
