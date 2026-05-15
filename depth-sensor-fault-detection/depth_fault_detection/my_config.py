# my_config.py
import torch

# ===============================
# 模型参数
# ===============================
INPUT_DIM = 40      # Stage 2: 20 raw features + 20 first-order difference features
NUM_CLASSES = 8     # 8-class fault diagnosis
SEQ_LEN = 50        # Time window length

# ===============================
# 训练参数
# ===============================
BATCH_SIZE = 256

# Stage 2 dataset is larger and feature dimension is higher.
# 60 epochs is acceptable, but early stopping can be added later.
EPOCHS = 35

# Recommended learning rate for Bi-LSTM + Attention on Stage 2 multi-sensor data.
# 0.001 can train fast, but 3e-4 is usually more stable.
LR = 3e-4

# Regularization to reduce overfitting / window leakage effect.
WEIGHT_DECAY = 1e-4

# ===============================
# 设备
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
