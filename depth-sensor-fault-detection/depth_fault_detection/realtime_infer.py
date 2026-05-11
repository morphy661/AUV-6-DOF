import sys
from pathlib import Path

# ====== 路径 ======
BASE_DIR = Path(__file__).resolve().parent.parent
AUV_SRC = BASE_DIR / "AUV-MathModel-main" / "src"

# ====== 清理冲突模块 ======
for mod in ["config", "utils"]:
    if mod in sys.modules:
        del sys.modules[mod]

# ====== 插入仿真路径（最高优先级）=====
sys.path.insert(0, str(AUV_SRC))
from collections import deque

# ====== 导入你已有模块 ======
from model import LSTMClassifier
from my_config import *

# ====== 导入AUV仿真 ======
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent / "AUV-MathModel-main" / "src"
sys.path.insert(0, str(PROJECT_ROOT))  # 用 insert

from main import create_auv, create_fault
from sensors.depth_sensor import DepthSensor
from environment.auv_simulator import Simulator
from simple_control.simple_control import simple_controller

# ====== label ======
label_names = {
    0: "Normal",
    1: "Bias",
    2: "Drift",
    3: "Stuck",
    4: "Spike",
    5: "Noise"
}


# ===============================
# 加载模型
# ===============================
def load_model():
    model = LSTMClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    model.load_state_dict(torch.load("results/best_model.pth", map_location=DEVICE))
    model.eval()

    print(" 模型加载完成")
    return model


# ===============================
# 实时检测核心
# ===============================
def run_realtime_detection(seq_len=50):

    model = load_model()

    # ====== 创建仿真 ======
    auv = create_auv()
    depth_sensor = DepthSensor()
    fault_injector = create_fault()

    simulator = Simulator(
        auv_model=auv,
        depth_sensor=depth_sensor,
        fault_injector=fault_injector
    )

    def controller(sensor_data):
        return simple_controller(sensor_data, auv)

    simulator.run_mission(
        duration=600,   # 可以先缩短测试
        control_function=controller,
        dt=0.5
    )

    print(" 开始实时检测...")

    # ====== 滑动窗口 ======
    buffer = deque(maxlen=seq_len)

    for i, log in enumerate(simulator.sensor_logs):

        depth = log["depth"]
        true_label = log["fault_label"]

        buffer.append(depth)

        # 不够长度不预测
        if len(buffer) < seq_len:
            continue

        # ====== 构造输入 ======
        input_seq = torch.tensor(buffer, dtype=torch.float32)\
                        .unsqueeze(0)\
                        .to(DEVICE)
        import numpy as np

        mean = np.load("results/mean.npy")
        std = np.load("results/std.npy")

        input_seq = (input_seq - mean) / (std + 1e-8)
        # ====== 推理 ======
        with torch.no_grad():
            output = model(input_seq)
            pred = torch.argmax(output, dim=1).item()

        # ====== 输出 ======
        print(f"[t={i}] True: {label_names[true_label]:<7} | Pred: {label_names[pred]:<7}")

    print(" 实时检测结束")


# ===============================
# main
# ===============================
if __name__ == "__main__":
    run_realtime_detection()