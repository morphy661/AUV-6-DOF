from collections import deque
from pathlib import Path
import sys

import numpy as np
import torch

from model import AUVFaultDetector
from my_config import DEVICE, INPUT_DIM, NUM_CLASSES, SEQ_LEN


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[1]
AUV_SRC = WORKSPACE_ROOT / "AUV-MathModel-main" / "src"

if str(AUV_SRC) not in sys.path:
    sys.path.insert(0, str(AUV_SRC))

from utils.feature_extractor import (  # noqa: E402
    MODEL_INPUT_DIM,
    RAW_FEATURE_DIM,
    extract_ai_features,
)


RESULTS_DIR = THIS_DIR / "results"
MODEL_PATH = RESULTS_DIR / "best_model_stage3_9class.pth"
MEAN_PATH = RESULTS_DIR / "mean_stage3_9class.npy"
STD_PATH = RESULTS_DIR / "std_stage3_9class.npy"

CLASS_NAMES = {
    0: "Normal",
    1: "Bias",
    2: "Drift",
    3: "Stuck",
    4: "Spike",
    5: "Increased Noise",
    6: "THRUSTER_ENTANGLED",
    7: "THRUSTER_NO_OUTPUT",
    8: "THRUSTER_THRUST_LOSS",
}


class RealtimeFaultDetector:
    """Sliding-window inference for the Stage 3 9-class model."""

    def __init__(self):
        if INPUT_DIM != MODEL_INPUT_DIM:
            raise ValueError(
                f"Configured input dimension {INPUT_DIM} does not match "
                f"feature extractor dimension {MODEL_INPUT_DIM}."
            )

        self.model = AUVFaultDetector(
            input_dim=INPUT_DIM,
            seq_len=SEQ_LEN,
            num_classes=NUM_CLASSES,
        ).to(DEVICE)

        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        )
        self.model.eval()

        self.mean = np.load(MEAN_PATH).astype(np.float32).reshape(-1)
        self.std = np.load(STD_PATH).astype(np.float32).reshape(-1)

        if self.mean.size != INPUT_DIM or self.std.size != INPUT_DIM:
            raise ValueError(
                "Stage 3 normalization dimension mismatch: "
                f"mean={self.mean.size}, std={self.std.size}, expected={INPUT_DIM}"
            )

        self.buffer = deque(maxlen=SEQ_LEN)

    def add_log(self, sensor_log):
        """Add one simulator/vehicle log and return a prediction when ready."""
        raw_features = np.asarray(
            extract_ai_features(sensor_log),
            dtype=np.float32,
        )

        if raw_features.size != RAW_FEATURE_DIM:
            raise ValueError(
                f"Expected {RAW_FEATURE_DIM} raw features, got {raw_features.size}"
            )

        self.buffer.append(raw_features)
        if len(self.buffer) < SEQ_LEN:
            return None

        raw_sequence = np.stack(self.buffer, axis=0)
        diff_sequence = np.diff(raw_sequence, axis=0, prepend=raw_sequence[:1])
        model_sequence = np.stack(
            (raw_sequence, diff_sequence),
            axis=-1,
        ).reshape(SEQ_LEN, INPUT_DIM)
        model_sequence = (model_sequence - self.mean) / (self.std + 1e-8)

        input_tensor = torch.from_numpy(model_sequence).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_label = int(torch.argmax(probabilities, dim=1).item())
            confidence = float(probabilities[0, predicted_label].item())

        return {
            "label": predicted_label,
            "name": CLASS_NAMES[predicted_label],
            "confidence": confidence,
        }


def load_model():
    """Compatibility helper for callers that only need the loaded model."""
    return RealtimeFaultDetector().model


if __name__ == "__main__":
    detector = RealtimeFaultDetector()
    print("Stage 3 9-class realtime detector loaded.")
    print(f"Model: {MODEL_PATH}")
    print(f"Classes: {CLASS_NAMES}")
