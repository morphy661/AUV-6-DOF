"""Generate publication-ready Stage 3 figures without retraining the model."""

import csv
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
MATH_MODEL_SRC = REPO_ROOT / "math-model" / "src"
RESULTS_DIR = THIS_DIR / "results"
THESIS_DIR = RESULTS_DIR / "thesis_figures"
LOG_DIR = THESIS_DIR / "source_logs"
DATASET_PATH = THIS_DIR / "data" / "simulation_dataset_stage3_9class.pth"

for path in (THIS_DIR, MATH_MODEL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model import AUVFaultDetector
from faults.system_faults import FaultType, SystemFaultInjector
from main import create_rich_training_auv
from environment.auv_simulator import Simulator
from sensors.battery_sensor import BatterySensor
from sensors.current_sensor import CurrentSensor
from sensors.depth_sensor import DepthSensor
from sensors.dvl_sensor import DVLSensor
from sensors.imu_sensor import IMUSensor
from simple_control.simple_control import simple_controller


SEED = 42
DT = 0.1
FAULT_TIME = 40.0
MISSION_DURATION = 100.0
CLASS_NAMES = [
    "Normal", "Bias", "Drift", "Stuck", "Spike", "Increased Noise",
    "Entangled", "No Output", "Thrust Loss",
]


def configure_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "stix",
    })
    sns.set_theme(style="whitegrid", rc={"grid.linestyle": "--", "grid.alpha": 0.3})


def save_figure(fig, filename):
    path = THESIS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def generate_training_figure():
    history = pd.read_csv(RESULTS_DIR / "training_history_stage3_9class.csv")
    epochs = history["epoch"].to_numpy()
    best_epoch = int(history["best_epoch"].iloc[-1])

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.2), sharex=True)
    axes[0].plot(epochs, history["train_loss"], color="#C73E1D", lw=2, label="Training loss")
    axes[0].plot(epochs, history["validation_loss"], color="#246A73", lw=2, label="Validation loss")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, history["validation_accuracy"], color="#2A6FBB", lw=2,
                 label="Validation accuracy")
    axes[1].plot(epochs, history["validation_macro_f1"], color="#B04A7A", lw=2,
                 label="Validation macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.75, 1.01)
    axes[1].legend(frameon=False, loc="lower right")

    for axis in axes:
        axis.axvline(best_epoch, color="#333333", ls=":", lw=1.5)
    best_score = history.loc[history["epoch"] == best_epoch, "validation_accuracy"].iloc[0]
    axes[1].annotate(
        f"Best epoch: {best_epoch}",
        (best_epoch, best_score),
        xytext=(10, 12),
        textcoords="offset points",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    fig.suptitle("Model Training and Validation Performance")
    fig.tight_layout()
    save_figure(fig, "fig_training_validation_performance.png")


def add_differences(raw):
    zero = torch.zeros_like(raw[:, :1, :])
    diff = torch.cat((zero, raw[:, 1:, :] - raw[:, :-1, :]), dim=1)
    return torch.stack((raw, diff), dim=-1)


def evaluate_test_set():
    dataset = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
    manifest = json.loads((RESULTS_DIR / "dataset_split_manifest_stage3_9class.json").read_text(encoding="utf-8"))
    test_missions = np.asarray(manifest["splits"]["test"]["mission_ids"], dtype=np.int64)
    mission_ids = dataset["mission_ids"].numpy()
    test_mask = np.isin(mission_ids, test_missions)
    X_test = dataset["X"][test_mask].float()
    y_test = dataset["y"][test_mask].long()

    mean = torch.from_numpy(np.load(RESULTS_DIR / "mean_stage3_9class.npy")).reshape(20, 2)
    std = torch.from_numpy(np.load(RESULTS_DIR / "std_stage3_9class.npy")).reshape(20, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AUVFaultDetector(input_dim=40, seq_len=50, num_classes=9).to(device)
    model.load_state_dict(torch.load(RESULTS_DIR / "best_model_stage3_9class.pth",
                                     map_location=device, weights_only=True))
    model.eval()

    predictions = []
    targets = []
    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=512, shuffle=False)
    with torch.no_grad():
        for raw_x, batch_y in loader:
            x = ((add_differences(raw_x) - mean) / (std + 1e-8)).reshape(-1, 50, 40)
            pred = model(x.to(device)).argmax(dim=1).cpu().numpy()
            predictions.extend(pred.tolist())
            targets.extend(batch_y.numpy().tolist())

    cm = confusion_matrix(targets, predictions, labels=range(9))
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0) * 100
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        THESIS_DIR / "test_confusion_matrix_counts.csv")
    pd.DataFrame(cm_pct, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        THESIS_DIR / "test_confusion_matrix_row_percent.csv", float_format="%.4f")

    for matrix, fmt, title, filename in (
        (cm, "d", "Test Confusion Matrix (Counts)", "fig_test_confusion_matrix_counts.png"),
        (cm_pct, ".1f", "Row-normalized Test Confusion Matrix (%)",
         "fig_test_confusion_matrix_normalized.png"),
    ):
        fig, ax = plt.subplots(figsize=(10.2, 8.2))
        sns.heatmap(matrix, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    cbar_kws={"shrink": 0.82})
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=42)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        save_figure(fig, filename)


def representative_injector(fault_type):
    return SystemFaultInjector(
        fault_type=fault_type,
        start_time=FAULT_TIME,
        bias=8.0,
        drift_rate=0.55,
        noise_std=0.75,
        spike_prob=0.005,
        spike_magnitude=11.0,
        random_seed=SEED + int(fault_type.value),
    )


def simulate_mission(fault_type):
    random.seed(SEED + int(fault_type.value))
    np.random.seed(SEED + int(fault_type.value))
    auv = create_rich_training_auv(route_profile="standard")
    simulator = Simulator(
        auv_model=auv,
        depth_sensor=DepthSensor(),
        fault_injector=representative_injector(fault_type),
        imu_sensor=IMUSensor(),
        dvl_sensor=DVLSensor(),
        current_sensor=CurrentSensor(),
        battery_sensor=BatterySensor(),
    )

    def controller(sensor_data):
        final_goal_z = sensor_data.get("target_z", 0.0)
        if not hasattr(controller, "dynamic_setpoint"):
            controller.dynamic_setpoint = float(sensor_data["position"][2])
        max_step = 1.2 * DT
        delta = final_goal_z - controller.dynamic_setpoint
        controller.dynamic_setpoint += np.clip(delta, -max_step, max_step)
        sensor_data["target_z"] = controller.dynamic_setpoint
        return simple_controller(sensor_data, auv)

    simulator.run_mission(duration=MISSION_DURATION, control_function=controller, dt=DT)
    if simulator.sensor_logs:
        final_log = simulator.sensor_logs[-1]
        final_goal_z = float(final_log.get("target_z", controller.dynamic_setpoint))
        final_delta = final_goal_z - controller.dynamic_setpoint
        controller.dynamic_setpoint += np.clip(final_delta, -1.2 * DT, 1.2 * DT)
        final_log["target_z"] = float(controller.dynamic_setpoint)
    rows = []
    for log in simulator.sensor_logs:
        thruster = log.get("thruster", {})
        current = log.get("current_sensor", {})
        rows.append({
            "time_s": float(log["time"]),
            "time_from_fault_s": float(log["time"] - FAULT_TIME),
            "fault_label": int(log["fault_label"]),
            "target_depth_m": float(log.get("target_z", 0.0)),
            "true_depth_m": float(log["true_depth"]),
            "measured_depth_m": float(log["depth"]),
            "depth_measurement_error_m": float(log["depth"] - log["true_depth"]),
            "tracking_error_m": float(log.get("target_z", 0.0) - log["true_depth"]),
            "commanded_vertical_speed_mps": float(thruster.get("cmd_vz", 0.0)),
            "actual_vertical_speed_mps": float(thruster.get("actual_vz", 0.0)),
            "expected_current_a": float(current.get("expected_current", thruster.get("expected_current", 0.0))),
            "measured_current_a": float(current.get("measured_current", thruster.get("current", 0.0))),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(LOG_DIR / f"mission_{fault_type.name.lower()}.csv", index=False)
    return frame


def generate_sensor_fault_figure(missions):
    faults = [FaultType.BIAS, FaultType.DRIFT, FaultType.STUCK,
              FaultType.SPIKE, FaultType.NOISE_INCREASE]
    titles = ["Bias", "Drift", "Stuck", "Spike", "Increased noise"]
    fig, axes = plt.subplots(len(faults), 1, figsize=(9.2, 10.2), sharex=True)
    for ax, fault, title in zip(axes, faults, titles):
        data = missions[fault]
        ax.plot(data["time_from_fault_s"], data["depth_measurement_error_m"],
                color="#2A6FBB", lw=1.35)
        ax.axvline(0, color="#C73E1D", ls="--", lw=1.2, label="Fault injection")
        ax.axhline(0, color="#333333", lw=0.7, alpha=0.65)
        ax.set_ylabel(r"$z_m-z_t$ (m)")
        ax.set_title(title, loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("Time relative to fault injection (s)")
    fig.suptitle("Depth-sensor Fault Signatures")
    fig.tight_layout()
    save_figure(fig, "fig_depth_sensor_fault_signatures.png")


def generate_thruster_fault_figure(missions):
    faults = [FaultType.THRUSTER_ENTANGLED, FaultType.THRUSTER_NO_OUTPUT,
              FaultType.THRUSTER_THRUST_LOSS]
    titles = ["Entanglement", "No output", "Thrust loss"]
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 9.0), sharex=True)
    for row, (fault, title) in enumerate(zip(faults, titles)):
        data = missions[fault]
        t = data["time_from_fault_s"]
        axes[row, 0].plot(t, data["commanded_vertical_speed_mps"], lw=1.3,
                          color="#2A6FBB", label=r"$v_{z,cmd}$")
        axes[row, 0].plot(t, data["actual_vertical_speed_mps"], lw=1.3,
                          color="#C73E1D", label=r"$v_{z,act}$")
        axes[row, 1].plot(t, data["expected_current_a"], lw=1.3,
                          color="#246A73", label=r"$I_e$")
        axes[row, 1].plot(t, data["measured_current_a"], lw=1.3,
                          color="#B04A7A", label=r"$I_m$")
        axes[row, 2].plot(t, data["tracking_error_m"], lw=1.3, color="#6A4C93")
        for ax in axes[row]:
            ax.axvline(0, color="#333333", ls="--", lw=1.0)
        axes[row, 0].set_ylabel(f"{title}\nSpeed (m/s)")
        axes[row, 1].set_ylabel("Current (A)")
        axes[row, 2].set_ylabel(r"$z_{ref}-z_t$ (m)")
    axes[0, 0].set_title("Vertical-speed response")
    axes[0, 1].set_title("Motor-current response")
    axes[0, 2].set_title("Depth-tracking error")
    axes[0, 0].legend(frameon=False, ncol=2)
    axes[0, 1].legend(frameon=False, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("Time relative to fault injection (s)")
    fig.suptitle("Thruster Fault Physical Signatures")
    fig.tight_layout()
    save_figure(fig, "fig_thruster_fault_signatures.png")


def generate_nominal_tracking_figure(data):
    time = data["time_s"].to_numpy()
    error = data["target_depth_m"].to_numpy() - data["true_depth_m"].to_numpy()
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    maximum = float(np.max(np.abs(error)))

    with open(THESIS_DIR / "nominal_depth_tracking_metrics.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value", "unit"])
        writer.writerow(["RMSE", f"{rmse:.6f}", "m"])
        writer.writerow(["MAE", f"{mae:.6f}", "m"])
        writer.writerow(["Maximum absolute error", f"{maximum:.6f}", "m"])

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(time, data["target_depth_m"], color="#333333", ls="--", lw=1.4,
                 label=r"Reference depth $z_{ref}$")
    axes[0].plot(time, data["true_depth_m"], color="#2A6FBB", lw=1.6,
                 label=r"True depth $z_t$")
    axes[0].plot(time, data["measured_depth_m"], color="#C73E1D", lw=1.0, alpha=0.75,
                 label=r"Measured depth $z_m$")
    axes[0].set_ylabel("Depth (m)")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(time, error, color="#6A4C93", lw=1.4)
    axes[1].axhline(0, color="#333333", lw=0.7)
    axes[1].set_xlabel("Mission time (s)")
    axes[1].set_ylabel(r"$z_{ref}-z_t$ (m)")
    axes[1].text(0.99, 0.95, f"RMSE = {rmse:.3f} m\nMAE = {mae:.3f} m\nMax = {maximum:.3f} m",
                 transform=axes[1].transAxes, ha="right", va="top",
                 bbox={"facecolor": "white", "edgecolor": "#777777", "alpha": 0.9})
    fig.suptitle("Nominal Depth Tracking Performance")
    fig.tight_layout()
    save_figure(fig, "fig_nominal_depth_tracking.png")


def main():
    THESIS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    generate_training_figure()
    evaluate_test_set()

    fault_types = [
        FaultType.NO_FAULT, FaultType.BIAS, FaultType.DRIFT, FaultType.STUCK,
        FaultType.SPIKE, FaultType.NOISE_INCREASE, FaultType.THRUSTER_ENTANGLED,
        FaultType.THRUSTER_NO_OUTPUT, FaultType.THRUSTER_THRUST_LOSS,
    ]
    missions = {fault: simulate_mission(fault) for fault in fault_types}
    generate_sensor_fault_figure(missions)
    generate_thruster_fault_figure(missions)
    generate_nominal_tracking_figure(missions[FaultType.NO_FAULT])
    print(f"All thesis figures are available in: {THESIS_DIR}")


if __name__ == "__main__":
    main()
