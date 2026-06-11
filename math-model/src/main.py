import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from my_utils import plot_ftc_response, plot_ftc_diagnosis_response
import sys
from pathlib import Path
import random
import torch
from model1 import AUVFaultDetector  # 导入模型类
from diagnosis import ResidualObserver, DiagnosisStrategy
from sensors.imu_sensor import IMUSensor
from sensors.dvl_sensor import DVLSensor
from sensors.current_sensor import CurrentSensor
from sensors.battery_sensor import BatterySensor
from utils.feature_extractor import extract_ai_features, RAW_FEATURE_NAMES, RAW_FEATURE_DIM, MODEL_INPUT_DIM
from actuators.thruster_motor_model import ThrusterMotorModel, ThrusterFaultMode
from diagnosis.diagnosis_strategy import FAULT_NAMES

# =======================================================
# FTC / monitoring level label merge
# =======================================================
# The neural network and training dataset still use 9 classes:
#   6 = THRUSTER_ENTANGLED
#   8 = THRUSTER_THRUST_LOSS
# At the monitoring and FTC decision levels, both are treated as
# THRUSTER_THRUST_LOSS because entanglement is one possible cause of
# effective thrust loss and requires the same safety response here.
FTC_FAULT_NAMES = {
    0: "NO_FAULT",
    1: "BIAS",
    2: "DRIFT",
    3: "STUCK",
    4: "SPIKE",
    5: "NOISE_INCREASE",
    6: "THRUSTER_THRUST_LOSS",  # merged: raw 6 ENTANGLED + raw 8 THRUST_LOSS
    7: "THRUSTER_NO_OUTPUT",
}


def merge_fault_for_monitoring_and_ftc(fault_id):
    """Map fine-grained 9-class fault IDs to FTC/monitoring IDs."""
    fault_id = int(fault_id)
    if fault_id in [6, 8]:
        return 6
    return fault_id


def get_ftc_fault_name(fault_id):
    return FTC_FAULT_NAMES.get(merge_fault_for_monitoring_and_ftc(fault_id), "UNKNOWN")


def get_monitoring_fault_name_from_raw_name(raw_fault_name):
    """Return the FTC/monitoring display name for a raw simulator fault name.

    Training and injection still keep the fine-grained 9-class labels.
    Images, videos, and FTC-level reports merge THRUSTER_ENTANGLED and
    THRUSTER_THRUST_LOSS into THRUSTER_THRUST_LOSS.
    """
    raw_fault_name = str(raw_fault_name)
    if raw_fault_name in [
        "THRUSTER_ENTANGLED",
        "ENTANGLED",
        "THRUSTER_THRUST_LOSS",
        "THRUST_LOSS",
    ]:
        return "THRUSTER_THRUST_LOSS"
    return raw_fault_name
# =======================================================
# 加载 AI 大脑 (Stage 3: 9-class, 40维多传感器融合架构)
# =======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Starting system...")

# 1. 按照新版模型参数进行实例化
model = AUVFaultDetector(
    input_dim=MODEL_INPUT_DIM,
    seq_len=50,
    num_classes=9
).to(DEVICE)

model_path = r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model\depth_fault_detection\results\best_model_stage3_9class.pth"
AI_MODEL_AVAILABLE = os.path.exists(model_path)

try:
    if AI_MODEL_AVAILABLE:
        model.load_state_dict(
            torch.load(model_path, map_location=DEVICE, weights_only=True)
        )
        print("Successfully loaded Stage 3 9-class multi-sensor model weights!")
        print("Model path:", model_path)
    else:
        print(
            "Stage 3 model has not been trained yet. Dataset generation remains "
            f"available; online AI inference is disabled until this file exists: {model_path}"
        )

except RuntimeError as e:
    raise RuntimeError(
        "Model structure does not match the saved weights. "
        "Please check model1.py and the trained model architecture.\n"
        f"{e}"
    )
model.eval()

# =======================================================
# 项目路径配置
# =======================================================
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.dataset_builder import build_sequences
from sensors.depth_sensor import DepthSensor
from faults.system_faults import SystemFaultInjector, FaultType

from environment.simulation_config import AUVModel
from environment.auv_simulator import Simulator
from config.auv_config import AUVConfig

from simple_control.simple_control import simple_controller
from utils.visualization import visualize_trajectory, animate_trajectory


# ======================
# AUV 模型配置
# ======================
def get_route_waypoints(route_profile="standard"):
    """Return waypoint sets for different mission profiles.

    standard:
        Original route used for baseline FTC evaluation.
    comprehensive:
        Rich full-mission route covering shallow cruise, deep cruise,
        large depth changes, turns, reverse motion, and return-to-surface.
    shallow_cruise:
        Shallow-water cruise with small depth changes.
    deep_cruise:
        Deep-water cruise and large-depth operation.
    zigzag_depth:
        Frequent depth changes to improve robustness against DRIFT/STUCK false alarms.
    """
    if route_profile == "standard":
        return [
            [0, 0, 0],
            [0, 0, 60],
            [20, 20, 60],
            [40, 20, 30],
            [80, 40, 500],
            [120, 60, 500],
            [80, 40, 300],
            [40, 20, 100],
            [0, 0, 0],
        ]

    if route_profile == "comprehensive":
        return [
            [0, 0, 0],
            [0, 0, 50],
            [80, 0, 50],
            [120, 40, 80],
            [160, 40, 120],
            [240, 40, 120],
            [280, 80, 350],
            [360, 120, 350],
            [420, 160, 500],
            [460, 160, 500],
            [380, 100, 500],
            [320, 80, 300],
            [240, 120, 300],
            [200, 80, 380],
            [160, 40, 180],
            [100, 20, 100],
            [40, -20, 40],
            [0, 0, 0],
        ]

    if route_profile == "shallow_cruise":
        return [
            [0, 0, 0],
            [0, 0, 40],
            [60, 0, 40],
            [120, 30, 60],
            [180, 0, 50],
            [240, -30, 70],
            [180, -60, 50],
            [80, -20, 40],
            [0, 0, 0],
        ]

    if route_profile == "deep_cruise":
        return [
            [0, 0, 0],
            [0, 0, 100],
            [40, 20, 300],
            [100, 40, 500],
            [200, 40, 500],
            [280, 100, 500],
            [200, 160, 500],
            [120, 100, 400],
            [40, 40, 200],
            [0, 0, 0],
        ]

    if route_profile == "zigzag_depth":
        return [
            [0, 0, 0],
            [0, 0, 80],
            [40, 20, 150],
            [80, 40, 80],
            [120, 60, 220],
            [160, 80, 120],
            [200, 100, 300],
            [160, 60, 180],
            [80, 20, 100],
            [0, 0, 0],
        ]

    raise ValueError(f"Unknown route_profile: {route_profile}")


TRAINING_ROUTE_PROFILES = [
    "standard",
    "comprehensive",
    "shallow_cruise",
    "deep_cruise",
    "zigzag_depth",
]


def create_auv(route_profile="standard"):
    config = AUVConfig(
        mass=50.0, length=1.5, width=0.5, height=0.5,
        max_velocity_x=3.0, max_velocity_y=3.0, max_velocity_z=3.0,
        max_angular_velocity=0.5, max_acceleration_x=0.5,
        max_acceleration_y=0.3, max_acceleration_z=0.3,
        max_angular_acceleration=0.2, battery_capacity=500.0,
        power_consumption_idle=10.0, power_consumption_per_velocity=5.0
    )

    auv = AUVModel(config)
    waypoints = get_route_waypoints(route_profile)
    auv.set_waypoints(waypoints)
    final_wp = waypoints[-1]
    auv.set_destination(final_wp[0], final_wp[1], final_wp[2])
    auv.route_profile = route_profile
    return auv


def create_rich_training_auv(route_profile="standard"):
    config = AUVConfig(
        mass=50.0, length=1.5, width=0.5, height=0.5,
        max_velocity_x=2.0, max_velocity_y=2.0, max_velocity_z=2.0,
        max_angular_velocity=0.5, max_acceleration_x=0.5,
        max_acceleration_y=0.3, max_acceleration_z=0.3,
        max_angular_acceleration=0.2, battery_capacity=500.0,
        power_consumption_idle=10.0, power_consumption_per_velocity=5.0
    )

    auv = AUVModel(config)
    waypoints = get_route_waypoints(route_profile)
    auv.set_waypoints(waypoints)

    final_wp = waypoints[-1]
    auv.set_destination(final_wp[0], final_wp[1], final_wp[2])
    auv.route_profile = route_profile
    return auv


# ======================
# 创建随机故障注入器
# ======================
def create_fault(target_fault_type=None, is_training=False, start_time_override=None):
    if start_time_override is not None:
        start_time = float(start_time_override)

        if target_fault_type is None:
            target_fault_type = random.choice(list(FaultType))

    elif is_training:
        start_time = random.uniform(30.0, 750.0)
        if target_fault_type is None:
            target_fault_type = random.choice(list(FaultType))

    elif target_fault_type is None:
        target_fault_type = random.choice(list(FaultType))
        start_time = random.uniform(100, 200.0)

    else:
        start_time = random.uniform(40.0, 60.0)

    # ======================================================
    # SPIKE probability policy
    # ======================================================
    # Training:
    #   Use a relatively higher probability so the model can learn enough
    #   transient spike patterns.
    #
    # Online / Mode 1 / Mode 2 / Mode 5:
    #   Use a lower probability so SPIKE represents isolated transient events,
    #   not continuous noise-like degradation.
    if is_training:
        spike_prob = 0.005
    else:
        spike_prob = 0.001

    return SystemFaultInjector(
        fault_type=target_fault_type,
        start_time=start_time,

        # 1. Bias fault
        bias=random.choice([-1, 1]) * random.uniform(5.0, 12.0),

        # 2. Spike fault
        spike_prob=spike_prob,
        spike_magnitude=random.choice([-1, 1]) * random.uniform(8.0, 15.0),

        # 3. Increased noise fault
        noise_std=random.uniform(0.5, 1.0),

        # 4. Drift fault
        drift_rate=random.choice([-1, 1]) * random.uniform(0.5, 1.0)
    )
# ======================
# FTC / Recovery Strategy Mapping
# ======================
def get_recovery_action(fault_id):
    """
    Map confirmed fault ID to a realistic FTC / recovery action.

    This keeps the current implementation focused on available emergency actions:
        - filtering for transient/noisy sensor faults
        - safe hover for bias
        - controlled emergency ascent for drift
        - emergency buoyancy ascent + acoustic beacon for severe faults
        - power cut + emergency ascent for thruster faults

    Future work can extend this to current-aware recovery probability optimization.
    """

    if fault_id == 0:
        return "Normal Cruising"

    if fault_id == 1:
        return "Safe Hover / Depth-Hold Using Estimated Depth"

    if fault_id == 2:
        return "Abort Mission + Controlled Emergency Ascent"

    if fault_id == 3:
        return "Emergency Buoyancy Ascent + Acoustic Beacon"

    if fault_id == 4:
        return "Spike Rejection Filter"

    if fault_id == 5:
        return "Adaptive Smoothing Filter"

    # FTC-level merge: raw 6 (ENTANGLED) and raw 8 (THRUST_LOSS)
    # are both handled as THRUSTER_THRUST_LOSS.
    if fault_id in [6, 8]:
        return "Degraded Thrust Mode + Controlled Emergency Ascent + Acoustic Beacon"

    if fault_id == 7:
        return "Power Cut + Emergency Buoyancy Ascent + Acoustic Beacon"

    return "Unknown Recovery Action"


def get_recovery_command(action_text, current_depth, safe_cmd_yaw):
    """
    Convert FTC / recovery action text into a simplified control command.

    Note:
    In this simulation, upward emergency ascent is represented by negative vz.
    In a real AUV, 'Emergency Buoyancy Ascent' would be implemented by
    buoyancy release / drop-weight / ballast mechanism rather than thruster thrust.
    """

    action_text = str(action_text)

    # Stop after reaching near surface
    if current_depth < 1.0:
        return np.array([0.0, 0.0, 0.0]), 0.0

    if "Safe Hover" in action_text or "Depth-Hold" in action_text:
        return np.array([0.0, 0.0, 0.0]), safe_cmd_yaw

    if "Controlled Emergency Ascent" in action_text:
        return np.array([0.0, 0.0, -0.5]), safe_cmd_yaw

    if "Emergency Buoyancy Ascent" in action_text:
        return np.array([0.0, 0.0, -2.0]), safe_cmd_yaw

    if "Power Cut" in action_text:
        return np.array([0.0, 0.0, 0.0]), 0.0

    return np.array([0.0, 0.0, 0.0]), safe_cmd_yaw


# ======================
# 核心任务调度器 (Mission Executor)
# ======================
def execute_mission(
        fault_type=None,
        is_demo=False,
        duration_override=None,
        fault_start_time=None,
        route_profile="standard"
):
    auv = create_auv(route_profile=route_profile)

    # Keep a fixed copy of the original planned route for visualization.
    # Do not use auv.waypoints after the mission, because it may be modified
    # during waypoint navigation.
    planned_waypoints = np.array(get_route_waypoints(route_profile), dtype=float)

    depth_sensor = DepthSensor()
    fault_injector = create_fault(
        target_fault_type=fault_type,
        is_training=False,
        start_time_override=fault_start_time
    )

    imu_sensor = IMUSensor()
    dvl_sensor = DVLSensor()
    current_sensor = CurrentSensor()
    battery_sensor = BatterySensor()
    residual_observer = ResidualObserver()
    diagnosis_strategy = DiagnosisStrategy()
    thruster_motor_model = ThrusterMotorModel(seed=42)
    simulator = Simulator(
        auv_model=auv,
        depth_sensor=depth_sensor,
        fault_injector=fault_injector,
        imu_sensor=imu_sensor,
        dvl_sensor=dvl_sensor,
        current_sensor=current_sensor,
        battery_sensor=battery_sensor
    )

    mean_path = r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model\depth_fault_detection\results\mean_stage3_9class.npy"
    std_path = r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model\depth_fault_detection\results\std_stage3_9class.npy"

    mean = np.load(mean_path).reshape(-1)
    std = np.load(std_path).reshape(-1)

    if mean.shape[0] != MODEL_INPUT_DIM or std.shape[0] != MODEL_INPUT_DIM:
        raise ValueError(
            f"Stage 3 normalization dimension mismatch: "
            f"mean={mean.shape}, std={std.shape}, expected={MODEL_INPUT_DIM}"
        )

    controller_buffer = [] # 给 AI 模型用，保存 Stage 3 的 20 维原始多传感器特征。
    diagnosis_history = [] #给规则诊断用，保存 sensor_data + residuals。

    # 系统状态标志
    system_locked = False
    is_safe_mode = False
    is_filtering = False  # 滤波器开关
    last_clean_depth = 0.0  # 存储上一帧干净的数据

    record_time = [-1.0]
    final_diagnosis = "NO_FAULT"
    final_action = "Normal Cruising"
    # BIAS is a soft fault. Delay confirmation to avoid confusing early DRIFT as BIAS.
    bias_candidate_start_time = None
    # BIAS needs a longer observation period on long complex routes,
    # otherwise early DRIFT or waypoint-transition residuals can be locked as BIAS.
    bias_confirm_delay = 15
    last_waypoint_change_time = -999.0
    stable_bias_start_time = None
    last_bias_candidate_print_time = -999.0

    locked_fault_id = 0          # 用于锁定型故障：BIAS / DRIFT / STUCK / ENTANGLED / NO_OUTPUT
    soft_fault_id = 0            # 用于非锁定型持续故障：NOISE_INCREASE
    confirmed_reason = "No diagnosis has been triggered yet."
    # 安全指令缓存与时间记录
    safe_cmd_vel = np.array([0.0, 0.0, 0.0])
    safe_cmd_yaw = 0.0
    spike_filtered_times = []
    last_spike_time = -999.0
    spike_cooldown = 3.0
    spike_recovery_window = 12
    def ftc_controller(sensor_data):
        nonlocal controller_buffer, diagnosis_history
        nonlocal system_locked, is_safe_mode
        nonlocal final_diagnosis, final_action, locked_fault_id, soft_fault_id, confirmed_reason
        nonlocal safe_cmd_vel, safe_cmd_yaw
        nonlocal is_filtering, last_clean_depth
        nonlocal last_spike_time
        nonlocal bias_candidate_start_time
        nonlocal stable_bias_start_time, last_bias_candidate_print_time
        nonlocal last_waypoint_change_time
        # 初始化状态机的持久化变量
        if not hasattr(ftc_controller, "dynamic_setpoint"):
            ftc_controller.dynamic_setpoint = sensor_data["depth"]
        if not hasattr(ftc_controller, "fault_counter"):
            ftc_controller.fault_counter = 0
            ftc_controller.current_pred = 0

        # Track waypoint switches. Immediately after a waypoint transition,
        # target/depth residuals can jump even in NO_FAULT, so sensor faults
        # should not be confirmed for a short guard window.
        current_wp_count = len(getattr(auv, "waypoints", []))
        if not hasattr(ftc_controller, "last_wp_count"):
            ftc_controller.last_wp_count = current_wp_count
        elif current_wp_count != ftc_controller.last_wp_count:
            last_waypoint_change_time = sensor_data["time"]
            ftc_controller.last_wp_count = current_wp_count
        # 诊断日志打印（每 30 秒打印一次）
        #if int(sensor_data["time"]) % 30 == 0 and abs(sensor_data["time"] - round(sensor_data["time"])) < 0.05:
            #print("Sensor check:")
            #print("DVL:", sensor_data.get("dvl", None))
            #print("Current Sensor:", sensor_data.get("current_sensor", None))
            #print("Battery:", sensor_data.get("battery", None))

        def finalize_return(cmd_vel, cmd_yaw):
            """统一写入 FTC 与诊断日志，保证图像中的 Final Diagnosis / Reason / Action 一致。"""
            sensor_data["ftc_diagnosis"] = final_diagnosis
            sensor_data["ftc_action"] = final_action
            sensor_data["ftc_is_locked"] = system_locked

            if "diagnosis_reason" not in sensor_data:
                sensor_data["diagnosis_reason"] = confirmed_reason

            if "ai_pred" not in sensor_data:
                sensor_data["ai_pred"] = 0

            if "rule_pred" not in sensor_data:
                sensor_data["rule_pred"] = 0

            if "final_pred" not in sensor_data:
                sensor_data["final_pred"] = 0

            # 锁定型故障：确认后持续保持最终故障标签
            if system_locked and locked_fault_id != 0:
                sensor_data["final_pred"] = locked_fault_id
                sensor_data["diagnosis_reason"] = confirmed_reason

            # 非锁定型持续故障：例如 NOISE，保持诊断显示，但不锁死控制器
            elif soft_fault_id != 0:
                sensor_data["final_pred"] = soft_fault_id
                sensor_data["diagnosis_reason"] = confirmed_reason

            return cmd_vel, cmd_yaw

        def get_recovery_depth(sensor_data):
            """
            Use true physical depth only for simulation-side recovery boundary.
            Diagnosis still uses measured sensor data.
            """
            return float(sensor_data.get("true_depth", sensor_data["depth"]))
        raw_depth = sensor_data["depth"]

        # ========================================================
        # 模块 1：物理底盘控制与自愈层 (Physics & Self-Healing)
        # ========================================================
        final_goal_z = sensor_data.get("target_z", 0.0)
        MAX_Z_STEP_SPEED = 1.8
        DT = 0.1
        max_step = MAX_Z_STEP_SPEED * DT

        # 计算平滑轨迹 (兔子逻辑)
        diff_to_goal = final_goal_z - ftc_controller.dynamic_setpoint

        # 牵引绳法则：只有当误差在 3.0 米内时，目标点才继续移动
        if abs(sensor_data["depth"] - ftc_controller.dynamic_setpoint) < 3.0:
            if abs(diff_to_goal) > max_step:
                ftc_controller.dynamic_setpoint += np.sign(diff_to_goal) * max_step
            else:
                ftc_controller.dynamic_setpoint = final_goal_z

        sensor_data["target_z"] = ftc_controller.dynamic_setpoint

        # 计算底层 PID 控制指令
        if is_filtering:
            alpha = 0.1
            smoothed_depth = alpha * raw_depth + (1.0 - alpha) * last_clean_depth
            last_clean_depth = smoothed_depth
            fake_sensor_data = sensor_data.copy()
            fake_sensor_data["depth"] = smoothed_depth
            normal_cmd_vel, normal_cmd_yaw = simple_controller(fake_sensor_data, auv)
        else:
            last_clean_depth = raw_depth
            normal_cmd_vel, normal_cmd_yaw = simple_controller(sensor_data, auv)

        # Cache the latest normal command as a safe fallback command.
        # This is used by Safe Hover / Controlled Ascent recovery modes.
        safe_cmd_vel = normal_cmd_vel.copy()
        safe_cmd_yaw = normal_cmd_yaw

        # ========================================================
        # Thruster motor equation simulation
        # 使用当前控制器输出的垂向速度作为 cmd_vz
        # command -> expected current / omega / thrust
        # fault mode -> measured current / omega / actual thrust
        # ========================================================
        cmd_vz = float(normal_cmd_vel[2])

        # 保证 thruster 字段存在
        if "thruster" not in sensor_data or not isinstance(sensor_data["thruster"], dict):
            sensor_data["thruster"] = {}

        sensor_data["thruster"]["cmd_vz"] = cmd_vz

        # --------------------------------------------------------
        # Current Stage 3.0-A:
        # label 7 represents THRUSTER_NO_OUTPUT / no_output
        # label 6 = entangled
        # label 7 = no_output
        # 其他故障暂时不作用到 motor model
        # --------------------------------------------------------
        fault_label = sensor_data.get("fault_label", 0)

        if fault_label == 6:
            motor_fault_mode = ThrusterFaultMode.ENTANGLED
        elif fault_label == 7:
            motor_fault_mode = ThrusterFaultMode.NO_OUTPUT
        elif fault_label == 8:
            motor_fault_mode = ThrusterFaultMode.THRUST_LOSS
        else:
            motor_fault_mode = ThrusterFaultMode.NORMAL

        motor_state = thruster_motor_model.simulate(
            cmd=cmd_vz,
            fault_mode=motor_fault_mode
        )

        # ========================================================
        # Write motor-equation output into unified sensor channels
        # ========================================================

        # 1) Current sensor channel
        # 电流传感器读数统一来自 ThrusterMotorModel
        sensor_data["current_sensor"] = {
            "measured_current": motor_state.measured_current,
            "expected_current": motor_state.expected_current,
            "current_residual": motor_state.current_residual,
        }

        # 2) Thruster channel
        # thruster 主要保存 command / omega / thrust / fault mode
        if "thruster" not in sensor_data or not isinstance(sensor_data["thruster"], dict):
            sensor_data["thruster"] = {}

        sensor_data["thruster"]["cmd_vz"] = cmd_vz

        sensor_data["thruster"]["omega"] = motor_state.measured_omega
        sensor_data["thruster"]["expected_omega"] = motor_state.expected_omega
        sensor_data["thruster"]["omega_residual"] = motor_state.omega_residual

        sensor_data["thruster"]["thrust"] = motor_state.actual_thrust
        sensor_data["thruster"]["expected_thrust"] = motor_state.expected_thrust
        sensor_data["thruster"]["thrust_residual"] = motor_state.thrust_residual

        sensor_data["thruster"]["motor_fault_mode"] = motor_state.fault_mode

        # Optional compatibility fields
        # 先保留，避免旧代码某些地方还在读 thruster["current"]
        sensor_data["thruster"]["current"] = motor_state.measured_current
        sensor_data["thruster"]["expected_current"] = motor_state.expected_current
        sensor_data["thruster"]["current_residual"] = motor_state.current_residual

        # ========================================================
        # Residual Observer
        # ========================================================
        residuals = residual_observer.compute(sensor_data)

        # Add motor-equation residuals to residuals as well.
        # Existing code can ignore these fields safely.
        residuals["omega_residual"] = motor_state.omega_residual
        residuals["thrust_residual"] = motor_state.thrust_residual
        residuals["expected_omega"] = motor_state.expected_omega
        residuals["measured_omega"] = motor_state.measured_omega
        residuals["expected_thrust"] = motor_state.expected_thrust
        residuals["actual_thrust"] = motor_state.actual_thrust
        residuals["motor_fault_mode"] = motor_state.fault_mode

        sensor_data["residuals"] = residuals

        diagnosis_history.append(sensor_data.copy())
        if len(diagnosis_history) > 50:
            diagnosis_history.pop(0)

        # ========================================================
        # 模块 2：最高优先级拦截 (Absolute Deadlock)
        # ========================================================
        if system_locked:
            recovery_depth = get_recovery_depth(sensor_data)

            recovery_cmd_vel, recovery_cmd_yaw = get_recovery_command(
                action_text=final_action,
                current_depth=recovery_depth,
                safe_cmd_yaw=safe_cmd_yaw
            )

            return finalize_return(recovery_cmd_vel, recovery_cmd_yaw)

        # ========================================================
        # 模块 3：AI 预测与物理护城河 (AI Inference & Physical Moat)
        # ========================================================
        f_depth = smoothed_depth if is_filtering else sensor_data["depth"]
        f_target = ftc_controller.dynamic_setpoint
        pred = 0

        # 默认诊断变量，防止窗口长度不足 50 时后续逻辑引用未定义变量
        ai_pred = 0
        rule_pred = 0
        diagnosis_source = "none"
        diagnosis_reason = confirmed_reason
        diagnosis_confidence = "Low"
        is_physical_spike = False

        # Defaults used by the final debouncer even before the AI window is full.
        # These guards are based on real navigation/control state, not on the
        # simulated fault_start_time.
        is_aggressive_vertical_maneuver = False
        persistent_sensor_guard_active = False
        latest_depth_error = 0.0
        recent_error_range = 0.0
        is_stable_bias_like_error = False
        is_drift_like_error = False

        # Stage 3:
        # Use the same multi-sensor feature extractor as dataset generation.
        # If adaptive filtering is active, feed the smoothed depth into the AI feature vector.
        ai_sensor_data = sensor_data.copy()
        ai_sensor_data["depth"] = f_depth
        ai_sensor_data["target_z"] = f_target

        current_features = extract_ai_features(ai_sensor_data)
        controller_buffer.append(current_features)
        if len(controller_buffer) > 50:
            controller_buffer.pop(0)

        if len(controller_buffer) == 50 and AI_MODEL_AVAILABLE:
            raw_seq = np.array(controller_buffer, dtype=np.float32)
            feature_dim = raw_seq.shape[1]
            if raw_seq.shape[0] >= 2:
                is_physical_spike = abs(raw_seq[-1, 0] - raw_seq[-2, 0]) >= 1.0

            if feature_dim != RAW_FEATURE_DIM:
                raise ValueError(
                    f"AI feature dimension mismatch: got {feature_dim}, expected {RAW_FEATURE_DIM}"
                )

            diff_seq = np.vstack([
                np.zeros((1, feature_dim), dtype=np.float32),
                np.diff(raw_seq, axis=0)
            ])

            input_seq_flat = np.stack((raw_seq, diff_seq), axis=-1).reshape(50, -1)

            if input_seq_flat.shape[1] != MODEL_INPUT_DIM:
                raise ValueError(
                    f"Model input dimension mismatch: got {input_seq_flat.shape[1]}, expected {MODEL_INPUT_DIM}"
                )

            input_seq_norm = (input_seq_flat - mean) / (std + 1e-8) if isinstance(std, np.ndarray) else input_seq_flat
            input_tensor = torch.tensor(input_seq_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred = torch.argmax(model(input_tensor), dim=1).item()

            # 物理护城河判断逻辑
            if pred == 3 and abs(raw_seq[-1, 0] - raw_seq[-5, 0]) > 0.5: pred = 2
            if sensor_data["time"] < 35.0: pred = 0  # 启动保护期
            dist_to_dest = np.linalg.norm(sensor_data["position"] - auv.destination)
            if dist_to_dest < 20.0: pred = 0
            if is_filtering and pred == 4: pred = 0  # 滤波态忽略脉冲

            # 动态意图保护
            tracking_error = abs(sensor_data["depth"] - ftc_controller.dynamic_setpoint)
            is_cruising = abs(ftc_controller.dynamic_setpoint - final_goal_z) > 0.1
            cmd_abs = abs(normal_cmd_vel[2])
            actual_vz_abs = abs(sensor_data["thruster"]["actual_vz"])

            is_aggressive_vertical_maneuver = (
                    is_cruising
                    and (
                            cmd_abs > 0.8
                            or actual_vz_abs > 0.6
                            or tracking_error > 2.0
                    )
            )

            # Do NOT force STUCK only from small depth change here.
            # On long routes this can create false STUCK/BIAS around waypoint transitions.
            # STUCK should come from the rule-based diagnosis layer.

            # 运行状态日志打印
            #if 40 <= sensor_data['time'] <= 70:
            #    state_str = "CRUISE" if is_cruising else "HOVER"
            #    print(
            #        f"Time: {sensor_data['time']:.1f}s | Pred: {pred} | Cmd: {normal_cmd_vel[2]:.4f} | Error: {tracking_error:.4f} | State: {state_str}")

            if is_cruising:
                # 巡航态下允许较大的跟随误差
                if pred == 1 and tracking_error < 5.0:
                    pred = 0
                if pred == 3:
                    depth_change = abs(raw_seq[-1, 0] - raw_seq[-5, 0])
                    if depth_change > 0.1:
                        pred = 0
            else:
                # 悬停态下严格限制误差
                if pred in [1, 3] and abs(normal_cmd_vel[2]) < 0.3 and tracking_error < 1.0:
                    pred = 0

            # 动力系统专属约束
            cmd_z = normal_cmd_vel[2]
            actual_vz = sensor_data["thruster"]["actual_vz"]
            if pred in [6, 7, 8] and abs(cmd_z) < 0.6: pred = 0
            if pred in [6, 7, 8] and (actual_vz * cmd_z > 0) and abs(actual_vz) > 0.1: pred = 0

            # 脉冲物理约束
            if pred == 4 and abs(raw_seq[-1, 0] - raw_seq[-2, 0]) < 1.0: pred = 0
            # AI-only DRIFT/STUCK predictions during aggressive vertical maneuver are unreliable.
            # Let the rule layer decide later; do not pass these weak AI candidates forward.
            if is_aggressive_vertical_maneuver and pred == 2:
                pred = 0
                diagnosis_reason = (
                    "Suppressed DRIFT during aggressive waypoint transition."
                )

            elif (
                    is_aggressive_vertical_maneuver
                    and pred == 3
                    and not (diagnosis_source == "rule" and rule_pred == 3)
            ):
                pred = 0
                diagnosis_reason = (
                    "Suppressed weak AI-only STUCK during aggressive waypoint transition."
                )
            # ========================================================
            # 新增：Rule-based diagnosis + AI fusion
            # ========================================================
            ai_pred = pred

            diagnosis_result = diagnosis_strategy.diagnose(
                sensor_data=sensor_data,
                residuals=residuals,
                history=diagnosis_history,
                ai_pred=ai_pred
            )

            rule_pred = diagnosis_result.fault_id
            diagnosis_source = diagnosis_result.source
            diagnosis_reason = diagnosis_result.reason
            diagnosis_confidence = diagnosis_result.confidence

            # ========================================================
            # New fusion priority
            # ========================================================
            # 1. 如果系统已经由锁定型故障接管，则保持锁定故障标签
            if system_locked and locked_fault_id != 0:
                pred = locked_fault_id

            # 2. 如果 NOISE 已经确认并进入滤波状态，保持 NOISE 显示。
            #    但更严重的锁定型故障应在确认阶段覆盖 soft fault。
            elif is_filtering and soft_fault_id == 5:
                pred = 5

            # 3. 如果规则诊断有明确物理证据，则优先使用规则结果
            elif diagnosis_source == "rule" and rule_pred != 0:
                pred = rule_pred

            # 4. 否则使用 AI 预测
            else:
                pred = ai_pred

            # 5. Waypoint transition / aggressive maneuver guard.
            # Right before and after a waypoint switch, suppress soft sensor candidates.


            dist_to_current_wp = 999.0
            if len(getattr(auv, "waypoints", [])) > 0:
                current_wp = np.array(auv.waypoints[0], dtype=float)
                dist_to_current_wp = float(np.linalg.norm(sensor_data["position"] - current_wp))

            time_since_waypoint = sensor_data["time"] - last_waypoint_change_time

            waypoint_guard_active = time_since_waypoint < 8.0

            near_waypoint_guard_active = (
                    dist_to_current_wp < 15.0
                    and time_since_waypoint < 8.0
            )

            # Approach guard:
            # Before the waypoint is officially reached, the vehicle may already be
            # in a strong vertical maneuver. This can create a residual trend similar
            # to DRIFT. This guard does not use fault_start_time, so it is also
            # realistic for online operation.
            approaching_waypoint_guard_active = (
                    dist_to_current_wp < 30.0
                    and (
                            is_aggressive_vertical_maneuver
                            or cmd_abs > 0.6
                            or tracking_error > 2.0
                    )
            )

            persistent_sensor_guard_active = (
                    waypoint_guard_active
                    or near_waypoint_guard_active
                    or approaching_waypoint_guard_active
                    or is_aggressive_vertical_maneuver
            )

            # Guard should suppress only weak AI-only persistent sensor candidates.
            # If the rule layer has already provided physical evidence, keep the
            # candidate and let the debouncer confirm it over time.
            rule_confirmed_persistent_fault = (
                    diagnosis_source == "rule"
                    and rule_pred in [1, 2, 3]
            )

            if (
                    persistent_sensor_guard_active
                    and pred in [1, 2, 3]
                    and not rule_confirmed_persistent_fault
            ):
                pred = 0
                diagnosis_reason = (
                    "Suppressed weak AI-only BIAS/DRIFT/STUCK during waypoint transition "
                    "or aggressive vertical maneuver guard."
                )
            # ========================================================
            # BIAS / DRIFT / STUCK arbitration
            # ========================================================
            # Purpose:
            #   1. Prevent early DRIFT from being locked as BIAS.
            #   2. Prevent confirmed BIAS candidates from being stolen by STUCK.
            #   3. Keep true STUCK detectable when no BIAS candidate exists.
            # ========================================================

            error_trend_5s = 0.0
            latest_depth_error = sensor_data["depth"] - sensor_data["target_z"]

            if len(diagnosis_history) >= 50:
                recent_errors = np.array([
                    h["depth"] - h.get("target_z", h["depth"])
                    for h in diagnosis_history[-50:]
                ], dtype=np.float32)

                error_trend_5s = float(recent_errors[-1] - recent_errors[0])
                recent_error_range = float(np.max(recent_errors) - np.min(recent_errors))
            else:
                recent_error_range = 0.0

            is_drift_like_error = (
                    abs(latest_depth_error) > 5.0
                    and abs(error_trend_5s) > 2.0
                    and recent_error_range > 2.0
            )

            is_stable_bias_like_error = (
                    abs(latest_depth_error) > 5.0
                    and abs(error_trend_5s) < 1.0
                    and recent_error_range < 2.0
            )

            # A slightly wider stable-offset evidence used to keep BIAS monitoring
            # alive even when individual frames temporarily become pred=0 or pred=3.
            stable_bias_evidence = (
                    abs(latest_depth_error) > 4.0
                    and recent_error_range < 3.0
                    and not is_drift_like_error
            )

            # Continuous BIAS evidence timer.
            # This timer is based on the physical residual pattern, not on every
            # single-frame class output. Therefore it is not reset by occasional
            # pred=0 or pred=3 frames in true BIAS cases.
            if stable_bias_evidence and (ai_pred == 1 or rule_pred == 1 or pred == 1):
                if stable_bias_start_time is None:
                    stable_bias_start_time = sensor_data["time"]
            elif is_drift_like_error or abs(latest_depth_error) < 4.0:
                stable_bias_start_time = None

            # Case 1:
            # If the current candidate is BIAS but the residual is still changing,
            # it is more likely DRIFT than BIAS.
            if pred == 1 and is_drift_like_error:
                pred = 2
                bias_candidate_start_time = None
                diagnosis_reason = (
                    "BIAS candidate converted to DRIFT because depth residual "
                    "is continuously changing."
                )

            # Case 2:
            # If the depth error is already a stable offset, do not let an
            # occasional STUCK / zero-motion frame steal the BIAS monitor.
            elif (
                    pred == 3
                    and stable_bias_evidence
                    and (
                            bias_candidate_start_time is not None
                            or ai_pred == 1
                            or rule_pred == 1
                    )
            ):
                pred = 1
                diagnosis_reason = (
                    "STUCK candidate converted back to BIAS because the depth error "
                    "is a stable offset and BIAS is already supported/monitored."
                )
            # During large vertical maneuvers, suppress weak DRIFT.
            # But allow rule-confirmed STUCK because a stuck depth reading can occur exactly
            # while the controller is demanding vertical motion.
            elif (
                    is_aggressive_vertical_maneuver
                    and pred == 2
                    and not rule_confirmed_persistent_fault
            ):
                pred = 0
                diagnosis_reason = (
                    "Suppressed weak AI-only DRIFT during aggressive waypoint transition."
                )
            elif is_aggressive_vertical_maneuver and pred == 3:
                if not (diagnosis_source == "rule" and rule_pred == 3):
                    pred = 0
                    diagnosis_reason = (
                        "Suppressed weak AI STUCK during aggressive waypoint transition."
                    )

            # Final safety gate after BIAS/DRIFT/STUCK arbitration.
            # This is necessary because the arbitration above may convert BIAS to DRIFT
            # after the first waypoint guard has already been applied.
            if (
                    persistent_sensor_guard_active
                    and pred in [1, 2, 3]
                    and not rule_confirmed_persistent_fault
            ):
                pred = 0
                diagnosis_reason = (
                    "Suppressed weak AI-only persistent depth-sensor candidate during "
                    "maneuver/waypoint guard after arbitration."
                )

            # --------------------------------------------------------
            # FTC / monitoring-level merge for similar thruster faults.
            # Keep AI and rule predictions as raw 9-class values, but merge
            # raw 6 (ENTANGLED) and raw 8 (THRUST_LOSS) before monitoring,
            # debouncing, locking, and FTC action selection.
            # --------------------------------------------------------
            raw_final_pred = int(pred)
            pred = merge_fault_for_monitoring_and_ftc(pred)

            sensor_data["ai_pred"] = ai_pred
            sensor_data["rule_pred"] = rule_pred
            sensor_data["raw_final_pred"] = raw_final_pred
            sensor_data["final_pred"] = pred
            sensor_data["ftc_final_pred"] = pred
            sensor_data["ftc_fault_name"] = get_ftc_fault_name(pred)
            sensor_data["diagnosis_reason"] = diagnosis_reason
            sensor_data["diagnosis_confidence"] = diagnosis_confidence
            sensor_data["diagnosis_source"] = diagnosis_source

            # Long-mission diagnostic print: one line every 30 seconds.
            if (
                    int(sensor_data["time"]) % 30 == 0
                    and abs(sensor_data["time"] - round(sensor_data["time"])) < 0.05
            ):
                print(
                    f"Time: {sensor_data['time']:.1f}s | "
                    f"TrueLabel={sensor_data.get('fault_label', -1)}, "
                    f"MotorMode={sensor_data['thruster'].get('motor_fault_mode', 'normal')}, "
                    f"I={sensor_data['thruster'].get('current', 0.0):.2f}, "
                    f"Iexp={sensor_data['thruster'].get('expected_current', 0.0):.2f}, "
                    f"rI={sensor_data['thruster'].get('current_residual', 0.0):.2f}, "
                    f"T={sensor_data['thruster'].get('thrust', 0.0):.2f}, "
                    f"Texp={sensor_data['thruster'].get('expected_thrust', 0.0):.2f}, "
                    f"AI={ai_pred}, Rule={rule_pred}, RawFinal={raw_final_pred}, Final={pred}, "
                    f"FTCFault={get_ftc_fault_name(pred)}, Source={diagnosis_source}, "
                    f"Depth={sensor_data['depth']:.2f}, "
                    f"Target={sensor_data['target_z']:.2f}"
                )
        # ========================================================
        # 模块 4：双轨防抖系统 (The "Two-Lane" Debouncer)
        # ========================================================
        # 轨道 A：瞬态故障 (SPIKE)
        # ========================================================
        # Transient fault lane: SPIKE
        # SPIKE is only filtered immediately when it is an isolated spike.
        # It must not override NOISE_INCREASE or other persistent faults.
        # ========================================================
        if (
                pred == 4
                and not is_filtering
                and is_physical_spike
        ):
            current_time = sensor_data["time"]

            if current_time - last_spike_time >= spike_cooldown:
                print(f"[{current_time:.1f}s] SPIKE detected and filtered.")

                spike_filtered_times.append(current_time)
                last_spike_time = current_time

            final_diagnosis = "SPIKE"
            final_action = get_recovery_action(4)
            confirmed_reason = diagnosis_reason if diagnosis_reason else "AI + physical depth jump detected transient spike."

            sensor_data["final_pred"] = 4
            sensor_data["diagnosis_reason"] = confirmed_reason

            return finalize_return(normal_cmd_vel, normal_cmd_yaw)
        # --------------------------------------------------------
        # Spike recovery guard
        # After a transient spike is filtered, do not allow
        # DRIFT/STUCK to be locked immediately due to residual shock.
        # --------------------------------------------------------
        time_since_spike = sensor_data["time"] - last_spike_time

        if time_since_spike < spike_recovery_window and pred in [2, 3]:
            pred = 0
            sensor_data["final_pred"] = 0
            sensor_data["diagnosis_reason"] = (
                f"Transient spike recovery guard active "
                f"({time_since_spike:.1f}s after spike)."
            )
        # Reset stale BIAS candidate only when the stable-offset evidence disappears.
        # In true BIAS cases, individual frames may temporarily become pred=0 or pred=3
        # because the biased depth reading becomes almost constant. If we reset the
        # timer on those frames, BIAS will print "candidate" forever and never lock.
        stable_bias_evidence_for_reset = (
                abs(latest_depth_error) > 4.0
                and recent_error_range < 3.0
                and not is_drift_like_error
        )

        if pred != 1 and not (
                bias_candidate_start_time is not None
                and stable_bias_evidence_for_reset
                and pred in [0, 3]
        ):
            bias_candidate_start_time = None

        # 轨道 B：稳态故障读条逻辑
        # Persistent depth-sensor faults must not be locked from short transient
        # candidates. In long routes, BIAS/DRIFT/STUCK can appear temporarily
        # around waypoint transitions or strong vertical maneuvers.
        if (
                pred in [1, 2, 3]
                and persistent_sensor_guard_active
                and not (
                        diagnosis_source == "rule"
                        and rule_pred in [1, 2, 3]
                )
        ):
            pred = 0
            sensor_data["final_pred"] = 0
            sensor_data["diagnosis_reason"] = (
                "Weak AI-only persistent depth-sensor candidate ignored by maneuver guard "
                "before debounce confirmation."
            )

        if pred != 0:
            if pred == ftc_controller.current_pred:
                ftc_controller.fault_counter += 1
            else:
                ftc_controller.current_pred = pred
                ftc_controller.fault_counter = 1
        else:
            ftc_controller.fault_counter = 0
            ftc_controller.current_pred = 0

        # 获取诊断阈值
        # Counts are measured at dt=0.1 s.
        # Keep BIAS/DRIFT/STUCK slower than transient faults, but not so slow
        # that real long-route faults can never lock.
        threshold_map = {
            1: 8,   # BIAS,  about 0.8 s
            2: 20,  # DRIFT, about 2.0 s
            3: 8,   # STUCK, about 0.8 s
            5: 20,  # NOISE_INCREASE
            6: 8,   # THRUSTER_THRUST_LOSS (FTC-merged raw 6 + raw 8)
            7: 8,   # THRUSTER_NO_OUTPUT
            8: 8,   # Fallback only; raw 8 is merged to 6 before debounce
        }
        confirm_threshold = threshold_map.get(ftc_controller.current_pred, 999)

        # ========================================================
        # 模块 5：最终执行决断
        # ========================================================
        if ftc_controller.fault_counter >= confirm_threshold:
            raw_confirmed_fault = int(ftc_controller.current_pred)
            confirmed_fault = merge_fault_for_monitoring_and_ftc(raw_confirmed_fault)
            sensor_data["raw_confirmed_fault"] = raw_confirmed_fault
            sensor_data["ftc_confirmed_fault"] = confirmed_fault
            sensor_data["ftc_fault_name"] = get_ftc_fault_name(confirmed_fault)

            confirmed_reason = sensor_data.get("diagnosis_reason", diagnosis_reason)

            # Reset BIAS candidate timer when the confirmed fault is not BIAS.
            # This prevents an old BIAS candidate from affecting later decisions.
            if confirmed_fault != 1:
                bias_candidate_start_time = None
                stable_bias_start_time = None

            if confirmed_fault == 5 and not is_filtering:
                print(f"[{sensor_data['time']:.1f}s] NOISE_INCREASE! Adaptive Smoothing Filter activated.")

                if record_time[0] < 0:
                    record_time[0] = sensor_data["time"]

                final_diagnosis = "NOISE"
                final_action = get_recovery_action(5)
                soft_fault_id = 5
                is_filtering = True

                return finalize_return(normal_cmd_vel, normal_cmd_yaw)


            elif confirmed_fault == 1 and not is_filtering:
                waypoint_guard_active = (sensor_data["time"] - last_waypoint_change_time) < 8.0

                dist_to_current_wp = 999.0
                if len(getattr(auv, "waypoints", [])) > 0:
                    current_wp = np.array(auv.waypoints[0], dtype=float)

                    dist_to_current_wp = float(np.linalg.norm(sensor_data["position"] - current_wp))
                near_waypoint_guard_active = dist_to_current_wp < 11.0

                # Do not delay a rule-confirmed BIAS only because the vehicle is in a
                # maneuver. A real BIAS creates a stable offset and should be allowed
                # to pass through the debouncer. Guard only weak AI-only BIAS.
                if (
                        (waypoint_guard_active or near_waypoint_guard_active or persistent_sensor_guard_active)
                        and not (
                                diagnosis_source == "rule"
                                and rule_pred == 1
                        )
                ):
                    bias_candidate_start_time = None
                    ftc_controller.fault_counter = 0
                    ftc_controller.current_pred = 0
                    final_diagnosis = "NO_FAULT"
                    final_action = "Normal Cruising"
                    confirmed_reason = "Weak BIAS confirmation delayed near waypoint/maneuver guard."
                    return finalize_return(normal_cmd_vel, normal_cmd_yaw)

                # Do not allow a changing residual to be locked as BIAS.
                # This protects DRIFT from being confirmed as BIAS.
                if is_drift_like_error:
                    bias_candidate_start_time = None
                    stable_bias_start_time = None
                    ftc_controller.current_pred = 2
                    ftc_controller.fault_counter = max(
                        ftc_controller.fault_counter,
                        threshold_map.get(2, 20) // 2
                    )
                    final_diagnosis = "NO_FAULT"
                    final_action = "Normal Cruising"
                    confirmed_reason = (
                        "BIAS confirmation blocked because the depth error is still changing; "
                        "monitoring as possible DRIFT."
                    )
                    return finalize_return(normal_cmd_vel, normal_cmd_yaw)

                # Use the continuous stable-bias timer instead of resetting the
                # age every time the debouncer re-enters this branch.
                if stable_bias_start_time is None:
                    stable_bias_start_time = sensor_data["time"]

                bias_candidate_age = sensor_data["time"] - stable_bias_start_time

                if (
                        sensor_data["time"] - last_bias_candidate_print_time >= 10.0
                        or bias_candidate_age < 0.2
                ):
                    print(
                        f"[{sensor_data['time']:.1f}s] BIAS candidate! "
                        f"Monitoring before lock. age={bias_candidate_age:.1f}s"
                    )
                    last_bias_candidate_print_time = sensor_data["time"]

                # Keep normal control during the confirmation delay.
                # A longer delay gives true DRIFT enough time to develop a clear
                # trend, preventing early DRIFT from being locked as BIAS.
                if bias_candidate_age < bias_confirm_delay:
                    final_diagnosis = "NO_FAULT"
                    final_action = "Normal Cruising"
                    return finalize_return(normal_cmd_vel, normal_cmd_yaw)

                # Confirm BIAS only if it remains stable for enough time.
                print(f"[{sensor_data['time']:.1f}s] BIAS confirmed! Safe Hover / Depth-Hold activated.")

                if record_time[0] < 0:
                    record_time[0] = sensor_data["time"]

                final_diagnosis = "BIAS"
                final_action = get_recovery_action(1)
                locked_fault_id = confirmed_fault
                system_locked = True

                return finalize_return(
                    *get_recovery_command(final_action, get_recovery_depth(sensor_data), safe_cmd_yaw)
                )

            elif confirmed_fault == 2 and not is_filtering:
                if (
                        persistent_sensor_guard_active
                        and not (
                                diagnosis_source == "rule"
                                and rule_pred in [1, 2]
                        )
                ):
                    ftc_controller.fault_counter = 0
                    ftc_controller.current_pred = 0
                    final_diagnosis = "NO_FAULT"
                    final_action = "Normal Cruising"
                    confirmed_reason = (
                        "Weak DRIFT confirmation delayed by maneuver/waypoint guard."
                    )
                    return finalize_return(normal_cmd_vel, normal_cmd_yaw)

                print(f"[{sensor_data['time']:.1f}s] DRIFT confirmed! Abort Mission + Controlled Emergency Ascent.")
                if record_time[0] < 0:
                    record_time[0] = sensor_data["time"]
                final_diagnosis = "DRIFT"
                final_action = get_recovery_action(2)
                locked_fault_id = confirmed_fault
                system_locked = True
                return finalize_return(
                    *get_recovery_command(final_action, get_recovery_depth(sensor_data), safe_cmd_yaw)
                )

            elif confirmed_fault == 3 and not is_filtering:
                if (
                        persistent_sensor_guard_active
                        and not (
                                diagnosis_source == "rule"
                                and rule_pred == 3
                        )
                ):
                    ftc_controller.fault_counter = 0
                    ftc_controller.current_pred = 0
                    final_diagnosis = "NO_FAULT"
                    final_action = "Normal Cruising"
                    confirmed_reason = (
                        "Weak STUCK confirmation delayed by maneuver/waypoint guard."
                    )
                    return finalize_return(normal_cmd_vel, normal_cmd_yaw)

                print(f"[{sensor_data['time']:.1f}s] STUCK confirmed! Emergency Buoyancy Ascent + Acoustic Beacon.")
                if record_time[0] < 0:
                    record_time[0] = sensor_data["time"]
                final_diagnosis = "STUCK"
                final_action = get_recovery_action(3)
                locked_fault_id = confirmed_fault
                system_locked = True
                return finalize_return(
                    *get_recovery_command(final_action, get_recovery_depth(sensor_data), safe_cmd_yaw)
                )

            elif confirmed_fault == 6:
                raw_fault_name = FAULT_NAMES.get(raw_confirmed_fault, "UNKNOWN")
                print(
                    f"[{sensor_data['time']:.1f}s] THRUSTER THRUST LOSS! "
                    f"(raw={raw_confirmed_fault}:{raw_fault_name}) "
                    "Degraded Thrust Mode + Controlled Emergency Ascent + Acoustic Beacon."
                )
                if record_time[0] < 0:
                    record_time[0] = sensor_data["time"]

                final_diagnosis = "THRUSTER_THRUST_LOSS"
                final_action = get_recovery_action(6)
                locked_fault_id = confirmed_fault
                system_locked = True
                sensor_data["raw_thruster_fault"] = raw_confirmed_fault
                sensor_data["ftc_fault_name"] = "THRUSTER_THRUST_LOSS"

                return finalize_return(
                    *get_recovery_command(final_action, get_recovery_depth(sensor_data), safe_cmd_yaw)
                )

            elif confirmed_fault == 7:
                print(f"[{sensor_data['time']:.1f}s] NO OUTPUT! Power Cut + Emergency Buoyancy Ascent + Acoustic Beacon.")

                if record_time[0] < 0:
                    record_time[0] = sensor_data["time"]
                final_diagnosis, final_action = "THRUSTER_NO_OUTPUT", get_recovery_action(7)
                locked_fault_id = confirmed_fault
                system_locked = True
                return finalize_return(
                    *get_recovery_command(final_action, get_recovery_depth(sensor_data), safe_cmd_yaw)
                )
        # 正常指令输出
        safe_cmd_vel = normal_cmd_vel * 0.5 if is_safe_mode else normal_cmd_vel
        return finalize_return(safe_cmd_vel, normal_cmd_yaw)

    # 设置任务时长
    if duration_override is not None:
        duration = duration_override
    else:
        duration = 1200 if is_demo else 1000

    true_fault_str = fault_injector.fault_type.name
    # Display-level true fault name used by reports, static 3D figures, and animation.
    # The raw simulator label is still kept in sensor_logs as fault_label.
    display_true_fault_str = get_monitoring_fault_name_from_raw_name(true_fault_str)

    if is_demo:
        print(f"\n Current scenario fault type: [{true_fault_str}]")
        print(f" AI FTC monitoring system is online and ready...\n")

    simulator.run_mission(duration=duration, control_function=ftc_controller, dt=0.1)

    if fault_injector.fault_type == FaultType.NO_FAULT:
        actual_fault_time = None
    else:
        actual_fault_time = fault_injector.start_time
    actual_ai_time = record_time[0] if record_time[0] > 0 else None

    import time  # 导入 time 模块

    # 生成保存文件名
    # Long-mission timing tests include the forced fault start time to avoid overwriting files.
    if fault_start_time is not None:
        time_tag = f"_T{int(fault_start_time)}s"
    else:
        time_tag = ""

    route_tag = f"_{route_profile}"

    if is_demo:
        time_str = time.strftime("%H%M%S")
        save_name = f"results/FTC_Response_Demo_{true_fault_str}{time_tag}{route_tag}_{time_str}.png"
    else:
        time_str = ""
        save_name = f"results/FTC_Response_{true_fault_str}{time_tag}{route_tag}.png"

    # 绘制 2D 容错响应图
    plot_ftc_response(
        logs=simulator.sensor_logs,
        fault_time=actual_fault_time,
        ai_intervention_time=actual_ai_time,
        save_path=save_name,
        true_fault_name=display_true_fault_str,
        ai_diagnosis=final_diagnosis,
        ai_action=final_action,
        spike_times=spike_filtered_times
    )
    # 绘制增强版诊断 FTC 响应图
    enhanced_save_name = save_name.replace(".png", "_Enhanced_Diagnosis.png")

    plot_ftc_diagnosis_response(
        logs=simulator.sensor_logs,
        fault_time=actual_fault_time,
        ai_intervention_time=actual_ai_time,
        save_path=enhanced_save_name,
        true_fault_name=display_true_fault_str,
        ai_diagnosis=final_diagnosis,
        ai_action=final_action,
        spike_times=spike_filtered_times
    )
    # Mode 1 演示模式专属输出
    if is_demo:
        static_traj_name = f"results/Trajectory_3D_Demo_{true_fault_str}_{route_profile}_{time_str}.png"
        print(f" Generating static 3D trajectory map...")

        visualize_trajectory(
            trajectory=np.array(simulator.trajectory),
            visited_waypoints=planned_waypoints,
            destination=auv.destination,
            sensor_logs=simulator.sensor_logs,
            fault_time=actual_fault_time,
            true_fault_name=display_true_fault_str,
            ai_time=actual_ai_time,
            ai_diagnosis=final_diagnosis,
            save_path=static_traj_name,
            show=False
        )

        print(" Generating 3D animation for USV collaborative mission...")

        animation_save_name = f"results/Animation_3D_Demo_{true_fault_str}_{route_profile}_{time_str}.mp4"

        animate_trajectory(
            trajectory=np.array(simulator.trajectory),
            waypoints=planned_waypoints,
            destination=auv.destination,
            sensor_logs=simulator.sensor_logs,
            dt=0.1,
            playback_speed=20,
            save_path=animation_save_name,
            show=False
        )

        print(f"3D animation saved to: {animation_save_name}")

# ======================
# 训练数据集生成
# ======================
def generate_dataset(num_missions=1000):
    print("Generating dataset (Stage 3, 9-class multi-sensor mode: 40-D input)...")
    SEQ_LEN = 50
    all_X = []
    all_y = []
    all_mission_ids = []
    training_fault_types = list(FaultType)

    if num_missions < len(training_fault_types):
        raise ValueError(
            f"At least {len(training_fault_types)} missions are required so every "
            "class from label 0 through 8 is generated."
        )

    for mission in range(num_missions):
        if (mission + 1) % 10 == 0:
            print(f"Progress: {mission + 1}/{num_missions}")

        route_profile = random.choice(TRAINING_ROUTE_PROFILES)
        auv = create_rich_training_auv(route_profile=route_profile)
        depth_sensor = DepthSensor()
        # Round-robin assignment guarantees that label 8 and every other class
        # enter the dataset instead of relying on random fault selection.
        target_fault_type = training_fault_types[mission % len(training_fault_types)]
        fault_injector = create_fault(
            target_fault_type=target_fault_type,
            is_training=True
        )
        imu_sensor = IMUSensor()
        dvl_sensor = DVLSensor()
        current_sensor = CurrentSensor()
        battery_sensor = BatterySensor()

        simulator = Simulator(
            auv_model=auv,
            depth_sensor=depth_sensor,
            fault_injector=fault_injector,
            imu_sensor=imu_sensor,
            dvl_sensor=dvl_sensor,
            current_sensor=current_sensor,
            battery_sensor=battery_sensor
        )

        def controller(sensor_data):
            final_goal_z = sensor_data.get("target_z", 0.0)
            if not hasattr(controller, "dynamic_setpoint"):
                controller.dynamic_setpoint = sensor_data["position"][2]

            MAX_Z_STEP_SPEED = 1.2
            DT = 0.1
            max_step = MAX_Z_STEP_SPEED * DT

            diff_to_goal = final_goal_z - controller.dynamic_setpoint
            if abs(diff_to_goal) > max_step:
                controller.dynamic_setpoint += np.sign(diff_to_goal) * max_step
            else:
                controller.dynamic_setpoint = final_goal_z

            sensor_data["target_z"] = controller.dynamic_setpoint
            return simple_controller(sensor_data, auv)

        sim_duration = fault_injector.start_time + 100.0
        simulator.run_mission(duration=sim_duration, control_function=controller, dt=0.1)

        X, y = build_sequences(
            sensor_logs=simulator.sensor_logs,
            seq_len=SEQ_LEN
        )

        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)

            mission_id_array = np.full(len(y), mission, dtype=np.int64)
            all_mission_ids.append(mission_id_array)

    X_raw = np.concatenate(all_X)
    y_final = np.concatenate(all_y)
    mission_ids_final = np.concatenate(all_mission_ids)
    expected_labels = set(range(9))
    present_labels = set(np.unique(y_final).tolist())
    missing_labels = sorted(expected_labels - present_labels)
    if missing_labels:
        raise ValueError(
            f"Stage 3 dataset is missing labels {missing_labels}. "
            f"Present labels: {sorted(present_labels)}"
        )
    print(f"Verified dataset labels: {sorted(present_labels)}")
    idx_normal = np.where(y_final == 0)[0]
    idx_faults = np.where(y_final != 0)[0]
    print(f" Data volume before balancing -> Normal: {len(idx_normal)}, Faults: {len(idx_faults)}")

    np.random.shuffle(idx_normal)
    max_normal_count = int(len(idx_faults) * 1.5)
    idx_normal_kept = idx_normal[:max_normal_count]

    final_indices = np.concatenate([idx_normal_kept, idx_faults])
    np.random.shuffle(final_indices)

    X_raw = X_raw[final_indices]
    y_final = y_final[final_indices]
    mission_ids_final = mission_ids_final[final_indices]
    print(f" Balanced data volume -> Normal: {np.sum(y_final == 0)}, Faults: {np.sum(y_final != 0)}")

    from src.utils.dataset_builder import preprocess_dataset
    X_processed, stats = preprocess_dataset(X_raw)

    save_mean = np.array(stats['mean'], dtype=np.float32).flatten()
    save_std = np.array(stats['std'], dtype=np.float32).flatten()

    if save_mean.size != MODEL_INPUT_DIM:
        raise ValueError(
            f"CRITICAL ERROR: Mean size is {save_mean.size}, expected {MODEL_INPUT_DIM}!"
        )

    res_dir = Path(
        r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model\depth_fault_detection\results")
    res_dir.mkdir(parents=True, exist_ok=True)

    np.save(res_dir / "mean_stage3_9class.npy", save_mean)
    np.save(res_dir / "std_stage3_9class.npy", save_std)

    data_dir = Path(
        r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model\depth_fault_detection\data")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Mission IDs shape: {mission_ids_final.shape}")
    print(f"Unique missions saved: {len(np.unique(mission_ids_final))}")
    save_path = data_dir / "simulation_dataset_stage3_9class.pth"

    torch.save({
        "X": torch.tensor(X_processed, dtype=torch.float32),
        "y": torch.tensor(y_final, dtype=torch.long),
        "mission_ids": torch.tensor(mission_ids_final, dtype=torch.long),

        "feature_names": RAW_FEATURE_NAMES,
        "raw_feature_dim": RAW_FEATURE_DIM,
        "model_input_dim": MODEL_INPUT_DIM,
        "num_classes": 9,
        "label_names": {i: name for i, name in FAULT_NAMES.items()},
    }, save_path)

    print(f"Success! Dataset shape: {X_processed.shape}")
    print(f"Raw feature dimension: {RAW_FEATURE_DIM}")
    print(f"Model input dimension after flatten: {MODEL_INPUT_DIM}")
    print(f"Mean shape before flatten: {np.array(stats['mean']).shape}")
    print(f"Std shape before flatten: {np.array(stats['std']).shape}")
    print(f"DEBUG: Saved Stage 3 Means ({MODEL_INPUT_DIM} dims): {save_mean}")
    print(f"Dataset saved to: {save_path}")


# ======================
# 长航线故障时间测试
# ======================
def run_long_mission_timing_evaluation():
    """Run long-mission timing tests with fixed fault start times.

    Recommended workflow:
        1. First run NO_FAULT on the comprehensive route to verify no false alarm.
        2. Then enable the difficult sensor faults.
        3. Finally use test_times = [80, 300, 600, 900] for full evaluation.
    """
    print("\n Starting Channel 5: Long mission fault timing evaluation.")

    route_profile = "comprehensive"
    duration = 1200

    # First safe default:
    #   test_times = [300]
    # Full evaluation:
    #   test_times = [80, 300, 600, 900]
    test_times = [730]

    # Start with NO_FAULT enabled for route robustness checking.
    # Uncomment more faults after NO_FAULT passes.
    test_faults = [
        FaultType.NO_FAULT,
        FaultType.BIAS,
        FaultType.DRIFT,
        FaultType.STUCK,
        FaultType.SPIKE,
        FaultType.NOISE_INCREASE,
        FaultType.THRUSTER_ENTANGLED,
        FaultType.THRUSTER_NO_OUTPUT,
        FaultType.THRUSTER_THRUST_LOSS,
    ]

    print(f" Route profile: {route_profile}")
    print(f" Mission duration: {duration}s")
    print(f" Test times: {test_times}")
    print(f" Test faults: {[f.name for f in test_faults]}")

    for fault_time in test_times:
        print("\n" + "=" * 70)
        print(f" Long mission test group: fault_start_time = {fault_time}s")
        print("=" * 70)

        for f_type in test_faults:
            print(f"\n Testing [{f_type.name}] at t = {fault_time}s on route [{route_profile}]")

            execute_mission(
                fault_type=f_type,
                duration_override=duration,
                fault_start_time=fault_time,
                is_demo=False,
                route_profile=route_profile
            )

            print(f" Finished [{f_type.name}] at t = {fault_time}s on route [{route_profile}]")

    print("\n Long mission timing evaluation completed.")


# ======================
# 控制台多模式启动菜单
# ======================
if __name__ == "__main__":
    print("=" * 60)
    print(" Welcome to the AUV Fault-Tolerant Control (FTC) Simulation Framework ")
    print(" [1] Random fault simulation (with 3D animation)")
    print(" [2] Batch evaluation mode (traverses all fault types)")
    print(" [3] Generate training dataset (automatically build simulation data)")
    print(" [4] Generate Baseline Trajectory (Perfect NO_FAULT case)")
    print(" [5] Long mission fault timing evaluation")
    print("=" * 60)

    mode_choice = input("Please enter your choice (1, 2, 3, 4, or 5): ").strip()

    if mode_choice == '2':
        print("\n Starting Channel 2: Batch evaluation mode.")
        demo_faults = list(FaultType)
        for f_type in demo_faults:
            print(f"\n Testing scenario: [{f_type.name}]")
            execute_mission(fault_type=f_type, duration_override=180, route_profile="standard")
            print(f" {f_type.name} materials have been saved to the results folder.")
        print("\n All reports generated successfully!")

    elif mode_choice == '3':
        print("\n Starting Channel 3: Dataset generation.")
        try:
            num = input("Enter number of missions (default 1000): ").strip()
            n_missions = int(num) if num else 1000
        except ValueError:
            n_missions = 1000
        generate_dataset(num_missions=n_missions)
        print("\n Dataset generation complete!")

    elif mode_choice == '4':
        print("\n Starting Channel 4: Generating Baseline Trajectory...")
        execute_mission(fault_type=FaultType.NO_FAULT, duration_override=800, is_demo=True, route_profile="standard")
        print("\n Baseline trajectory saved successfully!")

    elif mode_choice == '5':
        run_long_mission_timing_evaluation()


    else:
        print("\n Starting Channel 1: Random fault simulation on comprehensive route...")

        # Mode 1 demo:
        #   - Keep random fault type for demonstration.
        #   - Use the same comprehensive route as Mode 5.
        #   - Randomize the fault injection time within the stable long-mission window.
        # This makes Mode 1 visually consistent with the final FTC validation,
        # while still preserving its original random-demo behavior.
        demo_route_profile = "comprehensive"
        demo_duration = 1200
        demo_fault_type = FaultType.THRUSTER_ENTANGLED  # Random fault type will be selected automatically in the controller.
        demo_fault_start_time = random.uniform(300.0, 900.0)

        print(f" Demo route profile: {demo_route_profile}")
        print(f" Demo duration: {demo_duration}s")
        print(f" Random fault start time: {demo_fault_start_time:.1f}s")
        print(" Random fault type will be selected automatically.\n")

        execute_mission(
            fault_type=demo_fault_type,
            duration_override=demo_duration,
            fault_start_time=demo_fault_start_time,
            is_demo=True,
            route_profile=demo_route_profile
        )
