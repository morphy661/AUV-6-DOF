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

# =======================================================
# 加载 AI 大脑 (适配最新的 14维 PINN 架构)
# =======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Starting system...")

# 1. 按照新版模型参数进行实例化
model = AUVFaultDetector(
    input_dim=14,  # 输入维度为 14
    seq_len=50,
    num_classes=8
).to(DEVICE)

try:
    model_path = r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model\depth_fault_detection\results\best_model.pth"

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print("Successfully loaded the highest precision model weights (best_model.pth)!")
except FileNotFoundError:
    print("Warning: best_model.pth still not found!")

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
def create_auv():
    config = AUVConfig(
        mass=50.0, length=1.5, width=0.5, height=0.5,
        max_velocity_x=3.0, max_velocity_y=3.0, max_velocity_z=3.0,
        max_angular_velocity=0.5, max_acceleration_x=0.5,
        max_acceleration_y=0.3, max_acceleration_z=0.3,
        max_angular_acceleration=0.2, battery_capacity=500.0,
        power_consumption_idle=10.0, power_consumption_per_velocity=5.0
    )
    auv = AUVModel(config)
    waypoints = [
        [0, 0, 0], [0, 0, 60], [20, 20, 60], [40, 20, 30],
        [80, 40, 500], [120, 60, 500], [80, 40, 300], [40, 20, 100], [0, 0, 0]
    ]
    auv.set_waypoints(waypoints)
    auv.set_destination(350, 0, 100)
    return auv


def create_rich_training_auv():
    config = AUVConfig(
        mass=50.0, length=1.5, width=0.5, height=0.5,
        max_velocity_x=2.0, max_velocity_y=2.0, max_velocity_z=2.0,
        max_angular_velocity=0.5, max_acceleration_x=0.5,
        max_acceleration_y=0.3, max_acceleration_z=0.3,
        max_angular_acceleration=0.2, battery_capacity=500.0,
        power_consumption_idle=10.0, power_consumption_per_velocity=5.0
    )
    auv = AUVModel(config)
    waypoints = [
        [0, 0, 0], [0, 0, 60], [20, 20, 60], [40, 20, 30],
        [80, 40, 500], [120, 60, 500], [80, 40, 300], [40, 20, 100], [0, 0, 0]
    ]
    auv.set_waypoints(waypoints)
    auv.set_destination(350, 0, 100)
    return auv


# ======================
# 创建随机故障注入器
# ======================
def create_fault(target_fault_type=None, is_training=False):
    if is_training:
        start_time = random.uniform(30.0, 750.0)
        target_fault_type = random.choice(list(FaultType))
    elif target_fault_type is None:
        target_fault_type = random.choice(list(FaultType))
        start_time = random.uniform(100, 200.0)
    else:
        start_time = random.uniform(40.0, 60.0)

    return SystemFaultInjector(
        fault_type=target_fault_type,
        start_time=start_time,
        # 1. 偏差故障：设置在 5.0 到 12.0 米之间
        bias=random.choice([-1, 1]) * random.uniform(5.0, 12.0),
        # 2. 脉冲故障：设置触发概率和幅值
        spike_prob=0.005,
        spike_magnitude=random.choice([-1, 1]) * random.uniform(8.0, 15.0),
        # 3. 噪声故障：设置噪声标准差
        noise_std=random.uniform(0.5, 1.0),
        # 4. 漂移故障：设置漂移率
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

    if fault_id in [6, 7]:
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
def execute_mission(fault_type=None, is_demo=False, duration_override=None):
    auv = create_auv()
    depth_sensor = DepthSensor()
    fault_injector = create_fault(target_fault_type=fault_type)

    imu_sensor = IMUSensor()
    dvl_sensor = DVLSensor()
    current_sensor = CurrentSensor()
    battery_sensor = BatterySensor()
    residual_observer = ResidualObserver()
    diagnosis_strategy = DiagnosisStrategy()

    simulator = Simulator(
        auv_model=auv,
        depth_sensor=depth_sensor,
        fault_injector=fault_injector,
        imu_sensor=imu_sensor,
        dvl_sensor=dvl_sensor,
        current_sensor=current_sensor,
        battery_sensor=battery_sensor
    )

    try:
        mean = np.load(
            r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model\depth_fault_detection\results\mean.npy")
        std = np.load(
            r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model\depth_fault_detection\results\std.npy")
    except FileNotFoundError:
        mean, std = 0, 1

    controller_buffer = [] #给 AI 模型用，保存 7 维特征。
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
    bias_confirm_delay = 10.0

    locked_fault_id = 0          # 用于锁定型故障：BIAS / DRIFT / STUCK / ENTANGLED / BROKEN
    soft_fault_id = 0            # 用于非锁定型持续故障：NOISE_INCREASE
    confirmed_reason = "No diagnosis has been triggered yet."
    # 安全指令缓存与时间记录
    safe_cmd_vel = np.array([0.0, 0.0, 0.0])
    safe_cmd_yaw = 0.0
    spike_filtered_times = []
    last_spike_time = -999.0
    spike_cooldown = 3.0
    def ftc_controller(sensor_data):
        nonlocal controller_buffer, diagnosis_history
        nonlocal system_locked, is_safe_mode
        nonlocal final_diagnosis, final_action, locked_fault_id, soft_fault_id, confirmed_reason
        nonlocal safe_cmd_vel, safe_cmd_yaw
        nonlocal is_filtering, last_clean_depth
        nonlocal last_spike_time
        nonlocal bias_candidate_start_time
        # 初始化状态机的持久化变量
        if not hasattr(ftc_controller, "dynamic_setpoint"):
            ftc_controller.dynamic_setpoint = sensor_data["depth"]
        if not hasattr(ftc_controller, "fault_counter"):
            ftc_controller.fault_counter = 0
            ftc_controller.current_pred = 0
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
        # 新增：Residual Observer
        # 使用当前控制器输出的垂向速度作为 cmd_vz
        # ========================================================
        sensor_data["thruster"]["cmd_vz"] = float(normal_cmd_vel[2])
        residuals = residual_observer.compute(sensor_data)
        sensor_data["residuals"] = residuals

        diagnosis_history.append(sensor_data.copy())
        if len(diagnosis_history) > 50:
            diagnosis_history.pop(0)

        # ========================================================
        # 模块 2：最高优先级拦截 (Absolute Deadlock)
        # ========================================================
        if system_locked:
            recovery_cmd_vel, recovery_cmd_yaw = get_recovery_command(
                action_text=final_action,
                current_depth=sensor_data["depth"],
                safe_cmd_yaw=safe_cmd_yaw
            )
            return finalize_return(recovery_cmd_vel, recovery_cmd_yaw)

        # ========================================================
        # 模块 3：AI 预测与物理护城河 (AI Inference & Physical Moat)
        # ========================================================
        f_depth = smoothed_depth if is_filtering else sensor_data["depth"]
        f_target = ftc_controller.dynamic_setpoint
        f_error = f_target - f_depth
        pred = 0

        # 默认诊断变量，防止窗口长度不足 50 时后续逻辑引用未定义变量
        ai_pred = 0
        rule_pred = 0
        diagnosis_source = "none"
        diagnosis_reason = confirmed_reason
        diagnosis_confidence = "Low"

        current_features = [
            f_depth, f_target, f_error,
            sensor_data["thruster"]["current"],
            sensor_data["thruster"]["actual_vz"],
            sensor_data["relative_pos"]["delta_x"],
            sensor_data["relative_pos"]["delta_y"]
        ]
        controller_buffer.append(current_features)
        if len(controller_buffer) > 50: controller_buffer.pop(0)

        if len(controller_buffer) == 50:
            raw_seq = np.array(controller_buffer)
            diff_seq = np.vstack([np.zeros((1, 7)), np.diff(raw_seq, axis=0)])
            input_seq_flat = np.stack((raw_seq, diff_seq), axis=-1).reshape(50, -1)

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

            # 【真理拦截】：巡航态下深度变化过小判定为卡死
            if is_cruising and abs(raw_seq[-1, 0] - raw_seq[-10, 0]) < 0.01:
                pred = 3

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
            if pred in [6, 7] and abs(cmd_z) < 0.6: pred = 0
            if pred in [6, 7] and (actual_vz * cmd_z > 0) and abs(actual_vz) > 0.1: pred = 0

            # 脉冲物理约束
            if pred == 4 and abs(raw_seq[-1, 0] - raw_seq[-2, 0]) < 1.0: pred = 0
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

            # 2. 如果 NOISE 已经确认并进入滤波状态，不允许后续局部 SPIKE 抢走最终诊断
            elif is_filtering and soft_fault_id == 5:
                pred = 5

            # 3. 如果规则诊断有明确物理证据，则优先使用规则结果
            elif diagnosis_source == "rule" and rule_pred != 0:
                pred = rule_pred

            # 4. 否则使用 AI 预测
            else:
                pred = ai_pred

            sensor_data["ai_pred"] = ai_pred
            sensor_data["rule_pred"] = rule_pred
            sensor_data["final_pred"] = pred
            sensor_data["diagnosis_reason"] = diagnosis_reason
            sensor_data["diagnosis_confidence"] = diagnosis_confidence
            sensor_data["diagnosis_source"] = diagnosis_source

            if 40 <= sensor_data["time"] <= 80:
                print(
                    f"Time: {sensor_data['time']:.1f}s | "
                    f"AI={ai_pred}, Rule={rule_pred}, Final={pred}, "
                    f"Source={diagnosis_source}"
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
                and diagnosis_source == "rule"
                and rule_pred == 4
        ):
            current_time = sensor_data["time"]

            # 防止同一个 spike 在历史窗口中被重复触发很多次
            if current_time - last_spike_time >= spike_cooldown:
                print(f"[{current_time:.1f}s] SPIKE! Filtered.")

                spike_filtered_times.append(current_time)
                last_spike_time = current_time

            final_diagnosis = "SPIKE"
            final_action = get_recovery_action(4)
            confirmed_reason = diagnosis_reason

            sensor_data["final_pred"] = 4
            sensor_data["diagnosis_reason"] = confirmed_reason

            return finalize_return(normal_cmd_vel, normal_cmd_yaw)
        # 轨道 B：稳态故障读条逻辑
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
        threshold_map = {
            1: 5,  # BIAS
            2: 6,  # DRIFT
            3: 6,  # STUCK
            5: 8,  # NOISE_INCREASE
            6: 8,  # THRUSTER_ENTANGLED
            7: 8,  # THRUSTER_BROKEN
        }
        confirm_threshold = threshold_map.get(ftc_controller.current_pred, 999)

        # ========================================================
        # 模块 5：最终执行决断
        # ========================================================
        if ftc_controller.fault_counter >= confirm_threshold:
            confirmed_fault = ftc_controller.current_pred

            confirmed_reason = sensor_data.get("diagnosis_reason", diagnosis_reason)

            # Reset BIAS candidate timer when the confirmed fault is not BIAS.
            # This prevents an old BIAS candidate from affecting later decisions.
            if confirmed_fault != 1:
                bias_candidate_start_time = None

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
                # BIAS is a soft sensor fault.
                # Do not hard-lock immediately, because early DRIFT can look like BIAS.
                if bias_candidate_start_time is None:
                    bias_candidate_start_time = sensor_data["time"]

                # Print only around whole seconds to avoid flooding the console.
                if abs(sensor_data["time"] - round(sensor_data["time"])) < 0.05:
                    print(
                        f"[{sensor_data['time']:.1f}s] BIAS candidate! "
                        f"Monitoring before lock."
                    )

                # Keep normal control during the confirmation delay.
                # This gives DRIFT time to develop a clear residual trend.
                if sensor_data["time"] - bias_candidate_start_time < bias_confirm_delay:
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
                    *get_recovery_command(final_action, sensor_data["depth"], safe_cmd_yaw)
                )

            elif confirmed_fault == 2 and not is_filtering:
                print(f"[{sensor_data['time']:.1f}s] DRIFT! Abort Mission + Controlled Emergency Ascent.")
                if record_time[0] < 0:
                    record_time[0] = sensor_data["time"]
                final_diagnosis = "DRIFT"
                final_action = get_recovery_action(2)
                locked_fault_id = confirmed_fault
                system_locked = True
                return finalize_return(*get_recovery_command(final_action, sensor_data["depth"], safe_cmd_yaw))

            elif confirmed_fault == 3 and not is_filtering:
                print(f"[{sensor_data['time']:.1f}s] STUCK! Emergency Buoyancy Ascent + Acoustic Beacon.")
                if record_time[0] < 0:
                    record_time[0] = sensor_data["time"]
                final_diagnosis = "STUCK"
                final_action = get_recovery_action(3)
                locked_fault_id = confirmed_fault
                system_locked = True
                return finalize_return(*get_recovery_command(final_action, sensor_data["depth"], safe_cmd_yaw))

            elif confirmed_fault == 6:
                print(f"[{sensor_data['time']:.1f}s] ENTANGLED! Power Cut + Emergency Buoyancy Ascent + Acoustic Beacon.")
                if record_time[0] < 0:
                    record_time[0] = sensor_data["time"]
                final_diagnosis, final_action = "ENTANGLED", get_recovery_action(6)
                locked_fault_id = confirmed_fault
                system_locked = True
                return finalize_return(*get_recovery_command(final_action, sensor_data["depth"], safe_cmd_yaw))

            elif confirmed_fault == 7:
                print(f"[{sensor_data['time']:.1f}s] BROKEN! Power Cut + Emergency Buoyancy Ascent + Acoustic Beacon.")
                if record_time[0] < 0:
                    record_time[0] = sensor_data["time"]
                final_diagnosis, final_action = "BROKEN", get_recovery_action(7)
                locked_fault_id = confirmed_fault
                system_locked = True
                return finalize_return(*get_recovery_command(final_action, sensor_data["depth"], safe_cmd_yaw))

        # 正常指令输出
        safe_cmd_vel = normal_cmd_vel * 0.5 if is_safe_mode else normal_cmd_vel
        return finalize_return(safe_cmd_vel, normal_cmd_yaw)

    # 设置任务时长
    if duration_override is not None:
        duration = duration_override
    else:
        duration = 1200 if is_demo else 1000

    true_fault_str = fault_injector.fault_type.name
    if is_demo:
        print(f"\n Current scenario fault type: [{true_fault_str}]")
        print(f" AI FTC monitoring system is online and ready...\n")

    simulator.run_mission(duration=duration, control_function=ftc_controller, dt=0.1)

    actual_fault_time = fault_injector.start_time
    actual_ai_time = record_time[0] if record_time[0] > 0 else None

    import time  # 导入 time 模块

    # 生成保存文件名
    if is_demo:
        time_str = time.strftime("%H%M%S")
        save_name = f"results/FTC_Response_Demo_{true_fault_str}_{time_str}.png"
    else:
        time_str = ""
        save_name = f"results/FTC_Response_{true_fault_str}.png"

    # 绘制 2D 容错响应图
    plot_ftc_response(
        logs=simulator.sensor_logs,
        fault_time=actual_fault_time,
        ai_intervention_time=actual_ai_time,
        save_path=save_name,
        true_fault_name=true_fault_str,
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
        true_fault_name=true_fault_str,
        ai_diagnosis=final_diagnosis,
        ai_action=final_action,
        spike_times=spike_filtered_times
    )
    # Mode 1 演示模式专属输出
    if is_demo:
        from utils.visualization import visualize_trajectory
        static_traj_name = f"results/Trajectory_3D_Demo_{true_fault_str}_{time_str}.png"
        print(f" Generating static 3D trajectory map...")

        visualize_trajectory(
            trajectory=np.array(simulator.trajectory),
            visited_waypoints=auv.waypoints if hasattr(auv, 'waypoints') else None,
            destination=auv.destination,
            sensor_logs=simulator.sensor_logs,
            fault_time=actual_fault_time,
            true_fault_name=true_fault_str,
            ai_time=actual_ai_time,
            ai_diagnosis=final_diagnosis,
            save_path=static_traj_name,
            show=False
        )

        from utils.visualization import animate_trajectory
        print(" Generating 3D animation for USV collaborative mission...")

        animate_trajectory(
            trajectory=np.array(simulator.trajectory),
            waypoints=auv.waypoints if hasattr(auv, 'waypoints') else None,
            destination=auv.destination,
            sensor_logs=simulator.sensor_logs,
            dt=0.1,
            playback_speed=10
        )


# ======================
# 训练数据集生成
# ======================
def generate_dataset(num_missions=1000):
    print("Generating dataset (14-Dimensional PINN System Mode)...")
    SEQ_LEN = 50
    all_X = []
    all_y = []

    for mission in range(num_missions):
        if (mission + 1) % 10 == 0:
            print(f"Progress: {mission + 1}/{num_missions}")

        auv = create_rich_training_auv()
        depth_sensor = DepthSensor()
        fault_injector = create_fault(is_training=True)
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

    X_raw = np.concatenate(all_X)
    y_final = np.concatenate(all_y)

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
    print(f" Balanced data volume -> Normal: {np.sum(y_final == 0)}, Faults: {np.sum(y_final != 0)}")

    from src.utils.dataset_builder import preprocess_dataset
    X_processed, stats = preprocess_dataset(X_raw)

    save_mean = np.array(stats['mean']).flatten()
    save_std = np.array(stats['std']).flatten()

    if save_mean.size != 14:
        raise ValueError(f"CRITICAL ERROR: Mean size is {save_mean.size}, expected 14!")

    res_dir = Path(
        r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model\depth_fault_detection\results")
    res_dir.mkdir(parents=True, exist_ok=True)

    np.save(res_dir / "mean.npy", save_mean)
    np.save(res_dir / "std.npy", save_std)

    save_path = r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model\depth_fault_detection\data\simulation_dataset.pth"
    torch.save({
        "X": torch.tensor(X_processed, dtype=torch.float32),
        "y": torch.tensor(y_final, dtype=torch.long)
    }, save_path)

    print(f"Success! Dataset shape: {X_processed.shape}")
    print(f"DEBUG: Saved Means (14 dims): {save_mean}")


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
    print("=" * 60)

    mode_choice = input("Please enter your choice (1, 2, 3, or 4): ").strip()

    if mode_choice == '2':
        print("\n Starting Channel 2: Batch evaluation mode.")
        demo_faults = list(FaultType)
        for f_type in demo_faults:
            print(f"\n Testing scenario: [{f_type.name}]")
            execute_mission(fault_type=f_type, duration_override=180)
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
        execute_mission(fault_type=FaultType.NO_FAULT, duration_override=1000, is_demo=True)
        print("\n Baseline trajectory saved successfully!")

    else:
        print("\n Starting Channel 1: Random fault simulation...")
        execute_mission(fault_type=None, is_demo=True)