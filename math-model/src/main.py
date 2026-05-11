import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from my_utils import plot_ftc_response
import sys
from pathlib import Path
import random
import torch
from model1 import AUVFaultDetector  # 导入模型类

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
# 核心任务调度器 (Mission Executor)
# ======================
def execute_mission(fault_type=None, is_demo=False, duration_override=None):
    auv = create_auv()
    depth_sensor = DepthSensor()
    fault_injector = create_fault(target_fault_type=fault_type)

    simulator = Simulator(
        auv_model=auv,
        depth_sensor=depth_sensor,
        fault_injector=fault_injector
    )

    try:
        mean = np.load(
            r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model\depth_fault_detection\results\mean.npy")
        std = np.load(
            r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model\depth_fault_detection\results\std.npy")
    except FileNotFoundError:
        mean, std = 0, 1

    controller_buffer = []

    # 系统状态标志
    system_locked = False
    is_safe_mode = False
    is_filtering = False  # 滤波器开关
    last_clean_depth = 0.0  # 存储上一帧干净的数据

    record_time = [-1.0]
    final_diagnosis = "NO_FAULT"
    final_action = "Normal Cruising"

    # 安全指令缓存与时间记录
    safe_cmd_vel = np.array([0.0, 0.0, 0.0])
    safe_cmd_yaw = 0.0
    spike_filtered_times = []

    def ftc_controller(sensor_data):
        nonlocal controller_buffer
        nonlocal system_locked, is_safe_mode
        nonlocal final_diagnosis, final_action
        nonlocal safe_cmd_vel, safe_cmd_yaw
        nonlocal is_filtering, last_clean_depth

        # 初始化状态机的持久化变量
        if not hasattr(ftc_controller, "dynamic_setpoint"):
            ftc_controller.dynamic_setpoint = sensor_data["depth"]
        if not hasattr(ftc_controller, "fault_counter"):
            ftc_controller.fault_counter = 0
            ftc_controller.current_pred = 0

        def finalize_return(cmd_vel, cmd_yaw):
            sensor_data["ftc_diagnosis"] = final_diagnosis
            sensor_data["ftc_action"] = final_action
            sensor_data["ftc_is_locked"] = system_locked
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

        # ========================================================
        # 模块 2：最高优先级拦截 (Absolute Deadlock)
        # ========================================================
        if system_locked:
            if "USV Winch" in final_action:
                if sensor_data["depth"] < 1.0: return finalize_return(np.array([0.0, 0.0, 0.0]), 0.0)
                return finalize_return(np.array([0.0, 0.0, -2.0]), safe_cmd_yaw)
            elif "Hover" in final_action:
                return finalize_return(np.array([0.0, 0.0, 0.0]), safe_cmd_yaw)
            elif "Emergency" in final_action:
                if sensor_data["depth"] < 1.0: return finalize_return(np.array([0.0, 0.0, 0.0]), 0.0)
                return finalize_return(np.array([0.0, 0.0, -2.0]), safe_cmd_yaw)
            elif "Slow Surface" in final_action:
                if sensor_data["depth"] < 1.0: return finalize_return(np.array([0.0, 0.0, 0.0]), 0.0)
                return finalize_return(np.array([0.0, 0.0, -0.5]), safe_cmd_yaw)
            elif "Power Cut" in final_action:
                return finalize_return(np.array([0.0, 0.0, 0.0]), 0.0)

        # ========================================================
        # 模块 3：AI 预测与物理护城河 (AI Inference & Physical Moat)
        # ========================================================
        f_depth = smoothed_depth if is_filtering else sensor_data["depth"]
        f_target = ftc_controller.dynamic_setpoint
        f_error = f_target - f_depth
        pred = 0

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
            if 40 <= sensor_data['time'] <= 70:
                state_str = "CRUISE" if is_cruising else "HOVER"
                print(
                    f"Time: {sensor_data['time']:.1f}s | Pred: {pred} | Cmd: {normal_cmd_vel[2]:.4f} | Error: {tracking_error:.4f} | State: {state_str}")

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
        # 模块 4：双轨防抖系统 (The "Two-Lane" Debouncer)
        # ========================================================
        # 轨道 A：瞬态故障 (SPIKE)
        if pred == 4:
            print(f"[{sensor_data['time']:.1f}s] SPIKE! Filtered.")
            spike_filtered_times.append(sensor_data['time'])
            final_diagnosis = "SPIKE"
            final_action = "Filtered"
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
        threshold_map = {1: 3, 2: 5, 3: 6, 5: 10, 6: 10, 7: 10}
        confirm_threshold = threshold_map.get(ftc_controller.current_pred, 999)

        # ========================================================
        # 模块 5：最终执行决断
        # ========================================================
        if ftc_controller.fault_counter >= confirm_threshold:
            confirmed_fault = ftc_controller.current_pred

            if record_time[0] < 0:
                record_time[0] = sensor_data['time']

            if confirmed_fault == 5 and not is_filtering:
                print(
                    f"[{sensor_data['time']:.1f}s] NOISE_INCREASE! Starting self-healing: Activating adaptive smoothing filter!")
                final_diagnosis = "NOISE"
                final_action = "Adaptive Filtering"
                is_filtering = True

            elif confirmed_fault == 1 and not is_filtering:
                print(f"[{sensor_data['time']:.1f}s] BIAS! Hover Locked.")
                final_diagnosis, final_action = "BIAS", "Hover Locked"
                system_locked = True
                return finalize_return(np.array([0.0, 0.0, 0.0]), safe_cmd_yaw)

            elif confirmed_fault == 2 and not is_filtering:
                print(f"[{sensor_data['time']:.1f}s] DRIFT! Abort.")
                final_diagnosis, final_action = "DRIFT", "Abort & Slow Surface"
                system_locked = True
                return finalize_return(np.array([0.0, 0.0, -0.5]), safe_cmd_yaw)

            elif confirmed_fault == 3 and not is_filtering:
                print(f"[{sensor_data['time']:.1f}s] STUCK! USV Winch Recovery")
                final_diagnosis, final_action = "STUCK", "USV Winch Recovery"
                system_locked = True
                return finalize_return(np.array([0.0, 0.0, -2.0]), safe_cmd_yaw)

            elif confirmed_fault == 6:
                print(f"[{sensor_data['time']:.1f}s] ENTANGLED! Power Cut & USV Winch Recovery")
                final_diagnosis, final_action = "ENTANGLED", "Power Cut & USV Winch Recovery"
                system_locked = True
                return finalize_return(np.array([0.0, 0.0, -2.0]), safe_cmd_yaw)

            elif confirmed_fault == 7:
                print(f"[{sensor_data['time']:.1f}s] BROKEN! Power Cut & USV Winch Recovery")
                final_diagnosis, final_action = "BROKEN", "Power Cut & USV Winch Recovery"
                system_locked = True
                return finalize_return(np.array([0.0, 0.0, -2.0]), safe_cmd_yaw)

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

        simulator = Simulator(
            auv_model=auv,
            depth_sensor=depth_sensor,
            fault_injector=fault_injector
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