import numpy as np


def simple_controller(sensor_data, auv):
    """
    升级版 PD 控制器：完美契合 0.1s 步长与 50kg 物理模型，新增航点追踪逻辑
    """
    dt = 0.1  # 确保与仿真步长严格一致
    current_pos = sensor_data["position"]

    # ==========================================
    # 🌟 修复 1：航点追踪与自动切换 (Waypoint Popping)
    # ==========================================
    # 1. 确定当前的 X/Y 追踪目标
    if hasattr(auv, 'waypoints') and len(auv.waypoints) > 0:
        current_target = auv.waypoints[0]

        # 检查是否到达当前航点 (三维距离阈值设为 2.0 米)
        dist_to_wp = np.linalg.norm(current_target - current_pos)
        if dist_to_wp < 4.0:
            auv.mark_waypoint_as_visited(current_target)
            auv.waypoints.pop(0)  # 到达后弹出该点

            # 更新为下一个点
            if len(auv.waypoints) > 0:
                current_target = auv.waypoints[0]
            else:
                current_target = auv.destination

            print(f"[{sensor_data.get('time', 0):.1f}s] Waypoint reached! Proceed to the next destination. Remaining waypoints: {len(auv.waypoints)}")
    else:
        current_target = auv.destination

    # ==========================================
    # 🌟 修复 2：正确构造 3D 目标点，防止 X 轴抢占动力
    # ==========================================
    # Z 轴继续使用外层传入的动态平滑目标，X 和 Y 使用当前航点的坐标
    target_z = sensor_data.get("target_z", current_target[2])
    target_pos = np.array([current_target[0], current_target[1], target_z])

    # 1. 计算三轴误差 (Error)
    error = target_pos - current_pos

    # ==========================================
    # 🌟 核心升级 1：分轴增益设置 (PID 参数)
    # ==========================================
    kp = np.array([0.8, 0.8, 2.5])
    ki = np.array([0.01, 0.01, 0.08])
    kd = np.array([0.0, 0.0, 0.3])

    # ==========================================
    # 🌟 核心升级 2 & 3：微积分计算
    # ==========================================
    if not hasattr(auv, 'last_error'):
        auv.last_error = np.zeros(3)
    error_derivative = (error - auv.last_error) / dt
    auv.last_error = error.copy()

    # ==========================================
    # 🌟 核心升级 3：积分项计算 (Integral) & 防积分饱和 (Anti-Windup)
    # ==========================================
    if not hasattr(auv, 'error_integral'):
        auv.error_integral = np.zeros(3)

    # 1. 允许误差随时累加（去掉 10m 的距离限制，让负误差能立刻消解正积分）
    auv.error_integral += error * dt

    # 2. 🌟 绝对的杀手锏：积分限幅钳位 (Anti-Windup Clamp)
    # 限制积分项的最大能量储备，防止长途跋涉时“油门被焊死”
    auv.error_integral = np.clip(auv.error_integral, -15.0, 15.0)

    # ==========================================
    # 🌟 核心升级 4：物理前馈补偿 (Buoyancy Offset)
    # ==========================================
    # 浮力向上为负，补偿向下为正
    buoyancy_offset = np.array([0.0, 0.0, 0.0017])

    # 2. 计算总指令
    command_velocity = (error * kp) + (auv.error_integral * ki) + (error_derivative * kd) + buoyancy_offset

    # ==========================================
    # 🌟 核心升级 5：动态限速
    # ==========================================
    max_allowable_speed = 2.0
    actual_speed = np.linalg.norm(command_velocity)

    if actual_speed > max_allowable_speed:
        command_velocity = (command_velocity / actual_speed) * max_allowable_speed

    # 3. 偏航角与停止逻辑
    direction = target_pos - current_pos
    target_yaw = np.arctan2(direction[1], direction[0])

    final_distance = np.linalg.norm(auv.destination - current_pos)
    if final_distance < 0.1:
        imu_data = sensor_data.get("imu", {})

        if "orientation" in imu_data:
            current_yaw = imu_data["orientation"][2]
        else:
            current_yaw = imu_data.get("yaw", 0.0)

        return np.zeros(3), current_yaw

    return command_velocity, target_yaw