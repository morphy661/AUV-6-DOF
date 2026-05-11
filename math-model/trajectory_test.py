import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import cast

# ==========================================
# 1. 复用之前的逻辑：生成光滑的位置和切线
# ==========================================
def get_path_geometry(s):
    # 一个简单的螺旋上升轨迹
    x = 5.0 * np.cos(0.2 * s)
    y = 5.0 * np.sin(0.2 * s)
    z = 0.2 * s
    return np.array([x, y, z])

def generate_trajectory_with_wobble():
    # --- A. 时间与路径参数化 ---
    T = 10.0
    dt = 0.01 # 100Hz
    times = np.arange(0, T, dt)
    
    # 简单的匀加速运动 s = 0.5 * a * t^2 (为了演示简单点，你可以换回之前的B样条速度)
    s_values = 0.5 * 0.5 * times**2 
    
    # 计算位置
    positions = np.array([get_path_geometry(s) for s in s_values])
    
    # --- B. 计算“名义姿态” (看着前方) ---
    # 计算切线向量 (Tangent) = 速度方向
    tangents = np.gradient(positions, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None] # 归一化
    
    # 我们需要构建一个旋转矩阵，让机器人的 X轴(假设为前进轴) 指向切线
    # 这里使用 Gram-Schmidt 正交化或者简单的 LookAt 方法
    # 假设 Z轴大致向上 (0, 0, 1)
    world_up = np.array([0, 0, 1])
    
    quats = []
    
    for i, t_vec in enumerate(tangents):
        # 1. X轴 = 切线方向
        x_axis = t_vec
        
        # 2. Y轴 = Z_world cross X (右手法则)
        y_axis = np.cross(world_up, x_axis)
        if np.linalg.norm(y_axis) < 1e-6: # 防止万一垂直向上
            y_axis = np.array([0, 1, 0])
        y_axis /= np.linalg.norm(y_axis)
        
        # 3. Z轴 = X cross Y (重新计算正交的Z)
        z_axis = np.cross(x_axis, y_axis)
        
        # 构建旋转矩阵 [x, y, z] (列向量)
        rot_mat = np.column_stack((x_axis, y_axis, z_axis))
        
        # 转换为 Scipy 的 Rotation 对象 (名义姿态)
        r_nominal = R.from_matrix(rot_mat)
        
        # --- C. 添加“轴向摆动” (核心步骤) ---
        # 定义摆动规则：正弦波
        # t = times[i]
        t_val = times[i]
        
        
        fade_in_duration = 2.0
        if t_val < fade_in_duration:
            # 使用 (1 - cos) 形状的平滑过渡，保证导数也是连续的
            w = (t_val / fade_in_duration) * np.pi
            fade_in = 0.5 * (1 - np.cos(w)) # 范围 0 -> 1
        else:
            fade_in = 1.0

        # 原来的摆动



        # Roll 摆动 (绕前进轴晃动)：频率 2Hz，幅度 15度
        wobble_roll = 15.0 * np.sin(2 * np.pi * 2.0 * t_val)
        # 保证摆动从0开始逐渐增大
        wobble_roll = fade_in * wobble_roll 
        # Pitch 摆动 (上下点头)：频率 1Hz，幅度 5度
        wobble_pitch = 5.0 * np.cos(2 * np.pi * 1.0 * t_val)
        # 保证摆动从0开始逐渐增大
        wobble_pitch = fade_in * wobble_pitch
        # Yaw 摆动 (左右摇头)：通常不需要太多，除非模拟水流干扰
        wobble_yaw = 0.0 
        
        # 将欧拉角转换为旋转对象 (注意是 'xyz' 顺序，且是内旋)
        # 这代表是在“当前机体坐标系”下的扰动
        r_wobble = R.from_euler('xyz', [wobble_roll, wobble_pitch, wobble_yaw], degrees=True)
        
        # 组合旋转：先有名义姿态，再叠加局部摆动
        # 注意乘法顺序：r_nominal * r_wobble 代表在 global frame 后再动 local
        r_final = r_nominal * r_wobble
        
        quats.append(r_final.as_quat())

    quats = np.array(quats)
    
    return times, positions, quats

# --- D. 后处理：计算角速度和角加速度 ---
# 有了连续的四元数，我们需要数值差分求角速度
def compute_kinematics(times, quats):
    dt = times[1] - times[0]
    rotations = R.from_quat(quats)
    
    # 1. 计算角速度 (Angular Velocity)
    # R_dot = R * skew(omega) -> omega 可以通过两个时刻的相对旋转算出
    omegas = []
    for i in range(len(times)-1):
        # R_diff = R_next * R_curr^T
        r_diff = rotations[i+1] * rotations[i].inv()
        # 把旋转差转为轴角 (Axis-Angle)
        rot_vec = r_diff.as_rotvec()
        # 角速度 = 旋转向量 / dt
        omegas.append(rot_vec / dt)
    omegas.append(omegas[-1]) # 补齐最后一个点
    omegas = np.array(omegas)
    
    # 2. 计算角加速度 (Angular Acceleration)
    # 直接对角速度差分
    alphas = np.gradient(omegas, dt, axis=0)
    
    return omegas, alphas

# 运行
times, pos, quats = generate_trajectory_with_wobble()
omegas, alphas = compute_kinematics(times, quats)

def plot_trajectory_3d(positions, quaternions, sample_step=50):
    fig = plt.figure(figsize=(10, 7))
    ax = cast(Axes3D, fig.add_subplot(111, projection='3d'))
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], lw=1.5, label='Reference path')

    rotations = R.from_quat(quaternions)
    scale = 0.6
    sample_indices = np.arange(0, len(positions), sample_step)
    for idx in sample_indices:
        origin = positions[idx]
        mat = rotations[idx].as_matrix()
        # 绘制机体坐标系轴 (X=red, Y=green, Z=blue)
        for axis_idx, color in enumerate(('r', 'g', 'b')):
            axis_vector = mat[:, axis_idx] * scale
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                axis_vector[0],
                axis_vector[1],
                axis_vector[2],
                color=color,
            )

    ax.set_title('3D Trajectory and Body Axes')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=20, azim=120)


# 绘图验证
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(times, omegas)
plt.title("Angular Velocity (with Wobble)")
plt.legend(['x', 'y', 'z'])
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(times, alphas)
plt.title("Angular Acceleration (True Value for Dual IMU)")
plt.ylabel("rad/s^2")
plt.legend(['x', 'y', 'z'])
plt.grid(True)

plt.tight_layout()

plot_trajectory_3d(pos, quats)

plt.show()