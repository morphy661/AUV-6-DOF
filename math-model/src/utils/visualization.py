import numpy as np
from matplotlib.animation import FuncAnimation
import os
import matplotlib.pyplot as plt

# 核心：手动指定 ffmpeg 的绝对路径
plt.rcParams[
    'animation.ffmpeg_path'] = r"C:\Users\Administrator\PycharmProjects\AUV Depth Sensor Fault Detection Model\depth_fault_detection\ffmpeg.exe"


def visualize_trajectory(trajectory: np.ndarray, visited_waypoints: np.ndarray = None,
                         destination: np.ndarray = None,
                         sensor_logs: list = None,
                         fault_time: float = None,
                         true_fault_name: str = None,
                         ai_time: float = None,
                         ai_diagnosis: str = None,
                         save_path=None, show=True):
    """
    可视化 AUV 的三维轨迹，并标注故障点与 AI 介入点，支持后台静默保存。
    """
    if len(trajectory) == 0:
        print("No trajectory data to visualize")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 画基础轨迹
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=2, alpha=0.8, label='Flight Path')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c='g', marker='o', s=100, label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c='r', marker='o', s=100, label='End')

    # 2. 画航点和终点
    if visited_waypoints is not None:
        for i, waypoint in enumerate(visited_waypoints, start=1):
            ax.scatter(waypoint[0], waypoint[1], waypoint[2], c='y', marker='x', s=150)
            ax.text(waypoint[0], waypoint[1], waypoint[2], f' WP {i}', color='black', fontsize=9)

    if destination is not None:
        ax.scatter(destination[0], destination[1], destination[2], c='m', marker='*', s=250, label='Destination')

    # ==========================================
    # 🌟 3. 寻找并标记“案发现场”
    # ==========================================
    fault_idx, ai_idx = None, None
    if sensor_logs is not None:
        for i, log in enumerate(sensor_logs):
            if fault_time is not None and log['time'] >= fault_time and fault_idx is None:
                fault_idx = i
            if ai_time is not None and log['time'] >= ai_time and ai_idx is None:
                ai_idx = i

    if fault_idx is not None and fault_idx < len(trajectory) and true_fault_name != "NO_FAULT":
        fx, fy, fz = trajectory[fault_idx]
        ax.scatter(fx, fy, fz, c='darkorange', marker='X', s=200, label='Fault Injected')
        ax.text(fx, fy, fz + 20, f' {true_fault_name}\n@{fault_time:.1f}s', color='darkorange', fontweight='bold')

    if ai_idx is not None and ai_idx < len(trajectory) and ai_diagnosis not in [None, "NO_FAULT"]:
        ai_x, ai_y, ai_z = trajectory[ai_idx]
        ax.scatter(ai_x, ai_y, ai_z, c='red', marker='P', s=200, label='AI Intervention')
        ax.text(ai_x, ai_y, ai_z - 30, f' AI: {ai_diagnosis}\n@{ai_time:.1f}s', color='red', fontweight='bold')

    # ==========================================
    # 🌟 4. 信息牌
    # ==========================================
    info_str = "=== FTC Mission Summary ===\n"
    if true_fault_name and true_fault_name != "NO_FAULT":
        info_str += f"Target Fault : [{true_fault_name}]\n"
        info_str += f"Occurred At  : {fault_time:.1f} s\n"
    else:
        info_str += "Mission Status : Normal Cruising\n"

    if ai_diagnosis and ai_diagnosis != "NO_FAULT":
        info_str += f"AI Diagnosis : [{ai_diagnosis}]\n"
        if ai_time is not None:
            info_str += f"Intervention : {ai_time:.1f} s\n"
            delay = ai_time - fault_time if fault_time is not None else 0
            if delay > 0:
                info_str += f"Response Delay: {delay:.1f} s\n"

    ax.text2D(0.02, 0.98, info_str, transform=ax.transAxes, fontsize=11,
              verticalalignment='top',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Depth Z (m)')
    ax.set_title('AUV 3D Trajectory & Fault Location', fontweight='bold')

    # 🌟 核心修复：翻转 Z 轴，让深海 (正数) 在下方！
    ax.invert_zaxis()

    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f" 3D trajectory map has been generated: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def animate_trajectory(trajectory: np.ndarray, waypoints=None, destination=None, sensor_logs=None, dt=0.1, playback_speed=10):
    if len(trajectory) == 0:
        print("No trajectory data to visualize")
        return

    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]

    usv_x = x
    usv_y = y
    usv_z = np.zeros_like(z)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    info_text = ax.text2D(
        0.02, 0.95,
        "",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    xx, yy = np.meshgrid(
        np.linspace(min(x) - 20, max(x) + 20, 30),
        np.linspace(min(y) - 20, max(y) + 20, 30)
    )
    zz_seabed = np.ones_like(xx) * (max(z) + 10)  # 这里的基准也要跟着正数走
    ax.plot_surface(xx, yy, zz_seabed, cmap="terrain", alpha=0.3)

    ax.plot_surface(xx, yy, np.zeros_like(xx), color='cyan', alpha=0.1)

    if waypoints is not None:
        for wp in waypoints:
            ax.scatter(wp[0], wp[1], wp[2], c='yellow', marker='^', s=120)

    if destination is not None:
        ax.scatter(destination[0], destination[1], destination[2],
                   c='green', marker='*', s=200)

    line_done, = ax.plot([], [], [], 'b-', linewidth=2)
    ax.plot(x, y, z, color='gray', alpha=0.3)
    point, = ax.plot([], [], [], 'ro', markersize=6, label='AUV')

    usv_point, = ax.plot([], [], [], 'gs', markersize=8, label='USV (Mother Ship)')
    tether_line, = ax.plot([], [], [], 'k--', linewidth=1.5, alpha=0.6, label='Tether')

    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    vz = np.gradient(z, dt)
    arrow = None

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Depth Z (m)')
    ax.set_title("AUV Mission Animation (with USV Tether)")

    ax.set_xlim(min(x) - 10, max(x) + 10)
    ax.set_ylim(min(y) - 10, max(y) + 10)

    # 🌟 核心修复：翻转 Z 轴，让深海 (正数) 在下方！
    ax.invert_zaxis()

    ax.legend(loc='upper right')

    def update(frame):
        nonlocal arrow
        time = frame * dt
        speed = np.sqrt(vx[frame] ** 2 + vy[frame] ** 2 + vz[frame] ** 2)

        fault_label = 0
        display_pred = "NO_FAULT"
        action_name = "NORMAL OPERATION"
        action_color = "green"
        is_locked = False

        if sensor_logs is not None and frame < len(sensor_logs):
            log = sensor_logs[frame]
            fault_label = log.get("fault_label", 0)
            display_pred = log.get("ftc_diagnosis", "NO_FAULT")
            action_name = log.get("ftc_action", "NORMAL OPERATION")
            is_locked = log.get("ftc_is_locked", False)

            if is_locked:
                action_color = "red"
            elif display_pred == "NOISE":
                action_color = "orange"
            elif display_pred == "SPIKE":
                action_color = "yellow"

        line_done.set_data(x[:frame], y[:frame])
        line_done.set_3d_properties(z[:frame])

        if display_pred != "NO_FAULT" and display_pred != "NOISE":
            line_done.set_color("red")
        else:
            line_done.set_color("blue")

        point.set_data([x[frame]], [y[frame]])
        point.set_3d_properties([z[frame]])
        usv_point.set_data([usv_x[frame]], [usv_y[frame]])
        usv_point.set_3d_properties([usv_z[frame]])
        tether_line.set_data([usv_x[frame], x[frame]], [usv_y[frame], y[frame]])
        tether_line.set_3d_properties([usv_z[frame], z[frame]])

        if arrow is not None: arrow.remove()
        arrow = ax.quiver(
            x[frame], y[frame], z[frame],
            vx[frame], vy[frame], vz[frame],
            length=5, color='red'
        )

        fault_map = {1: "BIAS", 2: "DRIFT", 3: "STUCK", 4: "SPIKE", 5: "NOISE", 6: "ENTANGLED", 7: "BROKEN"}
        fault_name = fault_map.get(fault_label, "NO_FAULT")

        info_text.set_text(
            f"Time: {time:.1f}s\n"
            f"Depth: {z[frame]:.2f} m\n"
            f"Speed: {speed:.2f} m/s\n"
            f"True Fault: {fault_name}\n"
            f"FTC Diagnosis: {display_pred}\n"
            f"Action: {action_name}"
        )

        bbox_patch = info_text.get_bbox_patch()
        if bbox_patch is not None:
            bbox_patch.set_edgecolor(action_color)
            bbox_patch.set_linewidth(2)

        return line_done, point, info_text, usv_point, tether_line
    # ==========================================
    # 🌟 核心修复：按 playback_speed 进行抽帧！
    # ==========================================
    frames_to_render = range(0, len(x), playback_speed)

    # 强制让 interval=30 (也就是保证 30 FPS 的流畅度)
    ani = FuncAnimation(fig, update, frames=frames_to_render, interval=30, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()