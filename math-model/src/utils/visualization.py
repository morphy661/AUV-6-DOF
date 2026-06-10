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
    ax.set_title('AUV Mission Animation with USV Acoustic Support', fontweight='bold')

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

def generate_usv_support_track(
        x,
        y,
        dt,
        update_interval=30.0,
        acoustic_delay=10.0,
        max_usv_speed=2.0,
        smoothing=0.25,
        estimation_uncertainty=15.0
):
    """
    Generate a realistic USV support trajectory.

    The USV does not stay exactly above the AUV.
    It follows a delayed and smoothed estimate of the AUV horizontal operating region,
    constrained by its maximum speed.

    Parameters
    ----------
    x, y : array-like
        AUV horizontal trajectory.
    dt : float
        Simulation time step.
    update_interval : float
        Low-frequency acoustic / mission-plan update interval in seconds.
    acoustic_delay : float
        Delay of acoustic/localization update in seconds.
    max_usv_speed : float
        Maximum horizontal USV speed in m/s.
    smoothing : float
        Smoothing factor for USV movement.
    estimation_uncertainty : float
        Deterministic uncertainty radius added to the estimated AUV region.
    """
    n = len(x)
    usv_x = np.zeros(n)
    usv_y = np.zeros(n)

    # Start near the mission support area, not necessarily exactly above the AUV.
    usv_x[0] = x[0]
    usv_y[0] = y[0]

    last_est_x = x[0]
    last_est_y = y[0]

    update_steps = max(1, int(update_interval / dt))
    delay_steps = max(0, int(acoustic_delay / dt))

    for i in range(1, n):
        # Low-frequency delayed estimate update.
        if i % update_steps == 0:
            est_idx = max(0, i - delay_steps)

            # Deterministic pseudo-current / localization uncertainty.
            # This avoids using randomness and keeps animation reproducible.
            t = i * dt
            uncertainty_x = estimation_uncertainty * np.sin(t / 180.0)
            uncertainty_y = estimation_uncertainty * np.cos(t / 220.0)

            last_est_x = x[est_idx] + uncertainty_x
            last_est_y = y[est_idx] + uncertainty_y

        dx = last_est_x - usv_x[i - 1]
        dy = last_est_y - usv_y[i - 1]
        dist = np.sqrt(dx ** 2 + dy ** 2)

        max_step = max_usv_speed * dt

        if dist > 1e-6:
            step = min(max_step, dist)
            target_x = usv_x[i - 1] + dx / dist * step
            target_y = usv_y[i - 1] + dy / dist * step
        else:
            target_x = usv_x[i - 1]
            target_y = usv_y[i - 1]

        # Smooth USV motion.
        usv_x[i] = (1 - smoothing) * usv_x[i - 1] + smoothing * target_x
        usv_y[i] = (1 - smoothing) * usv_y[i - 1] + smoothing * target_y

    return usv_x, usv_y
def animate_trajectory(
        trajectory: np.ndarray,
        waypoints=None,
        destination=None,
        sensor_logs=None,
        dt=0.1,
        playback_speed=20,
        save_path=None,
        show=True,
        show_direction_arrow=False,
        show_future_events=False
):
    """
    Enhanced 3D ocean-scene animation for AUV FTC demonstrations.

    Design notes:
    - AUV is autonomous.
    - USV is a surface support platform, not a tethered vehicle.
    - USV follows a delayed / smoothed estimate of the AUV operating region.
    - Acoustic link is only a communication visualization, not a physical cable.
    """
    if len(trajectory) == 0:
        print("No trajectory data to visualize")
        return

    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]

    # ======================================================
    # 1. USV support model: delayed, smoothed estimated track
    # ======================================================
    usv_x, usv_y = generate_usv_support_track(
        x=x,
        y=y,
        dt=dt,
        update_interval=30.0,
        acoustic_delay=10.0,
        max_usv_speed=2.0,
        smoothing=0.25,
        estimation_uncertainty=15.0
    )
    usv_z = np.zeros_like(z)

    # Use one range for logic and another smaller range for visual clarity.
    # This avoids a huge circle covering the whole figure.
    communication_range = 800.0      # Real acoustic-link logic range.
    display_comm_range = 60       # Visual support-zone radius shown in the plot.

    # ======================================================
    # 2. Figure / ocean scene
    # ======================================================
    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor("#071A2C")

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0B2239")
    ax.view_init(elev=24, azim=-55)

    # Lower grid density improves interactive rotation speed.
    margin = 35
    xx, yy = np.meshgrid(
        np.linspace(min(x) - margin, max(x) + margin, 25),
        np.linspace(min(y) - margin, max(y) + margin, 25)
    )

    # Wavy seabed. Depth is positive downward.
    seabed_base = max(z) + 25
    zz_seabed = seabed_base + 8 * np.sin(xx / 60.0) * np.cos(yy / 60.0)
    ax.plot_surface(
        xx, yy, zz_seabed,
        cmap="terrain",
        alpha=0.32,
        linewidth=0,
        antialiased=True
    )

    # Sea surface. Keep it very light so it does not block the route.
    ax.plot_surface(
        xx, yy,
        np.zeros_like(xx),
        color="#5EDFFF",
        alpha=0.045,
        linewidth=0
    )

    # ======================================================
    # 3. Waypoints and planned route
    # ======================================================
    # ======================================================
    # Dynamic waypoint display:
    # Only show the current target waypoint and a few future waypoints.
    # Reached waypoints are automatically hidden during animation.
    # ======================================================
    waypoints_np = None
    waypoint_markers = []
    waypoint_labels = []
    waypoint_reach_frames = []
    preview_waypoint_count = 3  # current target + next 3 waypoints

    planned_route_line = None

    if waypoints is not None:
        waypoints_np = np.array(waypoints, dtype=float)

        if waypoints_np.ndim == 2 and waypoints_np.shape[0] > 1:
            # Dynamic remaining planned route line.
            # It will be updated in update(frame).
            planned_route_line, = ax.plot(
                waypoints_np[:, 0],
                waypoints_np[:, 1],
                waypoints_np[:, 2],
                linestyle="--",
                color="#FFD54F",
                linewidth=2.8,
                alpha=0.95,
                label="Waypoint Connection"
            )

            # Estimate when each waypoint is reached.
            # Use sequential search to avoid the final return-to-origin waypoint
            # being incorrectly detected at frame 0.
            search_start = 0
            wp_reach_radius = 12.0
            wp_fallback_radius = 35.0

            for wp in waypoints_np:
                if search_start >= len(trajectory):
                    waypoint_reach_frames.append(None)
                    continue

                segment = trajectory[search_start:]
                distances = np.linalg.norm(segment[:, :3] - wp[:3], axis=1)

                candidates = np.where(distances <= wp_reach_radius)[0]

                if len(candidates) > 0:
                    reach_frame = search_start + int(candidates[0])
                    waypoint_reach_frames.append(reach_frame)
                    search_start = reach_frame + 1
                else:
                    closest_local = int(np.argmin(distances))
                    closest_dist = float(distances[closest_local])

                    if closest_dist <= wp_fallback_radius:
                        reach_frame = search_start + closest_local
                        waypoint_reach_frames.append(reach_frame)
                        search_start = reach_frame + 1
                    else:
                        waypoint_reach_frames.append(None)

            # Create waypoint artists, initially hidden.
            for i, wp in enumerate(waypoints_np, start=1):
                marker = ax.scatter(
                    wp[0], wp[1], wp[2],
                    c="#FFD54F",
                    marker="^",
                    s=95,
                    edgecolors="black",
                    linewidths=0.6,
                    visible=True
                )

                label = ax.text(
                    wp[0], wp[1], wp[2],
                    f" WP{i}",
                    color="#FFE082",
                    fontsize=8,
                    fontweight="bold",
                    visible=True
                )

                waypoint_markers.append(marker)
                waypoint_labels.append(label)

    if destination is not None:
        ax.scatter(
            destination[0], destination[1], destination[2],
            c="#00E676",
            marker="*",
            s=260,
            edgecolors="black",
            linewidths=0.5,
            label="Destination"
        )

    # ======================================================
    # 4. Detect first fault and first FTC intervention
    # ======================================================
    fault_idx = None
    ai_idx = None

    if sensor_logs is not None:
        for i, log in enumerate(sensor_logs):
            if fault_idx is None and log.get("fault_label", 0) != 0:
                fault_idx = i

            diagnosis = log.get("ftc_diagnosis", "NO_FAULT")
            if ai_idx is None and diagnosis not in [None, "NO_FAULT"]:
                ai_idx = i

    # Future event markers are optional. Keeping them hidden before the event
    # makes the animation look more natural.
    fault_marker = None
    fault_text = None
    ai_marker = None
    ai_text = None

    if fault_idx is not None and fault_idx < len(x):
        fault_marker = ax.scatter(
            [], [], [],
            c="#FF9800",
            marker="X",
            s=170,
            edgecolors="black",
            linewidths=0.6,
            label="Fault Injected"
        )
        fault_text = ax.text(
            x[fault_idx], y[fault_idx], z[fault_idx] + 15,
            " Fault",
            color="#FFCC80",
            fontsize=10,
            fontweight="bold",
            visible=False
        )

    if ai_idx is not None and ai_idx < len(x):
        ai_marker = ax.scatter(
            [], [], [],
            c="#FF1744",
            marker="P",
            s=180,
            edgecolors="white",
            linewidths=0.6,
            label="FTC Intervention"
        )
        ai_text = ax.text(
            x[ai_idx], y[ai_idx], z[ai_idx] - 18,
            " FTC",
            color="#FF8A80",
            fontsize=10,
            fontweight="bold",
            visible=False
        )

    # Show markers from the beginning only when explicitly requested.
    if show_future_events:
        if fault_marker is not None:
            fault_marker._offsets3d = ([x[fault_idx]], [y[fault_idx]], [z[fault_idx]])
            fault_text.set_visible(True)
        if ai_marker is not None:
            ai_marker._offsets3d = ([x[ai_idx]], [y[ai_idx]], [z[ai_idx]])
            ai_text.set_visible(True)

    # ======================================================
    # 5. Dynamic objects
    # ======================================================
    ax.plot(
        x, y, z,
        color="#CFD8DC",
        linestyle=":",
        linewidth=1.5,
        alpha=0.62,
        label="Full Trajectory"
    )

    line_done, = ax.plot([], [], [], color="#00E5FF", linewidth=3.4, label="AUV Path")
    point, = ax.plot([], [], [], marker="o", color="#FF5252", markersize=8, label="AUV")
    halo, = ax.plot([], [], [], marker="o", color="#FFCDD2", markersize=18, alpha=0.28)

    usv_point, = ax.plot(
        [], [], [],
        marker="s",
        color="#69F0AE",
        markersize=9,
        label="USV Support Vessel"
    )

    acoustic_link, = ax.plot(
        [], [], [],
        linestyle="--",
        color="#80DEEA",
        linewidth=1.7,
        alpha=0.75,
        label="Acoustic Link"
    )

    # Surface support-zone circle around the USV.
    theta = np.linspace(0, 2 * np.pi, 160)
    circle_z_level = -3.0  # Slightly above the sea surface for visibility.
    comm_circle_x = usv_x[0] + display_comm_range * np.cos(theta)
    comm_circle_y = usv_y[0] + display_comm_range * np.sin(theta)
    comm_circle_z = np.ones_like(theta) * circle_z_level

    comm_circle, = ax.plot(
        comm_circle_x,
        comm_circle_y,
        comm_circle_z,
        color="#00E5FF",
        linestyle="-.",
        linewidth=2.1,
        alpha=0.9,
        label="Surface Support Range"
    )

    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    vz = np.gradient(z, dt)
    arrow = None

    # ======================================================
    # 6. Mission HUD panel
    # ======================================================
    info_text = ax.text2D(
        0.02, 0.96,
        "",
        transform=ax.transAxes,
        fontsize=10,
        color="white",
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor="#102A43",
            alpha=0.82,
            edgecolor="#00E5FF",
            linewidth=1.6
        )
    )

    # ======================================================
    # 7. Axes styling
    # ======================================================
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Y (m)", color="white")
    ax.set_zlabel("Depth Z (m)", color="white")
    ax.set_title(
        "AUV Fault-Tolerant Control Mission Animation",
        color="white",
        fontsize=14,
        fontweight="bold"
    )

    ax.tick_params(colors="white")

    # Make 3D panes less distracting.
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((0.05, 0.16, 0.25, 0.08))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.18))

    ax.set_xlim(min(x) - 20, max(x) + 20)
    ax.set_ylim(min(y) - 20, max(y) + 20)

    # Depth-positive downward:
    # 0 m surface at the top, larger depth lower in the plot.
    ax.set_zlim(-20, max(z) + 40)
    ax.invert_zaxis()

    legend = ax.legend(loc="upper right", fontsize=8)
    legend.get_frame().set_facecolor("#102A43")
    legend.get_frame().set_alpha(0.78)
    for text in legend.get_texts():
        text.set_color("white")

    fault_map = {
        1: "BIAS",
        2: "DRIFT",
        3: "STUCK",
        4: "SPIKE",
        5: "NOISE",
        6: "ENTANGLED",
        7: "NO_OUTPUT"
    }

    # ======================================================
    # 8. Update animation
    # ======================================================
    def update(frame):
        nonlocal arrow

        time = frame * dt
        speed = np.sqrt(vx[frame] ** 2 + vy[frame] ** 2 + vz[frame] ** 2)
        usv_horizontal_range = np.sqrt(
            (usv_x[frame] - x[frame]) ** 2 +
            (usv_y[frame] - y[frame]) ** 2
        )

        link_status = "ACTIVE" if usv_horizontal_range <= communication_range else "OUT OF RANGE"

        fault_label = 0
        display_pred = "NO_FAULT"
        action_name = "NORMAL OPERATION"
        action_color = "#00E676"
        is_locked = False
        target_z = None
        tracking_error = None

        if sensor_logs is not None and frame < len(sensor_logs):
            log = sensor_logs[frame]
            fault_label = log.get("fault_label", 0)
            display_pred = log.get("ftc_diagnosis", "NO_FAULT")
            action_name = log.get("ftc_action", "NORMAL OPERATION")
            is_locked = log.get("ftc_is_locked", False)
            target_z = log.get("target_z", None)

            if target_z is not None:
                tracking_error = target_z - z[frame]

            if is_locked:
                action_color = "#FF1744"
            elif display_pred == "NOISE":
                action_color = "#FF9800"
            elif display_pred == "SPIKE":
                action_color = "#FFEB3B"
            else:
                action_color = "#00E676"

        line_done.set_data(x[:frame], y[:frame])
        line_done.set_3d_properties(z[:frame])
        # ======================================================
        # Update dynamic waypoint display
        # ======================================================
        current_wp_text = "N/A"

        if waypoints_np is not None and planned_route_line is not None:
            reached_indices = [
                idx for idx, reach_frame in enumerate(waypoint_reach_frames)
                if reach_frame is not None and frame >= reach_frame
            ]

            if len(reached_indices) > 0:
                current_wp_idx = max(reached_indices) + 1
            else:
                current_wp_idx = 0

            current_wp_idx = min(current_wp_idx, len(waypoints_np) - 1)

            # Show only current target waypoint and several future waypoints.
            visible_start = current_wp_idx
            visible_end = min(len(waypoints_np), current_wp_idx + preview_waypoint_count)

            # Update remaining planned route.
            remaining_route = waypoints_np[visible_start:]

            if len(remaining_route) > 0:
                planned_route_line.set_data(
                    remaining_route[:, 0],
                    remaining_route[:, 1]
                )
                planned_route_line.set_3d_properties(remaining_route[:, 2])
            else:
                planned_route_line.set_data([], [])
                planned_route_line.set_3d_properties([])

            # Hide reached waypoints and distant future waypoints.
            for idx, (marker, label) in enumerate(zip(waypoint_markers, waypoint_labels)):
                is_visible = visible_start <= idx < visible_end

                marker.set_visible(is_visible)
                label.set_visible(is_visible)

                # Highlight the current target waypoint.
                if is_visible and idx == current_wp_idx:
                    marker.set_sizes([180])
                    label.set_fontsize(10)
                elif is_visible:
                    marker.set_sizes([95])
                    label.set_fontsize(8)

            current_wp_text = f"WP{current_wp_idx + 1}/{len(waypoints_np)}"
        if is_locked:
            line_done.set_color("#FF1744")
        elif display_pred == "NOISE":
            line_done.set_color("#FF9800")
        elif display_pred == "SPIKE":
            line_done.set_color("#FFEB3B")
        else:
            line_done.set_color("#00E5FF")

        point.set_data([x[frame]], [y[frame]])
        point.set_3d_properties([z[frame]])
        halo.set_data([x[frame]], [y[frame]])
        halo.set_3d_properties([z[frame]])

        usv_point.set_data([usv_x[frame]], [usv_y[frame]])
        usv_point.set_3d_properties([usv_z[frame]])

        # Update support-zone circle around the USV.
        comm_circle_x = usv_x[frame] + display_comm_range * np.cos(theta)
        comm_circle_y = usv_y[frame] + display_comm_range * np.sin(theta)
        comm_circle.set_data(comm_circle_x, comm_circle_y)
        comm_circle.set_3d_properties(np.ones_like(theta) * circle_z_level)

        # Acoustic link is shown only when the AUV is inside real communication range.
        if usv_horizontal_range <= communication_range:
            acoustic_link.set_data(
                [usv_x[frame], x[frame]],
                [usv_y[frame], y[frame]]
            )
            acoustic_link.set_3d_properties([usv_z[frame], z[frame]])
        else:
            acoustic_link.set_data([], [])
            acoustic_link.set_3d_properties([])

        # Optional direction arrow. Disabled by default for smoother interaction.
        if show_direction_arrow:
            if arrow is not None:
                arrow.remove()
            arrow = ax.quiver(
                x[frame], y[frame], z[frame],
                vx[frame], vy[frame], vz[frame],
                length=8,
                color="#FF5252",
                linewidth=1.2,
                normalize=True
            )

        # Reveal event markers only after the event has happened.
        if not show_future_events:
            if fault_marker is not None and frame >= fault_idx:
                fault_marker._offsets3d = ([x[fault_idx]], [y[fault_idx]], [z[fault_idx]])
                fault_text.set_visible(True)

            if ai_marker is not None and frame >= ai_idx:
                ai_marker._offsets3d = ([x[ai_idx]], [y[ai_idx]], [z[ai_idx]])
                ai_text.set_visible(True)

        fault_name = fault_map.get(fault_label, "NO_FAULT")
        target_text = f"{target_z:.2f} m" if target_z is not None else "N/A"
        error_text = f"{tracking_error:.2f} m" if tracking_error is not None else "N/A"
        locked_text = "YES" if is_locked else "NO"

        info_text.set_text(
            f"Mission Time: {time:.1f} s\n"
            f"Current Target: {current_wp_text}\n"
            f"Depth: {z[frame]:.2f} m\n"
            f"Target Depth: {target_text}\n"
            f"Tracking Error: {error_text}\n"
            f"Speed: {speed:.2f} m/s\n"
            f"True Fault: {fault_name}\n"
            f"FTC Diagnosis: {display_pred}\n"
            f"Locked: {locked_text}\n"
            f"Action: {action_name}\n"
            f"USV Range: {usv_horizontal_range:.1f} m\n"
            f"Acoustic Link: {link_status}"
        )

        bbox_patch = info_text.get_bbox_patch()
        if bbox_patch is not None:
            bbox_patch.set_edgecolor(action_color)
            bbox_patch.set_linewidth(2.0)

        artists = [line_done, point, halo, info_text, usv_point, acoustic_link, comm_circle]
        if fault_marker is not None:
            artists.append(fault_marker)
        if ai_marker is not None:
            artists.append(ai_marker)
        return tuple(artists)

    frames_to_render = range(0, len(x), playback_speed)

    ani = FuncAnimation(
        fig,
        update,
        frames=frames_to_render,
        interval=30,
        blit=False,
        repeat=False
    )

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(save_path, writer="ffmpeg", fps=30, dpi=150)
        print(f"Animation saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
