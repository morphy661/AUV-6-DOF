"""Validate all single-fault cases of the six-thruster AUV layout."""

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from actuators.six_dof_thruster_faults import (
    SingleThrusterFault,
    SixDOFThrusterFaultMode,
)
from environment.six_dof_simulator import SixDOFSimulator
from simple_control.six_dof_controller import PoseTarget


FAULT_TIME = 55.0
THRUSTER_NAMES = ("H1", "H2", "H3", "H4", "V1", "V2")
AXIS_NAMES = ("X", "Y", "Z", "K", "M", "N")
FAULT_DEFINITIONS = (
    ("No Output", SixDOFThrusterFaultMode.NO_OUTPUT, 0.0),
    ("Thrust Loss", SixDOFThrusterFaultMode.THRUST_LOSS, 0.45),
)
MISSION_TARGETS = [
    (0.0, np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.0])),
    (25.0, np.array([6.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.0])),
    (55.0, np.array([6.0, 4.0, 3.0]), np.array([0.0, 0.0, np.pi / 2.0])),
    (85.0, np.array([1.0, 4.0, 2.0]), np.array([0.0, 0.0, np.pi])),
]


def target_provider(time_s, _state):
    selected = MISSION_TARGETS[0]
    for candidate in MISSION_TARGETS:
        if time_s >= candidate[0]:
            selected = candidate
        else:
            break
    return PoseTarget(selected[1], selected[2])


def scenarios():
    """Return normal plus 6 thrusters x 2 fault modes = 13 cases."""
    cases = {"Normal": None}
    for thruster_name in THRUSTER_NAMES:
        for label, mode, efficiency in FAULT_DEFINITIONS:
            cases[f"{thruster_name} {label}"] = SingleThrusterFault(
                thruster_name=thruster_name,
                mode=mode,
                start_time=FAULT_TIME,
                thrust_efficiency=efficiency,
            )
    return cases


def run_scenarios(duration, dt):
    return {
        name: SixDOFSimulator(fault=fault).run(duration, dt, target_provider)
        for name, fault in scenarios().items()
    }


def angle_difference(values, reference):
    return np.arctan2(np.sin(values - reference), np.cos(values - reference))


def scenario_metadata(name):
    if name == "Normal":
        return "none", "normal"
    thruster_name, fault_mode = name.split(" ", 1)
    return thruster_name, fault_mode.lower().replace(" ", "_")


def extract_metrics(results):
    normal_logs = results["Normal"]
    nominal_positions = np.array([log["position_ned"] for log in normal_logs])
    nominal_attitudes = np.array([log["euler_rpy"] for log in normal_logs])
    rows = []
    peak_residuals = {}

    for name, logs in results.items():
        times = np.array([log["time"] for log in logs])
        mask = times >= FAULT_TIME
        if not np.any(mask):
            raise ValueError("simulation duration must extend beyond FAULT_TIME")

        positions = np.array([log["position_ned"] for log in logs])
        attitudes = np.array([log["euler_rpy"] for log in logs])
        deviations = np.linalg.norm(positions - nominal_positions, axis=1)[mask]
        attitude_delta = angle_difference(attitudes, nominal_attitudes)[mask]
        residual_matrix = np.array([
            log["actuation_residual_body"] for log in logs
        ])[mask]
        peak_residual = np.max(np.abs(residual_matrix), axis=0)
        peak_residuals[name] = peak_residual
        thruster_name, fault_mode = scenario_metadata(name)

        current_residual = 0.0
        force_residual = 0.0
        if name != "Normal":
            fault_index = THRUSTER_NAMES.index(thruster_name)
            current_residual = max(
                abs(
                    log["thruster_measured_currents"][fault_index]
                    - log["thruster_expected_currents"][fault_index]
                )
                for log, keep in zip(logs, mask)
                if keep
            )
            force_residual = max(
                abs(
                    log["commanded_thruster_forces"][fault_index]
                    - log["actual_thruster_forces"][fault_index]
                )
                for log, keep in zip(logs, mask)
                if keep
            )

        direct_axes = "/".join(
            axis for axis, value in zip(AXIS_NAMES, peak_residual)
            if value > 1e-8
        ) or "none"
        final_target = logs[-1]["target_position_ned"]
        rows.append({
            "scenario": name,
            "faulted_thruster": thruster_name,
            "fault_mode": fault_mode,
            "direct_wrench_axes": direct_axes,
            "max_trajectory_deviation_m": float(np.max(deviations)),
            "rms_trajectory_deviation_m": float(
                np.sqrt(np.mean(deviations ** 2))
            ),
            "max_roll_deviation_deg": float(
                np.rad2deg(np.max(np.abs(attitude_delta[:, 0])))
            ),
            "max_pitch_deviation_deg": float(
                np.rad2deg(np.max(np.abs(attitude_delta[:, 1])))
            ),
            "max_yaw_deviation_deg": float(
                np.rad2deg(np.max(np.abs(attitude_delta[:, 2])))
            ),
            "final_position_error_m": float(
                np.linalg.norm(final_target - positions[-1])
            ),
            "max_force_residual_n": float(force_residual),
            "max_current_residual_a": float(current_residual),
        })

    return rows, peak_residuals


def validate_coverage(rows):
    """Fail fast when a scenario or direct fault signature is missing."""
    if len(rows) != 13:
        raise AssertionError(f"expected 13 scenarios, got {len(rows)}")
    for row in rows:
        thruster_name = row["faulted_thruster"]
        if thruster_name == "none":
            if row["direct_wrench_axes"] != "none":
                raise AssertionError("normal scenario has a non-zero fault residual")
            continue
        expected_axes = "X/Y/N" if thruster_name.startswith("H") else "Z/M"
        if row["direct_wrench_axes"] != expected_axes:
            raise AssertionError(
                f"{row['scenario']} signature {row['direct_wrench_axes']} "
                f"does not match {expected_axes}"
            )
        if row["max_force_residual_n"] <= 0.0:
            raise AssertionError(f"{row['scenario']} has no force residual")


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_axis_residual_csv(peak_residuals, path):
    rows = []
    for name, values in peak_residuals.items():
        thruster_name, fault_mode = scenario_metadata(name)
        row = {
            "scenario": name,
            "faulted_thruster": thruster_name,
            "fault_mode": fault_mode,
        }
        row.update({axis: float(value) for axis, value in zip(AXIS_NAMES, values)})
        rows.append(row)
    save_csv(rows, path)


def _flatten_log(log, scenario_name):
    thruster_name, fault_mode = scenario_metadata(scenario_name)
    row = {
        "scenario": scenario_name,
        "faulted_thruster": thruster_name,
        "fault_mode": fault_mode,
        "time_s": float(log["time"]),
        "thruster_fault_active": int(log["thruster_fault_active"]),
        "ideal_ftc_enabled": int(log["ideal_ftc_enabled"]),
        "ftc_active": int(log["ftc_active"]),
        "faulted_thruster_index": (
            -1
            if log["faulted_thruster_index"] is None
            else int(log["faulted_thruster_index"])
        ),
    }
    vector_fields = {
        "position_ned": ("north_m", "east_m", "depth_m"),
        "euler_rpy": ("roll_rad", "pitch_rad", "yaw_rad"),
        "body_velocity": (
            "u_mps", "v_mps", "w_mps", "p_radps", "q_radps", "r_radps"
        ),
        "target_position_ned": (
            "target_north_m", "target_east_m", "target_depth_m"
        ),
        "target_euler_rpy": (
            "target_roll_rad", "target_pitch_rad", "target_yaw_rad"
        ),
        "desired_wrench_body": tuple(f"desired_{axis}" for axis in AXIS_NAMES),
        "allocated_wrench_body": tuple(
            f"allocated_{axis}" for axis in AXIS_NAMES
        ),
        "achieved_wrench_body": tuple(f"actual_{axis}" for axis in AXIS_NAMES),
        "allocation_residual_body": tuple(
            f"allocation_residual_{axis}" for axis in AXIS_NAMES
        ),
        "actuation_residual_body": tuple(
            f"residual_{axis}" for axis in AXIS_NAMES
        ),
    }
    for field, columns in vector_fields.items():
        row.update({column: float(value) for column, value in zip(columns, log[field])})

    thruster_fields = {
        "commanded_thruster_forces": "command_force_n",
        "actual_thruster_forces": "actual_force_n",
        "thruster_expected_currents": "expected_current_a",
        "thruster_measured_currents": "measured_current_a",
        "thruster_force_efficiencies": "force_efficiency",
        "allocation_thruster_effectiveness": "allocation_effectiveness",
    }
    for field, suffix in thruster_fields.items():
        row.update({
            f"{name}_{suffix}": float(value)
            for name, value in zip(THRUSTER_NAMES, log[field])
        })
    return row


def save_scenario_logs(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, logs in results.items():
        rows = [_flatten_log(log, name) for log in logs]
        filename = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        save_csv(rows, output_dir / f"{filename}.csv")


def _short_label(name):
    return name.replace(" No Output", "-NO").replace(" Thrust Loss", "-TL")


def save_plot(results, rows, peak_residuals, path):
    fault_names = [name for name in results if name != "Normal"]
    row_by_name = {row["scenario"]: row for row in rows}
    thruster_colors = dict(zip(
        THRUSTER_NAMES,
        plt.get_cmap("tab10").colors[:len(THRUSTER_NAMES)],
    ))
    short_labels = [_short_label(name) for name in fault_names]
    colors = [thruster_colors[name.split(" ", 1)[0]] for name in fault_names]

    normal_positions = np.array([
        log["position_ned"] for log in results["Normal"]
    ])
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    deviation_axis, position_axis = axes[0]
    attitude_axis, residual_axis = axes[1]

    for name in fault_names:
        logs = results[name]
        times = np.array([log["time"] for log in logs])
        positions = np.array([log["position_ned"] for log in logs])
        thruster_name, fault_mode = scenario_metadata(name)
        deviation_axis.plot(
            times,
            np.linalg.norm(positions - normal_positions, axis=1),
            color=thruster_colors[thruster_name],
            linestyle="-" if fault_mode == "no_output" else "--",
            linewidth=1.3,
            label=_short_label(name),
        )
    deviation_axis.axvline(
        FAULT_TIME, color="black", linestyle=":", linewidth=1.4
    )
    deviation_axis.set_xlabel("Time (s)")
    deviation_axis.set_ylabel("Trajectory deviation from nominal (m)")
    deviation_axis.grid(True, alpha=0.3)
    deviation_axis.legend(fontsize=7, ncol=3)

    x = np.arange(len(fault_names))
    final_errors = [
        row_by_name[name]["final_position_error_m"] for name in fault_names
    ]
    position_axis.bar(x, final_errors, color=colors)
    position_axis.set_xticks(x, short_labels, rotation=60, ha="right")
    position_axis.set_ylabel("Final position error (m)")
    position_axis.grid(True, axis="y", alpha=0.3)

    width = 0.25
    attitude_metrics = (
        ("max_roll_deviation_deg", "Roll"),
        ("max_pitch_deviation_deg", "Pitch"),
        ("max_yaw_deviation_deg", "Yaw"),
    )
    for index, (field, label) in enumerate(attitude_metrics):
        attitude_axis.bar(
            x + (index - 1) * width,
            [row_by_name[name][field] for name in fault_names],
            width,
            label=label,
        )
    attitude_axis.set_xticks(x, short_labels, rotation=60, ha="right")
    attitude_axis.set_ylabel("Maximum deviation from nominal (deg)")
    attitude_axis.grid(True, axis="y", alpha=0.3)
    attitude_axis.legend()

    residual_matrix = np.array([peak_residuals[name] for name in fault_names])
    image = residual_axis.imshow(residual_matrix, aspect="auto", cmap="magma")
    residual_axis.set_xticks(np.arange(len(AXIS_NAMES)), AXIS_NAMES)
    residual_axis.set_yticks(np.arange(len(fault_names)), short_labels)
    residual_axis.set_xlabel("Body wrench axis")
    residual_axis.set_ylabel("Fault scenario")
    colorbar = figure.colorbar(image, ax=residual_axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Peak absolute actuation residual")

    figure.suptitle(
        "All single-thruster faults: 6 thrusters x 2 modes",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "six_dof_fault_coverage",
    )
    args = parser.parse_args()
    if args.duration <= FAULT_TIME:
        parser.error(f"--duration must be greater than fault time {FAULT_TIME:g} s")

    results = run_scenarios(args.duration, args.dt)
    rows, peak_residuals = extract_metrics(results)
    validate_coverage(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "six_dof_fault_coverage_summary.csv"
    residual_path = args.output_dir / "six_dof_fault_axis_residuals.csv"
    plot_path = args.output_dir / "six_dof_fault_coverage.png"
    log_dir = args.output_dir / "source_logs"
    save_csv(rows, summary_path)
    save_axis_residual_csv(peak_residuals, residual_path)
    save_scenario_logs(results, log_dir)
    save_plot(results, rows, peak_residuals, plot_path)

    for row in rows:
        print(row)
    print(f"Scenarios: {len(results)}")
    print("Coverage checks: PASS")
    print(f"Summary: {summary_path}")
    print(f"Axis residuals: {residual_path}")
    print(f"Source logs: {log_dir}")
    print(f"Figure: {plot_path}")


if __name__ == "__main__":
    main()
