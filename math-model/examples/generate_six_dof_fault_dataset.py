"""Generate a leakage-safe six-thruster fault-diagnosis dataset."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from actuators.six_dof_thruster_faults import (
    SingleThrusterFault,
    SixDOFThrusterFaultMode,
    ThrusterActuatorBank,
)
from actuators.thruster_array import default_six_thruster_array
from config.six_dof_config import SixDOFConfig
from environment.six_dof_dynamics import (
    SixDOFDynamics,
    SixDOFState,
    euler_to_quaternion,
)
from environment.six_dof_simulator import SixDOFSimulator
from sensors.depth_sensor import DepthSensor
from sensors.dvl_sensor import DVLSensor
from sensors.imu_sensor import IMUSensor
from sensors.six_dof_sensor_suite import SixDOFSensorSuite
from simple_control.six_dof_controller import PoseTarget
from utils.six_dof_dataset_builder import build_six_dof_sequence_dataset
from utils.six_dof_feature_extractor import (
    FAULT_LOCATION_NAMES,
    FAULT_MODE_NAMES,
    JOINT_FAULT_NAMES,
    THRUSTER_NAMES,
)


DEPTH_BANDS_M = (
    (0.0, 50.0),
    (50.0, 150.0),
    (150.0, 300.0),
    (300.0, 500.0),
)
DEPTH_BAND_PROBABILITIES = np.array([0.25, 0.35, 0.30, 0.10])
# A deterministic 20-mission cycle matching the probabilities above.  The
# interleaving avoids long runs in one band while keeping every split close to
# the requested distribution.
DEPTH_BAND_WEIGHTED_CYCLE = (
    1, 2, 0, 1, 3, 2, 1, 0, 2, 1,
    0, 2, 1, 3, 0, 2, 1, 0, 2, 1,
)
CRUISE_SPEEDS_MPS = np.array([1.2, 1.35, 1.5])
CRUISE_SPEED_PROBABILITIES = np.array([0.30, 0.40, 0.30])
VERTICAL_SPEED_RANGE_MPS = (0.15, 0.35)


@dataclass(frozen=True)
class MissionPlan:
    """A depth-stratified local mission with continuous position targets."""

    waypoints: tuple
    initial_position_ned: np.ndarray
    initial_euler_rpy: np.ndarray
    depth_band_index: int
    depth_band_m: tuple
    cruise_speed_mps: float
    vertical_speed_mps: float
    ramp_duration_s: float

    def to_metadata(self):
        return {
            "profile": "depth_weighted_local_smooth_v2",
            "depth_band_index": self.depth_band_index,
            "depth_band_m": list(self.depth_band_m),
            "initial_position_ned_m": self.initial_position_ned.tolist(),
            "cruise_speed_mps": self.cruise_speed_mps,
            "vertical_speed_mps": self.vertical_speed_mps,
            "ramp_duration_s": self.ramp_duration_s,
            "waypoints": [
                {
                    "time_s": float(time_s),
                    "position_ned_m": position.tolist(),
                    "euler_rpy_rad": attitude.tolist(),
                }
                for time_s, position, attitude in self.waypoints
            ],
        }


def scenario_definitions():
    definitions = [("Normal", None, None)]
    for thruster_name in THRUSTER_NAMES:
        definitions.append((
            f"{thruster_name} No Output",
            thruster_name,
            SixDOFThrusterFaultMode.NO_OUTPUT,
        ))
    for thruster_name in THRUSTER_NAMES:
        definitions.append((
            f"{thruster_name} Thrust Loss",
            thruster_name,
            SixDOFThrusterFaultMode.THRUST_LOSS,
        ))
    return definitions


def _split_counts(missions_per_scenario):
    if missions_per_scenario < 3:
        raise ValueError("at least three missions per scenario are required")
    validation_count = max(1, int(round(0.15 * missions_per_scenario)))
    test_count = max(1, int(round(0.15 * missions_per_scenario)))
    if validation_count + test_count >= missions_per_scenario:
        validation_count = 1
        test_count = 1
    return {
        "train": missions_per_scenario - validation_count - test_count,
        "validation": validation_count,
        "test": test_count,
    }


def _split_for_repetition(repetition, missions_per_scenario):
    counts = _split_counts(missions_per_scenario)
    if repetition < counts["train"]:
        return "train"
    if repetition < counts["train"] + counts["validation"]:
        return "validation"
    return "test"


def _randomized_dynamics(rng, split, mass_kg_override=None):
    """Sample a broad deployment domain for every leakage-safe split."""
    del split
    mass_scale = rng.uniform(0.82, 1.18)
    inertia_scale = np.array([
        rng.uniform(0.70, 1.30)
        for _ in range(3)
    ])
    added_mass_scale = np.array([
        rng.uniform(0.65, 1.35)
        for _ in range(6)
    ])
    linear_damping_scale = np.array([
        rng.uniform(0.60, 1.40)
        for _ in range(6)
    ])
    quadratic_damping_scale = np.array([
        rng.uniform(0.60, 1.40)
        for _ in range(6)
    ])
    sampled_mass = 50.0 * mass_scale
    mass = (
        sampled_mass
        if mass_kg_override is None
        else float(mass_kg_override)
    )
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass_kg_override must be finite and positive")
    weight = mass * 9.81
    buoyancy_ratio = rng.uniform(0.990, 1.010)
    xy_offset = 0.018
    cg = np.array([
        rng.uniform(-xy_offset, xy_offset),
        rng.uniform(-xy_offset, xy_offset),
        rng.uniform(0.015, 0.035),
    ])
    cb = np.array([
        rng.uniform(-xy_offset, xy_offset),
        rng.uniform(-xy_offset, xy_offset),
        rng.uniform(-0.035, -0.015),
    ])
    config = SixDOFConfig(
        mass=mass,
        inertia=np.diag(np.array([4.0, 12.0, 12.0]) * inertia_scale),
        added_mass=np.diag(
            np.array([5.0, 20.0, 25.0, 0.5, 1.5, 1.5])
            * added_mass_scale
        ),
        linear_damping=(
            np.array([15.0, 30.0, 35.0, 2.0, 5.0, 5.0])
            * linear_damping_scale
        ),
        quadratic_damping=(
            np.array([8.0, 18.0, 22.0, 1.0, 2.5, 2.5])
            * quadratic_damping_scale
        ),
        weight=weight,
        buoyancy=weight * buoyancy_ratio,
        center_of_gravity=cg,
        center_of_buoyancy=cb,
    )
    metadata = {
        "mass_kg": mass,
        "inertia_diagonal": np.diag(config.inertia).tolist(),
        "added_mass_diagonal": np.diag(config.added_mass).tolist(),
        "linear_damping": config.linear_damping.tolist(),
        "quadratic_damping": config.quadratic_damping.tolist(),
        "buoyancy_to_weight_ratio": buoyancy_ratio,
        "center_of_gravity_m": cg.tolist(),
        "center_of_buoyancy_m": cb.tolist(),
    }
    return SixDOFDynamics(config=config), metadata


def _randomized_thrusters(
    rng,
    split,
    vertical_force_limit_override=None,
):
    del split
    length = rng.uniform(1.00, 1.40)
    width = rng.uniform(0.48, 0.72)
    horizontal_limit = rng.uniform(34.0, 46.0)
    sampled_vertical_limit = rng.uniform(34.0, 46.0)
    vertical_limit = (
        sampled_vertical_limit
        if vertical_force_limit_override is None
        else float(vertical_force_limit_override)
    )
    if not np.isfinite(vertical_limit) or vertical_limit <= 0.0:
        raise ValueError(
            "vertical_force_limit_override must be finite and positive"
        )
    array = default_six_thruster_array(
        length=length,
        width=width,
        horizontal_force_limit=horizontal_limit,
        vertical_force_limit=vertical_limit,
    )
    return array, {
        "length_m": length,
        "width_m": width,
        "horizontal_force_limit_n": horizontal_limit,
        "vertical_force_limit_n": vertical_limit,
    }


def _depth_band_index_for_mission(
    scenario_index,
    repetition,
    missions_per_scenario,
):
    """Apply the 0--300 m focused depth mix independently to every split."""
    counts = _split_counts(missions_per_scenario)
    split = _split_for_repetition(repetition, missions_per_scenario)
    if split == "train":
        local_index = repetition
    elif split == "validation":
        local_index = repetition - counts["train"]
    else:
        local_index = repetition - counts["train"] - counts["validation"]
    split_index = int(scenario_index) * counts[split] + local_index
    return DEPTH_BAND_WEIGHTED_CYCLE[
        split_index % len(DEPTH_BAND_WEIGHTED_CYCLE)
    ]


def _mission_schedule(duration, rng, depth_band_index=None):
    """Build a local mission, with 300--500 m reserved for sparse stress runs."""
    duration = float(duration)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and positive")
    if depth_band_index is None:
        depth_band_index = int(rng.choice(
            len(DEPTH_BANDS_M),
            p=DEPTH_BAND_PROBABILITIES,
        ))
    depth_band_index = int(depth_band_index)
    if depth_band_index not in range(len(DEPTH_BANDS_M)):
        raise ValueError("depth_band_index is outside the configured bands")

    depth_low, depth_high = DEPTH_BANDS_M[depth_band_index]
    depth_margin = min(5.0, 0.05 * (depth_high - depth_low))
    working_low = depth_low + depth_margin
    working_high = depth_high - depth_margin
    initial_depth = rng.uniform(working_low, working_high)
    cruise_speed = float(rng.choice(
        CRUISE_SPEEDS_MPS,
        p=CRUISE_SPEED_PROBABILITIES,
    ))
    vertical_speed = float(rng.uniform(*VERTICAL_SPEED_RANGE_MPS))

    turn_duration = min(5.0, 0.08 * duration)
    cruise_duration = (duration - 3.0 * turn_duration) / 4.0
    ramp_duration = min(4.0, 0.25 * cruise_duration)
    effective_travel_time = cruise_duration - ramp_duration
    headings = [float(rng.uniform(-np.pi, np.pi))]
    for _ in range(3):
        headings.append(headings[-1] + float(rng.uniform(-0.55, 0.55)))
    initial_position = np.array([0.0, 0.0, initial_depth])
    waypoints = [(0.0, initial_position.copy(), np.array([
        0.0, 0.0, headings[0],
    ]))]

    first_vertical_direction = float(rng.choice((-1.0, 1.0)))
    vertical_directions = (
        first_vertical_direction,
        0.0,
        -first_vertical_direction,
        float(rng.choice((-1.0, 0.0, 1.0))),
    )
    current_time = 0.0
    current_position = initial_position.copy()
    for segment_index, vertical_direction in enumerate(vertical_directions):
        heading = headings[segment_index]
        horizontal_distance = cruise_speed * effective_travel_time
        depth_delta = (
            vertical_direction * vertical_speed * effective_travel_time
        )
        next_depth = np.clip(
            current_position[2] + depth_delta,
            working_low,
            working_high,
        )
        next_position = current_position + np.array([
            horizontal_distance * np.cos(heading),
            horizontal_distance * np.sin(heading),
            next_depth - current_position[2],
        ])
        current_time += cruise_duration
        waypoints.append((
            current_time,
            next_position.copy(),
            np.array([0.0, 0.0, heading]),
        ))
        current_position = next_position
        if segment_index < len(vertical_directions) - 1:
            current_time += turn_duration
            waypoints.append((
                current_time,
                current_position.copy(),
                np.array([0.0, 0.0, headings[segment_index + 1]]),
            ))

    waypoints = tuple(waypoints)
    return MissionPlan(
        waypoints=waypoints,
        initial_position_ned=initial_position,
        initial_euler_rpy=waypoints[0][2].copy(),
        depth_band_index=depth_band_index,
        depth_band_m=(depth_low, depth_high),
        cruise_speed_mps=cruise_speed,
        vertical_speed_mps=vertical_speed,
        ramp_duration_s=ramp_duration,
    )


def _trapezoidal_profile(elapsed_s, duration_s, ramp_duration_s):
    """Return normalized distance and its rate for a trapezoidal profile."""
    duration_s = float(duration_s)
    elapsed_s = float(np.clip(elapsed_s, 0.0, duration_s))
    ramp = float(np.clip(ramp_duration_s, 0.0, 0.5 * duration_s))
    if ramp <= 1e-12:
        return elapsed_s / duration_s, 1.0 / duration_s
    total_distance_scale = duration_s - ramp
    if elapsed_s < ramp:
        distance_scale = 0.5 * elapsed_s * elapsed_s / ramp
        rate_scale = elapsed_s / ramp
    elif elapsed_s <= duration_s - ramp:
        distance_scale = 0.5 * ramp + elapsed_s - ramp
        rate_scale = 1.0
    else:
        remaining = duration_s - elapsed_s
        distance_scale = total_distance_scale - 0.5 * remaining * remaining / ramp
        rate_scale = remaining / ramp
    return (
        float(distance_scale / total_distance_scale),
        float(rate_scale / total_distance_scale),
    )


def _target_provider(plan):
    schedule = plan.waypoints

    def provider(time_s, _state):
        if time_s <= schedule[0][0]:
            segment_index = 0
            progress = 0.0
            progress_rate = 0.0
        elif time_s >= schedule[-1][0]:
            segment_index = len(schedule) - 2
            progress = 1.0
            progress_rate = 0.0
        else:
            segment_index = 0
            for index in range(len(schedule) - 1):
                if schedule[index][0] <= time_s < schedule[index + 1][0]:
                    segment_index = index
                    break
            start_time = schedule[segment_index][0]
            end_time = schedule[segment_index + 1][0]
            progress, progress_rate = _trapezoidal_profile(
                time_s - start_time,
                end_time - start_time,
                plan.ramp_duration_s,
            )

        start = schedule[segment_index]
        end = schedule[segment_index + 1]
        position = (1.0 - progress) * start[1] + progress * end[1]
        linear_velocity_ned = progress_rate * (end[1] - start[1])
        attitude = (1.0 - progress) * start[2] + progress * end[2]
        yaw_delta = np.arctan2(
            np.sin(end[2][2] - start[2][2]),
            np.cos(end[2][2] - start[2][2]),
        )
        attitude[2] = start[2][2] + progress * yaw_delta
        return PoseTarget(
            position,
            attitude,
            guidance_context_id=segment_index,
            linear_velocity_ned=linear_velocity_ned,
        )

    return provider


def _disturbance_provider(
    rng,
    lateral_force_amplitudes_override=None,
):
    amplitudes = rng.uniform(
        low=np.zeros(6),
        high=np.array([1.5, 1.5, 0.8, 0.03, 0.03, 0.08]),
    )
    frequencies = rng.uniform(0.03, 0.12, size=6)
    phases = rng.uniform(-np.pi, np.pi, size=6)
    if lateral_force_amplitudes_override is not None:
        lateral = np.asarray(
            lateral_force_amplitudes_override, dtype=float
        )
        if (
            lateral.shape != (2,)
            or not np.all(np.isfinite(lateral))
            or np.any(lateral < 0.0)
        ):
            raise ValueError(
                "lateral_force_amplitudes_override must contain two "
                "finite nonnegative values"
            )
        amplitudes[:2] = lateral

    def provider(time_s, _state):
        return amplitudes * np.sin(frequencies * time_s + phases)

    return provider, {
        "amplitudes": amplitudes.tolist(),
        "frequencies_radps": frequencies.tolist(),
        "phases_rad": phases.tolist(),
    }


def run_mission(
    thruster_name,
    fault_mode,
    duration,
    dt,
    seed,
    split="train",
    depth_band_index=None,
    fault_seed=None,
    rpm_noise_std_override=None,
    lateral_force_amplitudes_override=None,
    vertical_force_limit_override=None,
    mass_kg_override=None,
):
    rng = np.random.default_rng(seed)
    dynamics, dynamics_metadata = _randomized_dynamics(
        rng,
        split,
        mass_kg_override=mass_kg_override,
    )
    thruster_array, thruster_metadata = _randomized_thrusters(
        rng,
        split,
        vertical_force_limit_override=vertical_force_limit_override,
    )
    fault = None
    if fault_mode is not None:
        fault_rng = (
            rng
            if fault_seed is None
            else np.random.default_rng(fault_seed)
        )
        fault = SingleThrusterFault(
            thruster_name=thruster_name,
            mode=fault_mode,
            start_time=fault_rng.uniform(0.35 * duration, 0.60 * duration),
            thrust_efficiency=(
                fault_rng.uniform(0.30, 0.70)
                if fault_mode is SixDOFThrusterFaultMode.THRUST_LOSS
                else 0.0
            ),
        )

    depth_noise = rng.uniform(0.02, 0.12)
    dvl_noise = rng.uniform(0.01, 0.08)
    dvl_dropout = rng.uniform(0.0, 0.06)
    current_noise = rng.uniform(0.02, 0.14)
    sampled_rpm_noise = rng.uniform(10.0, 70.0)
    rpm_noise = (
        sampled_rpm_noise
        if rpm_noise_std_override is None
        else float(rpm_noise_std_override)
    )
    if not np.isfinite(rpm_noise) or rpm_noise < 0.0:
        raise ValueError("rpm_noise_std_override must be finite and nonnegative")
    voltage_noise = rng.uniform(0.02, 0.15)
    temperature_noise = rng.uniform(0.05, 0.50)
    sensor_suite = SixDOFSensorSuite(
        depth_sensor=DepthSensor(
            noise_std=depth_noise,
            drift_std=rng.uniform(0.0003, 0.0020),
            seed=seed + 1,
        ),
        imu_sensor=IMUSensor(
            attitude_noise_std=rng.uniform(0.001, 0.005),
            gyro_noise_std=rng.uniform(0.0005, 0.003),
            accel_noise_std=rng.uniform(0.005, 0.03),
            seed=seed + 2,
        ),
        dvl_sensor=DVLSensor(
            velocity_noise_std=dvl_noise,
            dropout_prob=dvl_dropout,
            seed=seed + 3,
        ),
    )
    actuator_bank = ThrusterActuatorBank(
        thruster_array,
        fault=fault,
        idle_current=rng.uniform(0.32, 0.48),
        current_gain=rng.uniform(7.2, 8.8),
        no_output_current_fraction=rng.uniform(0.02, 0.08),
        current_noise_std=current_noise,
        max_rpm=rng.uniform(3300.0, 3700.0),
        no_output_rpm_fraction=rng.uniform(0.01, 0.06),
        rpm_noise_std=rpm_noise,
        nominal_voltage=rng.uniform(46.0, 50.0),
        voltage_droop_per_amp=rng.uniform(0.015, 0.030),
        voltage_noise_std=voltage_noise,
        ambient_temperature=rng.uniform(12.0, 25.0),
        full_load_temperature_rise=rng.uniform(25.0, 40.0),
        thermal_time_constant=rng.uniform(35.0, 60.0),
        temperature_noise_std=temperature_noise,
        seed=seed + 4,
    )
    disturbance_provider, disturbance_metadata = _disturbance_provider(
        rng,
        lateral_force_amplitudes_override=(
            lateral_force_amplitudes_override
        ),
    )
    simulator = SixDOFSimulator(
        dynamics=dynamics,
        thruster_array=thruster_array,
        actuator_bank=actuator_bank,
        sensor_suite=sensor_suite,
    )
    mission_plan = _mission_schedule(
        duration,
        rng,
        depth_band_index=depth_band_index,
    )
    simulator.reset(SixDOFState(
        position_ned=mission_plan.initial_position_ned,
        quaternion_nb=euler_to_quaternion(*mission_plan.initial_euler_rpy),
    ))
    logs = simulator.run(
        duration=duration,
        dt=dt,
        target_provider=_target_provider(mission_plan),
        disturbance_provider=disturbance_provider,
    )
    metadata = {
        "split": split,
        "domain": "independent_broad_deployment_domain",
        "dynamics": dynamics_metadata,
        "thrusters": thruster_metadata,
        "sensors": {
            "depth_noise_std_m": depth_noise,
            "dvl_noise_std_mps": dvl_noise,
            "dvl_dropout_probability": dvl_dropout,
            "current_noise_std_a": current_noise,
            "rpm_noise_std": rpm_noise,
            "voltage_noise_std_v": voltage_noise,
            "temperature_noise_std_c": temperature_noise,
        },
        "disturbance": disturbance_metadata,
        "mission": mission_plan.to_metadata(),
        "fault_start_time_s": None if fault is None else fault.start_time,
        "thrust_efficiency": (
            None if fault is None else fault.thrust_efficiency
        ),
    }
    return logs, metadata


def generate_dataset(
    missions_per_scenario,
    duration,
    dt,
    seq_len,
    stride,
    seed,
):
    chunks = []
    mission_metadata = {}
    mission_id = 0
    for scenario_index, (name, thruster_name, mode) in enumerate(
        scenario_definitions()
    ):
        for repetition in range(missions_per_scenario):
            split = _split_for_repetition(
                repetition, missions_per_scenario
            )
            split_namespace = {
                "train": 0,
                "validation": 1_000_000,
                "test": 2_000_000,
            }[split]
            mission_seed = (
                seed
                + split_namespace
                + scenario_index * 10_000
                + repetition
            )
            logs, parameter_metadata = run_mission(
                thruster_name=thruster_name,
                fault_mode=mode,
                duration=duration,
                dt=dt,
                seed=mission_seed,
                split=split,
                depth_band_index=_depth_band_index_for_mission(
                    scenario_index,
                    repetition,
                    missions_per_scenario,
                ),
            )
            chunk = build_six_dof_sequence_dataset(
                {mission_id: logs},
                seq_len=seq_len,
                stride=stride,
            )
            chunks.append(chunk)
            mission_metadata[mission_id] = {
                "scenario": name,
                "seed": mission_seed,
                "split": split,
                "parameters": parameter_metadata,
            }
            print(
                f"Mission {mission_id + 1:03d}/"
                f"{len(scenario_definitions()) * missions_per_scenario}: "
                f"{name}, {split}, windows={len(chunk['X'])}"
            )
            mission_id += 1

    array_keys = (
        "X",
        "y_mode",
        "y_location",
        "y_joint",
        "mission_ids",
        "window_end_times",
        "guidance_context_ids",
        "guidance_context_stable",
    )
    dataset = {
        key: np.concatenate([chunk[key] for chunk in chunks], axis=0)
        for key in array_keys
    }
    dataset.update({
        key: chunks[0][key]
        for key in (
            "feature_names",
            "raw_feature_dim",
            "model_input_dim",
            "sequence_length",
        )
    })
    return dataset, mission_metadata


def fixed_split_indices(mission_ids, mission_metadata):
    mission_ids = np.asarray(mission_ids, dtype=np.int64)
    split_indices = {}
    split_missions = {}
    for split in ("train", "validation", "test"):
        selected_missions = sorted(
            mission_id
            for mission_id, metadata in mission_metadata.items()
            if metadata["split"] == split
        )
        split_missions[split] = selected_missions
        split_indices[split] = np.flatnonzero(
            np.isin(mission_ids, selected_missions)
        )

    if set(split_missions["train"]) & set(split_missions["validation"]):
        raise RuntimeError("train/validation mission leakage")
    if set(split_missions["train"]) & set(split_missions["test"]):
        raise RuntimeError("train/test mission leakage")
    if set(split_missions["validation"]) & set(split_missions["test"]):
        raise RuntimeError("validation/test mission leakage")
    return split_indices


def _torch_payload(dataset, splits, mission_metadata):
    payload = {}
    for key, value in dataset.items():
        payload[key] = (
            torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        )
    payload.update({
        "dataset_version": "six_dof_hybrid_telemetry_0_300m_focus_v3",
        "label_format": "multitask_mode_and_location_with_joint_baseline",
        "split_policy": (
            "fixed mission seeds; 90% mission focus at 0-300 m and 10% "
            "deep-stress coverage at 300-500 m; broad randomized deployment "
            "domain with disjoint train/validation/test mission seeds"
        ),
        "depth_bands_m": [list(band) for band in DEPTH_BANDS_M],
        "depth_band_probabilities": DEPTH_BAND_PROBABILITIES.tolist(),
        "mode_names": FAULT_MODE_NAMES,
        "location_names": FAULT_LOCATION_NAMES,
        "joint_names": JOINT_FAULT_NAMES,
        "thruster_names": THRUSTER_NAMES,
        "mission_metadata": mission_metadata,
        "split_indices": {
            name: torch.from_numpy(indices.astype(np.int64))
            for name, indices in splits.items()
        },
    })
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--missions-per-scenario", type=int, default=20)
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "depth-sensor-fault-detection"
            / "depth_fault_detection"
            / "data"
            / "simulation_dataset_six_dof_hybrid_telemetry_0_300m_focus_v3.pth"
        ),
    )
    args = parser.parse_args()
    if args.missions_per_scenario < 3:
        parser.error("--missions-per-scenario must be at least 3")
    if args.duration <= 0.0 or args.dt <= 0.0:
        parser.error("--duration and --dt must be positive")

    dataset, mission_metadata = generate_dataset(
        missions_per_scenario=args.missions_per_scenario,
        duration=args.duration,
        dt=args.dt,
        seq_len=args.seq_len,
        stride=args.stride,
        seed=args.seed,
    )
    splits = fixed_split_indices(
        dataset["mission_ids"],
        mission_metadata,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_torch_payload(dataset, splits, mission_metadata), args.output)

    summary = {
        "missions": len(mission_metadata),
        "windows": int(len(dataset["X"])),
        "X_shape": list(dataset["X"].shape),
        "mode_labels": sorted(np.unique(dataset["y_mode"]).tolist()),
        "location_labels": sorted(np.unique(dataset["y_location"]).tolist()),
        "joint_labels": sorted(np.unique(dataset["y_joint"]).tolist()),
        "split_windows": {
            name: int(len(indices)) for name, indices in splits.items()
        },
        "split_missions": {
            name: int(len(np.unique(dataset["mission_ids"][indices])))
            for name, indices in splits.items()
        },
        "split_depth_band_missions": {
            split: {
                f"{low:g}-{high:g}m": sum(
                    1
                    for metadata in mission_metadata.values()
                    if metadata["split"] == split
                    and metadata["parameters"]["mission"][
                        "depth_band_index"
                    ] == band_index
                )
                for band_index, (low, high) in enumerate(DEPTH_BANDS_M)
            }
            for split in ("train", "validation", "test")
        },
        "test_domain": "independent_broad_deployment_domain",
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
