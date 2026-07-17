"""Generate a leakage-safe six-thruster fault-diagnosis dataset."""

import argparse
import json
import sys
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
from environment.six_dof_dynamics import SixDOFDynamics
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


def _edge_scale(rng, lower_outer, lower_inner, upper_inner, upper_outer):
    """Sample outside the in-domain interval for a held-out OOD test."""
    if rng.random() < 0.5:
        return rng.uniform(lower_outer, lower_inner)
    return rng.uniform(upper_inner, upper_outer)


def _scale(rng, split, in_domain, held_out):
    if split != "test":
        return rng.uniform(*in_domain)
    return _edge_scale(rng, *held_out)


def _randomized_dynamics(rng, split):
    mass_scale = _scale(rng, split, (0.90, 1.10), (0.82, 0.90, 1.10, 1.18))
    inertia_scale = np.array([
        _scale(rng, split, (0.82, 1.18), (0.70, 0.82, 1.18, 1.30))
        for _ in range(3)
    ])
    added_mass_scale = np.array([
        _scale(rng, split, (0.80, 1.20), (0.65, 0.80, 1.20, 1.35))
        for _ in range(6)
    ])
    linear_damping_scale = np.array([
        _scale(rng, split, (0.75, 1.25), (0.60, 0.75, 1.25, 1.40))
        for _ in range(6)
    ])
    quadratic_damping_scale = np.array([
        _scale(rng, split, (0.75, 1.25), (0.60, 0.75, 1.25, 1.40))
        for _ in range(6)
    ])
    mass = 50.0 * mass_scale
    weight = mass * 9.81
    buoyancy_ratio = rng.uniform(
        0.995 if split != "test" else 0.990,
        1.005 if split != "test" else 1.010,
    )
    xy_offset = 0.010 if split != "test" else 0.018
    cg = np.array([
        rng.uniform(-xy_offset, xy_offset),
        rng.uniform(-xy_offset, xy_offset),
        rng.uniform(0.015, 0.030 if split != "test" else 0.035),
    ])
    cb = np.array([
        rng.uniform(-xy_offset, xy_offset),
        rng.uniform(-xy_offset, xy_offset),
        rng.uniform(-0.030 if split != "test" else -0.035, -0.015),
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


def _randomized_thrusters(rng, split):
    if split == "test":
        length = rng.uniform(1.00, 1.40)
        width = rng.uniform(0.48, 0.72)
        horizontal_limit = rng.uniform(34.0, 46.0)
        vertical_limit = rng.uniform(29.0, 41.0)
    else:
        length = rng.uniform(1.10, 1.30)
        width = rng.uniform(0.54, 0.66)
        horizontal_limit = rng.uniform(37.0, 43.0)
        vertical_limit = rng.uniform(32.0, 38.0)
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


def _mission_schedule(duration, rng):
    times = np.array([0.0, 0.25, 0.50, 0.75]) * duration
    positions = [
        np.array([0.0, 0.0, rng.uniform(1.5, 2.5)]),
        np.array([
            rng.uniform(3.5, 7.0),
            rng.uniform(-2.5, 2.5),
            rng.uniform(1.5, 3.5),
        ]),
        np.array([
            rng.uniform(2.0, 7.0),
            rng.uniform(2.5, 6.0) * rng.choice([-1.0, 1.0]),
            rng.uniform(2.0, 4.0),
        ]),
        np.array([
            rng.uniform(-1.0, 3.0),
            rng.uniform(-4.0, 4.0),
            rng.uniform(1.0, 3.0),
        ]),
    ]
    attitudes = [
        np.array([0.0, 0.0, rng.uniform(-0.2, 0.2)]),
        np.array([0.0, 0.0, rng.uniform(-np.pi, np.pi)]),
        np.array([0.0, 0.0, rng.uniform(-np.pi, np.pi)]),
        np.array([0.0, 0.0, rng.uniform(-np.pi, np.pi)]),
    ]
    return list(zip(times, positions, attitudes))


def _target_provider(schedule):
    def provider(time_s, _state):
        selected_index = 0
        for index, candidate in enumerate(schedule):
            if time_s >= candidate[0]:
                selected_index = index
            else:
                break
        selected = schedule[selected_index]
        return PoseTarget(
            selected[1],
            selected[2],
            guidance_context_id=selected_index,
        )

    return provider


def _disturbance_provider(rng):
    amplitudes = rng.uniform(
        low=np.zeros(6),
        high=np.array([1.5, 1.5, 0.8, 0.03, 0.03, 0.08]),
    )
    frequencies = rng.uniform(0.03, 0.12, size=6)
    phases = rng.uniform(-np.pi, np.pi, size=6)

    def provider(time_s, _state):
        return amplitudes * np.sin(frequencies * time_s + phases)

    return provider, {
        "amplitudes": amplitudes.tolist(),
        "frequencies_radps": frequencies.tolist(),
        "phases_rad": phases.tolist(),
    }


def run_mission(thruster_name, fault_mode, duration, dt, seed, split="train"):
    rng = np.random.default_rng(seed)
    dynamics, dynamics_metadata = _randomized_dynamics(rng, split)
    thruster_array, thruster_metadata = _randomized_thrusters(rng, split)
    fault = None
    if fault_mode is not None:
        fault = SingleThrusterFault(
            thruster_name=thruster_name,
            mode=fault_mode,
            start_time=rng.uniform(0.35 * duration, 0.60 * duration),
            thrust_efficiency=(
                rng.uniform(0.30, 0.70)
                if fault_mode is SixDOFThrusterFaultMode.THRUST_LOSS
                else 0.0
            ),
        )

    if split == "test":
        depth_noise = rng.uniform(0.08, 0.12)
        dvl_noise = rng.uniform(0.05, 0.08)
        dvl_dropout = rng.uniform(0.03, 0.06)
        current_noise = rng.uniform(0.08, 0.14)
        rpm_noise = rng.uniform(35.0, 70.0)
        voltage_noise = rng.uniform(0.08, 0.15)
        temperature_noise = rng.uniform(0.20, 0.50)
    else:
        depth_noise = rng.uniform(0.02, 0.08)
        dvl_noise = rng.uniform(0.01, 0.05)
        dvl_dropout = rng.uniform(0.0, 0.03)
        current_noise = rng.uniform(0.02, 0.08)
        rpm_noise = rng.uniform(10.0, 35.0)
        voltage_noise = rng.uniform(0.02, 0.08)
        temperature_noise = rng.uniform(0.05, 0.25)
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
    disturbance_provider, disturbance_metadata = _disturbance_provider(rng)
    simulator = SixDOFSimulator(
        dynamics=dynamics,
        thruster_array=thruster_array,
        actuator_bank=actuator_bank,
        sensor_suite=sensor_suite,
    )
    logs = simulator.run(
        duration=duration,
        dt=dt,
        target_provider=_target_provider(_mission_schedule(duration, rng)),
        disturbance_provider=disturbance_provider,
    )
    metadata = {
        "split": split,
        "domain": "held_out_ood" if split == "test" else "in_domain",
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
        "dataset_version": "six_dof_hybrid_telemetry_v3",
        "label_format": "multitask_mode_and_location_with_joint_baseline",
        "split_policy": (
            "fixed mission seeds; in-domain train/validation; "
            "held-out OOD test"
        ),
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
            / "simulation_dataset_six_dof_hybrid_telemetry.pth"
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
        "test_domain": "held_out_ood",
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
