"""Generate the frozen, one-shot six-DOF fault-diagnosis blind set."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
MATH_MODEL_ROOT = REPO_ROOT / "math-model"
MATH_MODEL_SRC = MATH_MODEL_ROOT / "src"
MATH_MODEL_EXAMPLES = MATH_MODEL_ROOT / "examples"
for path in (MATH_MODEL_SRC, MATH_MODEL_EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_six_dof_fault_dataset import run_mission, scenario_definitions
from utils.six_dof_dataset_builder import build_six_dof_sequence_dataset
from utils.six_dof_feature_extractor import (
    FAULT_LOCATION_NAMES,
    FAULT_MODE_NAMES,
    JOINT_FAULT_NAMES,
    THRUSTER_NAMES,
)


DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "six_dof_final_blind_protocol.json"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_and_verify_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("state") != "frozen_before_evaluation":
        raise ValueError("blind protocol is not frozen before evaluation")
    for artifact in protocol["frozen_artifacts"]:
        artifact_path = REPO_ROOT / artifact["path"]
        actual = _sha256(artifact_path)
        if actual != artifact["sha256"]:
            raise RuntimeError(
                f"frozen artifact hash mismatch: {artifact['path']}"
            )
    return protocol


def _concatenate(chunks):
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
    return dataset


def generate_blind_dataset(protocol):
    parameters = protocol["dataset"]
    repeats = int(parameters["repeats_per_scenario"])
    duration = float(parameters["duration_s"])
    dt = float(parameters["dt_s"])
    seq_len = int(parameters["sequence_length"])
    stride = int(parameters["stride"])
    base_seed = int(parameters["base_seed"])
    chunks = []
    mission_metadata = {}
    mission_id = 0
    definitions = scenario_definitions()
    for scenario_index, (name, thruster_name, mode) in enumerate(definitions):
        for repetition in range(repeats):
            mission_seed = base_seed + scenario_index * 10_000 + repetition
            logs, parameter_metadata = run_mission(
                thruster_name=thruster_name,
                fault_mode=mode,
                duration=duration,
                dt=dt,
                seed=mission_seed,
                split="test",
            )
            chunk = build_six_dof_sequence_dataset(
                {mission_id: logs},
                seq_len=seq_len,
                stride=stride,
            )
            chunks.append(chunk)
            parameter_metadata = dict(parameter_metadata)
            parameter_metadata["split"] = "final_blind_test"
            parameter_metadata["domain"] = "held_out_ood_final_blind"
            mission_metadata[mission_id] = {
                "scenario": name,
                "seed": mission_seed,
                "split": "final_blind_test",
                "parameters": parameter_metadata,
            }
            print(
                f"Mission {mission_id + 1:03d}/"
                f"{len(definitions) * repeats}: {name}, "
                f"windows={len(chunk['X'])}"
            )
            mission_id += 1

    dataset = _concatenate(chunks)
    payload = {
        key: (
            torch.from_numpy(value)
            if isinstance(value, np.ndarray) else value
        )
        for key, value in dataset.items()
    }
    payload.update({
        "dataset_version": parameters.get(
            "dataset_version", "six_dof_final_blind_v2"
        ),
        "label_format": "multitask_mode_and_location_with_joint_baseline",
        "split_policy": "all missions are frozen held-out OOD blind tests",
        "mode_names": FAULT_MODE_NAMES,
        "location_names": FAULT_LOCATION_NAMES,
        "joint_names": JOINT_FAULT_NAMES,
        "thruster_names": THRUSTER_NAMES,
        "mission_metadata": mission_metadata,
        "blind_protocol_id": protocol["protocol_id"],
        "blind_indices": torch.arange(len(dataset["X"]), dtype=torch.long),
    })
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol = _load_and_verify_protocol(args.protocol)
    output_path = REPO_ROOT / protocol["dataset"]["output_path"]
    if output_path.exists():
        raise FileExistsError(
            f"blind dataset already exists and will not be overwritten: "
            f"{output_path}"
        )

    payload = generate_blind_dataset(protocol)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    dataset_hash = _sha256(output_path)
    sidecar = output_path.with_suffix(output_path.suffix + ".sha256.json")
    sidecar.write_text(json.dumps({
        "protocol_id": protocol["protocol_id"],
        "dataset_path": str(output_path.relative_to(REPO_ROOT)),
        "dataset_bytes": output_path.stat().st_size,
        "dataset_sha256": dataset_hash,
        "mission_count": len(payload["mission_metadata"]),
        "window_count": len(payload["X"]),
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "sha256": dataset_hash,
        "missions": len(payload["mission_metadata"]),
        "windows": len(payload["X"]),
        "shape": list(payload["X"].shape),
    }, indent=2))


if __name__ == "__main__":
    main()
