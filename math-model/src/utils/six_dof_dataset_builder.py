"""Mission-safe sequence construction for six-thruster fault diagnosis."""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

try:
    from .six_dof_feature_extractor import (
        SIX_DOF_MODEL_INPUT_DIM,
        SIX_DOF_RAW_FEATURE_DIM,
        SIX_DOF_RAW_FEATURE_NAMES,
        extract_six_dof_fault_labels,
        extract_six_dof_features,
    )
except ImportError:
    from utils.six_dof_feature_extractor import (
        SIX_DOF_MODEL_INPUT_DIM,
        SIX_DOF_RAW_FEATURE_DIM,
        SIX_DOF_RAW_FEATURE_NAMES,
        extract_six_dof_fault_labels,
        extract_six_dof_features,
    )


def _iter_missions(missions: Any) -> Iterable[tuple[int, list[Mapping]]]:
    items = missions.items() if isinstance(missions, Mapping) else missions
    for mission_id, logs in items:
        yield int(mission_id), list(logs)


def build_six_dof_sequence_dataset(
    missions,
    seq_len: int = 50,
    stride: int = 10,
    include_transition_windows: bool = True,
) -> dict[str, Any]:
    """Build causal windows without ever crossing a mission boundary.

    Labels are taken from the final timestamp of each window.  Extra windows
    around label changes improve fault-onset coverage while remaining causal:
    every window contains only samples available at its prediction time.
    """

    seq_len = int(seq_len)
    stride = int(stride)
    if seq_len <= 0 or stride <= 0:
        raise ValueError("seq_len and stride must be positive")

    windows = []
    mode_labels = []
    location_labels = []
    joint_labels = []
    mission_ids = []
    end_times = []
    guidance_context_ids = []
    guidance_context_stable = []

    for mission_id, logs in _iter_missions(missions):
        if len(logs) < seq_len:
            continue
        features = np.stack([
            extract_six_dof_features(
                log,
                previous_log=(logs[index - 1] if index > 0 else None),
            )
            for index, log in enumerate(logs)
        ]).astype(np.float32)
        labels = [extract_six_dof_fault_labels(log) for log in logs]

        end_indices = set(range(seq_len - 1, len(logs), stride))
        end_indices.add(len(logs) - 1)
        if include_transition_windows:
            joint_series = np.array([label.joint for label in labels])
            changes = np.flatnonzero(np.diff(joint_series) != 0) + 1
            for change in changes:
                for offset in (0, seq_len // 4, seq_len // 2):
                    end_index = int(change + offset)
                    if seq_len - 1 <= end_index < len(logs):
                        end_indices.add(end_index)

        for end_index in sorted(end_indices):
            start_index = end_index - seq_len + 1
            if start_index < 0:
                continue
            label = labels[end_index]
            windows.append(features[start_index:end_index + 1])
            mode_labels.append(label.mode)
            location_labels.append(label.location)
            joint_labels.append(label.joint)
            mission_ids.append(mission_id)
            end_times.append(float(logs[end_index].get("time", end_index)))
            context_id = int(logs[end_index].get("guidance_context_id", 0))
            if context_id < 0:
                raise ValueError("guidance_context_id must be non-negative")
            guidance_context_ids.append(context_id)
            window_contexts = [
                int(logs[index].get("guidance_context_id", 0))
                for index in range(start_index, end_index + 1)
            ]
            guidance_context_stable.append(bool(
                all(value == context_id for value in window_contexts)
            ))

    if not windows:
        empty_x = np.empty(
            (0, seq_len, SIX_DOF_RAW_FEATURE_DIM), dtype=np.float32
        )
        empty_y = np.empty((0,), dtype=np.int64)
        return {
            "X": empty_x,
            "y_mode": empty_y.copy(),
            "y_location": empty_y.copy(),
            "y_joint": empty_y.copy(),
            "mission_ids": empty_y.copy(),
            "window_end_times": np.empty((0,), dtype=np.float32),
            "guidance_context_ids": empty_y.copy(),
            "guidance_context_stable": np.empty((0,), dtype=bool),
            "feature_names": SIX_DOF_RAW_FEATURE_NAMES,
            "raw_feature_dim": SIX_DOF_RAW_FEATURE_DIM,
            "model_input_dim": SIX_DOF_MODEL_INPUT_DIM,
            "sequence_length": seq_len,
        }

    return {
        "X": np.asarray(windows, dtype=np.float32),
        "y_mode": np.asarray(mode_labels, dtype=np.int64),
        "y_location": np.asarray(location_labels, dtype=np.int64),
        "y_joint": np.asarray(joint_labels, dtype=np.int64),
        "mission_ids": np.asarray(mission_ids, dtype=np.int64),
        "window_end_times": np.asarray(end_times, dtype=np.float32),
        "guidance_context_ids": np.asarray(
            guidance_context_ids, dtype=np.int64
        ),
        "guidance_context_stable": np.asarray(
            guidance_context_stable, dtype=bool
        ),
        "feature_names": SIX_DOF_RAW_FEATURE_NAMES,
        "raw_feature_dim": SIX_DOF_RAW_FEATURE_DIM,
        "model_input_dim": SIX_DOF_MODEL_INPUT_DIM,
        "sequence_length": seq_len,
    }


def _mission_scenario_labels(
    mission_ids: np.ndarray,
    joint_labels: np.ndarray,
) -> dict[int, int]:
    result = {}
    for mission_id in np.unique(mission_ids):
        labels = np.unique(joint_labels[mission_ids == mission_id])
        fault_labels = labels[labels != 0]
        if len(fault_labels) > 1:
            raise ValueError(
                f"mission {mission_id} contains multiple fault scenarios"
            )
        result[int(mission_id)] = (
            int(fault_labels[0]) if len(fault_labels) else 0
        )
    return result


def stratified_mission_split(
    mission_ids,
    joint_labels,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Split whole missions, stratified by their injected fault scenario."""

    mission_ids = np.asarray(mission_ids, dtype=np.int64)
    joint_labels = np.asarray(joint_labels, dtype=np.int64)
    if mission_ids.shape != joint_labels.shape or mission_ids.ndim != 1:
        raise ValueError("mission_ids and joint_labels must be matching vectors")
    if len(mission_ids) == 0:
        raise ValueError("cannot split an empty dataset")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be within (0, 1)")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be within (0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions must sum to less than 1")

    scenario_by_mission = _mission_scenario_labels(mission_ids, joint_labels)
    rng = np.random.default_rng(seed)
    split_missions = {"train": set(), "validation": set(), "test": set()}

    scenario_values = sorted(set(scenario_by_mission.values()))
    for scenario in scenario_values:
        group = np.array([
            mission_id
            for mission_id, value in scenario_by_mission.items()
            if value == scenario
        ], dtype=np.int64)
        if len(group) < 3:
            raise ValueError(
                f"fault scenario {scenario} needs at least 3 missions for a "
                "leakage-free train/validation/test split"
            )
        rng.shuffle(group)
        test_count = max(1, int(round(len(group) * test_fraction)))
        validation_count = max(
            1, int(round(len(group) * validation_fraction))
        )
        if test_count + validation_count >= len(group):
            validation_count = 1
            test_count = 1

        split_missions["test"].update(group[:test_count].tolist())
        split_missions["validation"].update(
            group[test_count:test_count + validation_count].tolist()
        )
        split_missions["train"].update(
            group[test_count + validation_count:].tolist()
        )

    if (
        split_missions["train"] & split_missions["validation"]
        or split_missions["train"] & split_missions["test"]
        or split_missions["validation"] & split_missions["test"]
    ):
        raise RuntimeError("mission leakage detected while building splits")

    return {
        name: np.flatnonzero(np.isin(mission_ids, sorted(values)))
        for name, values in split_missions.items()
    }


def add_temporal_differences(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[-1] != SIX_DOF_RAW_FEATURE_DIM:
        raise ValueError(
            "expected X with shape (N, seq_len, "
            f"{SIX_DOF_RAW_FEATURE_DIM}), got {X.shape}"
        )
    difference = np.diff(X, axis=1, prepend=X[:, :1, :])
    return np.concatenate([X, difference], axis=-1).astype(np.float32)


def fit_standardizer(X_train: np.ndarray) -> dict[str, np.ndarray]:
    augmented = add_temporal_differences(X_train)
    return {
        "mean": augmented.mean(axis=(0, 1)).astype(np.float32),
        "std": augmented.std(axis=(0, 1)).astype(np.float32),
    }


def apply_standardizer(
    X: np.ndarray,
    stats: Mapping[str, np.ndarray],
    epsilon: float = 1e-8,
) -> np.ndarray:
    augmented = add_temporal_differences(X)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    expected_shape = (SIX_DOF_MODEL_INPUT_DIM,)
    if mean.shape != expected_shape or std.shape != expected_shape:
        raise ValueError(
            f"standardizer statistics must have shape {expected_shape}"
        )
    return ((augmented - mean) / (std + epsilon)).astype(np.float32)
