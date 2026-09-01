"""Counterfactual feature ablation for the frozen 0-300 m Focus V3 model.

This is a diagnostic attribution experiment, not a replacement-model result.
Each selected normalized input channel is set to zero (its training-reference
value) for both the raw signal and its temporal difference.  The checkpoint,
labels, decision thresholds, temporal policy, ticket policy, and FTC logic stay
unchanged.  Validation is the primary localisation split; test is reported only
as a descriptive replication and must not be used to tune a later model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
MATH_MODEL_SRC = REPO_ROOT / "math-model" / "src"
if str(MATH_MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MATH_MODEL_SRC))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from audit_six_dof_false_alarms import _build_split_audit  # noqa: E402
from diagnosis.maintenance_ticket_policy import (  # noqa: E402
    MaintenanceTicketConfig,
)
from diagnosis.temporal_fault_decision import (  # noqa: E402
    TemporalDecisionConfig,
)
from model_six_dof_multitask import AUVSixDOFMultiTaskDetector  # noqa: E402
from train_six_dof_multitask import (  # noqa: E402
    NormalizedWindowDataset,
    class_weights,
    evaluate,
    load_and_validate_dataset,
    set_seed,
)
from utils.six_dof_feature_extractor import (  # noqa: E402
    FAULT_MODE_NAMES,
    SIX_DOF_RAW_FEATURE_DIM,
    THRUSTER_NAMES,
)


ABLATION_GROUPS = {
    "baseline": (),
    "motion_loss_evidence": tuple(
        f"{name}_motion_loss_evidence" for name in THRUSTER_NAMES
    ),
    "simplified_response_residual": (
        "nominal_expected_accel_x_mps2",
        "nominal_expected_accel_y_mps2",
        "nominal_expected_accel_z_mps2",
        "linear_response_residual_x_mps2",
        "linear_response_residual_y_mps2",
        "linear_response_residual_z_mps2",
        "nominal_expected_angular_accel_p_radps2",
        "nominal_expected_angular_accel_q_radps2",
        "nominal_expected_angular_accel_r_radps2",
    ),
    "tracking_error": (
        "depth_tracking_error_m",
        "attitude_error_roll_rad",
        "attitude_error_pitch_rad",
        "attitude_error_yaw_rad",
    ),
}
ABLATION_GROUPS["motion_plus_response"] = tuple(dict.fromkeys(
    ABLATION_GROUPS["motion_loss_evidence"]
    + ABLATION_GROUPS["simplified_response_residual"]
))
ABLATION_GROUPS["motion_response_tracking"] = tuple(dict.fromkeys(
    ABLATION_GROUPS["motion_plus_response"]
    + ABLATION_GROUPS["tracking_error"]
))


class MaskedNormalizedWindowDataset(NormalizedWindowDataset):
    """Set selected normalized raw and difference channels to zero."""

    def __init__(self, *args, masked_raw_indices=(), **kwargs):
        super().__init__(*args, **kwargs)
        raw_indices = tuple(int(index) for index in masked_raw_indices)
        self.masked_input_indices = torch.as_tensor(
            raw_indices
            + tuple(index + SIX_DOF_RAW_FEATURE_DIM for index in raw_indices),
            dtype=torch.long,
        )

    def __getitem__(self, item):
        values, mode, location = super().__getitem__(item)
        if len(self.masked_input_indices):
            values = values.clone()
            values[:, self.masked_input_indices] = 0.0
        return values, mode, location


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _masked_loader(
    dataset,
    indices,
    checkpoint,
    masked_raw_indices,
    batch_size,
):
    windows = MaskedNormalizedWindowDataset(
        dataset["X"],
        dataset["y_mode"].long(),
        dataset["y_location"].long(),
        indices,
        checkpoint["mean"],
        checkpoint["std"],
        masked_raw_indices=masked_raw_indices,
    )
    return DataLoader(
        windows,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def _window_metrics(predictions):
    true = np.asarray(predictions["mode_true"], dtype=int)
    predicted = np.asarray(predictions["mode_pred"], dtype=int)

    def recall(label):
        selected = true == label
        return float(np.mean(predicted[selected] == label))

    fault = true != 0
    return {
        "normal_recall": recall(0),
        "no_output_recall": recall(1),
        "thrust_loss_recall": recall(2),
        "fault_detection_recall": float(np.mean(predicted[fault] != 0)),
        "fault_mode_recall": float(np.mean(predicted[fault] == true[fault])),
    }


def _compact_split_result(predictions, audit):
    raw = audit["summary"]["raw"]
    temporal = audit["summary"]["temporal"]
    operator = audit["summary"]["operator"]
    ticket = audit["summary"]["ticket"]
    formal = audit["summary"]["formal_ticket_audit"]
    return {
        "mode_macro_f1": float(predictions["mode_macro_f1"]),
        "location_macro_f1": float(predictions["location_macro_f1"]),
        "joint_macro_f1": float(predictions["joint_macro_f1"]),
        "window_metrics": _window_metrics(predictions),
        "false_alarm": {
            "raw": float(raw["false_alarm_rate"]),
            "temporal": float(temporal["false_alarm_rate"]),
            "operator_attention_window_coverage": float(
                operator["false_alarm_rate"]
            ),
            "formal_ticket_window_coverage": float(
                ticket["false_alarm_rate"]
            ),
            "wholly_normal_mission_raw": float(
                raw["by_mission_kind"]["normal_mission"][
                    "false_alarm_rate"
                ]
            ),
            "pre_fault_segment_raw": float(
                raw["by_mission_kind"]["pre_fault_segment"][
                    "false_alarm_rate"
                ]
            ),
            "raw_episode_count": int(raw["episode_count"]),
            "raw_episodes_at_least_8_s": int(
                raw["episodes_at_least_8_s"]
            ),
            "temporal_episode_count": int(temporal["episode_count"]),
            "temporal_episodes_at_least_8_s": int(
                temporal["episodes_at_least_8_s"]
            ),
        },
        "formal_ticket_metrics": formal["ticket_metrics"],
        "graded_log_metrics": formal["graded_log_metrics"],
        "wholly_normal_missions": [
            {
                "mission_id": int(record["mission_id"]),
                "depth_band": record["depth_band"],
                "normal_windows": int(record["normal_windows"]),
                "raw_false_windows": int(record["raw_false_windows"]),
                "raw_false_alarm_rate": float(
                    record["raw_false_alarm_rate"]
                ),
                "temporal_false_alarm_rate": float(
                    record["temporal_false_alarm_rate"]
                ),
                "operator_false_windows": int(
                    record["operator_false_windows"]
                ),
                "ticket_false_windows": int(
                    record["ticket_false_windows"]
                ),
            }
            for record in audit["mission_summary"]
            if record["mission_kind"] == "normal_mission"
        ],
    }


def _csv_rows(result):
    rows = []
    baseline = result["variants"]["baseline"]
    for variant, variant_result in result["variants"].items():
        for split in ("validation", "test"):
            values = variant_result[split]
            false_alarm = values["false_alarm"]
            windows = values["window_metrics"]
            tickets = values["formal_ticket_metrics"]
            baseline_far = baseline[split]["false_alarm"]["raw"]
            rows.append({
                "variant": variant,
                "split": split,
                "masked_raw_feature_count": len(
                    variant_result["masked_raw_features"]
                ),
                "raw_false_alarm_rate": false_alarm["raw"],
                "raw_far_change_percentage_points": 100.0 * (
                    false_alarm["raw"] - baseline_far
                ),
                "temporal_false_alarm_rate": false_alarm["temporal"],
                "wholly_normal_mission_raw_far": false_alarm[
                    "wholly_normal_mission_raw"
                ],
                "raw_episode_count": false_alarm["raw_episode_count"],
                "raw_episodes_at_least_8_s": false_alarm[
                    "raw_episodes_at_least_8_s"
                ],
                "mode_macro_f1": values["mode_macro_f1"],
                "joint_macro_f1": values["joint_macro_f1"],
                "normal_recall": windows["normal_recall"],
                "no_output_recall": windows["no_output_recall"],
                "thrust_loss_recall": windows["thrust_loss_recall"],
                "fault_detection_recall": windows[
                    "fault_detection_recall"
                ],
                "formal_ticket_count": tickets[
                    "formal_maintenance_ticket_count"
                ],
                "false_maintenance_tickets": tickets[
                    "false_maintenance_tickets"
                ],
                "no_output_ticket_recall": tickets[
                    "no_output_ticket_recall"
                ],
                "thrust_loss_ticket_recall": tickets[
                    "thrust_loss_ticket_recall"
                ],
            })
    return rows


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            THIS_DIR / "data" /
            "simulation_dataset_six_dof_hybrid_telemetry_0_300m_focus_v3.pth"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            THIS_DIR / "results" /
            "six_dof_hybrid_telemetry_0_300m_focus_v3" / "best_model.pth"
        ),
    )
    parser.add_argument(
        "--temporal-config",
        type=Path,
        default=(
            THIS_DIR / "results" /
            "six_dof_hybrid_telemetry_0_300m_focus_v3_temporal" /
            "temporal_decision_config.json"
        ),
    )
    parser.add_argument(
        "--ticket-config",
        type=Path,
        default=(
            THIS_DIR / "results" /
            "six_dof_hybrid_telemetry_0_300m_focus_v3_temporal" /
            "maintenance_ticket_config.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "math-model" / "results" /
            "six_dof_feature_ablation_v1_20260821"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, split_indices, _ = load_and_validate_dataset(args.dataset)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model = AUVSixDOFMultiTaskDetector(
        input_dim=int(checkpoint["input_dim"]),
        structured_fusion=bool(checkpoint.get("structured_fusion", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_modes = dataset["y_mode"][split_indices["train"]]
    train_locations = dataset["y_location"][split_indices["train"]]
    mode_weights, _ = class_weights(train_modes, len(FAULT_MODE_NAMES))
    location_weights, _ = class_weights(
        train_locations[train_modes != 0] - 1, len(THRUSTER_NAMES)
    )
    mode_loss = torch.nn.CrossEntropyLoss(weight=mode_weights.to(device))
    location_loss = torch.nn.CrossEntropyLoss(
        weight=location_weights.to(device)
    )
    temporal_config = TemporalDecisionConfig(**_load_json(
        args.temporal_config
    ))
    ticket_config = MaintenanceTicketConfig(**_load_json(
        args.ticket_config
    ))

    feature_names = tuple(dataset["feature_names"])
    variants = {}
    for variant, masked_names in ABLATION_GROUPS.items():
        missing = sorted(set(masked_names) - set(feature_names))
        if missing:
            raise ValueError(f"{variant} has unknown features: {missing}")
        masked_indices = tuple(
            feature_names.index(name) for name in masked_names
        )
        variants[variant] = {
            "masked_raw_features": list(masked_names),
        }
        for split in ("validation", "test"):
            predictions = evaluate(
                model,
                _masked_loader(
                    dataset,
                    split_indices[split],
                    checkpoint,
                    masked_indices,
                    args.batch_size,
                ),
                device,
                mode_loss,
                location_loss,
            )
            audit = _build_split_audit(
                dataset,
                split,
                split_indices[split].cpu().numpy(),
                predictions,
                temporal_config,
                ticket_config,
            )
            variants[variant][split] = _compact_split_result(
                predictions, audit
            )

    result = {
        "experiment_id": "six_dof_feature_ablation_v1_20260821",
        "experiment_type": "frozen-model_counterfactual_input_masking",
        "interpretation_limit": (
            "localises frozen-model feature reliance; does not estimate the "
            "performance of a model retrained without the masked features"
        ),
        "selection_policy": (
            "validation is primary; test is descriptive replication only and "
            "must not be used for later model or threshold tuning"
        ),
        "device": str(device),
        "dataset_version": dataset["dataset_version"],
        "dataset_sha256": _sha256(args.dataset),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "temporal_config": asdict(temporal_config),
        "ticket_config": asdict(ticket_config),
        "variants": variants,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "feature_ablation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = _csv_rows(result)
    _write_csv(args.output_dir / "feature_ablation_summary.csv", rows)

    compact = [
        {
            "variant": row["variant"],
            "split": row["split"],
            "raw_far": row["raw_false_alarm_rate"],
            "delta_pp": row["raw_far_change_percentage_points"],
            "mode_f1": row["mode_macro_f1"],
            "no_output_recall": row["no_output_recall"],
            "thrust_loss_recall": row["thrust_loss_recall"],
            "thrust_loss_ticket_recall": row[
                "thrust_loss_ticket_recall"
            ],
        }
        for row in rows
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
