"""Calibrate causal health monitoring on validation missions."""

import argparse
import json
import sys
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
MATH_MODEL_SRC = REPO_ROOT / "math-model" / "src"
if str(MATH_MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MATH_MODEL_SRC))

from diagnosis.temporal_fault_decision import (
    TemporalDecisionConfig,
    apply_temporal_decision_layer,
)
from diagnosis.maintenance_health_decision import (
    apply_maintenance_decision_layer,
    maintenance_event_metrics,
)
from diagnosis.maintenance_ticket_policy import (
    MaintenanceTicketConfig,
    apply_maintenance_ticket_policy,
    extract_maintenance_ticket_evidence,
    maintenance_ticket_metrics,
)
from diagnosis.maintenance_log_policy import (
    MaintenanceLogConfig,
    apply_maintenance_log_policy,
    maintenance_log_metrics,
)
from model_six_dof_multitask import AUVSixDOFMultiTaskDetector
from train_six_dof_multitask import (
    class_weights,
    evaluate,
    load_and_validate_dataset,
    make_loader,
    operational_test_metrics,
    set_seed,
)
from utils.six_dof_feature_extractor import (
    FAULT_MODE_NAMES,
    JOINT_FAULT_NAMES,
    THRUSTER_NAMES,
)


def classification_metrics(predictions):
    mode_true = predictions["mode_true"]
    mode_pred = predictions["mode_pred"]
    location_true = predictions["location_true"]
    location_pred = predictions["location_pred"]
    joint_true = predictions["joint_true"]
    joint_pred = predictions["joint_pred"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        mode_true,
        mode_pred,
        labels=range(len(FAULT_MODE_NAMES)),
        zero_division=0,
    )
    thrust_mask = mode_true == 2
    no_output_mask = mode_true == 1
    return {
        "mode_macro_f1": float(f1_score(
            mode_true,
            mode_pred,
            labels=range(3),
            average="macro",
            zero_division=0,
        )),
        "location_macro_f1": float(f1_score(
            location_true,
            location_pred,
            labels=range(7),
            average="macro",
            zero_division=0,
        )),
        "joint_macro_f1": float(f1_score(
            joint_true,
            joint_pred,
            labels=range(len(JOINT_FAULT_NAMES)),
            average="macro",
            zero_division=0,
        )),
        "joint_accuracy": float(accuracy_score(joint_true, joint_pred)),
        "thrust_loss_mode_precision": float(precision[2]),
        "thrust_loss_mode_recall": float(recall[2]),
        "thrust_loss_mode_f1": float(f1[2]),
        "thrust_loss_exact_window_accuracy": float(np.mean(
            joint_pred[thrust_mask] == joint_true[thrust_mask]
        )),
        "no_output_exact_window_accuracy": float(np.mean(
            joint_pred[no_output_mask] == joint_true[no_output_mask]
        )),
        "thrust_loss_location_macro_f1": float(f1_score(
            joint_true[thrust_mask],
            joint_pred[thrust_mask],
            labels=range(7, 13),
            average="macro",
            zero_division=0,
        )),
    }


def evaluate_config(dataset, indices, raw_predictions, config):
    filtered = apply_temporal_decision_layer(
        dataset,
        indices,
        raw_predictions,
        config,
    )
    metrics = classification_metrics(filtered)
    operations = operational_test_metrics(dataset, indices, filtered)
    return filtered, metrics, operations


def _delay_or_penalty(value, penalty=20.0):
    return penalty if value is None else float(value)


def calibrate_mode_state(dataset, indices, raw_predictions):
    base = TemporalDecisionConfig(
        location_probability_threshold=0.0,
        location_confirmation_s=0.0,
    )
    records = []
    grid = product(
        (0.55, 0.65, 0.75, 0.85),
        (0.0, 1.25),
        (1.25, 2.50, 3.75, 5.00),
        (0.60, 0.75),
        (1.25, 2.50, 3.75),
        (0.0, 1.25),
    )
    for (
        enter_probability,
        no_output_duration,
        thrust_loss_duration,
        exit_probability,
        recovery_duration,
        smoothing_time,
    ) in grid:
        config = replace(
            base,
            enter_fault_probability=enter_probability,
            no_output_confirmation_s=no_output_duration,
            thrust_loss_confirmation_s=thrust_loss_duration,
            exit_normal_probability=exit_probability,
            recovery_confirmation_s=recovery_duration,
            probability_time_constant_s=smoothing_time,
        )
        _, metrics, operations = evaluate_config(
            dataset, indices, raw_predictions, config
        )
        delay = _delay_or_penalty(
            operations["mean_mode_detection_delay_s"]
        )
        detection_shortfall = max(
            0.0, 0.95 - operations["mode_detection_rate"]
        )
        score = (
            metrics["mode_macro_f1"]
            - 0.65 * operations["normal_window_false_alarm_rate"]
            - 0.008 * delay
            - 2.0 * detection_shortfall
        )
        records.append({
            "score": float(score),
            "config": config,
            "metrics": metrics,
            "operations": operations,
        })
    records.sort(key=lambda item: item["score"], reverse=True)
    return records[0], records[:5], len(records)


def calibrate_ticket_policy(dataset, indices, decisions):
    """Select conservative formal-ticket thresholds on validation missions."""

    base = MaintenanceTicketConfig(evidence_tail_steps=20)
    evidence = extract_maintenance_ticket_evidence(
        dataset, indices, base
    )
    records = []
    grid = product(
        (0.05, 0.10, 0.15),
        (0.10, 0.20, 0.30),
        (0.50, 0.70, 0.85),
        (0.0, 1.25, 2.50),
        (1.25, 2.50),
        (15.0, 30.0, 60.0),
    )
    for (
        minimum_excitation,
        minimum_motion_evidence,
        minimum_local_anomaly,
        confirmation_duration,
        recovery_duration,
        merge_gap,
    ) in grid:
        config = replace(
            base,
            minimum_excitation_ratio=minimum_excitation,
            minimum_thrust_loss_motion_evidence=minimum_motion_evidence,
            minimum_no_output_local_anomaly=minimum_local_anomaly,
            ticket_confirmation_s=confirmation_duration,
            ticket_recovery_s=recovery_duration,
            merge_gap_s=merge_gap,
        )
        ticket_decisions = apply_maintenance_ticket_policy(
            dataset,
            indices,
            decisions,
            config,
            ticket_evidence=evidence,
        )
        metrics = maintenance_ticket_metrics(
            dataset, indices, ticket_decisions
        )
        precision = metrics["ticket_event_precision"] or 0.0
        no_output_recall = metrics["no_output_ticket_recall"] or 0.0
        thrust_loss_recall = metrics["thrust_loss_ticket_recall"] or 0.0
        stable_rate = (
            metrics["single_ticket_per_detected_mission_rate"] or 0.0
        )
        delay = _delay_or_penalty(
            metrics["mean_ticket_detection_delay_s"], penalty=30.0
        )
        false_rate = metrics["false_tickets_per_hour"] or 0.0
        severe_shortfall = max(0.0, 0.95 - no_output_recall)
        score = (
            1.50 * precision
            + 1.50 * no_output_recall
            + 0.25 * thrust_loss_recall
            + 0.10 * stable_rate
            - 4.00 * severe_shortfall
            - 0.003 * delay
            - 0.010 * false_rate
        )
        records.append({
            "score": float(score),
            "config": config,
            "metrics": metrics,
        })
    records.sort(key=lambda item: item["score"], reverse=True)
    return records[0], records[:5], len(records), evidence


def _record_for_json(record):
    payload = {
        "score": record["score"],
        "config": asdict(record["config"]),
    }
    for name in ("metrics", "operations"):
        if name in record:
            payload[name] = record[name]
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            THIS_DIR
            / "data"
            / "simulation_dataset_six_dof_hybrid_telemetry.pth"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            THIS_DIR
            / "results"
            / "six_dof_hybrid_telemetry"
            / "best_model.pth"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            THIS_DIR
            / "results"
            / "six_dof_hybrid_telemetry_temporal"
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

    train_indices = split_indices["train"]
    train_modes = dataset["y_mode"][train_indices]
    train_locations = dataset["y_location"][train_indices]
    mode_weights, _ = class_weights(train_modes, len(FAULT_MODE_NAMES))
    location_weights, _ = class_weights(
        train_locations[train_modes != 0] - 1,
        len(THRUSTER_NAMES),
    )
    mode_loss_fn = torch.nn.CrossEntropyLoss(weight=mode_weights.to(device))
    location_loss_fn = torch.nn.CrossEntropyLoss(
        weight=location_weights.to(device)
    )

    predictions = {}
    for split in ("validation", "test"):
        loader = make_loader(
            dataset,
            split_indices[split],
            checkpoint["mean"],
            checkpoint["std"],
            args.batch_size,
            shuffle=False,
            seed=args.seed,
        )
        predictions[split] = evaluate(
            model,
            loader,
            device,
            mode_loss_fn,
            location_loss_fn,
        )

    validation_indices = split_indices["validation"].cpu().numpy()
    test_indices = split_indices["test"].cpu().numpy()
    mode_best, mode_top, mode_candidate_count = calibrate_mode_state(
        dataset,
        validation_indices,
        predictions["validation"],
    )
    # Exact six-way location is no longer a calibration objective.  The
    # location probabilities are retained only for group/Top-2 inspection
    # advice, while the selected policy is based on persistent fault events.
    selected_config = mode_best["config"]

    raw_validation_metrics = classification_metrics(
        predictions["validation"]
    )
    raw_validation_operations = operational_test_metrics(
        dataset,
        validation_indices,
        predictions["validation"],
    )
    raw_test_metrics = classification_metrics(predictions["test"])
    raw_test_operations = operational_test_metrics(
        dataset,
        test_indices,
        predictions["test"],
    )
    _, filtered_validation_metrics, filtered_validation_operations = (
        evaluate_config(
            dataset,
            validation_indices,
            predictions["validation"],
            selected_config,
        )
    )
    _, filtered_test_metrics, filtered_test_operations = evaluate_config(
        dataset,
        test_indices,
        predictions["test"],
        selected_config,
    )
    maintenance_validation = apply_maintenance_decision_layer(
        dataset,
        validation_indices,
        predictions["validation"],
        selected_config,
    )
    maintenance_test = apply_maintenance_decision_layer(
        dataset,
        test_indices,
        predictions["test"],
        selected_config,
    )
    maintenance_validation_metrics = maintenance_event_metrics(
        dataset, validation_indices, maintenance_validation
    )
    maintenance_test_metrics = maintenance_event_metrics(
        dataset, test_indices, maintenance_test
    )
    (
        ticket_best,
        ticket_top,
        ticket_candidate_count,
        validation_ticket_evidence,
    ) = calibrate_ticket_policy(
        dataset, validation_indices, maintenance_validation
    )
    selected_ticket_config = ticket_best["config"]
    maintenance_validation = apply_maintenance_ticket_policy(
        dataset,
        validation_indices,
        maintenance_validation,
        selected_ticket_config,
        ticket_evidence=validation_ticket_evidence,
    )
    maintenance_test = apply_maintenance_ticket_policy(
        dataset,
        test_indices,
        maintenance_test,
        selected_ticket_config,
    )
    log_config = MaintenanceLogConfig()
    maintenance_validation = apply_maintenance_log_policy(
        dataset,
        validation_indices,
        maintenance_validation,
        log_config,
    )
    maintenance_test = apply_maintenance_log_policy(
        dataset,
        test_indices,
        maintenance_test,
        log_config,
    )
    ticket_validation_metrics = maintenance_ticket_metrics(
        dataset, validation_indices, maintenance_validation
    )
    ticket_test_metrics = maintenance_ticket_metrics(
        dataset, test_indices, maintenance_test
    )
    graded_log_validation_metrics = maintenance_log_metrics(
        dataset, validation_indices, maintenance_validation
    )
    graded_log_test_metrics = maintenance_log_metrics(
        dataset, test_indices, maintenance_test
    )

    summary = {
        "device": str(device),
        "calibration_policy": (
            "temporal thresholds preserve causal raw health observations; "
            "formal ticket thresholds are selected on validation missions "
            "for severe no-output recall and maintenance precision; exact "
            "thruster location is not an optimization target; held-out OOD "
            "missions are evaluated only after selection"
        ),
        "maintenance_policy": {
            "normal": "no health event",
            "transient_observation": "record and observe; no FTC",
            "persistent_degradation": (
                "retain the raw health log; after excitation and independent "
                "motion evidence, count only qualified time from stable "
                "guidance-context windows toward an 8-second confirmation; "
                "cancel a recovered episode after 3.75 seconds but preserve "
                "it as an observation"
            ),
            "critical_fault": (
                "retain the raw health log and use the direct no-output path "
                "when excitation and local thruster telemetry agree"
            ),
            "recurrent_degradation": (
                "recovered thrust-loss episodes within one guidance context "
                "may create an intermittent advisory but never a formal "
                "ticket; context changes cut the recurrence history"
            ),
            "location_output": (
                "horizontal/vertical/uncertain group plus Top-2 candidates; "
                "candidate identity is advisory"
            ),
            "ticket_merge": (
                "same-mission and same-mode ticket segments separated by a "
                "short recovery are merged into one maintenance incident"
            ),
            "graded_log": (
                "retain all raw events; merge same-context and same-mode "
                "segments within five seconds; only formal thrust-loss and "
                "no-output tickets enter the operator attention queue"
            ),
        },
        "selected_config": asdict(selected_config),
        "selected_ticket_config": asdict(selected_ticket_config),
        "maintenance_log_config": asdict(log_config),
        "candidate_counts": {
            "mode_state": mode_candidate_count,
            "maintenance_ticket": ticket_candidate_count,
        },
        "validation": {
            "raw": {
                "metrics": raw_validation_metrics,
                "operations": raw_validation_operations,
            },
            "temporal": {
                "metrics": filtered_validation_metrics,
                "operations": filtered_validation_operations,
            },
            "maintenance_log": maintenance_validation_metrics,
            "graded_maintenance_log": graded_log_validation_metrics,
            "maintenance_tickets": ticket_validation_metrics,
        },
        "test": {
            "raw": {
                "metrics": raw_test_metrics,
                "operations": raw_test_operations,
            },
            "temporal": {
                "metrics": filtered_test_metrics,
                "operations": filtered_test_operations,
            },
            "maintenance_log": maintenance_test_metrics,
            "graded_maintenance_log": graded_log_test_metrics,
            "maintenance_tickets": ticket_test_metrics,
        },
        "top_mode_candidates": [
            _record_for_json(record) for record in mode_top
        ],
        "top_ticket_candidates": [
            _record_for_json(record) for record in ticket_top
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "temporal_decision_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "temporal_decision_config.json").write_text(
        json.dumps(asdict(selected_config), indent=2), encoding="utf-8"
    )
    (args.output_dir / "maintenance_ticket_config.json").write_text(
        json.dumps(asdict(selected_ticket_config), indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "maintenance_event_log.json").write_text(
        json.dumps({
            "validation": {
                "raw_health_events": maintenance_validation[
                    "maintenance_events"
                ],
                "graded_health_events": maintenance_validation[
                    "maintenance_graded_events"
                ],
                "operator_attention_events": maintenance_validation[
                    "maintenance_operator_events"
                ],
                "collapsed_observations": maintenance_validation[
                    "maintenance_observation_events"
                ],
                "background_traces": maintenance_validation[
                    "maintenance_trace_events"
                ],
                "formal_maintenance_tickets": maintenance_validation[
                    "maintenance_tickets"
                ],
                "pending_maintenance_observations": maintenance_validation[
                    "maintenance_pending_observations"
                ],
            },
            "test": {
                "raw_health_events": maintenance_test[
                    "maintenance_events"
                ],
                "graded_health_events": maintenance_test[
                    "maintenance_graded_events"
                ],
                "operator_attention_events": maintenance_test[
                    "maintenance_operator_events"
                ],
                "collapsed_observations": maintenance_test[
                    "maintenance_observation_events"
                ],
                "background_traces": maintenance_test[
                    "maintenance_trace_events"
                ],
                "formal_maintenance_tickets": maintenance_test[
                    "maintenance_tickets"
                ],
                "pending_maintenance_observations": maintenance_test[
                    "maintenance_pending_observations"
                ],
            },
        }, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
