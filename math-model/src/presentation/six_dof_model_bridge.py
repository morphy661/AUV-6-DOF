"""Causal bridge from the frozen six-DOF model to presentation fields."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import numpy as np

from diagnosis.maintenance_health_decision import MaintenanceHealthDecision
from diagnosis.temporal_fault_decision import TemporalDecisionConfig
from presentation.advisory_context_gate import (
    AdvisoryContextGate,
    AdvisoryGateConfig,
)
from utils.six_dof_feature_extractor import (
    SIX_DOF_MODEL_INPUT_DIM,
    SIX_DOF_RAW_FEATURE_DIM,
    SIX_DOF_RAW_FEATURE_NAMES,
    extract_six_dof_features,
)


MODE_NAMES = ("normal", "no_output", "thrust_loss")
THRUSTER_NAMES = ("H1", "H2", "H3", "H4", "V1", "V2")


def _numpy(values):
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


class SixDOFModelBridge:
    """Run fixed-length model windows without using future samples.

    Model output is maintenance advice only. It is not fed into the FTC
    supervisor and cannot isolate a thruster.
    """

    def __init__(
        self,
        checkpoint_path,
        temporal_config_path,
        model_class,
        *,
        sequence_length=50,
        inference_stride=10,
        advisory_stabilization_time_s=3.0,
        device=None,
    ):
        import torch

        self.torch = torch
        self.checkpoint_path = Path(checkpoint_path)
        self.temporal_config_path = Path(temporal_config_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(self.checkpoint_path)
        if not self.temporal_config_path.exists():
            raise FileNotFoundError(self.temporal_config_path)
        self.sequence_length = int(sequence_length)
        self.inference_stride = int(inference_stride)
        if self.sequence_length <= 1 or self.inference_stride <= 0:
            raise ValueError("sequence_length and inference_stride must be positive")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        if int(checkpoint.get("input_dim", -1)) != SIX_DOF_MODEL_INPUT_DIM:
            raise ValueError("checkpoint model input dimension is incompatible")
        if int(checkpoint.get("raw_feature_dim", -1)) != SIX_DOF_RAW_FEATURE_DIM:
            raise ValueError("checkpoint raw feature dimension is incompatible")
        if tuple(checkpoint.get("feature_names", ())) != SIX_DOF_RAW_FEATURE_NAMES:
            raise ValueError("checkpoint feature schema is incompatible")
        self.model = model_class(
            input_dim=SIX_DOF_MODEL_INPUT_DIM,
            structured_fusion=bool(checkpoint.get("structured_fusion", True)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.mean = torch.as_tensor(
            _numpy(checkpoint["mean"]), dtype=torch.float32,
            device=self.device,
        )
        self.std = torch.as_tensor(
            _numpy(checkpoint["std"]), dtype=torch.float32,
            device=self.device,
        )
        temporal_config = TemporalDecisionConfig(**json.loads(
            self.temporal_config_path.read_text(encoding="utf-8")
        ))
        self.decision = MaintenanceHealthDecision(
            temporal_config=temporal_config
        )
        self.context_gate = AdvisoryContextGate(AdvisoryGateConfig(
            stabilization_time_s=float(advisory_stabilization_time_s)
        ))
        self.reset()

    def reset(self):
        self.features = deque(maxlen=self.sequence_length)
        self.previous_log = None
        self.sample_index = 0
        self.decision.reset()
        self.context_gate.reset()
        self.last_result = None

    def _reset_model_context(self):
        """Discard model history contaminated by a causal context transition."""

        self.features.clear()
        self.previous_log = None
        self.sample_index = 0
        self.decision.reset()
        self.last_result = None

    @staticmethod
    def _warmup(time_s, observed, required):
        return {
            "available": False,
            "updated": False,
            "time_s": float(time_s),
            "status": "window_warmup",
            "warmup_samples": int(observed),
            "required_samples": int(required),
            "health_level": 0,
            "health_state": "normal",
            "temporal_state": "normal",
            "probable_mode": 0,
            "probable_mode_name": "normal",
            "confirmed_mode": 0,
            "confirmed_mode_name": "normal",
            "fault_probability": 0.0,
            "suspected_group": "none",
            "group_confidence": 0.0,
            "location_confidence": "none",
            "candidates": [],
            "mode_probabilities": [1.0, 0.0, 0.0],
            "location_probabilities": [0.0] * 6,
            "action": "none",
            "record_event": False,
            "requires_ftc": False,
        }

    @staticmethod
    def _apply_context_gate(result, gate):
        output = dict(result)
        output.update({
            "advisory_gate_active": bool(gate.active),
            "advisory_gate_reasons": list(gate.reasons),
            "advisory_suppress_until_s": float(gate.suppress_until_s),
            "advisory_suppressed": False,
        })
        if not gate.active:
            return output

        output.update({
            "raw_health_level": int(output.get("health_level", 0)),
            "raw_health_state": str(output.get("health_state", "normal")),
            "raw_temporal_state": str(output.get("temporal_state", "normal")),
            "raw_probable_mode": int(output.get("probable_mode", 0)),
            "raw_probable_mode_name": str(
                output.get("probable_mode_name", "normal")
            ),
            "raw_fault_probability": float(
                output.get("fault_probability", 0.0)
            ),
            "raw_suspected_group": str(
                output.get("suspected_group", "none")
            ),
            "raw_group_confidence": float(
                output.get("group_confidence", 0.0)
            ),
            "raw_candidates": list(output.get("candidates", ())),
        })
        output["advisory_suppressed"] = bool(
            output["raw_health_level"] > 0
            or output["raw_probable_mode"] != 0
        )
        output.update({
            "health_level": 0,
            "health_state": "context_suppressed",
            "temporal_state": "normal",
            "probable_mode": 0,
            "probable_mode_name": "normal",
            "confirmed_mode": 0,
            "confirmed_mode_name": "normal",
            "fault_probability": 0.0,
            "suspected_group": "none",
            "group_confidence": 0.0,
            "location_confidence": "none",
            "candidates": [],
            "mode_probabilities": [1.0, 0.0, 0.0],
            "location_probabilities": [0.0] * 6,
            "action": "none",
            "record_event": False,
            "requires_ftc": False,
        })
        return output

    @staticmethod
    def _control_saturation(log):
        commands = np.asarray(
            log.get("commanded_thruster_forces", np.zeros(6)), dtype=float
        )
        limits = np.asarray(
            log.get("thruster_force_limits", np.ones(6)), dtype=float
        )
        if commands.shape != (6,) or limits.shape != (6,):
            return 0.0
        return float(np.max(np.clip(
            np.abs(commands) / np.maximum(np.abs(limits), 1e-6), 0.0, 1.0
        )))

    def _infer(self, log):
        torch = self.torch
        raw = np.asarray(self.features, dtype=np.float32)[None, :, :]
        difference = np.diff(raw, axis=1, prepend=raw[:, :1, :])
        augmented = np.concatenate((raw, difference), axis=-1)
        tensor = torch.as_tensor(
            augmented, dtype=torch.float32, device=self.device
        )
        normalized = (tensor - self.mean) / (self.std + 1e-8)
        with torch.no_grad():
            mode_logits, location_logits = self.model(normalized)
            mode = torch.softmax(mode_logits, dim=-1)[0].cpu().numpy()
            location = torch.softmax(location_logits, dim=-1)[0].cpu().numpy()
        result = self.decision.update(
            float(log.get("time", 0.0)),
            mode,
            location,
            tracking_error_ratio=float(
                log.get("ftc_tracking_error_ratio", 0.0)
            ),
            control_saturation_ratio=self._control_saturation(log),
        )
        probable_mode = int(result.probable_mode)
        confirmed_mode = int(result.confirmed_mode)
        return {
            "available": True,
            "updated": True,
            "time_s": float(log.get("time", 0.0)),
            "status": "model_inference",
            "warmup_samples": self.sequence_length,
            "required_samples": self.sequence_length,
            "health_level": int(result.health_level),
            "health_state": result.health_state,
            "temporal_state": result.temporal_state,
            "probable_mode": probable_mode,
            "probable_mode_name": MODE_NAMES[probable_mode],
            "confirmed_mode": confirmed_mode,
            "confirmed_mode_name": MODE_NAMES[confirmed_mode],
            "fault_probability": float(result.fault_probability),
            "suspected_group": result.suspected_group,
            "group_confidence": float(result.group_confidence),
            "location_confidence": result.location_confidence,
            "candidates": [
                {
                    "index": int(candidate.index),
                    "name": candidate.name,
                    "probability": float(candidate.probability),
                }
                for candidate in result.candidates
            ],
            "mode_probabilities": [
                float(value) for value in result.smoothed_mode_probabilities
            ],
            "location_probabilities": [
                float(value)
                for value in result.smoothed_location_probabilities
            ],
            "action": result.action,
            "record_event": bool(result.record_event),
            "requires_ftc": bool(result.requires_ftc),
        }

    def update(self, log):
        gate = self.context_gate.update(log)
        if gate.reset_model_context:
            self._reset_model_context()
        feature = extract_six_dof_features(log, self.previous_log)
        self.previous_log = log
        self.features.append(feature)
        self.sample_index += 1
        time_s = float(log.get("time", 0.0))
        if len(self.features) < self.sequence_length:
            warmup = self._warmup(
                time_s, len(self.features), self.sequence_length
            )
            if gate.active:
                warmup["status"] = "context_warmup"
            return self._apply_context_gate(warmup, gate)
        inference_due = (
            self.last_result is None
            or (self.sample_index - self.sequence_length)
            % self.inference_stride == 0
        )
        if inference_due:
            self.last_result = self._infer(log)
            gate = self.context_gate.mark_model_inference(time_s)
            return self._apply_context_gate(self.last_result, gate)
        held = dict(self.last_result)
        held.update({"updated": False, "time_s": time_s, "status": "held"})
        return self._apply_context_gate(held, gate)

    def enrich_logs(self, logs):
        self.reset()
        for log in logs:
            log["maintenance_diagnosis"] = self.update(log)
        return logs
