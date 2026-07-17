"""Multi-task BiLSTM-attention detector for six-thruster AUV faults."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AUVSixDOFMultiTaskDetector(nn.Module):
    """Fuse vehicle motion and per-thruster telemetry without mixing early."""

    HYBRID_INPUT_DIM = 218
    BASELINE_RAW_DIM = 61
    HYBRID_RAW_DIM = 109
    EXTRA_RAW_DIM = HYBRID_RAW_DIM - BASELINE_RAW_DIM
    THRUSTER_COUNT = 6
    LOCAL_FEATURES_PER_THRUSTER = 2 * EXTRA_RAW_DIM // THRUSTER_COUNT

    def __init__(
        self,
        input_dim=218,
        num_fault_modes=3,
        num_locations=6,
        hidden_size=128,
        num_layers=2,
        lstm_dropout=0.3,
        classifier_dropout=0.4,
        structured_fusion=None,
        local_embedding_size=16,
        local_hidden_size=48,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_fault_modes = int(num_fault_modes)
        self.num_locations = int(num_locations)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.structured_fusion = (
            self.input_dim == self.HYBRID_INPUT_DIM
            if structured_fusion is None
            else bool(structured_fusion)
        )
        self.local_embedding_size = int(local_embedding_size)
        self.local_hidden_size = int(local_hidden_size)
        if min(
            self.input_dim,
            self.num_fault_modes,
            self.num_locations,
            self.hidden_size,
            self.num_layers,
            self.local_embedding_size,
            self.local_hidden_size,
        ) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.structured_fusion and self.input_dim != self.HYBRID_INPUT_DIM:
            raise ValueError(
                "structured fusion requires the 218-dimensional hybrid input"
            )

        if self.structured_fusion:
            baseline_input_dim = 2 * self.BASELINE_RAW_DIM
            self.global_input_norm = nn.LayerNorm(baseline_input_dim)
            self.global_lstm = nn.LSTM(
                input_size=baseline_input_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=lstm_dropout if self.num_layers > 1 else 0.0,
            )
            self.local_feature_projection = nn.Sequential(
                nn.Linear(
                    self.LOCAL_FEATURES_PER_THRUSTER,
                    24,
                ),
                nn.ReLU(),
                nn.Linear(24, self.local_embedding_size),
                nn.ReLU(),
            )
            self.local_lstm = nn.LSTM(
                input_size=(
                    self.THRUSTER_COUNT * self.local_embedding_size
                ),
                hidden_size=self.local_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            direct_input_dim = 3 * self.local_embedding_size
            self.direct_location_head = nn.Sequential(
                nn.Linear(direct_input_dim, 32),
                nn.ReLU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(32, 1),
            )
            self.direct_location_scale = nn.Parameter(torch.tensor(1.0))
            encoded_dim = 2 * (
                self.hidden_size + self.local_hidden_size
            )
        else:
            self.input_norm = nn.LayerNorm(self.input_dim)
            self.lstm = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=lstm_dropout if self.num_layers > 1 else 0.0,
            )
            encoded_dim = 2 * self.hidden_size
        self.mode_attention_layer = nn.Sequential(
            nn.Linear(encoded_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
        )
        self.location_attention_layer = nn.Sequential(
            nn.Linear(encoded_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
        )
        self.mode_projection = nn.Sequential(
            nn.Linear(encoded_dim, 128),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
        )
        self.location_projection = nn.Sequential(
            nn.Linear(encoded_dim, 128),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
        )
        self.mode_head = nn.Linear(64, self.num_fault_modes)
        self.location_head = nn.Linear(64, self.num_locations)

    def _structured_encode(self, x):
        raw_global = x[:, :, :self.BASELINE_RAW_DIM]
        difference_start = self.HYBRID_RAW_DIM
        difference_global = x[
            :,
            :,
            difference_start:difference_start + self.BASELINE_RAW_DIM,
        ]
        global_input = torch.cat((raw_global, difference_global), dim=-1)
        global_encoded, _ = self.global_lstm(
            self.global_input_norm(global_input)
        )

        raw_local = x[
            :,
            :,
            self.BASELINE_RAW_DIM:self.HYBRID_RAW_DIM,
        ]
        difference_local = x[
            :,
            :,
            difference_start + self.BASELINE_RAW_DIM:,
        ]
        batch_size, sequence_length, _ = raw_local.shape
        raw_local = raw_local.reshape(
            batch_size,
            sequence_length,
            -1,
            self.THRUSTER_COUNT,
        ).permute(0, 1, 3, 2)
        difference_local = difference_local.reshape(
            batch_size,
            sequence_length,
            -1,
            self.THRUSTER_COUNT,
        ).permute(0, 1, 3, 2)
        local_features = torch.cat((raw_local, difference_local), dim=-1)
        local_embeddings = self.local_feature_projection(local_features)
        local_sequence = local_embeddings.flatten(start_dim=2)
        local_encoded, _ = self.local_lstm(local_sequence)

        direct_statistics = torch.cat((
            local_embeddings.mean(dim=1),
            local_embeddings.amax(dim=1),
            local_embeddings[:, -1],
        ), dim=-1)
        direct_location_logits = self.direct_location_head(
            direct_statistics
        ).squeeze(-1)
        return (
            torch.cat((global_encoded, local_encoded), dim=-1),
            direct_location_logits,
        )

    def forward(self, x, return_attention=False):
        if x.ndim != 3:
            raise ValueError(
                "expected input shape [batch, sequence, input_dim], "
                f"got {tuple(x.shape)}"
            )
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"input feature dimension {x.size(-1)} does not match "
                f"configured dimension {self.input_dim}"
            )

        if self.structured_fusion:
            encoded, direct_location_logits = self._structured_encode(x)
        else:
            encoded, _ = self.lstm(self.input_norm(x))
            direct_location_logits = None
        mode_attention = F.softmax(self.mode_attention_layer(encoded), dim=1)
        location_attention = F.softmax(
            self.location_attention_layer(encoded), dim=1
        )
        mode_context = torch.sum(mode_attention * encoded, dim=1)
        location_context = torch.sum(location_attention * encoded, dim=1)
        mode_logits = self.mode_head(self.mode_projection(mode_context))
        location_logits = self.location_head(
            self.location_projection(location_context)
        )
        if direct_location_logits is not None:
            location_logits = (
                location_logits
                + self.direct_location_scale * direct_location_logits
            )

        if return_attention:
            return (
                mode_logits,
                location_logits,
                mode_attention.squeeze(-1),
                location_attention.squeeze(-1),
            )
        return mode_logits, location_logits


def combine_multitask_predictions(mode_predictions, location_predictions):
    """Convert mode/location predictions into the 13-class comparison label."""

    mode_predictions = torch.as_tensor(mode_predictions, dtype=torch.long)
    location_predictions = torch.as_tensor(
        location_predictions, dtype=torch.long, device=mode_predictions.device
    )
    if mode_predictions.shape != location_predictions.shape:
        raise ValueError("mode and location predictions must have matching shapes")

    joint = torch.zeros_like(mode_predictions)
    valid_location = (location_predictions >= 1) & (location_predictions <= 6)
    no_output = (mode_predictions == 1) & valid_location
    thrust_loss = (mode_predictions == 2) & valid_location
    joint[no_output] = location_predictions[no_output]
    joint[thrust_loss] = 6 + location_predictions[thrust_loss]
    return joint
