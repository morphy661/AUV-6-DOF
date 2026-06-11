import torch
import torch.nn as nn
import torch.nn.functional as F


class AUVFaultDetector(nn.Module):
    """
    Stage-2 multi-sensor AUV fault diagnosis model.

    Input:
        x shape = [batch_size, seq_len, input_dim]

    For the current Stage-2 dataset:
        raw features: 20
        raw + first-order temporal difference: 40
        therefore input_dim = 40

    Architecture:
        1. Input LayerNorm
        2. Bi-LSTM temporal encoder
        3. Attention pooling over time
        4. Fully-connected classifier
    """

    def __init__(
            self,
            input_dim=40,
            seq_len=50,
            num_classes=9,
            hidden_size=128,
            num_layers=2,
            lstm_dropout=0.3,
            classifier_dropout=0.4
    ):
        super(AUVFaultDetector, self).__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # --------------------------------------------------
        # 1. Input normalization
        # --------------------------------------------------
        # This is useful for Stage-2 multi-sensor fusion because the model
        # receives depth, current, DVL, IMU, and battery-related features.
        self.input_norm = nn.LayerNorm(input_dim)

        # --------------------------------------------------
        # 2. Bi-LSTM temporal encoder
        # --------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0
        )

        lstm_output_dim = hidden_size * 2

        # --------------------------------------------------
        # 3. Attention mechanism
        # --------------------------------------------------
        self.attention_layer = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # --------------------------------------------------
        # 4. Classifier
        # --------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Expected input shape:
            [B, 50, 40]

        Output shape:
            [B, 9]
        """

        if x.dim() != 3:
            raise ValueError(
                f"Expected input with shape [batch, seq_len, input_dim], got {x.shape}"
            )

        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Input feature dimension mismatch: got {x.size(-1)}, "
                f"expected {self.input_dim}"
            )

        # Input normalization for stable multi-sensor fusion learning
        x = self.input_norm(x)

        # LSTM output shape: [B, seq_len, hidden_size * 2]
        lstm_out, _ = self.lstm(x)

        # Attention scores shape: [B, seq_len, 1]
        attn_scores = self.attention_layer(lstm_out)

        # Attention weights shape: [B, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)

        # Context vector shape: [B, hidden_size * 2]
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)

        # Class logits shape: [B, num_classes]
        out = self.classifier(context_vector)

        return out
