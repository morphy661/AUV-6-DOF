import torch
import torch.nn as nn
import torch.nn.functional as F


class AUVFaultDetector(nn.Module):
    def __init__(self, input_dim=14, seq_len=50, num_classes=8):
        super(AUVFaultDetector, self).__init__()

        # 1. 砍掉 CNN！直接让双向 LSTM 读取完整的 25 帧原始数据
        # 这样任何微小的 Spike 和缓慢的 Drift 都会被按原样保留
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # 2. 注意力机制 (看懂哪里是故障，哪里是正常下潜)
        self.attention_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # 3. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: [B, 50, 10]

        # 直接输入 LSTM，输出形状 [B, 50, 256]
        lstm_out, _ = self.lstm(x)

        # 计算 50 个时间步的 Attention 权重
        # attn_scores: [B, 50, 1]
        attn_scores = self.attention_layer(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1)

        # 将权重与 LSTM 输出相乘，浓缩为一个包含核心上下文的向量
        # context_vector: [B, 256]
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)

        out = self.classifier(context_vector)
        return out