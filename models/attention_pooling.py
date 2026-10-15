"""注意力池化模块：学习 query 向量，通过 dot-product attention 聚合时间维度。

参数量: hidden_dim（一个可学习的 query 向量）≈ 128 参数。
"""

import torch
import torch.nn as nn
from typing import Optional


class AttentionPooling(nn.Module):
    """可学习的 query 向量 + dot-product attention + 加权求和池化。

    输入:
        x: [B, T, hidden_dim] 编码后的时间序列特征

    输出:
        pooled: [B, hidden_dim] 聚合后的场景表征向量
        attn_weights: [B, T] (可选) 注意力权重，用于可视化
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # 可学习的 query 向量
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> tuple:
        """前向传播。

        参数:
            x: [B, T, H] 编码特征
            return_weights: 是否返回注意力权重

        返回:
            (pooled [B, H], weights [B, T]) 或 (pooled [B, H], None)
        """
        # query: [H] → [1, H, 1]  与 x: [B, T, H] 做 batch 矩阵乘法
        q = self.query.unsqueeze(0).unsqueeze(-1)  # [1, H, 1]
        scores = torch.bmm(x, q.expand(x.shape[0], -1, -1)).squeeze(-1)  # [B, T]

        # 标准 dot-product attention 缩放: 1/sqrt(d_k)
        d_k = x.shape[-1]
        scores = scores / (d_k ** 0.5)

        # Softmax 归一化
        attn_weights = torch.softmax(scores, dim=-1)  # [B, T]

        # 加权求和
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [B, H]

        if return_weights:
            return pooled, attn_weights
        return pooled, None


class MeanPooling(nn.Module):
    """均值池化（消融实验用基准）：直接对时间维度取平均。"""

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> tuple:
        pooled = torch.mean(x, dim=1)  # [B, H]
        if return_weights:
            B, T = x.shape[:2]
            uniform_weights = torch.ones(B, T, device=x.device) / T
            return pooled, uniform_weights
        return pooled, None


class LSTMPooling(nn.Module):
    """LSTM 池化（消融实验用对比方案）：单层 LSTM + 最后隐状态。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> tuple:
        _, (h_n, _) = self.lstm(x)  # h_n: [1, B, H]
        pooled = h_n.squeeze(0)  # [B, H]
        if return_weights:
            return pooled, None  # LSTM 无显式权重
        return pooled, None


def build_pooling(pooling_type: str, hidden_dim: int) -> nn.Module:
    """根据配置构建池化模块。

    参数:
        pooling_type: "attention" | "mean" | "lstm"
        hidden_dim: 隐藏层维度

    返回:
        池化模块实例
    """
    if pooling_type == "attention":
        return AttentionPooling(hidden_dim)
    elif pooling_type == "mean":
        return MeanPooling()
    elif pooling_type == "lstm":
        return LSTMPooling(hidden_dim)
    else:
        raise ValueError(
            f"不支持的池化类型: {pooling_type}，可选: attention, mean, lstm"
        )
