"""MLPBlock：线性层 + LayerNorm + ReLU + Dropout 的基础构建块。"""

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """单个 MLP 块: Linear → LayerNorm → ReLU → Dropout。

    应用于每个时间帧独立处理（不混合时间维度），
    因此使用 LayerNorm 而非 BatchNorm，确保推理时确定性。
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数:
            x: [..., in_dim] 输入特征

        返回:
            [..., out_dim] 输出特征
        """
        x = self.fc(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
