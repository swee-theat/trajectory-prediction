"""轨迹编码器：特征投影 → 多层 MLP 编码 → 注意力池化 → 场景表征向量。

数据流:
    输入 [B, T_obs, input_dim]
    → 特征投影 Linear(input_dim → hidden_dim) + LayerNorm + ReLU
    → N 层 MLPBlock（逐帧处理，共享权重）
    → 池化（注意力/均值/LSTM）
    → 输出 [B, hidden_dim]
"""

import torch
import torch.nn as nn
from .mlp_block import MLPBlock
from .attention_pooling import build_pooling


class TrajectoryEncoder(nn.Module):
    """轨迹编码器：将原始轨迹特征编码为固定维度的场景表征向量。

    设计原则：
    - MLP 层逐帧独立处理（不混合时间维度），保持 O(N×d²) 而非 O(N²×d)
    - 池化层负责时间维度的信息聚合
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        num_mlp_layers: int = 3,
        dropout: float = 0.1,
        pooling_type: str = "attention",
        use_temporal_conv: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 特征投影层
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 多层 MLP 编码（逐帧处理）
        self.mlp_layers = nn.ModuleList([
            MLPBlock(hidden_dim, hidden_dim, dropout)
            for _ in range(num_mlp_layers)
        ])

        # 时序混合层（depthwise conv1d，约 384 参数，捕获局部时序模式）
        self.use_temporal_conv = use_temporal_conv
        if use_temporal_conv:
            self.temporal_mixer = nn.Conv1d(
                hidden_dim, hidden_dim, kernel_size=3,
                padding=1, groups=hidden_dim,  # depthwise
            )

        # 时间维度池化
        self.pooling = build_pooling(pooling_type, hidden_dim)

    def forward(
        self, x: torch.Tensor, return_attn_weights: bool = False, pool: bool = True
    ) -> torch.Tensor:
        """前向传播。

        参数:
            x: [B, T_obs, input_dim] 历史轨迹特征
            return_attn_weights: 是否返回注意力权重
            pool: 是否池化；False 时返回池化前 [B, T, hidden_dim]（供车道线融合）

        返回:
            encoded: [B, hidden_dim]（pool=True）或 [B, T, hidden_dim]（pool=False）
            attn_weights: [B, T_obs] 或 None（仅 return_attn_weights=True 时返回）
        """
        # 特征投影 [B, T, input_dim] → [B, T, hidden_dim]
        x = self.feature_proj(x)

        # 多层 MLP 编码（逐帧，不混合时间维度）
        for mlp in self.mlp_layers:
            residual = x
            x = mlp(x)
            # 残差连接（如果维度匹配）
            if x.shape == residual.shape:
                x = x + residual

        # 时序混合（depthwise conv1d 在时间维度上做局部交互）
        if self.use_temporal_conv:
            # [B, T, H] → [B, H, T] → conv1d → [B, H, T] → [B, T, H]
            x_t = x.transpose(1, 2)  # [B, H, T]
            x_t = self.temporal_mixer(x_t)
            x = x_t.transpose(1, 2)  # [B, T, H]

        # 返回池化前特征 [B, T, hidden_dim]（供车道线融合拼接后统一池化）
        if not pool:
            return x, None

        # 时间维度池化 [B, T, hidden_dim] → [B, hidden_dim]
        pooled, attn_weights = self.pooling(x, return_weights=return_attn_weights)

        if return_attn_weights:
            return pooled, attn_weights
        return pooled, None
