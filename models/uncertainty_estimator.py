"""不确定性估计模块：通过 Softplus 输出每个预测点的方差。

输出方差可用于：
- 量化模型对每个预测点的置信度
- 可视化不确定性椭圆
- 论文中图9的不确定性分析
"""

import torch
import torch.nn as nn


class UncertaintyEstimator(nn.Module):
    """预测不确定性估计器。

    从编码特征预测每个 (x, y) 预测点的方差（对角协方差）。
    使用 Softplus 激活确保方差为正。
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_modes: int = 3,
        pred_len: int = 30,
        dropout: float = 0.1,
        share_across_modes: bool = False,
    ):
        """初始化不确定性估计器。

        参数:
            hidden_dim: 隐藏层维度
            num_modes: 模态数量 K（仅在 share_across_modes=False 时区分）
            pred_len: 预测帧数
            dropout: Dropout 比率
            share_across_modes: 是否对所有模态共享不确定性头。
                默认 False（独立头），使每个模态输出各自不同的方差，
                让不确定性具备区分不同模态置信度的意义。
        """
        super().__init__()
        self.share_across_modes = share_across_modes
        self.num_modes = num_modes
        self.pred_len = pred_len
        self.output_dim = pred_len * 2  # 每预测点两个方差（x, y 独立）

        if share_across_modes:
            # 共享头：所有模态用同一个方差预测
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, self.output_dim),
            )
        else:
            # 独立头：每个模态有自己的方差预测
            self.uncertainty_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, self.output_dim),
                )
                for _ in range(num_modes)
            ])

        # Softplus 激活：确保方差 > 0
        self.softplus = nn.Softplus(beta=1.0)

    def forward(
        self, encoded_feature: torch.Tensor
    ) -> torch.Tensor:
        """前向传播。

        参数:
            encoded_feature: [B, hidden_dim] 编码后的场景表征

        返回:
            uncertainties: [B, K, T_pred, 2] 每预测点的方差（>= 0）
        """
        B = encoded_feature.shape[0]

        if self.share_across_modes:
            # 共享头：所有模态得到相同方差
            var_flat = self.uncertainty_head(encoded_feature)  # [B, T_pred*2]
            var = self.softplus(var_flat)
            var = var.view(B, self.pred_len, 2)  # [B, T_pred, 2]
            # 扩展到 K 个模态
            var = var.unsqueeze(1).expand(-1, self.num_modes, -1, -1)
        else:
            # 独立头：每个模态分别预测方差
            variances = []
            for head in self.uncertainty_heads:
                var_flat = head(encoded_feature)
                var = self.softplus(var_flat).view(B, self.pred_len, 2)
                variances.append(var)
            var = torch.stack(variances, dim=1)  # [B, K, T_pred, 2]

        return var
