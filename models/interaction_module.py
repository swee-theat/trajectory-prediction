"""交叉注意力交互编码模块：ego 特征作 query，邻居逐帧特征作 key/value。

数据流:
    ego_query [B, H] + neighbor_feats [B, K*T, H] + neighbor_mask [B, K]
    → 无 Q/K/V 投影的缩放点积注意力（直接加权求和原始邻居特征）
    → context [B, H]（本阶段不注入 ego，仅供交互编码需求项/可视化）

参数量（H=128）:
    norm    LayerNorm(128)      256
    合计                         256  ≤ 20K（计划预算）
    （已移除 v_proj 省 16,512：context 对 ego 无增益不注入，省下的预算给邻居独立预测头）
"""

import torch
import torch.nn as nn


class InteractionModule(nn.Module):
    """轻量级交叉注意力交互编码器。

    设计要点：
    - 无 Q/K/V 投影，ego 池化特征直接作 query，邻居逐帧特征作 key/value，
      直接加权求和，把新增参数压到约 256。
    - 邻居 mask 展开到逐帧（K*T），mask=0 的帧在 softmax 前置 -inf。
    - 全掩码守卫：无 OTHERS 邻居的样本 context 直接置 0，保证优雅退化为单车模型。
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 规范化交互上下文（约 256 参数）
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        ego_query: torch.Tensor,
        neighbor_feats: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> tuple:
        """前向传播。

        参数:
            ego_query:     [B, H] ego 池化特征（作 query）
            neighbor_feats: [B, K*T, H] 邻居逐帧编码特征（作 key/value）
            neighbor_mask: [B, K] 邻居槽位掩码，1 表示存在该邻居

        返回:
            context: [B, H] 交互上下文（无邻居样本置 0）
            attn_weights: [B, K*T] 逐帧注意力权重（用于身份混淆观察）
        """
        B = neighbor_feats.shape[0]
        H = neighbor_feats.shape[-1]
        K = neighbor_mask.shape[1]
        T = neighbor_feats.shape[1] // K  # 每邻居的观测帧数

        # 槽位掩码 [B, K] → 帧级掩码 [B, K*T]
        frame_mask = neighbor_mask.unsqueeze(-1).expand(B, K, T).reshape(B, K * T)

        # 无 Q/K 投影：ego_query 直接作 query，与原始邻居特征点积得分
        scores = (
            torch.bmm(neighbor_feats, ego_query.unsqueeze(-1)).squeeze(-1)
            / (H ** 0.5)
        )  # [B, K*T]

        # 屏蔽缺失邻居帧（softmax 前置 -inf）
        scores = scores.masked_fill(frame_mask == 0, float("-inf"))

        # 全掩码守卫：无邻居样本的 scores 全 -inf 会导致 softmax NaN，置 0 得均匀权重
        has_neighbor = frame_mask.any(dim=-1)  # [B] bool
        scores = scores.masked_fill(~has_neighbor.unsqueeze(-1), 0.0)

        attn_weights = torch.softmax(scores, dim=-1)  # [B, K*T]

        # 无 V 投影：直接加权求和原始邻居特征
        context = torch.bmm(attn_weights.unsqueeze(1), neighbor_feats).squeeze(1)  # [B, H]

        context = self.norm(context)
        context = self.dropout(context)

        # 无邻居样本 context 置 0，避免污染 ego（退化为单车模型）
        context = context * has_neighbor.float().unsqueeze(-1)

        return context, attn_weights
