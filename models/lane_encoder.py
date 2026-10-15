"""车道线图卷积编码器（GCN）：把车道线图编码为节点特征，供注意力池化阶段融合。

数据流:
    输入 nodes [B, N, 6] + adj [B, N, N] + node_mask [B, N]
    → 节点投影 Linear(6→128) + LayerNorm + ReLU
    → 3 层 GCN（邻接归一化消息传递 D^-1/2 A D^-1/2 + Linear(128→128) + 残差）
    → 输出节点特征 [B, N, 128]（不池化，保留给融合用）

设计要点:
    - 纯 PyTorch 实现（不引入 torch-geometric），手写稠密邻接归一化消息传递。
    - padding 节点（特征全 0）不参与消息传递（邻接行列置 0），输出乘 node_mask 置 0。
    - 参数量 ≈ 节点投影 1,152 + 3×GCN层(16,768) ≈ 51K，符合需求文档"GCN 约 50K"。
"""

import torch
import torch.nn as nn
from typing import Optional


class GCNLayer(nn.Module):
    """单层图卷积：邻接归一化消息传递 + 线性变换 + 残差。"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数:
            x: [B, N, C] 节点特征
            adj_norm: [B, N, N] 归一化邻接矩阵（含自环）

        返回:
            h: [B, N, C] 更新后的节点特征
        """
        # 邻接消息传递
        h = torch.bmm(adj_norm, x)          # [B, N, C]
        h = self.fc(h)
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        # 残差连接
        if h.shape == x.shape:
            h = h + x
        return h


class LaneEncoder(nn.Module):
    """车道线图编码器：3 层 GCN，输出车道节点特征 [B, N, H]。"""

    def __init__(
        self,
        node_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        # 节点特征投影
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # GCN 层
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])

    def _normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """邻接矩阵加自环 + 对称归一化 D^-1/2 A D^-1/2。

        参数:
            adj: [B, N, N]（0/1 整型或浮点）

        返回:
            adj_norm: [B, N, N] 归一化邻接矩阵
        """
        A = adj.float()
        N = A.shape[-1]
        eye = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)  # [1, N, N]
        A_hat = A + eye                       # 加自环
        D = A_hat.sum(dim=-1)                 # [B, N] 度
        D_inv_sqrt = torch.pow(D + 1e-6, -0.5).unsqueeze(-1)  # [B, N, 1]
        adj_norm = D_inv_sqrt * A_hat * D_inv_sqrt.transpose(-1, -2)  # [B, N, N]
        return adj_norm

    def forward(
        self,
        nodes: torch.Tensor,
        adj: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播。

        参数:
            nodes: [B, N, node_dim] 节点特征（padding 节点全 0）
            adj: [B, N, N] 邻接矩阵（padding 节点行列全 0）
            node_mask: [B, N] 有效节点掩码（1=有效，0=padding）

        返回:
            lane_feats: [B, N, hidden_dim] 车道节点特征
        """
        h = self.node_proj(nodes)             # [B, N, H]
        adj_norm = self._normalize_adj(adj)   # [B, N, N]

        for gcn in self.gcn_layers:
            h = gcn(h, adj_norm)

        # padding 节点输出置 0
        if node_mask is not None:
            h = h * node_mask.unsqueeze(-1).to(h.dtype)

        return h
