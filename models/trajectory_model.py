"""顶层轨迹预测模型：组装编码器 + 多模态预测器 + 不确定性估计器。

完整数据流:
    历史轨迹 [B, T_obs, input_dim]
    → TrajectoryEncoder → [B, hidden_dim]
    → MultimodalPredictor → trajectories [B, K, T_pred, 2] + mode_probs [B, K]
    → UncertaintyEstimator → uncertainties [B, K, T_pred, 2]
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from .trajectory_encoder import TrajectoryEncoder
from .multimodal_predictor import MultimodalPredictor
from .uncertainty_estimator import UncertaintyEstimator
from .lane_encoder import LaneEncoder
from .interaction_module import InteractionModule
from .attention_pooling import AttentionPooling


class NeighborEncoder(nn.Module):
    """邻居独立轻量编码器（约 13K 参数）。

    邻居特征分布（无 lane 融合、居中到 ego 原点）与 ego 差异大，复用 ego 的 encoder
    （含 lane 融合的注意力 pooling）会得到不适配的表示；且方案二冻结 ego 主干后 encoder
    无法适配邻居，邻居 predictor 容量再大也难恢复邻居运动（bh=48 仍只到 10.33）。
    独立可训练编码器让邻居表示随邻居损失优化，不污染 ego 的 encoder。
    """

    def __init__(self, input_dim: int = 11, hidden: int = 64, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.frame_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.pool_proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        """x: [B, T, input_dim] → (frame_feats [B, T, hidden], pooled [B, out_dim])。"""
        frame = self.frame_proj(x)           # [B, T, hidden] 逐帧特征（供 interaction）
        pooled = self.pool_proj(frame.mean(dim=1))  # [B, out_dim] 时序均值池化（供邻居预测）
        return frame, pooled


class TrajectoryModel(nn.Module):
    """轻量化多模态轨迹预测模型。

    目标参数量:
        - 独立不确定性头(share_uncertainty=False): ~244K
        - 共享不确定性头(share_uncertainty=True): ~195K（为车道线+交互扩展模块腾预算）
    目标硬件: RTX 3060 (6GB) 可完整训练
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        num_mlp_layers: int = 3,
        dropout: float = 0.1,
        num_modes: int = 3,
        pred_len: int = 30,
        pooling_type: str = "attention",
        use_uncertainty: bool = True,
        use_temporal_conv: bool = True,
        share_uncertainty: bool = False,
        use_lane: bool = True,
        use_city: bool = True,
        use_interaction: bool = False,
        use_lane_gate: bool = False,
        branch_hidden_dim: int = 96,
        num_cities: int = 7,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.pred_len = pred_len
        self.use_uncertainty = use_uncertainty
        self.use_lane = use_lane
        self.use_city = use_city
        self.use_interaction = use_interaction
        self.use_lane_gate = use_lane_gate

        # 轨迹编码器
        self.encoder = TrajectoryEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_mlp_layers=num_mlp_layers,
            dropout=dropout,
            pooling_type=pooling_type,
            use_temporal_conv=use_temporal_conv,
        )

        # 车道线编码器（可选）
        if use_lane:
            self.lane_encoder = LaneEncoder(
                node_dim=6, hidden_dim=hidden_dim, num_layers=3, dropout=dropout
            )
        else:
            self.lane_encoder = None

        # 车道线独立 query 池化 + 标量 gate（wen_ti/01 验证：修复共享 query 淹没车道线）。
        # gate 初始 0.0：训练早期 ego 特征等价纯轨迹（无车道线注入），与 interaction 的
        # context_gate 同理，避免随机 query 的 lane_context 噪声在早期破坏预测。
        if use_lane and use_lane_gate:
            self.lane_pooling = AttentionPooling(hidden_dim)
            self.lane_gate = nn.Parameter(torch.tensor(0.0))
        else:
            self.lane_pooling = None
            self.lane_gate = None

        # 城市 embedding（可选，域感知偏置）
        if use_city:
            self.city_embed = nn.Embedding(num_cities, hidden_dim)
        else:
            self.city_embed = None

        # 交互编码模块（第三阶段；默认关闭，旧脚本/stage2 权重不受影响）
        if use_interaction:
            self.interaction = InteractionModule(hidden_dim=hidden_dim, dropout=dropout)
            # 交互上下文门控：初始 0.0，让训练早期 ego 特征完全等同单车基线（组 A），
            # encoder 先在无交互梯度的干净状态下收敛，之后 gate 再从 0 自由学习，
            # 避免 gate 初始 0.1 时 interaction 路径对 encoder 的早期梯度污染（wen_ti/04）
            self.context_gate = nn.Parameter(torch.tensor(0.0))
            # 邻居独立预测头（wen_ti/04 方案二）：不复用 ego predictor，避免角色冲突。
            # 邻居特征分布（无 lane 融合、居中到 ego 原点）与 ego 差异大，复用 predictor 会崩
            # 到 17.49；独立头用小分支维度（16）压参数，专门服务邻居分布。
            self.neighbor_predictor = MultimodalPredictor(
                hidden_dim=hidden_dim,
                num_modes=num_modes,
                pred_len=pred_len,
                dropout=dropout,
                branch_hidden_dim=48,
                mode_hidden_dim=32,
            )
            # 邻居独立编码器（可训练，不复用 ego encoder 避免角色冲突）
            self.neighbor_encoder = NeighborEncoder(
                input_dim=input_dim, out_dim=hidden_dim, dropout=dropout
            )
        else:
            self.interaction = None
            self.context_gate = None
            self.neighbor_predictor = None
            self.neighbor_encoder = None

        # 多模态预测器（branch_hidden_dim 压缩中间层，第二阶段设 96 省约 30K）
        self.predictor = MultimodalPredictor(
            hidden_dim=hidden_dim,
            num_modes=num_modes,
            pred_len=pred_len,
            dropout=dropout,
            branch_hidden_dim=branch_hidden_dim,
        )

        # 不确定性估计器（可选）
        if use_uncertainty:
            self.uncertainty = UncertaintyEstimator(
                hidden_dim=hidden_dim,
                num_modes=num_modes,
                pred_len=pred_len,
                dropout=dropout,
                # 独立头(False)：每模态各自方差；共享头(True)：省约 48K 参数，
                # 用于容纳车道线+交互两个扩展模块、满足总参数 ≤288K。
                share_across_modes=share_uncertainty,
            )
        else:
            self.uncertainty = None

    def forward(
        self,
        history: torch.Tensor,
        lane_nodes: Optional[torch.Tensor] = None,
        lane_adj: Optional[torch.Tensor] = None,
        lane_mask: Optional[torch.Tensor] = None,
        city_id: Optional[torch.Tensor] = None,
        neighbor_history: Optional[torch.Tensor] = None,
        neighbor_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """前向传播。

        参数:
            history: [B, T_obs, input_dim] 历史轨迹特征
            lane_nodes: [B, N, 6] 车道线节点特征（可选）
            lane_adj: [B, N, N] 车道线邻接矩阵（可选）
            lane_mask: [B, N] 车道线节点掩码（可选）
            city_id: [B] 城市 ID（可选，域感知偏置）
            neighbor_history: [B, K, T_obs, input_dim] 邻居历史轨迹（第三阶段，可选）
            neighbor_mask: [B, K] 邻居槽位掩码（第三阶段，可选）
            return_attn_weights: 是否返回注意力权重（用于可视化）

        返回:
            {
                "trajectories":     [B, K, T_pred, 2],
                "mode_probs":       [B, K],
                "mode_logits":      [B, K],
                "uncertainties":    [B, K, T_pred, 2] (if use_uncertainty),
                "encoded_feature":  [B, hidden_dim],
                "attn_weights":     [B, T+N] 或 [B, T_obs] (if return_attn_weights),
                # use_interaction=True 时追加：
                "neighbor_trajectories": [B, K, T_pred, 2],
                "neighbor_mode_probs":   [B, K],
                "neighbor_mode_logits":  [B, K],
                "interaction_context":   [B, hidden_dim],
                "neighbor_attn_weights": [B, K*T_obs],
            }
        """
        # 编码（取池化前特征 [B, T, H]）
        enc_feats, _ = self.encoder(history, pool=False)

        # 车道线编码 + 融合
        if self.use_lane and lane_nodes is not None and self.lane_encoder is not None:
            lane_feats = self.lane_encoder(lane_nodes, lane_adj, lane_mask)  # [B, N, H]
            if self.use_lane_gate:
                # 新融合（wen_ti/01 验证）：轨迹帧与车道节点各自独立 query 池化，
                # 车道上下文经标量 gate 注入 ego 特征，避免共享 query 淹没车道线
                encoded, attn_weights = self.encoder.pooling(
                    enc_feats, return_weights=return_attn_weights
                )
                lane_context, _ = self.lane_pooling(lane_feats)  # [B, H]
                encoded = encoded + self.lane_gate * lane_context
            else:
                # 旧融合：轨迹帧 + 车道节点 concat 后共享 query 池化
                fused = torch.cat([enc_feats, lane_feats], dim=1)  # [B, T+N, H]
                encoded, attn_weights = self.encoder.pooling(
                    fused, return_weights=return_attn_weights
                )
        else:
            encoded, attn_weights = self.encoder.pooling(
                enc_feats, return_weights=return_attn_weights
            )

        # 交互编码（第三阶段）：ego 特征与邻居逐帧特征交叉注意力
        neighbor_pred = None
        neighbor_attn = None
        context = None
        if (
            self.use_interaction
            and self.interaction is not None
            and neighbor_history is not None
            and neighbor_mask is not None
        ):
            B, K, T_obs, D = neighbor_history.shape
            # 邻居独立轻量编码（可训练，不复用 ego encoder）。邻居损失反传只更新
            # neighbor_encoder + neighbor_predictor，不碰冻结的 ego 主干 —— wen_ti/04 方案二
            _, nbr_pooled = self.neighbor_encoder(
                neighbor_history.reshape(B * K, T_obs, D)
            )
            nbr_pooled = nbr_pooled.reshape(B, K, -1)[:, 0]  # top-1 邻居 [B, H]
            if self.use_city and city_id is not None and self.city_embed is not None:
                nbr_pooled = nbr_pooled + self.city_embed(city_id)
            # 邻居独立预测头（不复用 ego predictor，避免角色冲突）
            neighbor_pred = self.neighbor_predictor(nbr_pooled)

            # 交叉注意力交互编码（死参数，仅供邻居相关输出/可视化）：用 ego encoder 的
            # 邻居逐帧特征（no_grad，冻结），不注入 ego 预测路径（wen_ti/04 方案一）
            with torch.no_grad():
                nbr_ego_feats, _ = self.encoder(
                    neighbor_history.reshape(B * K, T_obs, D), pool=False
                )
                nbr_ego_feats = nbr_ego_feats.reshape(B, K, T_obs, -1)
            context, neighbor_attn = self.interaction(
                encoded.detach(), nbr_ego_feats.reshape(B, K * T_obs, -1), neighbor_mask
            )
            # ego 预测不经过交互（等价组 A），保证不退化
            ego_feat = encoded  # [B, H]

            # 城市偏置加给 ego
            if self.use_city and city_id is not None and self.city_embed is not None:
                ego_feat = ego_feat + self.city_embed(city_id)

            # ego 预测（正常梯度）
            pred_output = self.predictor(ego_feat)
        else:
            # 单车路径（与 stage2 完全一致）
            ego_feat = encoded
            if self.use_city and city_id is not None and self.city_embed is not None:
                ego_feat = ego_feat + self.city_embed(city_id)
            pred_output = self.predictor(ego_feat)

        # 不确定性估计（基于最终用于预测的 ego 特征）
        if self.uncertainty is not None:
            uncertainties = self.uncertainty(ego_feat)
        else:
            uncertainties = None

        output = {
            "trajectories": pred_output["trajectories"],
            "mode_probs": pred_output["mode_probs"],
            "mode_logits": pred_output["mode_logits"],
            "uncertainties": uncertainties,
            "encoded_feature": ego_feat,
        }

        # 邻居预测输出（第三阶段）
        if neighbor_pred is not None:
            output["neighbor_trajectories"] = neighbor_pred["trajectories"]
            output["neighbor_mode_probs"] = neighbor_pred["mode_probs"]
            output["neighbor_mode_logits"] = neighbor_pred["mode_logits"]
            output["interaction_context"] = context
            if neighbor_attn is not None:
                output["neighbor_attn_weights"] = neighbor_attn

        if return_attn_weights:
            output["attn_weights"] = attn_weights

        return output

    def predict(
        self,
        history: torch.Tensor,
        lane_nodes: Optional[torch.Tensor] = None,
        lane_adj: Optional[torch.Tensor] = None,
        lane_mask: Optional[torch.Tensor] = None,
        city_id: Optional[torch.Tensor] = None,
        neighbor_history: Optional[torch.Tensor] = None,
        neighbor_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """推理模式预测（no_grad + eval）。

        参数:
            history: [B, T_obs, input_dim]
            其余为可选车道线/城市/邻居输入，见 forward()

        返回:
            与 forward() 相同的输出字典
        """
        self.eval()
        with torch.no_grad():
            return self.forward(
                history,
                lane_nodes=lane_nodes,
                lane_adj=lane_adj,
                lane_mask=lane_mask,
                city_id=city_id,
                neighbor_history=neighbor_history,
                neighbor_mask=neighbor_mask,
            )

    def get_best_trajectory(
        self,
        history: torch.Tensor,
        lane_nodes: Optional[torch.Tensor] = None,
        lane_adj: Optional[torch.Tensor] = None,
        lane_mask: Optional[torch.Tensor] = None,
        city_id: Optional[torch.Tensor] = None,
        neighbor_history: Optional[torch.Tensor] = None,
        neighbor_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """获取最可能的单条预测轨迹。

        参数:
            history: [B, T_obs, input_dim]
            其余为可选车道线/城市/邻居输入，见 forward()

        返回:
            best_traj: [B, T_pred, 2] 概率最高的模态轨迹
        """
        output = self.predict(
            history,
            lane_nodes=lane_nodes,
            lane_adj=lane_adj,
            lane_mask=lane_mask,
            city_id=city_id,
            neighbor_history=neighbor_history,
            neighbor_mask=neighbor_mask,
        )
        best_mode_idx = torch.argmax(output["mode_probs"], dim=-1)  # [B]
        B = history.shape[0]
        best_traj = output["trajectories"][
            torch.arange(B, device=history.device), best_mode_idx
        ]  # [B, T_pred, 2]
        return best_traj

    def get_best_pair(
        self,
        history: torch.Tensor,
        lane_nodes: Optional[torch.Tensor] = None,
        lane_adj: Optional[torch.Tensor] = None,
        lane_mask: Optional[torch.Tensor] = None,
        city_id: Optional[torch.Tensor] = None,
        neighbor_history: Optional[torch.Tensor] = None,
        neighbor_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """获取 ego 与最近交互车的最可能预测轨迹对（供碰撞检测与 demo）。

        参数:
            history: [B, T_obs, input_dim]
            其余为可选车道线/城市/邻居输入，见 forward()

        返回:
            (ego_best [B, T_pred, 2], neighbor_best [B, T_pred, 2])
            未启用交互或无邻居时 neighbor_best 全 0
        """
        output = self.predict(
            history,
            lane_nodes=lane_nodes,
            lane_adj=lane_adj,
            lane_mask=lane_mask,
            city_id=city_id,
            neighbor_history=neighbor_history,
            neighbor_mask=neighbor_mask,
        )
        best_mode_idx = torch.argmax(output["mode_probs"], dim=-1)  # [B]
        B = history.shape[0]
        ego_best = output["trajectories"][
            torch.arange(B, device=history.device), best_mode_idx
        ]  # [B, T_pred, 2]

        if "neighbor_trajectories" in output:
            nbr_best_idx = torch.argmax(output["neighbor_mode_probs"], dim=-1)  # [B]
            neighbor_best = output["neighbor_trajectories"][
                torch.arange(B, device=history.device), nbr_best_idx
            ]  # [B, T_pred, 2]
        else:
            neighbor_best = torch.zeros_like(ego_best)

        return ego_best, neighbor_best
