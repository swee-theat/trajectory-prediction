"""损失函数模块：Winner-Takes-All 轨迹损失 + 模态分类损失 + 联合损失。

核心策略（Winner-Takes-All, WTA）:
1. 计算 K 个预测轨迹与真值之间的 ADE
2. 选择 ADE 最小的模态作为"胜者"
3. 仅胜者模态的轨迹损失反向传播梯度
4. 模态选择器始终接收交叉熵损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TrajectoryLoss(nn.Module):
    """Winner-Takes-All 轨迹预测损失。

    对 K 个模态分别计算 smooth L1 损失，
    仅最佳模态（最小 ADE）贡献梯度。
    """

    def __init__(self, loss_type: str = "smooth_l1"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        trajectories: torch.Tensor,  # [B, K, T_pred, 2]
        ground_truth: torch.Tensor,   # [B, T_pred, 2]
    ) -> tuple:
        """计算 WTA 轨迹损失。

        参数:
            trajectories: K 个模态的预测轨迹
            ground_truth: 真实未来轨迹

        返回:
            (traj_loss, best_mode_idx)
            - traj_loss: scalar，最佳模态的 smooth L1 损失
            - best_mode_idx: [B]，每个样本的最佳模态索引
        """
        B, K, T_pred, _ = trajectories.shape
        gt = ground_truth.unsqueeze(1)  # [B, 1, T_pred, 2]

        # 计算每个模态与真值的逐点 L2 距离
        diff = trajectories - gt  # [B, K, T_pred, 2]

        if self.loss_type == "smooth_l1":
            # Smooth L1 (Huber) loss
            per_point_loss = F.smooth_l1_loss(
                trajectories,
                gt.expand(-1, K, -1, -1),
                reduction="none",
                beta=1.0,
            )  # [B, K, T_pred, 2]
            # 每个模态的总损失（对 T_pred 和 2 维求和 / 平均）
            per_mode_loss = per_point_loss.sum(dim=(-1, -2))  # [B, K]
        elif self.loss_type == "mse":
            per_mode_loss = (diff ** 2).sum(dim=(-1, -2))  # [B, K]
        elif self.loss_type == "l1":
            per_mode_loss = diff.abs().sum(dim=(-1, -2))  # [B, K]
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")

        # WTA: 选择损失最小的模态
        best_mode_loss, best_mode_idx = per_mode_loss.min(dim=-1)  # [B], [B]

        # 平均所有样本的胜者损失
        traj_loss = best_mode_loss.mean()

        return traj_loss, best_mode_idx


class ModeLoss(nn.Module):
    """模态选择交叉熵损失。

    对模态选择器使用硬标签（one-hot at best_mode）。
    硬标签鼓励模态专业化，而非分摊概率。
    """

    def forward(
        self,
        mode_logits: torch.Tensor,  # [B, K]
        best_mode_idx: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """计算模态分类交叉熵损失。

        参数:
            mode_logits: 模态 logits（未经 softmax）
            best_mode_idx: 最佳模态索引（来自 WTA）

        返回:
            mode_loss: scalar 交叉熵损失
        """
        return F.cross_entropy(mode_logits, best_mode_idx)


class CombinedLoss(nn.Module):
    """联合损失: L = λ_traj * L_traj + λ_mode * L_mode + λ_unc * L_unc。

    L_unc 使用高斯负对数似然 (NLL)，让不确定性估计参与训练：
        L_unc = 0.5 * mean(log(σ²) + (y - ŷ)² / σ²)

    默认权重: λ_traj=0.7, λ_mode=0.3, λ_unc=0.0 (关闭以兼容旧配置)
    """

    def __init__(
        self,
        lambda_traj: float = 0.7,
        lambda_mode: float = 0.3,
        lambda_unc: float = 0.0,      # NLL 损失权重，设为 0.05~0.1 启用
        lambda_neighbor: float = 0.4,   # 邻居轨迹监督权重（第三阶段）
        lambda_collision: float = 0.1,  # 碰撞惩罚权重（第三阶段）
        collision_margin: float = 2.0,  # 碰撞距离阈值（米），与检测阈值一致
        traj_loss_type: str = "smooth_l1",
    ):
        super().__init__()
        self.lambda_traj = lambda_traj
        self.lambda_mode = lambda_mode
        self.lambda_unc = lambda_unc
        self.lambda_neighbor = lambda_neighbor
        self.lambda_collision = lambda_collision
        self.collision_margin = collision_margin
        self.traj_loss_fn = TrajectoryLoss(loss_type=traj_loss_type)
        self.mode_loss_fn = ModeLoss()

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        ground_truth: torch.Tensor,
        neighbor_future: Optional[torch.Tensor] = None,
        neighbor_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算联合损失。

        参数:
            model_output: 模型输出字典，包含 trajectories, mode_logits, uncertainties(可选)
                          （第三阶段另含 neighbor_trajectories/neighbor_mode_logits）
            ground_truth: [B, T_pred, 2] 真实未来轨迹
            neighbor_future: [B, T_pred, 2] top-1 邻居真实未来轨迹（第三阶段，可选）
            neighbor_valid: [B] 邻居存在掩码（= neighbor_mask[:,0]，第三阶段，可选）

        返回:
            {"loss", "traj_loss", "mode_loss", "unc_loss",
             "neighbor_loss", "collision_loss", "best_mode_idx"}
        """
        trajectories = model_output["trajectories"]  # [B, K, T_pred, 2]
        mode_logits = model_output["mode_logits"]    # [B, K]
        B = trajectories.shape[0]

        # 计算 WTA 轨迹损失
        traj_loss, best_mode_idx = self.traj_loss_fn(trajectories, ground_truth)

        # 计算模态分类损失
        mode_loss = self.mode_loss_fn(mode_logits, best_mode_idx)

        # 联合损失
        total_loss = self.lambda_traj * traj_loss + self.lambda_mode * mode_loss

        # 不确定性 NLL 损失（仅当启用且模型输出 uncertainties 时）
        unc_loss = torch.tensor(0.0, device=traj_loss.device)
        if self.lambda_unc > 0 and "uncertainties" in model_output:
            uncertainties = model_output["uncertainties"]  # [B, K, T_pred, 2]
            if uncertainties is not None:
                # 仅计算最佳模态的 NLL
                best_traj = trajectories[torch.arange(B, device=traj_loss.device), best_mode_idx]
                best_unc = uncertainties[torch.arange(B, device=traj_loss.device), best_mode_idx]
                # L = 0.5 * (log(σ²) + (μ - y)² / σ²)，加 eps 防 log(0)
                eps = 1e-6
                log_var = torch.log(best_unc + eps)
                sq_error = (best_traj - ground_truth) ** 2
                nll = 0.5 * (log_var + sq_error / (best_unc + eps))
                unc_loss = nll.mean()
                total_loss = total_loss + self.lambda_unc * unc_loss

        # 邻居监督 + 碰撞惩罚（第三阶段）
        neighbor_loss = torch.tensor(0.0, device=traj_loss.device)
        collision_loss = torch.tensor(0.0, device=traj_loss.device)
        if neighbor_future is not None and "neighbor_trajectories" in model_output:
            nbr_traj = model_output["neighbor_trajectories"]   # [B, K, T_pred, 2]
            nbr_logits = model_output["neighbor_mode_logits"]  # [B, K]
            if neighbor_valid is not None:
                valid = neighbor_valid.bool()  # [B]
            else:
                valid = torch.ones(B, dtype=torch.bool, device=traj_loss.device)

            if valid.any():
                # 邻居 WTA（仅在存在邻居的样本上监督）
                nbr_traj_valid = nbr_traj[valid]
                nbr_future_valid = neighbor_future[valid]
                nbr_traj_loss, nbr_best_idx = self.traj_loss_fn(nbr_traj_valid, nbr_future_valid)
                nbr_mode_loss = self.mode_loss_fn(nbr_logits[valid], nbr_best_idx)
                neighbor_loss = (
                    self.lambda_neighbor * nbr_traj_loss + self.lambda_mode * nbr_mode_loss
                )
                total_loss = total_loss + neighbor_loss

                # 碰撞惩罚：ego 与邻居的最可能轨迹逐帧最小距离 < margin 时惩罚
                if self.lambda_collision > 0:
                    num_valid = int(valid.sum())
                    ego_best = trajectories[
                        torch.arange(B, device=traj_loss.device), best_mode_idx
                    ]  # [B, T_pred, 2]
                    ego_best_valid = ego_best[valid]  # [V, T_pred, 2]
                    nbr_best_valid = nbr_traj_valid[
                        torch.arange(num_valid, device=traj_loss.device), nbr_best_idx
                    ]  # [V, T_pred, 2]
                    dist = torch.norm(ego_best_valid - nbr_best_valid, dim=-1)  # [V, T_pred]
                    # 平方 hinge：relu(margin - d)²，梯度连续有界
                    viol = torch.clamp(self.collision_margin - dist, min=0) ** 2
                    collision_loss = self.lambda_collision * viol.mean()
                    total_loss = total_loss + collision_loss

        return {
            "loss": total_loss,
            "traj_loss": traj_loss,
            "mode_loss": mode_loss,
            "unc_loss": unc_loss,
            "neighbor_loss": neighbor_loss,
            "collision_loss": collision_loss,
            "best_mode_idx": best_mode_idx,
        }
