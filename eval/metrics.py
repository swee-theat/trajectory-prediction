"""轨迹预测评估指标：ADE, FDE, minADE, minFDE, MR, 终点命中率等。

所有指标使用矢量化的 PyTorch 操作，支持批量计算。
"""

import torch
from typing import Optional


def _compute_l2_distances(
    pred: torch.Tensor, gt: torch.Tensor
) -> torch.Tensor:
    """计算预测轨迹与真值之间每个时间步的 L2 距离。

    参数:
        pred: [..., T, 2] 预测轨迹
        gt:   [..., T, 2] 真实轨迹

    返回:
        [..., T] 每个时间步的 L2 距离
    """
    return torch.norm(pred - gt, dim=-1)  # [... , T]


def compute_ade(
    trajectories: torch.Tensor,
    ground_truth: torch.Tensor,
    mode_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算平均位移误差 (Average Displacement Error)。

    参数:
        trajectories: [B, K, T_pred, 2] 预测轨迹
        ground_truth: [B, T_pred, 2] 真实轨迹
        mode_indices: [B] 指定使用哪个模态，为 None 时使用第 0 模态

    返回:
        [B] 每个样本的 ADE
    """
    B = trajectories.shape[0]

    if mode_indices is not None:
        # 使用指定模态
        pred = trajectories[torch.arange(B, device=trajectories.device), mode_indices]
    else:
        # 默认使用第 0 模态
        pred = trajectories[:, 0]

    distances = _compute_l2_distances(pred, ground_truth)  # [B, T]
    return distances.mean(dim=-1)  # [B]


def compute_fde(
    trajectories: torch.Tensor,
    ground_truth: torch.Tensor,
    mode_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算最终位移误差 (Final Displacement Error)。

    参数:
        trajectories: [B, K, T_pred, 2] 预测轨迹
        ground_truth: [B, T_pred, 2] 真实轨迹
        mode_indices: [B] 指定使用哪个模态

    返回:
        [B] 每个样本的 FDE
    """
    B = trajectories.shape[0]
    T = trajectories.shape[2]

    if mode_indices is not None:
        pred = trajectories[torch.arange(B, device=trajectories.device), mode_indices]
    else:
        pred = trajectories[:, 0]

    pred_final = pred[:, -1, :]  # [B, 2]
    gt_final = ground_truth[:, -1, :]  # [B, 2]
    return torch.norm(pred_final - gt_final, dim=-1)  # [B]


def compute_min_ade(
    trajectories: torch.Tensor,
    ground_truth: torch.Tensor,
) -> torch.Tensor:
    """计算最小 ADE（oracle：在 K 个模态中选择最佳）。

    参数:
        trajectories: [B, K, T_pred, 2]
        ground_truth: [B, T_pred, 2]

    返回:
        [B] 每个样本的最小 ADE
    """
    B, K = trajectories.shape[:2]
    gt_expanded = ground_truth.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, T, 2]
    distances = torch.norm(trajectories - gt_expanded, dim=-1)  # [B, K, T]
    ades = distances.mean(dim=-1)  # [B, K]
    return ades.min(dim=-1).values  # [B]


def compute_min_fde(
    trajectories: torch.Tensor,
    ground_truth: torch.Tensor,
) -> torch.Tensor:
    """计算最小 FDE（oracle：在 K 个模态中选择最佳）。

    参数:
        trajectories: [B, K, T_pred, 2]
        ground_truth: [B, T_pred, 2]

    返回:
        [B] 每个样本的最小 FDE
    """
    B, K = trajectories.shape[:2]
    pred_final = trajectories[:, :, -1, :]  # [B, K, 2]
    gt_final = ground_truth[:, -1, :]  # [B, 2]
    gt_expanded = gt_final.unsqueeze(1).expand(-1, K, -1)  # [B, K, 2]
    fdes = torch.norm(pred_final - gt_expanded, dim=-1)  # [B, K]
    return fdes.min(dim=-1).values  # [B]


def compute_miss_rate(
    trajectories: torch.Tensor,
    ground_truth: torch.Tensor,
    threshold: float = 2.0,
) -> float:
    """计算未命中率 (Miss Rate): minFDE > threshold 的样本比例。

    参数:
        trajectories: [B, K, T_pred, 2]
        ground_truth: [B, T_pred, 2]
        threshold: FDE 阈值（米），默认 2.0m

    返回:
        scalar，未命中率
    """
    min_fdes = compute_min_fde(trajectories, ground_truth)
    miss_count = (min_fdes > threshold).sum().float()
    return (miss_count / len(min_fdes)).item()


def compute_endpoint_hit_rate(
    trajectories: torch.Tensor,
    ground_truth: torch.Tensor,
    threshold: float = 3.0,
) -> float:
    """计算终点命中率：最佳模态终点距离 <= threshold 的比例。

    参数:
        trajectories: [B, K, T_pred, 2]
        ground_truth: [B, T_pred, 2]
        threshold: 距离阈值（米），默认 3.0m

    返回:
        scalar，命中率 [0, 1]
    """
    min_fdes = compute_min_fde(trajectories, ground_truth)
    hit_count = (min_fdes <= threshold).sum().float()
    return (hit_count / len(min_fdes)).item()


def compute_all_metrics(
    trajectories: torch.Tensor,
    ground_truth: torch.Tensor,
    mode_indices: Optional[torch.Tensor] = None,
) -> dict:
    """一键计算所有指标。

    返回:
        {
            "ade": ...,
            "fde": ...,
            "min_ade": ...,
            "min_fde": ...,
            "miss_rate_2m": ...,
            "endpoint_hit_3m": ...,
        }
    """
    return {
        "ade": compute_ade(trajectories, ground_truth, mode_indices).mean().item(),
        "fde": compute_fde(trajectories, ground_truth, mode_indices).mean().item(),
        "min_ade": compute_min_ade(trajectories, ground_truth).mean().item(),
        "min_fde": compute_min_fde(trajectories, ground_truth).mean().item(),
        "miss_rate_2m": compute_miss_rate(trajectories, ground_truth, threshold=2.0),
        "endpoint_hit_3m": compute_endpoint_hit_rate(trajectories, ground_truth, threshold=3.0),
    }
