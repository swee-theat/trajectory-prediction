"""数据增强模块：对轨迹历史帧进行随机旋转、缩放和加噪。

第三阶段扩展：邻居轨迹与 ego 共享同一旋转矩阵/角度/缩放，
保证相对几何不被破坏（否则碰撞惩罚会失真）。
"""

import torch
import numpy as np


class TrajectoryAugmentation:
    """轨迹数据增强，包含随机旋转、随机缩放和高斯噪声。

    所有增强仅作用于历史帧，未来帧同步变换以保证一致性。
    """

    def __init__(
        self,
        enable: bool = True,
        rotation_range: float = 3.14159,  # [-pi, pi]
        scale_range: tuple = (0.9, 1.1),
        noise_std: float = 0.02,
    ):
        self.enable = enable
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std

    def _random_rotation_matrix(self, batch_size: int, device: torch.device) -> tuple:
        """生成随机旋转矩阵 [B, 2, 2] 及对应的旋转角 [B]。"""
        angle = (torch.rand(batch_size, device=device) * 2 - 1) * self.rotation_range
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        rot = torch.stack([
            torch.stack([cos_a, -sin_a], dim=-1),
            torch.stack([sin_a, cos_a], dim=-1),
        ], dim=-2)
        return rot, angle

    def _augment_history(
        self,
        hist: torch.Tensor,   # [N, T_obs, 11]
        rot: torch.Tensor,    # [N, 2, 2]
        angle: torch.Tensor,  # [N]
        scale: torch.Tensor,  # [N, 1, 1]
        noise: torch.Tensor,  # [N, T_obs, 2]
    ) -> torch.Tensor:
        """对单批历史轨迹应用旋转/缩放/噪声（N 可为 ego 或展开后的邻居维度）。"""
        # 旋转 position (x, y)
        pos = hist[:, :, :2]  # [N, T, 2]
        pos_rot = torch.bmm(pos, rot.transpose(-2, -1))  # [N, T, 2]
        hist = torch.cat([pos_rot, hist[:, :, 2:]], dim=-1)

        # 旋转 velocity (vx, vy)
        vel = hist[:, :, 2:4]  # [N, T, 2]
        vel_rot = torch.bmm(vel, rot.transpose(-2, -1))  # [N, T, 2]
        hist[:, :, 2:4] = vel_rot

        # 更新航向角（index 9）：坐标旋转 angle 后，航向角同步 + angle
        if hist.shape[-1] > 9:
            hist[:, :, 9] = hist[:, :, 9] + angle.unsqueeze(-1)

        # 随机缩放
        hist[:, :, :2] *= scale
        hist[:, :, 2:4] *= scale
        # 距离相关特征（index 6/7/8）
        if hist.shape[-1] >= 7:
            hist[:, :, 6:9] *= scale

        # 高斯噪声（仅对位置坐标）
        hist[:, :, :2] += noise

        return hist

    def __call__(
        self,
        history: torch.Tensor,                       # [B, T_obs, 11]
        future: torch.Tensor,                         # [B, T_pred, 2]
        neighbor_history: torch.Tensor = None,        # [B, K, T_obs, 11]
        neighbor_future: torch.Tensor = None,         # [B, T_pred, 2]
    ) -> tuple:
        """对一批轨迹应用数据增强（ego 与邻居共享旋转/缩放）。

        参数:
            history: 历史轨迹 [B, T_obs, 11]
            future: 未来轨迹真值 [B, T_pred, 2]
            neighbor_history: 邻居历史轨迹 [B, K, T_obs, 11]（第三阶段，可选）
            neighbor_future: top-1 邻居未来真值 [B, T_pred, 2]（第三阶段，可选）

        返回:
            (history, future, neighbor_history, neighbor_future)
            未传入的邻居字段返回 None
        """
        if not self.enable:
            return history, future, neighbor_history, neighbor_future

        B = history.shape[0]
        device = history.device

        # 旋转矩阵/角度只生成一次，ego 与邻居共享
        rot, angle = self._random_rotation_matrix(B, device)  # rot [B,2,2], angle [B]

        # 随机缩放（只生成一次）
        scale = (
            torch.rand(B, 1, 1, device=device) * (self.scale_range[1] - self.scale_range[0])
            + self.scale_range[0]
        )  # [B, 1, 1]

        # ego 增强
        noise = torch.randn_like(history[:, :, :2]) * self.noise_std
        history = self._augment_history(history, rot, angle, scale, noise)

        # 未来轨迹同步旋转 + 缩放
        future = torch.bmm(future, rot.transpose(-2, -1)) * scale  # [B, T_pred, 2]

        # 邻居增强（与 ego 共享 rot/angle/scale，保证相对几何一致）
        if neighbor_history is not None:
            K = neighbor_history.shape[1]
            T_obs = neighbor_history.shape[2]
            D = neighbor_history.shape[3]
            # 旋转/缩放参数扩展到每个邻居 slot
            rot_n = rot.unsqueeze(1).expand(B, K, 2, 2).reshape(B * K, 2, 2)
            angle_n = angle.unsqueeze(1).expand(B, K).reshape(B * K)
            scale_n = scale.unsqueeze(1).expand(B, K, 1, 1).reshape(B * K, 1, 1)
            nbr_flat = neighbor_history.reshape(B * K, T_obs, D)
            noise_n = torch.randn_like(nbr_flat[:, :, :2]) * self.noise_std
            nbr_flat = self._augment_history(nbr_flat, rot_n, angle_n, scale_n, noise_n)
            neighbor_history = nbr_flat.reshape(B, K, T_obs, D)

            if neighbor_future is not None:
                neighbor_future = torch.bmm(neighbor_future, rot.transpose(-2, -1)) * scale

        return history, future, neighbor_history, neighbor_future
