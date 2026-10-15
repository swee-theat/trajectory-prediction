"""优化器和学习率调度器构建器。"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Tuple


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
) -> AdamW:
    """构建 AdamW 优化器。

    参数:
        model: PyTorch 模型
        lr: 学习率
        weight_decay: 权重衰减（L2 正则化强度）
        betas: Adam 动量参数

    返回:
        AdamW 优化器实例
    """
    return AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )


def build_scheduler(
    optimizer: AdamW,
    warmup_epochs: int = 2,
    total_epochs: int = 20,
    steps_per_epoch: int = 1,
    lr_min: float = 1e-6,
) -> torch.optim.lr_scheduler.LRScheduler:
    """构建 Warmup + CosineAnnealing 学习率调度器。

    调度策略:
    - 前 warmup_epochs 个 epoch: 线性 warmup (0 → lr)
    - 剩余 epoch: CosineAnnealing (lr → lr_min)

    参数:
        optimizer: AdamW 优化器
        warmup_epochs: 预热 epoch 数
        total_epochs: 总 epoch 数
        steps_per_epoch: 每个 epoch 的步数（用于计算总步数）
        lr_min: 最小学习率

    返回:
        组合调度器实例
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    cosine_steps = (total_epochs - warmup_epochs) * steps_per_epoch

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=lr_min,
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
