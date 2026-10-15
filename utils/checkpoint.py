"""Checkpoint 保存与加载工具。"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    filepath: str,
) -> None:
    """保存完整训练状态到 checkpoint。

    参数:
        model: 当前模型
        optimizer: 优化器（可为 None，推理时不需要）
        scheduler: 学习率调度器
        epoch: 当前 epoch 编号
        metrics: 当前指标字典（包含 val_minADE 等）
        filepath: 保存路径
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cuda",
    strict: bool = True,
) -> Dict[str, Any]:
    """从 checkpoint 恢复训练状态。

    参数:
        strict: 是否严格匹配 state_dict 键（第三阶段迁移加载用 False，
                允许缺失 interaction.* 等新增模块参数）

    返回:
        包含 epoch 和 metrics 的字典
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }
