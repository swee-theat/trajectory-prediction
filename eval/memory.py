"""显存占用测量工具。"""

import torch
import torch.nn as nn
from typing import Dict


def measure_training_memory(
    model: nn.Module,
    input_dim: int = 9,
    obs_len: int = 20,
    pred_len: int = 30,
    batch_size: int = 64,
    device: str = "cuda",
) -> Dict[str, float]:
    """测量训练时的显存占用峰值。

    参数:
        model: 模型实例（train 模式）
        input_dim: 输入特征维度
        obs_len: 观测帧数
        pred_len: 预测帧数
        batch_size: batch 大小
        device: 设备类型

    返回:
        {
            "peak_allocated_mb": 峰值分配显存 (MB),
            "peak_reserved_mb": 峰值保留显存 (MB),
            "model_params_mb": 模型参数显存 (MB),
        }
    """
    if device != "cuda" or not torch.cuda.is_available():
        return {"error": "CUDA 不可用，无法测量显存"}

    model = model.to(device)
    model.train()

    # 重置峰值统计
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # 模型参数量显存
    model_params_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    model_params_mb = model_params_bytes / (1024 ** 2)

    # 构造随机输入和目标
    history = torch.randn(batch_size, obs_len, input_dim, device=device)
    future = torch.randn(batch_size, pred_len, 2, device=device)

    # 前向 + 反向传播
    output = model(history)
    loss = (
        output["trajectories"].sum() * 0.001
        + output["mode_probs"].sum() * 0.001
    )
    loss.backward()

    # 记录峰值
    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

    torch.cuda.empty_cache()

    return {
        "peak_allocated_mb": peak_allocated,
        "peak_reserved_mb": peak_reserved,
        "model_params_mb": model_params_mb,
    }


def measure_inference_memory(
    model: nn.Module,
    input_dim: int = 9,
    obs_len: int = 20,
    batch_size: int = 64,
    device: str = "cuda",
) -> Dict[str, float]:
    """测量推理时的显存占用。

    参数和返回值同 measure_training_memory。
    """
    if device != "cuda" or not torch.cuda.is_available():
        return {"error": "CUDA 不可用，无法测量显存"}

    model = model.to(device)
    model.eval()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model_params_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    model_params_mb = model_params_bytes / (1024 ** 2)

    history = torch.randn(batch_size, obs_len, input_dim, device=device)

    with torch.no_grad():
        _ = model(history)

    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

    torch.cuda.empty_cache()

    return {
        "peak_allocated_mb": peak_allocated,
        "peak_reserved_mb": peak_reserved,
        "model_params_mb": model_params_mb,
    }
