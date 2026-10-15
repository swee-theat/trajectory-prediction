"""推理延迟基准测试：测量不同 batch size 下的前向传播时间。"""

import time
import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np


def benchmark_latency(
    model: nn.Module,
    input_dim: int = 9,
    obs_len: int = 20,
    batch_sizes: List[int] = [1, 8, 16, 32, 64],
    num_warmup: int = 50,
    num_iter: int = 200,
    device: str = "cuda",
) -> Dict[int, Dict[str, float]]:
    """测量模型在不同 batch size 下的推理延迟。

    使用 CUDA Events 精确计时（GPU），CPU 模式使用 time.perf_counter。

    参数:
        model: 待测模型（eval 模式）
        input_dim: 输入特征维度
        obs_len: 观测帧数
        batch_sizes: 要测试的 batch size 列表
        num_warmup: 预热迭代数
        num_iter: 测量迭代数
        device: 设备类型

    返回:
        {batch_size: {"mean_ms": ..., "std_ms": ..., "fps": ...}}
    """
    model.eval()
    results = {}

    for bs in batch_sizes:
        # 生成随机输入
        dummy_input = torch.randn(bs, obs_len, input_dim, device=device)

        # 预热
        for _ in range(num_warmup):
            _ = model(dummy_input)

        # 测量
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            times = []
            for _ in range(num_iter):
                start_event.record()
                _ = model(dummy_input)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))  # ms

            torch.cuda.empty_cache()
        else:
            times = []
            for _ in range(num_iter):
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()
                times.append((end - start) * 1000.0)  # ms

        times = np.array(times)
        results[bs] = {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "fps": float(bs / np.mean(times) * 1000),  # 每秒处理的样本数
        }

    return results


def benchmark_cpu_latency(
    model: nn.Module,
    input_dim: int = 9,
    obs_len: int = 20,
    batch_sizes: List[int] = [1, 8, 16],
    num_warmup: int = 10,
    num_iter: int = 50,
) -> Dict[int, Dict[str, float]]:
    """在 CPU 上测量推理延迟（用于部署场景评估）。"""
    cpu_model = model.cpu()
    return benchmark_latency(
        cpu_model, input_dim, obs_len, batch_sizes,
        num_warmup=num_warmup, num_iter=num_iter, device="cpu",
    )


def benchmark_throughput(
    model: nn.Module,
    input_dim: int = 9,
    obs_len: int = 20,
    batch_sizes: List[int] = [8, 16, 32, 64],
    num_iter: int = 100,
    device: str = "cuda",
) -> Dict[str, float]:
    """测量最大吞吐量（FPS）。

    返回:
        {"max_fps": float, "optimal_batch_size": int}
    """
    latencies = benchmark_latency(
        model, input_dim, obs_len, batch_sizes,
        num_warmup=20, num_iter=num_iter, device=device,
    )

    best_bs = max(latencies, key=lambda k: latencies[k]["fps"])
    return {
        "max_fps": latencies[best_bs]["fps"],
        "optimal_batch_size": best_bs,
        "per_batch_size": latencies,
    }
