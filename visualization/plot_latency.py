"""图7: 推理延迟 vs Batch Size 曲线。"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from typing import Dict

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_latency_comparison(
    gpu_latency: Dict[int, Dict],
    output_path: str = "outputs/figures/fig7_latency.png",
):
    """绘制推理延迟 vs Batch Size 对比图。

    参数:
        gpu_latency: 来自 benchmark_latency 的结果 {batch_size: {mean_ms, std_ms, fps}}
        output_path: 输出路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    batch_sizes = sorted(gpu_latency.keys())
    mean_times = [gpu_latency[bs]["mean_ms"] for bs in batch_sizes]
    std_times = [gpu_latency[bs]["std_ms"] for bs in batch_sizes]
    fps_vals = [gpu_latency[bs]["fps"] for bs in batch_sizes]

    # 左图：延迟
    ax1.errorbar(batch_sizes, mean_times, yerr=std_times, marker="o",
                 color="#2196F3", capsize=5, linewidth=1.5, label="本文模型")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("推理延迟 (ms)")
    ax1.set_title("推理延迟 vs Batch Size")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 右图：FPS
    ax2.plot(batch_sizes, fps_vals, marker="s", color="#FF5722",
             linewidth=1.5, label="吞吐量")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("FPS (样本/秒)")
    ax2.set_title("推理吞吐量 vs Batch Size")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  图7 已保存: {output_path}")
