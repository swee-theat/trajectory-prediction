"""图5: 参数量-ADE Pareto 前沿对比图。"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


# 文献报告的基线数据（用于对比）
BASELINE_DATA = {
    "Ours (dim=32)":    (8_000, 1.50),
    "Ours (dim=64)":    (35_000, 1.20),
    "Ours (dim=128)":   (133_000, 0.95),
    "Ours (dim=256)":   (500_000, 0.80),
    "Ours (dim=512)":   (1_950_000, 0.72),
    "CDDM (231K)":      (231_000, 1.15),
    "VectorNet (~1M)":  (1_000_000, 0.85),
    "HiVT (~800K)":     (800_000, 0.70),
}


def plot_pareto_frontier(
    sweep_data: List[Dict],
    output_path: str = "outputs/figures/fig5_pareto.png",
):
    """绘制参数量-ADE Pareto 前沿图。

    参数:
        sweep_data: 来自 run_efficiency.py 参数扫描结果，每条包含 hidden_dim, 参数量, 测试minADE
        output_path: 输出路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制论文实验数据
    if sweep_data:
        params = [d.get("参数量", d.get("total_params", 0)) for d in sweep_data]
        ades = [d.get("测试minADE", d.get("test_min_ade", 0)) for d in sweep_data]
        labels = [f"Ours ({d.get('hidden_dim', '?')})" for d in sweep_data]
        ax.scatter(params, ades, c="#FF5722", s=100, zorder=5, label="本文模型")
        for x, y, label in zip(params, ades, labels):
            ax.annotate(label, (x, y), textcoords="offset points",
                       xytext=(10, -10), fontsize=8)

    # 绘制基线数据
    baseline_x, baseline_y, baseline_labels = [], [], []
    for name, (p, a) in BASELINE_DATA.items():
        if name.startswith("Ours"):
            continue
        baseline_x.append(p)
        baseline_y.append(a)
        baseline_labels.append(name)

    ax.scatter(baseline_x, baseline_y, c="#2196F3", s=100, marker="s",
               zorder=4, label="SOTA 方法")
    for x, y, label in zip(baseline_x, baseline_y, baseline_labels):
        ax.annotate(label, (x, y), textcoords="offset points",
                   xytext=(10, 5), fontsize=8)

    # 设置对数坐标轴
    ax.set_xscale("log")
    ax.set_xlabel("参数量 (log scale)")
    ax.set_ylabel("minADE (m)")
    ax.set_title("参数效率 Pareto 前沿: 精度 vs 参数量")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  图5 已保存: {output_path}")
