"""图6: 消融实验柱状图 — 各消融条件下的 ADE/FDE 对比。"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_ablation_bars(
    ablation_results: list,
    baseline_ade: float = None,
    baseline_fde: float = None,
    output_path: str = "outputs/figures/fig6_ablation.png",
):
    """绘制分组的消融实验柱状图。

    参数:
        ablation_results: list of dict，每个 dict 包含 实验名称, 测试minADE, 测试minFDE
        baseline_ade: 完整模型的 minADE（画水平虚线）
        baseline_fde: 完整模型的 minFDE（画水平虚线）
        output_path: 输出路径
    """
    names = [r["实验名称"] for r in ablation_results]
    ades = [r.get("测试minADE", 0) for r in ablation_results]
    fdes = [r.get("测试minFDE", 0) for r in ablation_results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width / 2, ades, width, label="minADE", color="#2196F3")
    bars2 = ax.bar(x + width / 2, fdes, width, label="minFDE", color="#FF5722")

    # 基线水平线
    if baseline_ade is not None:
        ax.axhline(y=baseline_ade, color="#2196F3", linestyle="--",
                   alpha=0.5, linewidth=1, label=f"基线 minADE={baseline_ade:.3f}")
    if baseline_fde is not None:
        ax.axhline(y=baseline_fde, color="#FF5722", linestyle="--",
                   alpha=0.5, linewidth=1, label=f"基线 minFDE={baseline_fde:.3f}")

    ax.set_xlabel("实验条件")
    ax.set_ylabel("误差 (m)")
    ax.set_title("消融实验: 各组件对预测精度的影响")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  图6 已保存: {output_path}")
