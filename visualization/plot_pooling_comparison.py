"""图3: 池化方式对比示意图 — 注意力池化 vs 均值池化 vs LSTM。"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_pooling_comparison(
    ablation_results: list,
    output_path: str = "outputs/figures/fig3_pooling.png",
):
    """绘制池化方式对比柱状图。

    参数:
        ablation_results: 消融实验结果（需包含注意力/均值/LSTM三种池化的数据）
        output_path: 输出路径
    """
    pooling_data = {
        r["实验名称"]: (r.get("测试minADE", 0), r.get("测试minFDE", 0))
        for r in ablation_results
        if "池化" in r.get("实验名称", "")
    }

    if not pooling_data:
        print("  图3: 缺少池化实验数据，跳过")
        return

    names = list(pooling_data.keys())
    ades = [pooling_data[n][0] for n in names]
    fdes = [pooling_data[n][1] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    width = 0.3

    ax.bar(x - width / 2, ades, width, label="minADE", color="#2196F3")
    ax.bar(x + width / 2, fdes, width, label="minFDE", color="#FF5722")

    ax.set_xlabel("池化方式")
    ax.set_ylabel("误差 (m)")
    ax.set_title("池化方式对预测精度的影响")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  图3 已保存: {output_path}")


def plot_mode_distribution(
    ablation_results: list,
    output_path: str = "outputs/figures/mode_distribution.png",
):
    """绘制不同场景下模态概率分布的补充图。"""
    # 此图需要逐样本的 mode_probs 数据，在完整评估后生成
    # 这里是占位实现
    pass
