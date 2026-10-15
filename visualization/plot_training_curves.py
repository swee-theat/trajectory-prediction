"""图4: 训练曲线 — 训练损失 + 验证 ADE 下降曲线。"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# 中文字体设置
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_training_curves(
    train_losses: list,
    val_ades: list,
    best_epoch: int = None,
    output_path: str = "outputs/figures/fig4_training_curves.png",
):
    """绘制双轴训练曲线。

    参数:
        train_losses: 每个 epoch 的训练损失
        val_ades: 每个 epoch 的验证 minADE
        best_epoch: 最佳 epoch 编号（画竖虚线）
        output_path: 输出路径
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    epochs = np.arange(1, len(train_losses) + 1)

    # 左轴：训练损失
    color1 = "#2196F3"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("训练损失", color=color1)
    ax1.plot(epochs, train_losses, color=color1, linewidth=1.5, label="训练损失")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # 右轴：验证 minADE
    ax2 = ax1.twinx()
    color2 = "#FF5722"
    ax2.set_ylabel("验证 minADE", color=color2)
    ax2.plot(epochs, val_ades, color=color2, linewidth=1.5, label="验证 minADE")
    ax2.tick_params(axis="y", labelcolor=color2)

    # 最佳 epoch 标记
    if best_epoch is not None and best_epoch < len(epochs):
        ax1.axvline(x=best_epoch + 1, color="green", linestyle="--",
                     alpha=0.7, label=f"最佳 Epoch {best_epoch + 1}")

    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("训练曲线: 损失 + 验证 minADE")
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  图4 已保存: {output_path}")
