"""图8: 轨迹预测效果可视化 — 8个典型场景的轨迹叠加图。"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from pathlib import Path

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

CATEGORY_NAMES = {0: "直行", 1: "左转", 2: "右转", 3: "掉头", 4: "路口/停车"}


def plot_trajectory_samples(
    model,
    dataloader,
    num_samples: int = 8,
    output_path: str = "outputs/figures/fig8_trajectories.png",
    device: str = "cuda",
):
    """挑选并可视化 8 个代表性样本的轨迹预测结果。

    参数:
        model: 训练好的 TrajectoryModel
        dataloader: 测试集 DataLoader
        num_samples: 可视化样本数（2x4 网格）
        output_path: 输出路径
    """
    model.eval()
    samples = []

    # 收集样本
    with torch.no_grad():
        for batch in dataloader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)
            categories = batch["category"]

            output = model(history)
            trajectories = output["trajectories"].cpu().numpy()  # [B, K, T, 2]
            mode_probs = output["mode_probs"].cpu().numpy()      # [B, K]
            history_np = history.cpu().numpy()                    # [B, T_obs, 9]
            future_np = future.cpu().numpy()                      # [B, T_pred, 2]

            for i in range(len(history)):
                cat = int(categories[i].item())
                samples.append({
                    "history": history_np[i, :, :2],      # [T_obs, 2]
                    "future_gt": future_np[i],              # [T_pred, 2]
                    "trajectories": trajectories[i],        # [K, T_pred, 2]
                    "mode_probs": mode_probs[i],            # [K]
                    "category": cat,
                    "category_name": CATEGORY_NAMES.get(cat, "未知"),
                })
                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                break

    # 绘制 2x4 网格
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    colors = ["#2196F3", "#FF5722", "#4CAF50"]  # 3 个模态的颜色

    for idx, sample in enumerate(samples[:num_samples]):
        ax = axes[idx]

        # 历史轨迹（蓝色实线）
        hist = sample["history"]
        ax.plot(hist[:, 0], hist[:, 1], "b-", linewidth=2, label="历史轨迹")
        ax.scatter(hist[-1, 0], hist[-1, 1], c="blue", s=50, marker="o", zorder=5)

        # 真实未来（黑色实线）
        gt = sample["future_gt"]
        ax.plot(gt[:, 0], gt[:, 1], "k-", linewidth=2, label="真实轨迹")
        ax.scatter(gt[-1, 0], gt[-1, 1], c="black", s=50, marker="*", zorder=5)

        # K 个预测模态（彩色虚线）
        for k in range(len(sample["trajectories"])):
            traj = sample["trajectories"][k]
            prob = sample["mode_probs"][k]
            ax.plot(traj[:, 0], traj[:, 1], "--", color=colors[k % 3],
                    linewidth=1.5, alpha=0.7, label=f"模态{k+1} (p={prob:.2f})")

        ax.set_title(f"{sample['category_name']}", fontsize=11)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("轨迹预测效果可视化 (8 个典型场景)", fontsize=14, y=1.02)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  图8 已保存: {output_path}")
