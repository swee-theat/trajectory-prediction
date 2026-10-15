"""图10: 失败案例分析 — 掉头和路口场景的预测偏差。"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from pathlib import Path

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_failure_cases(
    model,
    dataloader,
    output_path: str = "outputs/figures/fig10_failures.png",
    device: str = "cuda",
    max_failure_samples: int = 4,
):
    """查找并可视化 FDE 最大的失败案例。

    参数:
        model: 训练好的模型
        dataloader: 测试集 DataLoader
        output_path: 输出路径
        max_failure_samples: 最大失败样本数
    """
    model.eval()

    failures = []

    with torch.no_grad():
        for batch in dataloader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)
            categories = batch["category"]

            output = model(history)
            trajectories = output["trajectories"].cpu().numpy()
            mode_probs = output["mode_probs"].cpu().numpy()

            for i in range(len(history)):
                cat = int(categories[i].item())
                # 仅关注掉头和路口
                if cat not in [3, 4]:
                    continue

                gt = future[i].cpu().numpy()
                # 找到最佳模态的 FDE
                best_k = mode_probs[i].argmax()
                best_fde = np.linalg.norm(trajectories[i, best_k, -1] - gt[-1])

                failures.append({
                    "history": history[i, :, :2].cpu().numpy(),
                    "future_gt": gt,
                    "trajectories": trajectories[i],
                    "mode_probs": mode_probs[i],
                    "best_fde": best_fde,
                    "category": "掉头" if cat == 3 else "路口/停车",
                })

        # 按 FDE 降序排列，取前 max_failure_samples
        failures.sort(key=lambda x: -x["best_fde"])
        failures = failures[:max_failure_samples]

    # 绘制 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for idx, sample in enumerate(failures):
        ax = axes[idx]

        hist = sample["history"]
        gt = sample["future_gt"]
        ax.plot(hist[:, 0], hist[:, 1], "b-", linewidth=2, label="历史轨迹")
        ax.plot(gt[:, 0], gt[:, 1], "k-", linewidth=2, label="真实轨迹")

        for k in range(len(sample["trajectories"])):
            traj = sample["trajectories"][k]
            prob = sample["mode_probs"][k]
            ax.plot(traj[:, 0], traj[:, 1], "--", color=colors[k % 3],
                    linewidth=1.5, alpha=0.7, label=f"模态{k+1}")

        ax.set_title(f"失败案例: {sample['category']} | FDE={sample['best_fde']:.2f}m")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("失败案例: 掉头/路口场景预测偏差", fontsize=14)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  图10 已保存: {output_path}")
