"""图9: 不确定性估计可视化 — 高/低置信度场景对比。"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from pathlib import Path

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_uncertainty_comparison(
    model,
    dataloader,
    output_path: str = "outputs/figures/fig9_uncertainty.png",
    device: str = "cuda",
):
    """挑选高/低置信度样本，可视化不确定性椭圆。

    参数:
        model: 训练好的模型
        dataloader: 测试集 DataLoader
        output_path: 输出路径
    """
    model.eval()

    high_conf_samples = []
    low_conf_samples = []

    with torch.no_grad():
        for batch in dataloader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)

            output = model(history)
            mode_probs = output["mode_probs"].cpu().numpy()
            trajectories = output["trajectories"].cpu().numpy()
            uncertainties = output.get("uncertainties")
            if uncertainties is not None:
                uncertainties = uncertainties.cpu().numpy()
            history_np = history.cpu().numpy()

            for i in range(len(history)):
                max_prob = mode_probs[i].max()
                best_mode = mode_probs[i].argmax()
                sample = {
                    "history": history_np[i, :, :2],
                    "future_gt": future[i].cpu().numpy(),
                    "best_traj": trajectories[i, best_mode],
                    "mode_probs": mode_probs[i],
                    "uncertainty": (uncertainties[i, best_mode]
                                    if uncertainties is not None else None),
                }
                if max_prob > 0.8 and len(high_conf_samples) < 2:
                    high_conf_samples.append(sample)
                elif max_prob < 0.4 and len(low_conf_samples) < 2:
                    low_conf_samples.append(sample)
            if len(high_conf_samples) >= 2 and len(low_conf_samples) >= 2:
                break

    # 绘制 2x2 对比
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    row_labels = ["高置信度 (p>0.8)", "低置信度 (p<0.4)"]
    all_samples = [high_conf_samples, low_conf_samples]

    for row_idx, (samples, row_title) in enumerate(zip(all_samples, row_labels)):
        for col_idx, sample in enumerate(samples[:2]):
            ax = axes[row_idx, col_idx]

            hist = sample["history"]
            gt = sample["future_gt"]
            traj = sample["best_traj"]

            ax.plot(hist[:, 0], hist[:, 1], "b-", linewidth=2, label="历史")
            ax.plot(gt[:, 0], gt[:, 1], "k-", linewidth=2, label="真值")
            ax.plot(traj[:, 0], traj[:, 1], "r--", linewidth=2, label="预测")

            # 绘制不确定性椭圆（在 t+5, t+15, t+29 处）
            if sample["uncertainty"] is not None:
                for t_idx in [5, 15, 28]:
                    if t_idx < len(traj):
                        cx, cy = traj[t_idx]
                        sx = np.sqrt(max(sample["uncertainty"][t_idx, 0], 1e-6)) * 2
                        sy = np.sqrt(max(sample["uncertainty"][t_idx, 1], 1e-6)) * 2
                        ellipse = plt.matplotlib.patches.Ellipse(
                            (cx, cy), sx, sy, fill=False, color="red",
                            alpha=0.3, linewidth=0.8,
                        )
                        ax.add_patch(ellipse)

            ax.set_title(f"{row_title} | 概率={sample['mode_probs'].max():.2f}")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.axis("equal")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    fig.suptitle("不确定性估计: 高/低置信度场景对比", fontsize=14)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  图9 已保存: {output_path}")
