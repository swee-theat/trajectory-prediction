#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第三阶段评估：两车交互 vs 单车对照 + 碰撞指标 + 碰撞可视化。

验收口径（同数据对照）：
    - 组 B ego minADE ≤ 组 A ego minADE + 0.02m（交互不退化精度）
    - 组 B 邻居 minADE 如实报告（neighbor_mask[0]=1 样本）
    - 碰撞指标：true_collision_rate / warn_trigger_rate / warn_precision / warn_recall（阈值 2m）

注意：显式传全输入（车道线/城市/邻居），不用已过时的 eval/evaluator.py。

用法:
    python scripts/eval_stage3_interaction.py
"""

import sys
import os
import json
import argparse
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from models.trajectory_model import TrajectoryModel
from data.dataset import ArgoverseTrajectoryDataset
from data.dataloader import create_dataloader
from eval.metrics import compute_min_ade, compute_min_fde

CATEGORY_NAMES = {0: "直行", 1: "左转", 2: "右转", 3: "掉头", 4: "路口/停车"}

# 两组对照（名称, use_interaction, checkpoint 路径）
GROUPS = [
    ("组A对照(无交互)", False, "outputs/checkpoints/s3_组A对照/best_model.pt"),
    ("组B交互(两车)", True, "outputs/checkpoints/s3_组B交互_scheme2c/best_model.pt"),
]

COLLISION_MARGIN = 2.0  # 碰撞检测阈值（米），与训练碰撞惩罚一致


def build_model(use_interaction: bool) -> TrajectoryModel:
    """按组构造模型（参数须与训练脚本完全一致）。

    方案二（组 B）uncertainty 共享化（share_uncertainty=True）省预算给邻居独立头，
    组 A 保持独立头。
    """
    return TrajectoryModel(
        input_dim=11,
        hidden_dim=128,
        num_mlp_layers=3,
        dropout=0.1,
        num_modes=5,
        pred_len=30,
        pooling_type="attention",
        use_uncertainty=True,
        use_temporal_conv=True,
        share_uncertainty=use_interaction,
        use_lane=True,
        use_city=True,
        use_interaction=use_interaction,
        branch_hidden_dim=96,
    )


def _forward(model, batch, device):
    """构造完整输入并前向，返回 (output, future, neighbor_future, neighbor_mask)。"""
    history = batch["history"].to(device)
    future = batch["future"].to(device)

    kwargs = {}
    if "lane_nodes" in batch:
        kwargs["lane_nodes"] = batch["lane_nodes"].to(device)
        kwargs["lane_adj"] = batch["lane_adj"].to(device)
        kwargs["lane_mask"] = batch["lane_mask"].to(device)
    if "city_id" in batch:
        kwargs["city_id"] = batch["city_id"].to(device)
    if "neighbor_history" in batch:
        kwargs["neighbor_history"] = batch["neighbor_history"].to(device)
        kwargs["neighbor_mask"] = batch["neighbor_mask"].to(device)

    neighbor_future = batch["neighbor_future"].to(device) if "neighbor_future" in batch else None
    neighbor_mask = batch["neighbor_mask"].to(device) if "neighbor_mask" in batch else None

    output = model(history, **kwargs)
    return output, future, neighbor_future, neighbor_mask


def evaluate(model, loader, device, use_interaction: bool) -> dict:
    """评估一组模型：ego 按类别 minADE/minFDE + （组 B）邻居精度 + 碰撞指标。"""
    model.eval()
    per_cat_ade = {name: [] for name in CATEGORY_NAMES.values()}
    overall_ade = []
    overall_fde = []

    nbr_ade = []          # 邻居 minADE（组 B，有邻居样本）
    nbr_fde = []          # 邻居 minFDE
    nbr1_attn_ratio = []  # 邻居1 注意力占比（身份混淆观察）

    # 碰撞指标累积
    tp = fp = fn = tn = 0       # 预警 vs 真实碰撞 混淆矩阵
    n_true_collision = 0
    n_warn = 0
    n_nbr = 0

    with torch.no_grad():
        for batch in loader:
            output, future, neighbor_future, neighbor_mask = _forward(model, batch, device)
            categories = batch["category"]
            B = future.shape[0]

            trajectories = output["trajectories"]  # [B, K, T, 2]
            ego_min_ade = compute_min_ade(trajectories, future)  # [B]
            ego_min_fde = compute_min_fde(trajectories, future)  # [B]

            for i in range(B):
                cat = int(categories[i].item())
                per_cat_ade[CATEGORY_NAMES.get(cat, "直行")].append(ego_min_ade[i].item())
                overall_ade.append(ego_min_ade[i].item())
                overall_fde.append(ego_min_fde[i].item())

            if not use_interaction or "neighbor_trajectories" not in output:
                continue

            # 邻居精度 + 碰撞（只在有 top-1 邻居的样本上统计）
            nbr_traj = output["neighbor_trajectories"]  # [B, K, T, 2]
            if neighbor_future is not None and neighbor_mask is not None:
                valid = neighbor_mask[:, 0] > 0.5  # [B] bool
                if valid.any():
                    n_nbr += int(valid.sum())
                    nbr_min_ade = compute_min_ade(nbr_traj[valid], neighbor_future[valid])
                    nbr_min_fde = compute_min_fde(nbr_traj[valid], neighbor_future[valid])
                    nbr_ade.extend(nbr_min_ade.tolist())
                    nbr_fde.extend(nbr_min_fde.tolist())

                    # 碰撞检测：best mode 的 ego 与邻居轨迹逐帧最小距离
                    ego_best_idx = output["mode_probs"].argmax(-1)  # [B]
                    nbr_best_idx = output["neighbor_mode_probs"].argmax(-1)  # [B]
                    ego_best = trajectories[torch.arange(B, device=device), ego_best_idx]  # [B, T, 2]
                    nbr_best = nbr_traj[torch.arange(B, device=device), nbr_best_idx]      # [B, T, 2]

                    pred_dmin = (ego_best - nbr_best).norm(dim=-1).min(dim=-1).values  # [B]
                    gt_dmin = (future - neighbor_future).norm(dim=-1).min(dim=-1).values  # [B]

                    pred_dmin = pred_dmin[valid]
                    gt_dmin = gt_dmin[valid]

                    true_col = gt_dmin < COLLISION_MARGIN      # 真实碰撞
                    warn = pred_dmin < COLLISION_MARGIN         # 预警触发

                    n_true_collision += int(true_col.sum())
                    n_warn += int(warn.sum())
                    tp += int((warn & true_col).sum())
                    fp += int((warn & ~true_col).sum())
                    fn += int((~warn & true_col).sum())
                    tn += int((~warn & ~true_col).sum())

            # 邻居注意力占比（有 2 个邻居时，邻居1 应占前 20 帧的注意力）
            if "neighbor_attn_weights" in output and neighbor_mask is not None:
                attn = output["neighbor_attn_weights"]  # [B, K*T]
                both = (neighbor_mask[:, 1] > 0.5)      # 有第二邻居的样本
                if both.any():
                    T_obs = attn.shape[1] // 2
                    attn_both = attn[both]
                    ratio = attn_both[:, :T_obs].sum(dim=-1) / (
                        attn_both.sum(dim=-1) + 1e-8
                    )  # 邻居1 注意力占比
                    nbr1_attn_ratio.extend(ratio.tolist())

    result = {
        "overall_min_ade": sum(overall_ade) / len(overall_ade) if overall_ade else 0.0,
        "overall_min_fde": sum(overall_fde) / len(overall_fde) if overall_fde else 0.0,
    }
    for name, vals in per_cat_ade.items():
        if vals:
            result[f"{name}_min_ade"] = sum(vals) / len(vals)
            result[f"{name}_样本数"] = len(vals)

    if use_interaction:
        result["neighbor_min_ade"] = sum(nbr_ade) / len(nbr_ade) if nbr_ade else 0.0
        result["neighbor_min_fde"] = sum(nbr_fde) / len(nbr_fde) if nbr_fde else 0.0
        result["n_neighbor_samples"] = n_nbr
        result["true_collision_rate"] = n_true_collision / n_nbr if n_nbr else 0.0
        result["warn_trigger_rate"] = n_warn / n_nbr if n_nbr else 0.0
        result["warn_precision"] = tp / n_warn if n_warn else 0.0
        result["warn_recall"] = tp / n_true_collision if n_true_collision else 0.0
        result["confusion_tp_fp_fn_tn"] = [tp, fp, fn, tn]
        result["nbr1_attn_ratio_mean"] = (
            sum(nbr1_attn_ratio) / len(nbr1_attn_ratio) if nbr1_attn_ratio else 0.0
        )

    return result


def plot_collision_samples(model, loader, device, output_path: str):
    """可视化有邻居且发生（预测/真实）碰撞的样本：ego/邻居 历史+预测+GT，距离<2m 标红。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    model.eval()
    samples = []
    with torch.no_grad():
        for batch in loader:
            output, future, neighbor_future, neighbor_mask = _forward(model, batch, device)
            if "neighbor_trajectories" not in output or neighbor_mask is None:
                continue
            history = batch["history"].to(device)
            nbr_hist = batch["neighbor_history"].to(device)
            categories = batch["category"]

            B = future.shape[0]
            trajectories = output["trajectories"]
            nbr_traj = output["neighbor_trajectories"]
            ego_best_idx = output["mode_probs"].argmax(-1)
            nbr_best_idx = output["neighbor_mode_probs"].argmax(-1)
            ego_best = trajectories[torch.arange(B, device=device), ego_best_idx]
            nbr_best = nbr_traj[torch.arange(B, device=device), nbr_best_idx]

            valid = neighbor_mask[:, 0] > 0.5
            if not valid.any():
                continue
            pred_dmin = (ego_best - nbr_best).norm(dim=-1).min(dim=-1).values  # [B]
            gt_dmin = (future - neighbor_future).norm(dim=-1).min(dim=-1).values  # [B]

            for i in range(B):
                if not valid[i]:
                    continue
                if pred_dmin[i] < COLLISION_MARGIN or gt_dmin[i] < COLLISION_MARGIN:
                    samples.append({
                        "ego_hist": history[i, :, :2].cpu().numpy(),
                        "ego_gt": future[i].cpu().numpy(),
                        "ego_pred": ego_best[i].cpu().numpy(),
                        "nbr_hist": nbr_hist[i, 0, :, :2].cpu().numpy(),
                        "nbr_gt": neighbor_future[i].cpu().numpy(),
                        "nbr_pred": nbr_best[i].cpu().numpy(),
                        "cat": CATEGORY_NAMES.get(int(categories[i].item()), "直行"),
                    })
                if len(samples) >= 4:
                    break
            if len(samples) >= 4:
                break

    if not samples:
        print("  未找到碰撞样本，跳过可视化")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, s in enumerate(samples[:4]):
        ax = axes[idx]
        # ego
        ax.plot(s["ego_hist"][:, 0], s["ego_hist"][:, 1], "b-", lw=2, label="ego 历史")
        ax.plot(s["ego_gt"][:, 0], s["ego_gt"][:, 1], "b--", lw=1.5, label="ego 真值")
        ax.plot(s["ego_pred"][:, 0], s["ego_pred"][:, 1], "b:", lw=1.5, label="ego 预测")
        # 邻居
        ax.plot(s["nbr_hist"][:, 0], s["nbr_hist"][:, 1], "r-", lw=2, label="邻居 历史")
        ax.plot(s["nbr_gt"][:, 0], s["nbr_gt"][:, 1], "r--", lw=1.5, label="邻居 真值")
        ax.plot(s["nbr_pred"][:, 0], s["nbr_pred"][:, 1], "r:", lw=1.5, label="邻居 预测")
        # 未来帧距离 < 2m 的时刻标红点
        d = ((s["ego_pred"] - s["nbr_pred"]) ** 2).sum(-1) ** 0.5
        t_risk = [t for t in range(len(d)) if d[t] < COLLISION_MARGIN]
        if t_risk:
            ax.scatter(s["ego_pred"][t_risk, 0], s["ego_pred"][t_risk, 1],
                       c="purple", s=60, marker="x", zorder=6, label="碰撞风险帧")
        ax.set_title(f"{s['cat']}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("两车交互碰撞可视化（未来帧最小距离 < 2m 标红）", fontsize=14, y=1.02)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  碰撞可视化已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-b-ckpt", default=None,
                        help="组 B checkpoint 路径（覆盖默认，用于门控等变体对比）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    test_ds = ArgoverseTrajectoryDataset(
        "data/processed/test.npz",
        lane_graphs_path="data/processed/lane_graphs.npz",
    )
    test_loader = create_dataloader(test_ds, batch_size=64, shuffle=False, num_workers=1)
    print(f"测试集样本数: {len(test_ds)}")

    # 可选覆盖组 B checkpoint（如门控变体）
    groups = GROUPS
    if args.group_b_ckpt:
        groups = [(n, u, args.group_b_ckpt if u else c) for n, u, c in GROUPS]

    results = {}
    models = {}
    for name, use_interaction, ckpt_path in groups:
        if not Path(ckpt_path).exists():
            print(f"\n跳过 {name}: checkpoint 不存在 ({ckpt_path})，请先训练")
            continue
        print(f"\n评估: {name}")
        model = build_model(use_interaction)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        models[name] = model

        r = evaluate(model, test_loader, device, use_interaction)
        results[name] = r
        print(f"  总体 minADE: {r['overall_min_ade']:.4f}  总体 minFDE: {r['overall_min_fde']:.4f}")
        for cat in CATEGORY_NAMES.values():
            k = f"{cat}_min_ade"
            if k in r:
                print(f"    {cat}: {r[k]:.4f}  (n={r[f'{cat}_样本数']})")
        if use_interaction:
            print(f"  邻居 minADE: {r['neighbor_min_ade']:.4f}  minFDE: {r['neighbor_min_fde']:.4f}"
                  f"  (n={r['n_neighbor_samples']})")
            print(f"  碰撞指标: 真碰撞率={r['true_collision_rate']:.3f} "
                  f"预警触发率={r['warn_trigger_rate']:.3f} "
                  f"精确率={r['warn_precision']:.3f} 召回率={r['warn_recall']:.3f}")
            print(f"  邻居1注意力占比(均值): {r['nbr1_attn_ratio_mean']:.3f}（有2邻居样本，1.0=完全聚焦邻居1）")

    # 同数据对照结论
    if "组A对照(无交互)" in results and "组B交互(两车)" in results:
        a = results["组A对照(无交互)"]["overall_min_ade"]
        b = results["组B交互(两车)"]["overall_min_ade"]
        diff = b - a
        print("\n" + "=" * 60)
        print(f"同数据对照: 组A minADE={a:.4f}  组B minADE={b:.4f}  差值={diff:+.4f}m")
        if diff <= 0.02:
            print("  结论: 组 B 不退化（≤ 组 A + 0.02m）✅ 交互未牺牲 ego 精度")
            if diff < 0:
                print(f"        交互带来 {abs(diff)*100:.1f}cm 精度增益")
        else:
            print(f"  结论: 组 B 退化 {diff*100:.1f}cm，需调查交互模块")

    # 碰撞可视化（组 B）
    if "组B交互(两车)" in models:
        print("\n生成碰撞可视化...")
        plot_collision_samples(
            models["组B交互(两车)"], test_loader, device,
            "outputs/figures/stage3_collision.png",
        )

    # 保存结果
    out_path = Path("outputs/results/stage3_interaction_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "collision_margin": COLLISION_MARGIN},
                  f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
