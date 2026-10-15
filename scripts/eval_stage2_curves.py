#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二阶段弯道评估：三组模型在 test 集上按类别统计 minADE，计算弯道降幅。

验收硬指标：
    - 总体 minADE < 0.8m
    - 弯道（左转 + 右转 + 掉头）minADE 降幅 ≥ 50%（相对①仅轨迹）

注意：第一阶段的 eval/evaluator.py 只传 history、未传车道线/城市输入，
会让第二阶段模型退化为「不融合车道线」而失真。故本脚本独立实现评估循环，
显式把 lane_nodes / lane_adj / lane_mask / city_id 传给 model.forward。

用法:
    python scripts/eval_stage2_curves.py
"""

import sys
import os
import json
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from models.trajectory_model import TrajectoryModel
from data.dataset import ArgoverseTrajectoryDataset
from data.dataloader import create_dataloader
from eval.metrics import compute_min_ade, compute_min_fde

CATEGORY_NAMES = {0: "直行", 1: "左转", 2: "右转", 3: "掉头", 4: "路口/停车"}
CURVE_CATS = ["左转", "右转", "掉头"]

# 三组消融配置（与 scripts/run_ablation_stage2.py 的 ABLATIONS 保持一致）
GROUPS = [
    ("仅轨迹(基线)", False, False, "outputs/checkpoints/s2_仅轨迹基线/best_model.pt"),
    ("+车道线", True, False, "outputs/checkpoints/s2_加车道线/best_model.pt"),
    ("+车道线+城市特征", True, True, "outputs/checkpoints/s2_加车道线加城市特征/best_model.pt"),
]


def build_model(use_lane: bool, use_city: bool) -> TrajectoryModel:
    """按消融组构造模型（参数须与训练时完全一致，否则 state_dict 形状不匹配）。"""
    return TrajectoryModel(
        input_dim=11,
        hidden_dim=128,
        num_mlp_layers=3,
        dropout=0.1,
        num_modes=5,               # K=5，须与训练 checkpoint 一致
        pred_len=30,
        pooling_type="attention",
        use_uncertainty=True,
        use_temporal_conv=True,
        share_uncertainty=False,   # 独立头
        use_lane=use_lane,
        use_city=use_city,
        branch_hidden_dim=96,      # 压缩 predictor
    )


def evaluate(model, loader, device):
    """在测试集上评估，按类别统计 minADE / minFDE。"""
    model.eval()
    per_cat = {name: [] for name in CATEGORY_NAMES.values()}
    overall_ade = []
    overall_fde = []

    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)
            categories = batch["category"]

            # 第二阶段可选输入（车道线 + 城市）
            kwargs = {}
            if "lane_nodes" in batch:
                kwargs["lane_nodes"] = batch["lane_nodes"].to(device)
                kwargs["lane_adj"] = batch["lane_adj"].to(device)
                kwargs["lane_mask"] = batch["lane_mask"].to(device)
            if "city_id" in batch:
                kwargs["city_id"] = batch["city_id"].to(device)

            output = model(history, **kwargs)
            trajectories = output["trajectories"]       # [B, K, T, 2]
            min_ade = compute_min_ade(trajectories, future)   # [B]
            min_fde = compute_min_fde(trajectories, future)   # [B]

            B = trajectories.shape[0]
            for i in range(B):
                cat = int(categories[i].item())
                cat_name = CATEGORY_NAMES.get(cat, "未知")
                per_cat[cat_name].append(min_ade[i].item())
                overall_ade.append(min_ade[i].item())
                overall_fde.append(min_fde[i].item())

    result = {
        "overall_min_ade": sum(overall_ade) / len(overall_ade),
        "overall_min_fde": sum(overall_fde) / len(overall_fde),
    }
    for name, vals in per_cat.items():
        if vals:
            result[f"{name}_min_ade"] = sum(vals) / len(vals)
            result[f"{name}_样本数"] = len(vals)
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    test_ds = ArgoverseTrajectoryDataset(
        "data/processed/test.npz",
        lane_graphs_path="data/processed/lane_graphs.npz",
    )
    test_loader = create_dataloader(
        test_ds, batch_size=64, shuffle=False, num_workers=1
    )
    print(f"测试集样本数: {len(test_ds)}")

    results = {}
    for name, use_lane, use_city, ckpt_path in GROUPS:
        print(f"\n评估: {name}")
        model = build_model(use_lane, use_city)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)

        r = evaluate(model, test_loader, device)
        results[name] = r
        print(f"  总体 minADE: {r['overall_min_ade']:.4f}")
        for cat in CATEGORY_NAMES.values():
            k = f"{cat}_min_ade"
            if k in r:
                print(f"    {cat}: {r[k]:.4f}  (n={r[f'{cat}_样本数']})")

    # 弯道降幅（相对①仅轨迹）
    base = results["仅轨迹(基线)"]
    lane = results["+车道线"]
    print("\n" + "=" * 60)
    print("弯道降幅（+车道线 相对 ①仅轨迹）")
    print("=" * 60)
    for cat in CURVE_CATS:
        b = base[f"{cat}_min_ade"]
        l = lane[f"{cat}_min_ade"]
        drop = (b - l) / b * 100
        print(f"  {cat}: {b:.4f} -> {l:.4f}   降幅 {drop:+.1f}%")

    # 弯道整体（按样本数加权）
    def curve_weighted(r):
        total = 0.0
        n = 0
        for cat in CURVE_CATS:
            k = f"{cat}_min_ade"
            if k in r:
                cnt = r[f"{cat}_样本数"]
                total += r[k] * cnt
                n += cnt
        return total / n if n else 0.0

    base_curve = curve_weighted(base)
    lane_curve = curve_weighted(lane)
    drop_overall = (base_curve - lane_curve) / base_curve * 100
    print(f"\n  弯道整体(加权): {base_curve:.4f} -> {lane_curve:.4f}   降幅 {drop_overall:+.1f}%")

    # 保存结果
    out_path = Path("outputs/results/stage2_curve_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": results, "curve_cats": CURVE_CATS}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
