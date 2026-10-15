#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查车道线在注意力池化中是否被模型利用。

加载 +车道线 模型，在 test 集上统计注意力权重在「轨迹帧(20)」vs「车道节点(32)」的分布。
判断标准：softmax 后 52 个权重之和为 1；均匀分布下车道节点(32)应占 61.5%。
若车道节点权重占比接近 0，说明车道线信息被模型忽略（融合无效）。

用法:
    python scripts/check_lane_attention.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from models.trajectory_model import TrajectoryModel
from data.dataset import ArgoverseTrajectoryDataset
from data.dataloader import create_dataloader

CATEGORY_NAMES = {0: "直行", 1: "左转", 2: "右转", 3: "掉头", 4: "路口/停车"}
T_OBS = 20     # 轨迹帧数
N_LANE = 32    # 车道节点数


def build_model() -> TrajectoryModel:
    """构造 +车道线 模型（参数与训练时一致）。"""
    return TrajectoryModel(
        input_dim=11, hidden_dim=128, num_mlp_layers=3, dropout=0.1,
        num_modes=5, pred_len=30, pooling_type="attention",
        use_uncertainty=True, use_temporal_conv=True,
        share_uncertainty=False, use_lane=True, use_city=False,
        branch_hidden_dim=96,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    ckpt = torch.load(
        "outputs/checkpoints/s2_加车道线/best_model.pt",
        map_location="cpu", weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    ds = ArgoverseTrajectoryDataset(
        "data/processed/test.npz", lane_graphs_path="data/processed/lane_graphs.npz"
    )
    loader = create_dataloader(ds, batch_size=64, shuffle=False, num_workers=1)

    total_n = 0
    total_traj = 0.0
    total_lane = 0.0
    per_cat_lane = {name: [] for name in CATEGORY_NAMES.values()}

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= 40:   # 约 2560 个样本，足够统计
                break
            history = batch["history"].to(device)
            categories = batch["category"]

            out = model(
                history,
                lane_nodes=batch["lane_nodes"].to(device),
                lane_adj=batch["lane_adj"].to(device),
                lane_mask=batch["lane_mask"].to(device),
                return_attn_weights=True,
            )
            attn = out["attn_weights"]                 # [B, T+N] = [B, 52]
            traj_share = attn[:, :T_OBS].sum(dim=1)     # 轨迹帧权重占比 [B]
            lane_share = attn[:, T_OBS:].sum(dim=1)     # 车道节点权重占比 [B]

            total_n += attn.shape[0]
            total_traj += traj_share.sum().item()
            total_lane += lane_share.sum().item()

            for i in range(attn.shape[0]):
                cat = int(categories[i].item())
                per_cat_lane[CATEGORY_NAMES[cat]].append(lane_share[i].item())

    print(f"统计样本数: {total_n}")
    print(f"轨迹帧(20) 权重占比: {total_traj / total_n:.4f}")
    print(f"车道节点(32) 权重占比: {total_lane / total_n:.4f}")
    print(f"均匀基线(车道 32/52): 0.6154")
    print(f"平均每轨迹帧权重: {total_traj / total_n / T_OBS:.4f}")
    print(f"平均每车道节点权重: {total_lane / total_n / N_LANE:.4f}")
    print()
    print("分类别车道节点权重占比:")
    for name in CATEGORY_NAMES.values():
        vals = per_cat_lane[name]
        if vals:
            print(f"  {name}: {sum(vals) / len(vals):.4f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
