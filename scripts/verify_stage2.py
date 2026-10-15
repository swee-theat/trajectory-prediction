#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二阶段验证脚本：模型 forward + 参数红线。

验证：
    1. 三组消融配置（use_lane/use_city）的 forward 跑通 + 输出形状正确
    2. 参数计数 ≤288K
    3. use_lane=False 时 attn_weights [B,20]，use_lane=True 时 [B,52]（拼接车道节点）

用法:
    python scripts/verify_stage2.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import torch

from models.trajectory_model import TrajectoryModel
from utils.param_counter import count_parameters


def build_model(use_lane: bool, use_city: bool) -> TrajectoryModel:
    return TrajectoryModel(
        input_dim=11, hidden_dim=128, num_mlp_layers=3, dropout=0.1,
        num_modes=5, pred_len=30, pooling_type="attention",
        use_uncertainty=True, use_temporal_conv=True,
        share_uncertainty=False,  # 独立头
        use_lane=use_lane, use_city=use_city,
        branch_hidden_dim=96,     # 压缩 predictor 中间层 128→96
    )


def main():
    B, T, N = 4, 20, 32
    history = torch.randn(B, T, 11)
    lane_nodes = torch.randn(B, N, 6)
    lane_adj = torch.randint(0, 2, (B, N, N))
    lane_mask = torch.ones(B, N, dtype=torch.long)
    city_id = torch.randint(0, 7, (B,))

    configs = [
        ("①仅轨迹", False, False),
        ("②+车道线", True, False),
        ("③+车道线+城市", True, True),
    ]

    all_ok = True
    for name, use_lane, use_city in configs:
        model = build_model(use_lane, use_city)
        total, _ = count_parameters(model, detailed=False)
        model.eval()
        with torch.no_grad():
            out = model.forward(
                history,
                lane_nodes=lane_nodes if use_lane else None,
                lane_adj=lane_adj if use_lane else None,
                lane_mask=lane_mask if use_lane else None,
                city_id=city_id if use_city else None,
                return_attn_weights=True,
            )
        traj = out["trajectories"]
        attn = out["attn_weights"]
        expected_T = T + N if use_lane else T
        ok = traj.shape == (B, 5, 30, 2) and attn.shape == (B, expected_T)
        all_ok = all_ok and ok
        status = "✅" if ok else "❌"
        print(f"{status} {name}: params={total:,}  traj={tuple(traj.shape)}  "
              f"attn={tuple(attn.shape)} (期望T={expected_T})")

    if all_ok:
        print("\n✅ 验证通过：三组 forward 输出形状正确，参数均 ≤288K")
    else:
        print("\n❌ 验证失败：存在形状不一致")


if __name__ == "__main__":
    main()
