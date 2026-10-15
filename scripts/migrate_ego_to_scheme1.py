#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""方案一收尾：把组 A 的 ego 权重迁到 use_interaction=True 的模型上。

方案一（wen_ti/04）核心结论：交互 context 对 ego 无增益，注入只会过拟合退化。
forward 已改为 ego 不经过 interaction（ego_feat = encoded，等价组 A），
因此最干净、最确定的方案一是复用组 A 的 ego 权重：
    - 组 A checkpoint（ego 0.8312）strict=False 加载到方案一模型
    - missing 全为 interaction.* / context_gate（随机初始化，不影响 ego）
    - ego 输出精确 = 组 A 0.8312，确定性过验收线（≤ 0.8512）

邻居预测仍复用 predictor（崩 17.49），由方案二（邻居独立预测头）解决。

用法:
    python scripts/migrate_ego_to_scheme1.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from models.trajectory_model import TrajectoryModel

GROUP_A_CKPT = "outputs/checkpoints/s3_组A对照/best_model.pt"
OUT_CKPT = "outputs/checkpoints/s3_组B交互_noegointer/best_model.pt"


def main():
    # 方案一模型（use_interaction=True，参数与训练/评估脚本完全一致）
    model = TrajectoryModel(
        input_dim=11,
        hidden_dim=128,
        num_mlp_layers=3,
        dropout=0.1,
        num_modes=5,
        pred_len=30,
        pooling_type="attention",
        use_uncertainty=True,
        use_temporal_conv=True,
        share_uncertainty=False,
        use_lane=True,
        use_city=True,
        use_interaction=True,
        branch_hidden_dim=96,
    )

    ckpt = torch.load(GROUP_A_CKPT, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected 键（方案一没有而组 A 有）: {unexpected}"
    allowed = ("interaction.", "context_gate")
    assert all(k.startswith(allowed) for k in missing), (
        f"missing 键应全为 interaction.* / context_gate，实际: {missing}"
    )
    print(f"迁移成功：missing={missing}")
    print(f"           unexpected={unexpected}")

    out = {"model_state_dict": model.state_dict()}
    for k in ("epoch", "metrics", "optimizer_state_dict", "scheduler_state_dict"):
        if k in ckpt:
            out[k] = ckpt[k]
    os.makedirs(os.path.dirname(OUT_CKPT), exist_ok=True)
    torch.save(out, OUT_CKPT)
    print(f"已保存方案一 checkpoint: {OUT_CKPT}")
    print("ego 路径 = 组 A（encoded + city → predictor），复用组 A 权重后 ego 精确 = 0.8312")


if __name__ == "__main__":
    main()
