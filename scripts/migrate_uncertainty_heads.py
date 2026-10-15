#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性权重迁移脚本：把旧共享不确定性头权重复制 5 份到新独立头结构。

背景：
    M2 审计将 uncertainty_estimator 从 share_across_modes=True（共享头）
    改为 False（K=5 独立头），模型结构变化导致旧 checkpoint 无法直接加载。
    本脚本把共享头权重复制到 5 个独立头，使迁移后各模态不确定性头
    初始权重完全相同（等价于旧的共享行为），后续可在 Stage B 微调中
    由 NLL 损失驱动逐渐分化出各模态不同的方差。

用法:
    python scripts/migrate_uncertainty_heads.py

输入:
    outputs/checkpoints/full_train_K5/best_model.pt   （旧共享头 checkpoint）

输出:
    outputs/checkpoints/full_train_K5/best_model_indep.pt （新独立头 checkpoint）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from models.trajectory_model import TrajectoryModel

OLD_CKPT = "outputs/checkpoints/full_train_K5/best_model.pt"
NEW_CKPT = "outputs/checkpoints/full_train_K5/best_model_indep.pt"
NUM_MODES = 5
# 独立头 Sequential 里 Linear 层的索引（0 和 3，中间是 ReLU/Dropout）
LINEAR_LAYER_IDX = [0, 3]


def main():
    # 1. 加载旧 checkpoint
    old = torch.load(OLD_CKPT, map_location="cpu", weights_only=False)
    old_sd = old["model_state_dict"]

    # 2. 构造新模型（独立不确定性头）
    model = TrajectoryModel(
        input_dim=11, hidden_dim=128, num_mlp_layers=3, dropout=0.1,
        num_modes=NUM_MODES, pred_len=30, pooling_type="attention",
        use_uncertainty=True, use_temporal_conv=True,
    )
    new_sd = model.state_dict()

    # 3. 非不确定性层：直接复制（这些层结构未变）
    copied = 0
    for key, value in old_sd.items():
        if key.startswith("uncertainty.uncertainty_head"):
            continue  # 旧共享头，跳过
        if key in new_sd:
            new_sd[key] = value
            copied += 1

    # 4. 共享头 → 复制 5 份到独立头
    shared_prefix = "uncertainty.uncertainty_head"
    for k in range(NUM_MODES):
        for layer_idx in LINEAR_LAYER_IDX:
            for suffix in ["weight", "bias"]:
                old_key = f"{shared_prefix}.{layer_idx}.{suffix}"
                new_key = f"uncertainty.uncertainty_heads.{k}.{layer_idx}.{suffix}"
                if old_key not in old_sd:
                    raise KeyError(f"旧 checkpoint 缺少 {old_key}")
                new_sd[new_key] = old_sd[old_key].clone()

    model.load_state_dict(new_sd)

    # 5. 保存（保留 epoch/metrics，不保留 optimizer/scheduler，Stage B 会重建）
    new_ckpt = {
        "epoch": old.get("epoch"),
        "model_state_dict": model.state_dict(),
        "metrics": old.get("metrics"),
    }
    torch.save(new_ckpt, NEW_CKPT)

    # 6. 验证
    n_params = sum(p.numel() for p in model.parameters())
    w0 = model.uncertainty.uncertainty_heads[0][0].weight
    w4 = model.uncertainty.uncertainty_heads[4][0].weight
    max_diff = (w0 - w4).abs().max().item()

    print("=" * 58)
    print("  权重迁移完成")
    print("=" * 58)
    print(f"  输入: {OLD_CKPT}")
    print(f"  输出: {NEW_CKPT}")
    print(f"  复制非不确定性层: {copied} 个")
    print(f"  共享头 → {NUM_MODES} 个独立头")
    print(f"  新模型参数量: {n_params:,}")
    print(f"  验证 5 个头初始权重相同: 最大差 {max_diff:.10f} (应≈0)")


if __name__ == "__main__":
    main()
