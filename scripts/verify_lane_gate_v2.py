#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""01 车道线融合干净验证 v2：从「仅轨迹」checkpoint 迁移 encoder/predictor。

与 v1（verify_lane_gate.py）的区别：
    v1 从「+车道线」checkpoint 迁移，encoder.pooling.query 是针对 52 个位置
    （20 轨迹帧 + 32 车道节点）训练的，改成新融合后输入变 20 帧纯轨迹，
    query 不适配，污染了 baseline（直行 0.8591 vs 仅轨迹 0.8359）。

    v2 从「仅轨迹」checkpoint 迁移 encoder/predictor/uncertainty（query 本就
    针对 20 帧训练），lane_encoder 单独从「+车道线」checkpoint 取（GCN 已训练好），
    使 gate=0 时模型严格等价「仅轨迹」——干净的 baseline。

用法:
    python scripts/verify_lane_gate_v2.py --subset 100000 --epochs 8 --lr 3e-3
"""

import sys
import os
import json
import time
import argparse
import torch
from torch.utils.data import Subset
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from utils.config import load_config, config_to_dict
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.param_counter import count_parameters
from models.trajectory_model import TrajectoryModel
from data.dataset import ArgoverseTrajectoryDataset
from data.dataloader import create_dataloader
from train.trainer import Trainer
from eval.metrics import compute_min_ade

BASE_CKPT = "outputs/checkpoints/s2_仅轨迹基线/best_model.pt"   # encoder/predictor/uncertainty
LANE_CKPT = "outputs/checkpoints/s2_加车道线/best_model.pt"      # lane_encoder

CATEGORY_NAMES = {0: "直行", 1: "左转", 2: "右转", 3: "掉头", 4: "路口/停车"}
CURVE_CATS = ["左转", "右转", "掉头"]

# 旧融合 baseline（来自 eval_stage2_curves.py，test 98,166 样本）
BASELINE_CURVE_TRAJ = 1.4624       # 仅轨迹 弯道加权 minADE
BASELINE_CURVE_LANE_OLD = 1.4382   # 旧融合 +车道线 弯道加权 minADE（降幅 1.7%）


def freeze_except_lane_gate(model):
    """冻结除车道线独立池化 + gate 外的所有参数（wen_ti/01 验证）。"""
    frozen = trainable = 0
    names = []
    for name, param in model.named_parameters():
        if name.startswith("lane_pooling") or name == "lane_gate":
            param.requires_grad = True
            trainable += param.numel()
            names.append(name)
        else:
            param.requires_grad = False
            frozen += param.numel()
    print(f"  冻结: {frozen:,}  可训练(车道线融合): {trainable:,}")
    print(f"  可训练参数名: {names}")
    assert all(n.startswith("lane_pooling") or n == "lane_gate" for n in names), \
        f"冻结名单有遗漏: {names}"
    return trainable


def evaluate_curve(model, loader, device):
    """在 test 集评估，返回各类别 minADE 与弯道加权 minADE。"""
    model.eval()
    per_cat = {name: [] for name in CATEGORY_NAMES.values()}
    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)
            categories = batch["category"]

            kwargs = {}
            if "lane_nodes" in batch:
                kwargs["lane_nodes"] = batch["lane_nodes"].to(device)
                kwargs["lane_adj"] = batch["lane_adj"].to(device)
                kwargs["lane_mask"] = batch["lane_mask"].to(device)

            output = model(history, **kwargs)
            trajectories = output["trajectories"]
            min_ade = compute_min_ade(trajectories, future)  # [B]

            for i in range(trajectories.shape[0]):
                cat = int(categories[i].item())
                per_cat[CATEGORY_NAMES.get(cat, "未知")].append(min_ade[i].item())

    result = {}
    for name, vals in per_cat.items():
        if vals:
            result[name] = sum(vals) / len(vals)
            result[f"{name}_n"] = len(vals)

    curve_total = curve_n = 0.0
    for cat in CURVE_CATS:
        if cat in result:
            curve_total += result[cat] * result[f"{cat}_n"]
            curve_n += result[f"{cat}_n"]
    result["弯道加权"] = curve_total / curve_n if curve_n else 0.0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=100000, help="train 子集大小")
    parser.add_argument("--epochs", type=int, default=8, help="微调 epoch 数")
    parser.add_argument("--lr", type=float, default=3e-3, help="学习率")
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    cfg = config_to_dict(load_config("configs/default.yaml"))
    mc = cfg["model"]

    # ── 数据 ──
    train_ds = ArgoverseTrajectoryDataset(
        "data/processed/train.npz", lane_graphs_path="data/processed/lane_graphs.npz")
    val_ds = ArgoverseTrajectoryDataset(
        "data/processed/val.npz", lane_graphs_path="data/processed/lane_graphs.npz")
    n_sub = min(args.subset, len(train_ds))
    train_sub = Subset(train_ds, range(n_sub))
    print(f"train 子集: {n_sub} 样本（全量 {len(train_ds)}）")
    train_loader = create_dataloader(train_sub, batch_size=64, shuffle=True, num_workers=1)
    val_loader = create_dataloader(val_ds, batch_size=64, shuffle=False, num_workers=1)

    # ── 模型（+车道线 + 新融合）──
    model = TrajectoryModel(
        input_dim=mc["input_dim"], hidden_dim=mc["hidden_dim"],
        num_mlp_layers=mc.get("num_mlp_layers", 3), dropout=mc.get("dropout", 0.1),
        num_modes=5, pred_len=cfg["data"].get("pred_len", 30),
        pooling_type="attention", use_uncertainty=True,
        use_temporal_conv=True, share_uncertainty=False,
        use_lane=True, use_city=False,
        use_lane_gate=True,
        branch_hidden_dim=96,
    )
    total_params, _ = count_parameters(model)
    print(f"模型参数量: {total_params:,}")

    # ── 迁移：encoder/predictor/uncertainty 从 base，lane_encoder 从 lane ──
    base = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    lane = torch.load(LANE_CKPT, map_location="cpu", weights_only=False)
    merged = dict(base["model_state_dict"])
    n_lane = 0
    for k, v in lane["model_state_dict"].items():
        if k.startswith("lane_encoder"):
            merged[k] = v
            n_lane += 1
    print(f"从 {LANE_CKPT} 迁移 lane_encoder 权重 {n_lane} 个张量")

    missing, unexpected = model.load_state_dict(merged, strict=False)
    print(f"missing keys: {missing}")
    print(f"unexpected keys: {unexpected}")
    allowed = {"lane_pooling.query", "lane_gate"}
    assert set(missing) <= allowed, f"意外 missing: {missing}"
    assert len(unexpected) == 0, f"unexpected: {unexpected}"

    # ── 冻结 ──
    trainable = freeze_except_lane_gate(model)
    model = model.to(device)

    # ── 训练前 sanity：gate=0 时弯道应 ≈ 仅轨迹 1.4624（迁移正确性）──
    test_ds = ArgoverseTrajectoryDataset(
        "data/processed/test.npz", lane_graphs_path="data/processed/lane_graphs.npz")
    test_loader = create_dataloader(test_ds, batch_size=64, shuffle=False, num_workers=1)
    r0 = evaluate_curve(model, test_loader, device)
    print(f"\n[训练前 sanity] gate=0 弯道: {r0['弯道加权']:.4f}（应 ≈ 仅轨迹 {BASELINE_CURVE_TRAJ:.4f}）")

    # ── 训练配置 ──
    cfg["training"]["epochs"] = args.epochs
    cfg["training"]["warmup_epochs"] = 1
    cfg["training"]["lr"] = args.lr
    cfg["logging"]["log_dir"] = "outputs/logs/verify_lane_gate_v2"
    cfg["logging"]["checkpoint_dir"] = "outputs/checkpoints/verify_lane_gate_v2"

    set_seed(42)
    logger = setup_logger(cfg["logging"]["log_dir"], "verify_lane_gate_v2")
    logger.info(f"可训练参数(车道线融合): {trainable:,}")

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        config=cfg, logger=logger, tb_writer=None,
    )
    result = trainer.train()

    # ── 训练后：test 集评估弯道降幅 ──
    best_path = Path(cfg["logging"]["checkpoint_dir"]) / "best_model.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    model = model.to(device)
    gate_val = best["model_state_dict"]["lane_gate"].item()

    r = evaluate_curve(model, test_loader, device)
    new_curve = r["弯道加权"]
    drop_new = (BASELINE_CURVE_TRAJ - new_curve) / BASELINE_CURVE_TRAJ * 100

    print("\n" + "=" * 60)
    print("  test 集弯道降幅对比（干净 baseline）")
    print("=" * 60)
    print(f"  仅轨迹(基线) 弯道:   {BASELINE_CURVE_TRAJ:.4f}")
    print(f"  旧融合 +车道线 弯道:  {BASELINE_CURVE_LANE_OLD:.4f}  降幅 +1.7%")
    print(f"  新融合 +车道线 弯道:  {new_curve:.4f}  降幅 {drop_new:+.1f}%")
    print(f"  训练后 lane_gate 值:  {gate_val:.4f}")
    print(f"\n  各类别 minADE:")
    for cat in CATEGORY_NAMES.values():
        if cat in r:
            print(f"    {cat}: {r[cat]:.4f}  (n={r[f'{cat}_n']})")

    # 保存结果
    out = {
        "subset": n_sub, "epochs": args.epochs, "lr": args.lr,
        "baseline_curve_traj": BASELINE_CURVE_TRAJ,
        "baseline_curve_lane_old": BASELINE_CURVE_LANE_OLD,
        "sanity_curve_gate0": r0["弯道加权"],
        "new_curve": new_curve, "new_drop_pct": drop_new,
        "lane_gate": gate_val,
        "per_category": {k: v for k, v in r.items()},
        "best_val_min_ade": result["best_val_min_ade"],
    }
    out_path = Path("outputs/results/verify_lane_gate_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")

    elapsed = (time.time() - t0) / 60
    print(f"总耗时: {elapsed:.1f} 分钟")


if __name__ == "__main__":
    main()
