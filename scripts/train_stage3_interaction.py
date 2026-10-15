#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第三阶段：两车交互轨迹预测 — 迁移训练 + 两组对照消融。

两组消融（都在 AV+AGENT 子集、同一迁移起点、同一训练策略，唯一区别 use_interaction）：
    组 A（对照）use_interaction=False：公平的 ego 基线，隔离交互模块这一个变量
    组 B（交互）use_interaction=True：验证交互增益 + 邻居预测 + 碰撞惩罚

迁移起点：第二阶段最佳（+车道线+城市特征，minADE 0.6107）的 best_model.pt，
strict=False 加载，断言 missing 全为 interaction.*（仅组 B）、unexpected 为空。

注意：需先重建数据（python scripts/preprocess_real_data.py），使 train/val.npz 含
neighbor_histories/neighbor_futures/neighbor_mask 三个键；未重建则邻居全 0、
组 B 退化为单车（脚本会打印警告）。

用法:
    python scripts/train_stage3_interaction.py --epochs 5 --group both      # 快速验证
    python scripts/train_stage3_interaction.py --epochs 80 --group both     # 全量训练
"""

import sys
import os
import json
import time
import argparse
import traceback
import torch

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
from pathlib import Path

RESULTS_PATH = "outputs/results/stage3_interaction_results.json"

# 两组消融（名称, use_interaction, checkpoint 目录名）
GROUPS = [
    ("组A对照(无交互)", False, "s3_组A对照"),
    ("组B交互(两车)", True, "s3_组B交互"),
]


def load_existing_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(results):
    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _find_latest_checkpoint(ckpt_dir: str):
    """在 checkpoint 目录中查找最新 epoch 的 checkpoint，用于续跑。"""
    d = Path(ckpt_dir)
    if not d.exists():
        return None
    ckpts = sorted(
        d.glob("checkpoint_epoch_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    return str(ckpts[-1]) if ckpts else None


def _load_prev_best(ckpt_dir: str):
    """从 best_model.pt 读取历史最佳 minADE 与 epoch，供续跑时恢复。"""
    best_path = Path(ckpt_dir) / "best_model.pt"
    if not best_path.exists():
        return None, None
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics", {})
    return metrics.get("val_min_ade"), ckpt.get("epoch", 0)


def _migrate_from_stage2(model, stage2_ckpt: str, use_interaction: bool, logger, drop_uncertainty: bool = False):
    """从 stage2 权重迁移加载（strict=False），断言新增/缺失键符合预期。

    参数:
        use_interaction: 组 B 为 True，缺失键应全为 interaction.* / neighbor_predictor.*；组 A 为 False，缺失键应空
        drop_uncertainty: 方案二 uncertainty 共享化（结构不兼容）时丢弃旧 uncertainty 键，随机初始化
    """
    ckpt = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    if drop_uncertainty:
        n_total = len(state)
        state = {k: v for k, v in state.items() if not k.startswith("uncertainty.")}
        logger.info(f"丢弃 {n_total - len(state)} 个 uncertainty 键（共享化结构不兼容，随机初始化）")

    missing, unexpected = model.load_state_dict(state, strict=False)

    assert not unexpected, f"unexpected 键（stage3 没有而 stage2 有）: {unexpected}"
    if use_interaction:
        allowed = ("interaction.", "context_gate", "neighbor_predictor.", "neighbor_encoder.", "uncertainty.")
        assert all(k.startswith(allowed) for k in missing), (
            f"组 B 的 missing 键应全为 interaction.* / context_gate / neighbor_predictor.* / neighbor_encoder.* / uncertainty.*，实际: {missing}"
        )
        logger.info(f"迁移加载成功：新增 {len(missing)} 个交互/邻居参数（随机初始化）")
    else:
        assert len(missing) == 0, f"组 A 的 missing 键应空，实际: {missing}"
        logger.info("迁移加载成功：组 A 与 stage2 完全同构，无缺失键")


def run_one(name, use_interaction, exp_dir, epochs, stage2_ckpt, lr, lambda_collision=None, freeze_ego=False, patience=None, share_uncertainty=False):
    print(f"\n{'=' * 60}")
    print(f"  第三阶段: {name}")
    print(f"{'=' * 60}")

    cfg = config_to_dict(load_config("configs/default.yaml"))
    cfg["training"]["epochs"] = epochs
    cfg["training"]["lr"] = lr
    cfg["training"]["warmup_epochs"] = max(1, epochs // 10)
    cfg["model"]["use_interaction"] = use_interaction
    # 复用 stage2 的压缩 predictor + 独立头
    cfg["model"]["branch_hidden_dim"] = 96
    cfg["model"]["share_uncertainty"] = share_uncertainty
    if lambda_collision is not None:
        cfg["loss"]["lambda_collision"] = lambda_collision
    if patience is not None:
        cfg["early_stopping"]["patience"] = patience

    cfg["logging"]["log_dir"] = f"outputs/logs/{exp_dir}"
    cfg["logging"]["checkpoint_dir"] = f"outputs/checkpoints/{exp_dir}"

    set_seed(42)
    logger = setup_logger(cfg["logging"]["log_dir"], exp_dir)

    train_ds = ArgoverseTrajectoryDataset(
        "data/processed/train.npz",
        lane_graphs_path="data/processed/lane_graphs.npz",
    )
    val_ds = ArgoverseTrajectoryDataset(
        "data/processed/val.npz",
        lane_graphs_path="data/processed/lane_graphs.npz",
    )
    # 邻居数据自检：若 mask 全 0，说明未重建数据，组 B 会退化为单车
    has_neighbor = bool((train_ds.neighbor_mask.sum() > 0))
    if use_interaction and not has_neighbor:
        logger.warning("警告：train.npz 的 neighbor_mask 全 0，疑似未重建数据，"
                       "组 B 将退化为单车模型，请先运行 scripts/preprocess_real_data.py")
    elif use_interaction:
        nbr_ratio = float(train_ds.neighbor_mask[:, 0].mean())
        logger.info(f"最近交互车覆盖率(neighbor_mask[0]=1): {nbr_ratio*100:.1f}%")

    train_loader = create_dataloader(train_ds, batch_size=64, shuffle=True, num_workers=1)
    val_loader = create_dataloader(val_ds, batch_size=64, shuffle=False, num_workers=1)

    mc = cfg["model"]
    model = TrajectoryModel(
        input_dim=mc.get("input_dim", 11),
        hidden_dim=mc.get("hidden_dim", 128),
        num_mlp_layers=mc.get("num_mlp_layers", 3),
        dropout=mc.get("dropout", 0.1),
        num_modes=mc.get("num_modes", 5),
        pred_len=cfg["data"].get("pred_len", 30),
        pooling_type=mc.get("pooling", "attention"),
        use_uncertainty=mc.get("use_uncertainty", True),
        use_temporal_conv=mc.get("use_temporal_conv", True),
        share_uncertainty=mc.get("share_uncertainty", False),
        use_lane=mc.get("use_lane", True),
        use_city=mc.get("use_city", True),
        use_interaction=mc.get("use_interaction", False),
        branch_hidden_dim=mc.get("branch_hidden_dim", 96),
    )
    total_params, _ = count_parameters(model)
    logger.info(f"参数量: {total_params:,}")

    # 迁移加载 / 恢复（先于 trainer 构建：冻结会影响 optimizer 收集参数）
    resume_from = _find_latest_checkpoint(cfg["logging"]["checkpoint_dir"])
    prev_best_ade = prev_best_epoch = None
    if resume_from:
        prev_best_ade, prev_best_epoch = _load_prev_best(cfg["logging"]["checkpoint_dir"])
        logger.info(f"检测到 checkpoint，从 epoch {Path(resume_from).stem.rsplit('_', 1)[-1]} 续跑")
    else:
        _migrate_from_stage2(model, stage2_ckpt, use_interaction, logger, drop_uncertainty=share_uncertainty)

    # 冻结 ego 主干（可选）：只训 interaction + context_gate，
    # 从机制上避免 interaction 路径的梯度污染 encoder（wen_ti/04）
    if freeze_ego:
        frozen_prefixes = ("encoder.", "lane_encoder.", "city_embed.", "predictor.", "uncertainty.")
        n_frozen = 0
        for pname, p in model.named_parameters():
            if pname.startswith(frozen_prefixes):
                p.requires_grad = False
                n_frozen += 1
        logger.info(f"冻结 ego 主干：{n_frozen} 个参数，仅训 interaction + context_gate")

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        config=cfg, logger=logger, tb_writer=None,
    )

    # 续跑时恢复历史最佳指标
    if resume_from and prev_best_ade is not None:
        trainer.best_metric = prev_best_ade
        trainer.early_stopping.best_score = prev_best_ade
        trainer.early_stopping.best_epoch = prev_best_epoch

    result = trainer.train(resume_from=resume_from)

    return {
        "name": name,
        "use_interaction": use_interaction,
        "params": total_params,
        "best_val_min_ade": result["best_val_min_ade"],
        "best_epoch": result["best_epoch"] + 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5,
                        help="训练 epoch 数（子集验证用 5，全量用 80）")
    parser.add_argument("--group", choices=["A", "B", "both"], default="both",
                        help="训练哪一组：A 对照 / B 交互 / both 都跑")
    parser.add_argument("--stage2-ckpt", default="outputs/checkpoints/s2_加车道线加城市特征/best_model.pt",
                        help="stage2 迁移起点 checkpoint 路径")
    parser.add_argument("--lr", type=float, default=3e-4, help="迁移学习率（不复用 stage2 optimizer）")
    parser.add_argument("--tag", default="", help="实验目录后缀（用于不覆盖旧结果的变体训练，如 _gate）")
    parser.add_argument("--lambda-collision", type=float, default=None,
                        help="覆盖碰撞惩罚权重（如 0.02，wen_ti/04 第二步）")
    parser.add_argument("--freeze-ego", action="store_true",
                        help="冻结 ego 主干（encoder/lane/city/predictor/uncertainty），只训 interaction + context_gate")
    parser.add_argument("--patience", type=int, default=None,
                        help="覆盖早停 patience（如 20，让训练更充分）")
    parser.add_argument("--share-unc", action="store_true",
                        help="uncertainty 头共享化（省约 48K 给邻居独立预测头，方案二）")
    args = parser.parse_args()

    t0 = time.time()
    results = load_existing_results()
    done_names = {r["name"] for r in results}
    print(f"已有 {len(results)} 个结果: {done_names}")

    groups = [g for g in GROUPS if args.group == "both" or (g[0].startswith("组" + args.group))]
    for name, use_interaction, exp_dir in groups:
        if args.tag:
            name = name + args.tag
            exp_dir = exp_dir + args.tag
        if name in done_names:
            print(f"跳过已完成: {name}")
            continue
        try:
            r = run_one(name, use_interaction, exp_dir, args.epochs, args.stage2_ckpt, args.lr, args.lambda_collision, args.freeze_ego, args.patience, args.share_unc)
            results.append(r)
            save_results(results)
            elapsed = (time.time() - t0) / 60
            print(f"  => {name}: minADE={r['best_val_min_ade']:.4f}, "
                  f"params={r['params']:,} | 耗时 {elapsed:.0f}分钟")
        except Exception as e:
            print(f"  => {name}: 失败 - {e}")
            traceback.print_exc()

    elapsed = (time.time() - t0) / 60
    print(f"\n第三阶段训练完成! 耗时 {elapsed:.0f}分钟")
    for r in results:
        print(f"  {r['name']}: minADE={r['best_val_min_ade']:.4f}, params={r['params']:,}")


if __name__ == "__main__":
    main()
